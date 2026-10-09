/* bench.c - latency harness for one kernel at one batch size.
 *
 *   bench --impl NAME --batch 1|1000 [--calls N] [--warmup W] [--cpu C]
 *   bench --list
 *
 * Prints one JSON object.  Time is CLOCK_MONOTONIC around each call; the cost
 * of the clock itself is measured and reported as timer_floor_ns and is NOT
 * subtracted.  Inputs rotate through a pool of POOL distinct tasks so a
 * repeated call is not served from one hot cache line.  Batch 1000 times the
 * whole pool in a single call.  bench/run_bench.py runs this once per
 * (impl, batch) in a fresh process so RSS is attributable.
 */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sfn.h"

#define POOL 1000
#define DAYS 3
#define D_IN 8
#define D_OUT 2

static volatile float sink;

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static float urand(void)               /* xorshift64*, uniform in [0, 1) */
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)((rng_state * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

/* ---- case state ---- */
static float *hist, *out;              /* reservation: POOL tasks */
static sfn_trig trig;
static sfn_ssm model;
static void *image;
static float *H, *X, *Y, *gain, *scratch, *Xseq, *Yseq;
static float *S, *Q, *Kv, *V;
static uint32_t n_state, flags_cfg, dk;

typedef struct {
    const char *name;
    int max_batch;
    void (*setup)(void);
    void (*run)(uint32_t i, int batch);
} bench_case;

/* ---- reservation arms ---- */
static void setup_reserve(void)
{
    size_t per = (size_t)DAYS * SFN_CH * SFN_SLOTS;
    hist = malloc(POOL * per * sizeof(float));
    out = malloc((size_t)POOL * SFN_CH * SFN_SLOTS * sizeof(float));
    for (size_t i = 0; i < POOL * per; i++)
        hist[i] = urand();
    sfn_trig_init(&trig);
}
#define RESERVE_RUN(fname, ARM, ONE)                                               \
    static void fname(uint32_t i, int batch)                                       \
    {                                                                              \
        const size_t per = (size_t)DAYS * SFN_CH * SFN_SLOTS;                      \
        if (batch == 1) {                                                          \
            uint32_t j = i % POOL;                                                 \
            float *o = out + (size_t)j * SFN_CH * SFN_SLOTS;                       \
            const float *h = hist + j * per;                                       \
            ONE;                                                                   \
        } else                                                                     \
            sfn_reserve_batch(ARM, &trig, hist, DAYS, 3.0f, POOL, out);            \
        sink = out[0];                                                             \
    }
RESERVE_RUN(run_peak, SFN_ARM_PEAK, sfn_reserve_peak(h, DAYS, o))
RESERVE_RUN(run_slot, SFN_ARM_SLOT, sfn_reserve_slot(h, DAYS, 3.0f, o))
RESERVE_RUN(run_harm, SFN_ARM_HARMONIC, sfn_reserve_harmonic(&trig, h, DAYS, 3.0f, o))

/* ---- linear recurrence ---- */
static void setup_ssm_common(uint32_t n, uint32_t flags)
{
    n_state = n;
    flags_cfg = flags;
    size_t bytes = sfn_ssm_file_size(n, D_IN, D_OUT, flags);
    image = malloc(bytes);
    uint32_t hdr[8] = {0x534e4653u /* "SFNS" */, 1, n, D_IN, D_OUT, flags, 0, 0};
    memcpy(image, hdr, 32);
    float *p = (float *)((char *)image + 32);
    size_t na = (flags & SFN_SSM_DIAG) ? n : (size_t)n * n;
    /* dense: entries in +-0.9*sqrt(3/n) give spectral radius about 0.9 */
    for (size_t i = 0; i < na; i++)
        p[i] = (flags & SFN_SSM_DIAG) ? 0.5f + 0.48f * urand()
                                       : (2.0f * urand() - 1.0f) * 0.9f * 1.7320508f / (float)__builtin_sqrtf((float)n);
    p += na;
    for (size_t i = 0; i < (size_t)n * D_IN + (size_t)D_OUT * n; i++)
        p[i] = (2.0f * urand() - 1.0f) * 0.3f;
    if (sfn_ssm_from_buffer(&model, image, bytes) != 0) {
        fprintf(stderr, "bad image\n");
        exit(2);
    }
    H = calloc((size_t)POOL * n, sizeof(float));
    X = malloc((size_t)POOL * D_IN * sizeof(float));
    Y = malloc((size_t)POOL * D_OUT * sizeof(float));
    gain = malloc((size_t)n * sizeof(float));
    scratch = malloc(sfn_ssm_scratch_floats(&model, POOL) * sizeof(float));
    Xseq = malloc(288 * D_IN * sizeof(float));
    Yseq = malloc(288 * D_OUT * sizeof(float));
    for (size_t i = 0; i < (size_t)POOL * D_IN; i++)
        X[i] = 2.0f * urand() - 1.0f;
    for (size_t i = 0; i < 288 * D_IN; i++)
        Xseq[i] = 2.0f * urand() - 1.0f;
    for (uint32_t i = 0; i < n; i++)
        gain[i] = 0.5f + 0.48f * urand();
}
static void setup_dense64(void) { setup_ssm_common(64, 0); }
static void setup_dense128(void) { setup_ssm_common(128, 0); }
static void setup_diag64(void) { setup_ssm_common(64, SFN_SSM_DIAG); }
static void setup_diag128(void) { setup_ssm_common(128, SFN_SSM_DIAG); }

static void run_ssm_step(uint32_t i, int batch)
{
    if (batch == 1) {
        uint32_t j = i % POOL;
        sfn_ssm_step(&model, H + (size_t)j * n_state, X + (size_t)j * D_IN,
                     Y + (size_t)j * D_OUT, scratch);
    } else
        sfn_ssm_step_batch(&model, H, X, Y, POOL, scratch);
    sink = Y[0];
}
static void run_ssm_gain(uint32_t i, int batch)
{
    if (batch == 1) {
        uint32_t j = i % POOL;
        sfn_ssm_step_gain(&model, H + (size_t)j * n_state, X + (size_t)j * D_IN, gain,
                          Y + (size_t)j * D_OUT);
    } else
        for (uint32_t j = 0; j < POOL; j++)
            sfn_ssm_step_gain(&model, H + (size_t)j * n_state, X + (size_t)j * D_IN, gain,
                              Y + (size_t)j * D_OUT);
    sink = Y[0];
}
static void run_ssm_scan(uint32_t i, int batch)
{
    (void)batch;                       /* batch 1 only: 288 steps x 1000 states is hours */
    uint32_t j = i % POOL;
    sfn_ssm_scan(&model, H + (size_t)j * n_state, Xseq, Yseq, 288, scratch);
    sink = Yseq[0];
}

/* ---- DeltaNet delta rule, dk = dv = 64 ---- */
static void setup_delta(void)
{
    dk = 64;
    S = calloc((size_t)POOL * dk * dk, sizeof(float));
    Q = malloc((size_t)POOL * dk * sizeof(float));
    Kv = malloc((size_t)POOL * dk * sizeof(float));
    V = malloc((size_t)POOL * dk * sizeof(float));
    Y = malloc((size_t)POOL * dk * sizeof(float));
    scratch = malloc(dk * sizeof(float));
    for (size_t j = 0; j < POOL; j++) {
        float nrm = 0.0f;
        for (uint32_t i = 0; i < dk; i++) {
            Q[j * dk + i] = 2.0f * urand() - 1.0f;
            Kv[j * dk + i] = 2.0f * urand() - 1.0f;
            V[j * dk + i] = 2.0f * urand() - 1.0f;
            nrm += Kv[j * dk + i] * Kv[j * dk + i];
        }
        nrm = 1.0f / __builtin_sqrtf(nrm);
        for (uint32_t i = 0; i < dk; i++)
            Kv[j * dk + i] *= nrm;
    }
}
static void run_delta(uint32_t i, int batch)
{
    uint32_t lo = batch == 1 ? i % POOL : 0, hi = batch == 1 ? lo + 1 : POOL;
    for (uint32_t j = lo; j < hi; j++)
        sfn_delta_step(S + (size_t)j * dk * dk, dk, dk, Q + j * dk, Kv + j * dk,
                       V + j * dk, 0.5f, Y + j * dk, scratch);
    sink = Y[lo * dk];
}

static const bench_case CASES[] = {
    {"reserve_peak", 1000, setup_reserve, run_peak},
    {"reserve_slot", 1000, setup_reserve, run_slot},
    {"reserve_harmonic_k4", 1000, setup_reserve, run_harm},
    {"ssm_dense_n64_step", 1000, setup_dense64, run_ssm_step},
    {"ssm_dense_n128_step", 1000, setup_dense128, run_ssm_step},
    {"ssm_diag_n64_step", 1000, setup_diag64, run_ssm_step},
    {"ssm_diag_n128_step", 1000, setup_diag128, run_ssm_step},
    {"ssm_gain_n64_step", 1000, setup_diag64, run_ssm_gain},
    {"ssm_dense_n64_scan288", 1, setup_dense64, run_ssm_scan},
    {"ssm_diag_n64_scan288", 1, setup_diag64, run_ssm_scan},
    {"deltanet_64x64_step", 1000, setup_delta, run_delta},
};
#define NCASES (sizeof CASES / sizeof *CASES)

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static long proc_kb(const char *key)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long v = -1;
    size_t kl = strlen(key);
    while (f && fgets(line, sizeof line, f))
        if (strncmp(line, key, kl) == 0)
            v = strtol(line + kl, NULL, 10);
    if (f)
        fclose(f);
    return v;
}

static uint64_t pct(const uint64_t *s, size_t n, double p)
{
    size_t r = (size_t)(p * (double)n + 0.999999999);   /* nearest rank, ceil */
    return s[(r < 1 ? 1 : r) - 1];
}

int main(int argc, char **argv)
{
    const char *impl = NULL;
    int batch = 1, cpu = -1;
    size_t calls = 0, warmup = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--list")) {
            for (size_t c = 0; c < NCASES; c++)
                printf("%s %d\n", CASES[c].name, CASES[c].max_batch);
            return 0;
        }
        if (i + 1 >= argc)
            break;
        if (!strcmp(argv[i], "--impl")) impl = argv[++i];
        else if (!strcmp(argv[i], "--batch")) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--calls")) calls = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")) warmup = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) cpu = atoi(argv[++i]);
    }
    const bench_case *bc = NULL;
    for (size_t c = 0; impl && c < NCASES; c++)
        if (!strcmp(CASES[c].name, impl))
            bc = &CASES[c];
    if (!bc || (batch != 1 && batch != 1000) || batch > bc->max_batch) {
        fprintf(stderr, "usage: bench --impl NAME --batch 1|1000 [--calls N] [--warmup W] [--cpu C] | --list\n");
        return 2;
    }
    if (!calls) calls = batch == 1 ? 20000 : 10000;
    if (!warmup) warmup = batch == 1 ? 5000 : 200;
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"   /* glibc CPU_SET macro */
        CPU_SET(cpu, &set);
#pragma GCC diagnostic pop
        if (sched_setaffinity(0, sizeof set, &set) != 0)
            perror("sched_setaffinity");
    }

    bc->setup();
    uint64_t *samples = malloc(calls * sizeof *samples);
    long rss_start = proc_kb("VmRSS:");

    for (size_t i = 0; i < warmup; i++)
        bc->run((uint32_t)i, batch);
    for (size_t i = 0; i < calls; i++) {
        uint64_t t0 = now_ns();
        bc->run((uint32_t)i, batch);
        uint64_t t1 = now_ns();
        samples[i] = t1 - t0;
    }
    long rss_end = proc_kb("VmRSS:"), hwm = proc_kb("VmHWM:");

    uint64_t floor_s[2001];
    for (int i = 0; i < 2001; i++) {
        uint64_t t0 = now_ns();
        floor_s[i] = now_ns() - t0;
    }
    qsort(floor_s, 2001, sizeof *floor_s, cmp_u64);

    double sum = 0.0;
    for (size_t i = 0; i < calls; i++)
        sum += (double)samples[i];
    qsort(samples, calls, sizeof *samples, cmp_u64);
    printf("{\"impl\":\"%s\",\"batch\":%d,\"calls\":%zu,\"warmup\":%zu,\"pool\":%d,"
           "\"p50_ns\":%llu,\"p99_ns\":%llu,\"p999_ns\":%llu,\"min_ns\":%llu,"
           "\"max_ns\":%llu,\"mean_ns\":%.1f,\"timer_floor_ns\":%llu,"
           "\"rss_kb_before\":%ld,\"rss_kb_after\":%ld,\"hwm_kb\":%ld,\"sink\":%g}\n",
           bc->name, batch, calls, warmup, POOL,
           (unsigned long long)pct(samples, calls, 0.50),
           (unsigned long long)pct(samples, calls, 0.99),
           (unsigned long long)pct(samples, calls, 0.999),
           (unsigned long long)samples[0], (unsigned long long)samples[calls - 1],
           sum / (double)calls, (unsigned long long)floor_s[1000], rss_start, rss_end,
           hwm, (double)sink);
    return 0;
}
