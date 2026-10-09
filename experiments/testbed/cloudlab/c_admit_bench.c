/* c_admit_bench.c - per-call latency of the src/native/ C reservation kernels.
 *
 * Consumes src/native/build/libsfn.a and src/native/include/sfn.h as build output; it
 * does not modify src/native/.  One call = one task's reservation from a
 * (days, 2, 288) history, which is the admission-path input.  Inputs rotate
 * through a pool of distinct tasks so calls are not served from one hot line.
 * Time is CLOCK_MONOTONIC around each call; the clock-read cost is measured
 * separately and NOT subtracted.  Output: one JSON object on stdout.
 *
 *   c_admit_bench --calls 200000 --days 3 [--cpu N] [--warmup 20000]
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
static volatile float sink;

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static float urand(void)
{
    rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
    return (float)((rs * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}
static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
static uint64_t pct(const uint64_t *s, size_t n, double p) /* nearest rank */
{
    size_t k = (size_t)((p / 100.0) * (double)n + 0.999999999);
    if (k < 1) k = 1;
    if (k > n) k = n;
    return s[k - 1];
}
static long rss_kb(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    char l[256]; long v = -1;
    if (!f) return -1;
    while (fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS:", 6)) { v = atol(l + 6); break; }
    fclose(f);
    return v;
}

static int first = 1;
static void report(const char *name, uint64_t *t, size_t n, double tot_s)
{
    qsort(t, n, sizeof *t, cmp_u64);
    double sum = 0; for (size_t i = 0; i < n; i++) sum += (double)t[i];
    printf("%s\"%s\":{\"calls\":%zu,\"p50_ns\":%llu,\"p99_ns\":%llu,\"p999_ns\":%llu,"
           "\"max_ns\":%llu,\"mean_ns\":%.1f,\"p999_samples_above\":%zu,\"loop_wall_s\":%.3f}",
           first ? "" : ",", name, n, (unsigned long long)pct(t, n, 50),
           (unsigned long long)pct(t, n, 99), (unsigned long long)pct(t, n, 99.9),
           (unsigned long long)t[n - 1], sum / (double)n, n - (size_t)(0.999 * (double)n), tot_s);
    first = 0;
}

int main(int argc, char **argv)
{
    size_t calls = 200000, warm = 20000; unsigned days = 3; int cpu = -1;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--calls")) calls = (size_t)atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--warmup")) warm = (size_t)atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--days")) days = (unsigned)atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--cpu")) cpu = atoi(argv[i + 1]);
    }
    if (cpu >= 0) {
        cpu_set_t m; CPU_ZERO(&m); CPU_SET(cpu, &m);
        if (sched_setaffinity(0, sizeof m, &m)) perror("sched_setaffinity");
    }
    size_t per = (size_t)days * SFN_CH * SFN_SLOTS;
    float *hist = malloc(POOL * per * sizeof(float));
    float *out = malloc((size_t)POOL * SFN_CH * SFN_SLOTS * sizeof(float));
    uint64_t *t = malloc(calls * sizeof *t);
    for (size_t i = 0; i < POOL * per; i++) hist[i] = urand();
    sfn_trig trig; sfn_trig_init(&trig);

    /* timer floor: back-to-back clock reads */
    for (size_t i = 0; i < calls; i++) { uint64_t a = now_ns(), b = now_ns(); t[i] = b - a; }

    printf("{\"days\":%u,\"pool\":%d,\"cpu\":%d,", days, POOL, cpu);
    printf("\"arms\":{");
    uint64_t floor_p50 = 0;
    { qsort(t, calls, sizeof *t, cmp_u64); floor_p50 = pct(t, calls, 50); }

    const char *names[3] = {"reserve_peak", "reserve_slot", "reserve_harmonic_k4"};
    for (int arm = 0; arm < 3; arm++) {
        for (size_t i = 0; i < warm; i++) {
            size_t j = i % POOL; const float *h = hist + j * per; float *o = out + j * SFN_CH * SFN_SLOTS;
            if (arm == 0) sfn_reserve_peak(h, days, o);
            else if (arm == 1) sfn_reserve_slot(h, days, 3.0f, o);
            else sfn_reserve_harmonic(&trig, h, days, 3.0f, o);
        }
        uint64_t w0 = now_ns();
        for (size_t i = 0; i < calls; i++) {
            size_t j = i % POOL; const float *h = hist + j * per; float *o = out + j * SFN_CH * SFN_SLOTS;
            uint64_t a = now_ns();
            if (arm == 0) sfn_reserve_peak(h, days, o);
            else if (arm == 1) sfn_reserve_slot(h, days, 3.0f, o);
            else sfn_reserve_harmonic(&trig, h, days, 3.0f, o);
            uint64_t b = now_ns();
            t[i] = b - a; sink = o[0];
        }
        report(names[arm], t, calls, (double)(now_ns() - w0) / 1e9);
    }
    printf("},\"timer_floor_p50_ns\":%llu,\"rss_kb\":%ld}\n", (unsigned long long)floor_p50, rss_kb());
    return 0;
}
