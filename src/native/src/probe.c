/* probe.c - one-kernel executables for measuring what linking a kernel costs.
 * Built once per PROBE_IMPL; bench/run_bench.py strips them and reports size,
 * the ldd closure and cold start.  PROBE_IMPL=0 is an empty main().  Each
 * probe makes one prediction and prints CLOCK_MONOTONIC (ns) right after it;
 * the harness subtracts its own spawn timestamp to get exec-to-first-prediction. */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "sfn.h"

int main(int argc, char **argv)
{
    static float hist[3 * SFN_CH * SFN_SLOTS], out[SFN_CH * SFN_SLOTS];
    static float S[64 * 64], q[64], k[64], v[64], y[64], scratch[64];
    static sfn_trig trig;
    (void)argv;
    if (argc > 99)              /* never true; keeps the calls from folding away */
        return 1;
    for (size_t i = 0; i < sizeof hist / sizeof *hist; i++)
        hist[i] = (float)(i % 17) / 17.0f;
#if PROBE_IMPL == 1
    sfn_reserve_peak(hist, 3, out);
#elif PROBE_IMPL == 2
    sfn_reserve_slot(hist, 3, 3.0f, out);
#elif PROBE_IMPL == 3
    sfn_trig_init(&trig);
    sfn_reserve_harmonic(&trig, hist, 3, 3.0f, out);
#elif PROBE_IMPL == 4
    sfn_ssm m;
    float h[64] = {0}, x[8] = {1}, yy[2];
    if (sfn_ssm_load(&m, argc > 1 ? argv[1] : "w.sfns") != 0)
        return 2;
    sfn_ssm_step(&m, h, x, yy, scratch);
    sfn_ssm_free(&m);
#elif PROBE_IMPL == 5
    sfn_delta_step(S, 64, 64, q, k, v, 0.5f, y, scratch);
#endif
    (void)trig; (void)S; (void)q; (void)k; (void)v; (void)y; (void)scratch;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    printf("%llu %g\n", (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec,
           (double)out[0]);
    return 0;
}
