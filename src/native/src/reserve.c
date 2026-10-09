/* reserve.c - closed-form reservation arms.
 *
 * Numerical approach for the harmonic arm: the signal is the 288-point
 * per-slot mean m[s].  Instead of a general FFT we project directly onto the
 * K+1 = 5 low harmonics (a direct DFT of those bins only):
 *     a_k = sum_s m[s] cos(2 pi k s / N)    b_k = sum_s m[s] sin(2 pi k s / N)
 *     r[s] = a_0/N + sum_{k=1..K} (2/N) (a_k cos(2 pi k s/N) + b_k sin(2 pi k s/N))
 * which is exactly irfft(rfft(m) with bins above K zeroed) because N is even
 * and K < N/2, so there is no Nyquist term.  That costs 2*(K+1)*N MACs,
 * less than a radix-2 FFT would for five bins, and N = 288 is not a power of
 * two anyway.  The twiddles are looked up as T[(k*s) mod N] from one
 * cos/sin table of N entries, so the angle is reduced exactly in integers
 * and no phase error grows with k*s.  Accumulation is in double.
 *
 * The reconstruction is not clamped at zero; a harmonic undershoot can go
 * negative and the consumer decides what that means.
 */
#include <math.h>
#include <string.h>

#include "sfn.h"

#define N SFN_SLOTS
#define K SFN_HARMONICS

void sfn_trig_init(sfn_trig *t)
{
    const double two_pi = 6.283185307179586476925286766559;
    for (int m = 0; m < N; m++) {
        t->c[m] = cos(two_pi * (double)m / (double)N);
        t->s[m] = sin(two_pi * (double)m / (double)N);
    }
}

void sfn_reserve_peak(const float *hist, uint32_t days, float *out)
{
    for (int c = 0; c < SFN_CH; c++) {
        /* elementwise max across days first (vectorises), then one scalar pass */
        float m[N];
        memcpy(m, hist + (size_t)c * N, sizeof m);
        for (uint32_t d = 1; d < days; d++) {
            const float *row = hist + ((size_t)d * SFN_CH + (size_t)c) * N;
            for (int s = 0; s < N; s++)
                m[s] = row[s] > m[s] ? row[s] : m[s];
        }
        float peak = m[0];
        for (int s = 1; s < N; s++)
            if (m[s] > peak)
                peak = m[s];
        for (int s = 0; s < N; s++)
            out[c * N + s] = peak;
    }
}

/* Per-slot mean and population std for one channel, in double. */
static void slot_stats(const float *hist, uint32_t days, int c, double *mean,
                       double *sd)
{
    const double inv = 1.0 / (double)days;
    for (int s = 0; s < N; s++)
        mean[s] = 0.0;
    for (uint32_t d = 0; d < days; d++) {
        const float *row = hist + ((size_t)d * SFN_CH + (size_t)c) * N;
        for (int s = 0; s < N; s++)
            mean[s] += (double)row[s];
    }
    for (int s = 0; s < N; s++) {
        mean[s] *= inv;
        sd[s] = 0.0;
    }
    for (uint32_t d = 0; d < days; d++) {
        const float *row = hist + ((size_t)d * SFN_CH + (size_t)c) * N;
        for (int s = 0; s < N; s++) {
            double e = (double)row[s] - mean[s];
            sd[s] += e * e;
        }
    }
    for (int s = 0; s < N; s++)
        sd[s] = sqrt(sd[s] * inv);
}

void sfn_reserve_slot(const float *hist, uint32_t days, float z, float *out)
{
    double mean[N], sd[N];
    for (int c = 0; c < SFN_CH; c++) {
        slot_stats(hist, days, c, mean, sd);
        for (int s = 0; s < N; s++)
            out[c * N + s] = (float)(mean[s] + (double)z * sd[s]);
    }
}

void sfn_reserve_harmonic(const sfn_trig *t, const float *hist, uint32_t days,
                          float z, float *out)
{
    double mean[N], sd[N];
    for (int c = 0; c < SFN_CH; c++) {
        double a[K + 1], b[K + 1];
        slot_stats(hist, days, c, mean, sd);
        for (int k = 0; k <= K; k++) {
            double ak = 0.0, bk = 0.0;
            unsigned idx = 0;
            for (int s = 0; s < N; s++) {
                ak += mean[s] * t->c[idx];
                bk += mean[s] * t->s[idx];
                idx += (unsigned)k;
                if (idx >= N)
                    idx -= N;
            }
            a[k] = ak;
            b[k] = bk;
        }
        unsigned ix[K + 1] = {0};      /* ix[k] = (k*s) mod N, advanced without a divide */
        for (int s = 0; s < N; s++) {
            double r = a[0] / (double)N;
            for (int k = 1; k <= K; k++) {
                r += (2.0 / (double)N) * (a[k] * t->c[ix[k]] + b[k] * t->s[ix[k]]);
                ix[k] += (unsigned)k;
                if (ix[k] >= N)
                    ix[k] -= N;
            }
            out[c * N + s] = (float)(r + (double)z * sd[s]);
        }
    }
}

void sfn_reserve_batch(sfn_arm arm, const sfn_trig *t, const float *hist,
                       uint32_t days, float z, uint32_t n, float *out)
{
    const size_t in_stride = (size_t)days * SFN_CH * N;
    const size_t out_stride = (size_t)SFN_CH * N;
    for (uint32_t i = 0; i < n; i++) {
        const float *h = hist + i * in_stride;
        float *o = out + i * out_stride;
        switch (arm) {
        case SFN_ARM_PEAK:
            sfn_reserve_peak(h, days, o);
            break;
        case SFN_ARM_SLOT:
            sfn_reserve_slot(h, days, z, o);
            break;
        case SFN_ARM_HARMONIC:
            sfn_reserve_harmonic(t, h, days, z, o);
            break;
        }
    }
}
