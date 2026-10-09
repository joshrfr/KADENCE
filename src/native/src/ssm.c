/* ssm.c - linear-recurrent inference kernels (see sfn.h for the contract).
 *
 * Plain loops are the reference path.  Define SFN_USE_CBLAS to route the
 * batched step through cblas_sgemm; it is off by default because the build
 * host has no CBLAS, so that path is compiled only where one is installed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sfn.h"

#if defined(SFN_USE_CBLAS)
#include <cblas.h>
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "weight files are little-endian float32; add a byte swap for this target"
#endif

#define HDR_BYTES 32u

size_t sfn_ssm_sizeof(void) { return sizeof(sfn_ssm); }

size_t sfn_ssm_file_size(uint32_t n, uint32_t d_in, uint32_t d_out, uint32_t flags)
{
    size_t a = (flags & SFN_SSM_DIAG) ? n : (size_t)n * n;
    size_t f = a + (size_t)n * d_in + (size_t)d_out * n;
    if (flags & SFN_SSM_HAS_D)
        f += (size_t)d_out * d_in;
    return HDR_BYTES + f * sizeof(float);
}

int sfn_ssm_from_buffer(sfn_ssm *m, const void *buf, size_t len)
{
    const uint32_t *h = (const uint32_t *)buf;
    if (len < HDR_BYTES || ((uintptr_t)buf & 3u))
        return -1;
    if (memcmp(buf, "SFNS", 4) != 0 || h[1] != 1u)
        return -2;
    if (h[2] == 0 || h[3] == 0 || h[4] == 0 || (h[5] & ~3u))
        return -3;
    if (len != sfn_ssm_file_size(h[2], h[3], h[4], h[5]))
        return -4;
    m->n = h[2];
    m->d_in = h[3];
    m->d_out = h[4];
    m->flags = h[5];
    m->owned = NULL;
    const float *p = (const float *)((const char *)buf + HDR_BYTES);
    m->A = p;
    p += (m->flags & SFN_SSM_DIAG) ? m->n : (size_t)m->n * m->n;
    m->B = p;
    p += (size_t)m->n * m->d_in;
    m->C = p;
    p += (size_t)m->d_out * m->n;
    m->D = (m->flags & SFN_SSM_HAS_D) ? p : NULL;
    return 0;
}

int sfn_ssm_load(sfn_ssm *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -10;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -11; }
    long len = ftell(f);
    if (len < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -11; }
    void *buf = malloc((size_t)len);   /* malloc returns 16-byte alignment */
    if (!buf) { fclose(f); return -12; }
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if (got != (size_t)len) { free(buf); return -13; }
    int rc = sfn_ssm_from_buffer(m, buf, (size_t)len);
    if (rc != 0) { free(buf); return rc; }
    m->owned = buf;
    return 0;
}

void sfn_ssm_free(sfn_ssm *m)
{
    free(m->owned);
    m->owned = NULL;
}

size_t sfn_ssm_scratch_floats(const sfn_ssm *m, uint32_t nb)
{
#if defined(SFN_USE_CBLAS)
    return (size_t)nb * m->n;
#else
    (void)nb;
    return m->n;
#endif
}

/* Dot product with 8 independent partial sums.  A single running sum is a
 * serial add chain that the compiler may not reorder without -ffast-math; the
 * fixed 8-lane order is explicit, deterministic, and vectorises on plain SSE2
 * with no -march flag, so the binary stays portable.  The summation order
 * differs from numpy's, which is inside the gate tolerance. */
static float dot(const float *restrict a, const float *restrict b, uint32_t n)
{
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t j = 0;
    for (; j + 8 <= n; j += 8)
        for (uint32_t l = 0; l < 8; l++)
            acc[l] += a[j + l] * b[j + l];
    float tail = 0.0f;
    for (; j < n; j++)
        tail += a[j] * b[j];
    return ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail;
}

void sfn_matvec(const float *W, uint32_t rows, uint32_t cols, const float *x, float *y)
{
    for (uint32_t i = 0; i < rows; i++)
        y[i] = dot(W + (size_t)i * cols, x, cols);
}

/* y = C h (+ D x) */
static void readout(const sfn_ssm *m, const float *h, const float *x, float *y)
{
    sfn_matvec(m->C, m->d_out, m->n, h, y);
    if (m->D)
        for (uint32_t i = 0; i < m->d_out; i++) {
            y[i] += dot(m->D + (size_t)i * m->d_in, x, m->d_in);
        }
}

void sfn_ssm_step(const sfn_ssm *m, float *h, const float *x, float *y, float *scratch)
{
    const uint32_t n = m->n;
    if (m->flags & SFN_SSM_DIAG) {
        for (uint32_t i = 0; i < n; i++) {
            const float *b = m->B + (size_t)i * m->d_in;
            float acc = m->A[i] * h[i];
            for (uint32_t j = 0; j < m->d_in; j++)
                acc += b[j] * x[j];
            h[i] = acc;
        }
    } else {
        /* h_new needs the old h throughout, hence the scratch copy. */
        for (uint32_t i = 0; i < n; i++) {
            scratch[i] = dot(m->A + (size_t)i * n, h, n) +
                         dot(m->B + (size_t)i * m->d_in, x, m->d_in);
        }
        memcpy(h, scratch, (size_t)n * sizeof(float));
    }
    if (y)
        readout(m, h, x, y);
}

void sfn_ssm_scan(const sfn_ssm *m, float *h, const float *X, float *Y, uint32_t T,
                  float *scratch)
{
    for (uint32_t t = 0; t < T; t++)
        sfn_ssm_step(m, h, X + (size_t)t * m->d_in,
                     Y ? Y + (size_t)t * m->d_out : NULL, scratch);
}

void sfn_ssm_step_batch(const sfn_ssm *m, float *H, const float *X, float *Y,
                        uint32_t nb, float *scratch)
{
#if defined(SFN_USE_CBLAS)
    if (!(m->flags & SFN_SSM_DIAG)) {
        const int n = (int)m->n, di = (int)m->d_in, dout = (int)m->d_out, B = (int)nb;
        /* scratch = X B^T ; H' = H A^T + scratch (row-major, one state per row) */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, n, di, 1.0f, X, di,
                    m->B, di, 0.0f, scratch, n);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, n, n, 1.0f, H, n,
                    m->A, n, 1.0f, scratch, n);
        memcpy(H, scratch, (size_t)nb * m->n * sizeof(float));
        if (Y) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, dout, n, 1.0f, H,
                        n, m->C, n, 0.0f, Y, dout);
            if (m->D)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, dout, di,
                            1.0f, X, di, m->D, di, 1.0f, Y, dout);
        }
        return;
    }
#endif
    for (uint32_t i = 0; i < nb; i++)
        sfn_ssm_step(m, H + (size_t)i * m->n, X + (size_t)i * m->d_in,
                     Y ? Y + (size_t)i * m->d_out : NULL, scratch);
}

void sfn_ssm_step_gain(const sfn_ssm *m, float *h, const float *x, const float *gain,
                       float *y)
{
    for (uint32_t i = 0; i < m->n; i++) {
        const float *b = m->B + (size_t)i * m->d_in;
        float acc = gain[i] * h[i];
        for (uint32_t j = 0; j < m->d_in; j++)
            acc += b[j] * x[j];
        h[i] = acc;
    }
    if (y)
        readout(m, h, x, y);
}

void sfn_delta_step(float *restrict S, uint32_t dk, uint32_t dv, const float *restrict q,
                    const float *restrict k, const float *restrict v, float beta,
                    float *restrict y, float *restrict scratch)
{
    /* scratch = S^T k (what the memory currently returns for key k) */
    for (uint32_t j = 0; j < dv; j++)
        scratch[j] = 0.0f;
    for (uint32_t i = 0; i < dk; i++) {
        const float *row = S + (size_t)i * dv;
        const float ki = k[i];
        for (uint32_t j = 0; j < dv; j++)
            scratch[j] += ki * row[j];
    }
    for (uint32_t j = 0; j < dv; j++)
        scratch[j] = beta * (v[j] - scratch[j]);
    for (uint32_t j = 0; j < dv; j++)
        y[j] = 0.0f;
    for (uint32_t i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float ki = k[i], qi = q[i];
        for (uint32_t j = 0; j < dv; j++) {
            row[j] += ki * scratch[j];
            y[j] += qi * row[j];
        }
    }
}
