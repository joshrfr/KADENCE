/* sfn.h - native reservation and recurrence kernels for the scheduler.
 *
 * Two families, both allocation-free on the call path:
 *   - closed-form reservation arms (peak, slot mean + z*std, K=4 harmonic)
 *   - linear-recurrent (state-space / delta-rule) inference steps
 *
 * Nothing here is claimed to be faster than anything until bench/ says so in a
 * configuration where inference sits on the scheduling path.  Every function
 * has a numpy twin in tests/reference.py and tests/gate.py must pass.
 *
 * Memory is caller-owned.  The only allocations are in sfn_ssm_load(), which
 * is an init-time call; sfn_ssm_from_buffer() never allocates.
 */
#ifndef SFN_H
#define SFN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SFN_SLOTS 288      /* five-minute slots per day */
#define SFN_CH 2           /* channels: cpu, memory */
#define SFN_HARMONICS 4    /* K of the harmonic arm */

/* ---- reservation arms -------------------------------------------------
 * History layout is row-major float32 (days, SFN_CH, SFN_SLOTS); the output
 * is (SFN_CH, SFN_SLOTS).  A batch is contiguous tasks, stride
 * days*SFN_CH*SFN_SLOTS in and SFN_CH*SFN_SLOTS out.  All arithmetic is in
 * double; only the final store rounds to float32.
 */

typedef enum { SFN_ARM_PEAK = 0, SFN_ARM_SLOT = 1, SFN_ARM_HARMONIC = 2 } sfn_arm;

/* Twiddle table for the harmonic arm: cos/sin(2*pi*m/SFN_SLOTS), m in
 * [0, SFN_SLOTS).  Caller-owned; fill once with sfn_trig_init(). */
typedef struct { double c[SFN_SLOTS]; double s[SFN_SLOTS]; } sfn_trig;
void sfn_trig_init(sfn_trig *t);

/* Per-task peak over every day and slot, one value per channel. */
void sfn_reserve_peak(const float *hist, uint32_t days, float *out);

/* mean_d + z * std_d per slot (population std, ddof = 0). */
void sfn_reserve_slot(const float *hist, uint32_t days, float z, float *out);

/* K=4 harmonic reconstruction of the per-slot mean, plus z * per-slot std. */
void sfn_reserve_harmonic(const sfn_trig *t, const float *hist, uint32_t days,
                          float z, float *out);

/* Dispatch over a contiguous batch of n tasks.  trig may be NULL unless
 * arm == SFN_ARM_HARMONIC. */
void sfn_reserve_batch(sfn_arm arm, const sfn_trig *t, const float *hist,
                       uint32_t days, float z, uint32_t n, float *out);

/* ---- linear recurrence -------------------------------------------------
 *   h_t = A h_{t-1} + B x_t        y_t = C h_t (+ D x_t)
 * A is dense (n x n) or diagonal (n).  Weights are float32 row-major.
 *
 * File format (little-endian, 32-byte header, then float32 arrays):
 *   char[4] "SFNS"; u32 version(=1); u32 n; u32 d_in; u32 d_out; u32 flags;
 *   u32 0; u32 0;  then A (n*n, or n if SFN_SSM_DIAG), B (n*d_in),
 *   C (d_out*n), and D (d_out*d_in) only if SFN_SSM_HAS_D.
 */
#define SFN_SSM_DIAG 1u
#define SFN_SSM_HAS_D 2u

typedef struct {
    uint32_t n, d_in, d_out, flags;
    const float *A, *B, *C, *D;   /* D is NULL unless SFN_SSM_HAS_D */
    void *owned;                  /* non-NULL only when sfn_ssm_load() allocated */
} sfn_ssm;

size_t sfn_ssm_sizeof(void);
size_t sfn_ssm_file_size(uint32_t n, uint32_t d_in, uint32_t d_out, uint32_t flags);
/* Parse in place; buf must be 4-byte aligned and outlive m.  Returns 0 on
 * success, negative on a malformed or truncated image.  No allocation. */
int sfn_ssm_from_buffer(sfn_ssm *m, const void *buf, size_t len);
/* Init-time only: reads path into a malloc'd buffer. */
int sfn_ssm_load(sfn_ssm *m, const char *path);
void sfn_ssm_free(sfn_ssm *m);

/* Floats of scratch needed by the step functions for nb parallel states. */
size_t sfn_ssm_scratch_floats(const sfn_ssm *m, uint32_t nb);

/* One fixed-A step.  h (n) is updated in place; y (d_out) may be NULL.
 * scratch: sfn_ssm_scratch_floats(m, 1) floats. */
void sfn_ssm_step(const sfn_ssm *m, float *h, const float *x, float *y,
                  float *scratch);

/* T steps; X is (T, d_in), Y is (T, d_out) or NULL. */
void sfn_ssm_scan(const sfn_ssm *m, float *h, const float *X, float *Y,
                  uint32_t T, float *scratch);

/* nb independent states sharing the weights: H (nb, n), X (nb, d_in),
 * Y (nb, d_out).  Uses CBLAS sgemm when built with -DSFN_USE_CBLAS, plain
 * loops otherwise; scratch is sfn_ssm_scratch_floats(m, nb) floats. */
void sfn_ssm_step_batch(const sfn_ssm *m, float *H, const float *X, float *Y,
                        uint32_t nb, float *scratch);

/* Input-dependent gain.  A fixed A is the special case of a constant gain;
 * selective SSMs (Mamba-style) and gated linear attention replace A by a
 * per-step diagonal a_t computed from x_t.  The caller computes gain (n) from
 * its own gate network; m must be a diagonal model (only B, C, D are used).
 *   h <- gain (.) h + B x       y = C h (+ D x) */
void sfn_ssm_step_gain(const sfn_ssm *m, float *h, const float *x,
                       const float *gain, float *y);

/* DeltaNet delta rule.  The state is a matrix S (dk x dv), and the effective
 * transition S -> S (I - beta k k^T) depends on the input k, so it is NOT a
 * fixed A and cannot be expressed as sfn_ssm_step.  One step:
 *   S <- S + beta * k (v - S^T k)^T        y = S^T q
 * q, k are length dk; v, y length dv; beta in (0, 1]; k should be L2-normalised
 * by the caller.  scratch: dv floats.  The q/k/v/beta projections are the
 * network's job (sfn_matvec below); this is only the recurrence. */
void sfn_delta_step(float *S, uint32_t dk, uint32_t dv, const float *q,
                    const float *k, const float *v, float beta, float *y,
                    float *scratch);

/* y (rows) = W (rows x cols, row-major) x. */
void sfn_matvec(const float *W, uint32_t rows, uint32_t cols, const float *x,
                float *y);

#ifdef __cplusplus
}
#endif
#endif
