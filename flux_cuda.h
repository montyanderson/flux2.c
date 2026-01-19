/*
 * FLUX CUDA Backend - High-Performance GPU Acceleration
 *
 * GPU-accelerated operations for NVIDIA GPUs.
 * Provides both host-device transfer operations (flux_cuda_sgemm)
 * and pure GPU operations (flux_cuda_*_gpu) for full GPU pipelines.
 */

#ifndef FLUX_CUDA_H
#define FLUX_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Initialization
 * ============================================================================ */

int flux_cuda_init(void);
void flux_cuda_cleanup(void);
int flux_cuda_available(void);
int flux_cuda_fp8_available(void);  /* FP8 tensor cores (SM 8.9+) */
void flux_cuda_synchronize(void);
void flux_cuda_memory_info(size_t *free, size_t *total);
void flux_cuda_reset_weights(void);  /* Clear weight cache between generations */

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/* Allocate/free GPU memory */
float *flux_cuda_malloc(size_t bytes);
void flux_cuda_free(float *ptr);

/* Transfer data between host and device */
void flux_cuda_memcpy_to_device(float *dst, const float *src, size_t bytes);
void flux_cuda_memcpy_to_host(float *dst, const float *src, size_t bytes);

/* Weight caching - upload weight to GPU, returns cached device pointer */
float *flux_cuda_upload_weight(const float *host_ptr, size_t bytes);

/* Activation buffer pool */
float *flux_cuda_get_buffer(size_t bytes);
void flux_cuda_release_buffer(float *ptr);

/* ============================================================================
 * Host-Device Operations (data copied each call)
 * ============================================================================ */

/* SGEMM: C = alpha * op(A) @ op(B) + beta * C
 * Row-major format. Handles host-device transfers internally.
 * Weight matrix B is cached on GPU for reuse.
 */
void flux_cuda_sgemm(int transA, int transB,
                     int M, int N, int K,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc);

/* Async SGEMM - queues operation without sync. Call flux_cuda_synchronize() when needed */
void flux_cuda_sgemm_async(int transA, int transB,
                           int M, int N, int K,
                           float alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           float beta,
                           float *C, int ldc);

/* Batched SGEMM */
void flux_cuda_sgemm_batched(int transA, int transB,
                             int M, int N, int K,
                             float alpha,
                             const float *A, int lda, int strideA,
                             const float *B, int ldb, int strideB,
                             float beta,
                             float *C, int ldc, int strideC,
                             int batch_count);

/* ============================================================================
 * Pure GPU Operations (for full GPU pipelines)
 * All pointers must be device pointers (allocated with flux_cuda_malloc
 * or flux_cuda_upload_weight or flux_cuda_get_buffer)
 * ============================================================================ */

/* Linear layer: out[seq, out_dim] = in[seq, in_dim] @ weight[out_dim, in_dim]^T + bias */
void flux_cuda_linear_gpu(float *d_out, const float *d_in,
                          const float *d_weight, const float *d_bias,
                          int seq_len, int in_dim, int out_dim);

/* Attention: out = softmax(Q @ K^T * scale) @ V (device pointers) */
void flux_cuda_attention_gpu(float *d_out,
                             const float *d_Q, const float *d_K, const float *d_V,
                             int seq_q, int seq_k, int heads, int head_dim,
                             float scale);

/* Batched attention with automatic memory transfers (host pointers)
 * Q, K, V, out: [heads, seq, head_dim] layout (transposed for efficient batching)
 */
void flux_cuda_attention(float *out,
                         const float *Q, const float *K, const float *V,
                         int heads, int seq_q, int seq_k, int head_dim,
                         float scale);

/* Softmax over rows */
void flux_cuda_softmax_gpu(float *d_x, int rows, int cols);

/* RMSNorm: out = x / rms(x) * weight */
void flux_cuda_rms_norm_gpu(float *d_out, const float *d_x,
                            const float *d_weight,
                            int seq_len, int hidden, float eps);

/* LayerNorm: out = (x - mean) / std * gamma + beta */
void flux_cuda_layer_norm_gpu(float *d_out, const float *d_x,
                              const float *d_gamma, const float *d_beta,
                              int seq_len, int hidden, float eps);

/* Activations */
void flux_cuda_silu_gpu(float *d_x, int n);
void flux_cuda_swiglu_gpu(float *d_out, const float *d_x, const float *d_gate, int n);

/* Element-wise operations */
void flux_cuda_add_gpu(float *d_out, const float *d_a, const float *d_b, int n);
void flux_cuda_add_inplace_gpu(float *d_a, const float *d_b, int n);
void flux_cuda_mul_gpu(float *d_out, const float *d_a, const float *d_b, int n);
void flux_cuda_scale_gpu(float *d_x, float s, int n);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_CUDA_H */
