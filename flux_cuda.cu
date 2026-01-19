/*
 * FLUX CUDA Backend - High-Performance GPU Acceleration
 *
 * Full GPU pipeline for NVIDIA GPUs (B200, H100, etc.)
 * Key features:
 * - GPU-resident tensors (data stays on GPU)
 * - Weight caching (upload once, reuse)
 * - Custom CUDA kernels for all operations
 * - cuBLAS for matrix multiplication
 * - Activation buffer pooling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {
#include "flux_cuda.h"
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

#define CUDA_BLOCK_SIZE 256
#define MAX_WEIGHT_ENTRIES 2048
#define MAX_BUFFER_POOL 128
#define WARP_SIZE 32

/* ============================================================================
 * Error Handling
 * ============================================================================ */

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
    } \
} while(0)

/* ============================================================================
 * Global State
 * ============================================================================ */

static cublasHandle_t g_cublas = NULL;
static cudaStream_t g_stream = NULL;
static int g_initialized = 0;

/* Weight cache - persistent GPU storage for model weights */
typedef struct {
    const void *host_ptr;
    void *device_ptr;
    size_t size;
} weight_entry_t;

static weight_entry_t g_weights[MAX_WEIGHT_ENTRIES];
static int g_weight_count = 0;
static size_t g_weight_bytes = 0;

/* Buffer pool - reusable GPU memory for activations */
typedef struct {
    void *ptr;
    size_t size;
    int in_use;
} buffer_entry_t;

static buffer_entry_t g_buffers[MAX_BUFFER_POOL];
static int g_buffer_count = 0;

/* ============================================================================
 * CUDA Kernels - Element-wise Operations
 * ============================================================================ */

__global__ void kernel_add(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void kernel_add_inplace(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void kernel_mul(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void kernel_scale(float *x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void kernel_add_bias(float *y, const float *bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        y[idx] += bias[col];
    }
}

/* ============================================================================
 * CUDA Kernels - Activations
 * ============================================================================ */

__global__ void kernel_silu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

__global__ void kernel_swiglu(float *out, const float *x, const float *gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = x[i] * silu_g;
    }
}

__global__ void kernel_gelu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

/* ============================================================================
 * CUDA Kernels - Softmax (numerically stable)
 * ============================================================================ */

__global__ void kernel_softmax_rows(float *x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float *row_ptr = x + row * cols;

    // Find max (for numerical stability)
    float max_val = -1e30f;
    for (int c = 0; c < cols; c++) {
        if (row_ptr[c] > max_val) max_val = row_ptr[c];
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        row_ptr[c] = expf(row_ptr[c] - max_val);
        sum += row_ptr[c];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int c = 0; c < cols; c++) {
        row_ptr[c] *= inv_sum;
    }
}

/* Optimized softmax using shared memory for reductions */
__global__ void kernel_softmax_optimized(float *x, int rows, int cols) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (row >= rows) return;

    float *row_ptr = x + row * cols;

    // Find max using parallel reduction
    float local_max = -1e30f;
    for (int c = tid; c < cols; c += stride) {
        if (row_ptr[c] > local_max) local_max = row_ptr[c];
    }
    shared[tid] = local_max;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared[tid + s] > shared[tid]) {
            shared[tid] = shared[tid + s];
        }
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    // Compute exp and local sum
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += stride) {
        float v = expf(row_ptr[c] - max_val);
        row_ptr[c] = v;
        local_sum += v;
    }
    shared[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float sum = shared[0];
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int c = tid; c < cols; c += stride) {
        row_ptr[c] *= inv_sum;
    }
}

/* ============================================================================
 * CUDA Kernels - Normalization
 * ============================================================================ */

__global__ void kernel_rms_norm(float *out, const float *x, const float *weight,
                                 int seq_len, int hidden, float eps) {
    int s = blockIdx.x;
    if (s >= seq_len) return;

    const float *x_row = x + s * hidden;
    float *out_row = out + s * hidden;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden; i++) {
        sum_sq += x_row[i] * x_row[i];
    }

    float rms_inv = rsqrtf(sum_sq / hidden + eps);

    // Normalize and scale
    for (int i = 0; i < hidden; i++) {
        out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

__global__ void kernel_layer_norm(float *out, const float *x,
                                   const float *gamma, const float *beta,
                                   int seq_len, int hidden, float eps) {
    int s = blockIdx.x;
    if (s >= seq_len) return;

    const float *x_row = x + s * hidden;
    float *out_row = out + s * hidden;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < hidden; i++) mean += x_row[i];
    mean /= hidden;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float diff = x_row[i] - mean;
        var += diff * diff;
    }
    var /= hidden;

    float std_inv = rsqrtf(var + eps);

    // Normalize and scale
    for (int i = 0; i < hidden; i++) {
        float norm = (x_row[i] - mean) * std_inv;
        out_row[i] = gamma[i] * norm + beta[i];
    }
}

/* ============================================================================
 * CUDA Kernels - RoPE (Rotary Position Embedding)
 * ============================================================================ */

__global__ void kernel_rope_2d(float *x, const float *freqs_h, const float *freqs_w,
                                int seq_len, int heads, int head_dim,
                                int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % heads;
    int s = idx / (head_dim * heads);

    if (d >= head_dim / 2) return;  // Only process first half

    int pos_h = s / W;
    int pos_w = s % W;

    // Get frequencies
    float cos_h = freqs_h[pos_h * (head_dim/4) + d % (head_dim/4)];
    float sin_h = freqs_h[pos_h * (head_dim/4) + d % (head_dim/4) + head_dim/4];
    float cos_w = freqs_w[pos_w * (head_dim/4) + d % (head_dim/4)];
    float sin_w = freqs_w[pos_w * (head_dim/4) + d % (head_dim/4) + head_dim/4];

    float *vec = x + (s * heads + h) * head_dim;
    int d2 = d + head_dim / 2;

    float x0 = vec[d];
    float x1 = vec[d2];

    // Apply rotation
    vec[d] = x0 * cos_h - x1 * sin_h;
    vec[d2] = x0 * sin_h + x1 * cos_h;
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

static void *get_weight(const void *host_ptr, size_t size) {
    // Check cache
    for (int i = 0; i < g_weight_count; i++) {
        if (g_weights[i].host_ptr == host_ptr) {
            return g_weights[i].device_ptr;
        }
    }

    // Not found - upload and cache
    if (g_weight_count >= MAX_WEIGHT_ENTRIES) {
        fprintf(stderr, "Weight cache full!\n");
        return NULL;
    }

    void *d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, size));
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, host_ptr, size, cudaMemcpyHostToDevice, g_stream));

    g_weights[g_weight_count].host_ptr = host_ptr;
    g_weights[g_weight_count].device_ptr = d_ptr;
    g_weights[g_weight_count].size = size;
    g_weight_count++;
    g_weight_bytes += size;

    return d_ptr;
}

static void *get_buffer(size_t size) {
    // Find existing buffer
    for (int i = 0; i < g_buffer_count; i++) {
        if (!g_buffers[i].in_use && g_buffers[i].size >= size) {
            g_buffers[i].in_use = 1;
            return g_buffers[i].ptr;
        }
    }

    // Allocate new
    if (g_buffer_count >= MAX_BUFFER_POOL) {
        // Pool full - direct allocation
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        return ptr;
    }

    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    g_buffers[g_buffer_count].ptr = ptr;
    g_buffers[g_buffer_count].size = size;
    g_buffers[g_buffer_count].in_use = 1;
    g_buffer_count++;

    return ptr;
}

static void release_buffer(void *ptr) {
    for (int i = 0; i < g_buffer_count; i++) {
        if (g_buffers[i].ptr == ptr) {
            g_buffers[i].in_use = 0;
            return;
        }
    }
    // Not in pool - free directly
    cudaFree(ptr);
}

/* ============================================================================
 * Initialization / Cleanup
 * ============================================================================ */

extern "C" int flux_cuda_init(void) {
    if (g_initialized) return 0;

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "CUDA: %s (%.1f GB, SM %d.%d)\n",
            prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);

    // Create stream and cuBLAS handle
    CUDA_CHECK(cudaStreamCreate(&g_stream));
    CUBLAS_CHECK(cublasCreate(&g_cublas));
    CUBLAS_CHECK(cublasSetStream(g_cublas, g_stream));
    CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));

    g_initialized = 1;
    return 0;
}

extern "C" void flux_cuda_cleanup(void) {
    if (!g_initialized) return;

    // Free weight cache
    for (int i = 0; i < g_weight_count; i++) {
        cudaFree(g_weights[i].device_ptr);
    }
    g_weight_count = 0;
    g_weight_bytes = 0;

    // Free buffer pool
    for (int i = 0; i < g_buffer_count; i++) {
        cudaFree(g_buffers[i].ptr);
    }
    g_buffer_count = 0;

    if (g_cublas) cublasDestroy(g_cublas);
    if (g_stream) cudaStreamDestroy(g_stream);

    g_initialized = 0;
}

extern "C" int flux_cuda_available(void) {
    return g_initialized;
}

extern "C" void flux_cuda_synchronize(void) {
    if (g_stream) cudaStreamSynchronize(g_stream);
}

extern "C" void flux_cuda_memory_info(size_t *free, size_t *total) {
    cudaMemGetInfo(free, total);
}

/* ============================================================================
 * High-Level Operations - SGEMM
 * ============================================================================ */

/* Pinned memory pool for faster H2D/D2H transfers */
#define MAX_PINNED_BUFFERS 16
static struct {
    void *ptr;
    size_t size;
    int in_use;
} g_pinned[MAX_PINNED_BUFFERS] = {0};

static void *get_pinned_buffer(size_t size) {
    for (int i = 0; i < MAX_PINNED_BUFFERS; i++) {
        if (!g_pinned[i].in_use && g_pinned[i].size >= size) {
            g_pinned[i].in_use = 1;
            return g_pinned[i].ptr;
        }
    }
    for (int i = 0; i < MAX_PINNED_BUFFERS; i++) {
        if (!g_pinned[i].ptr) {
            cudaHostAlloc(&g_pinned[i].ptr, size, cudaHostAllocDefault);
            g_pinned[i].size = size;
            g_pinned[i].in_use = 1;
            return g_pinned[i].ptr;
        }
    }
    return NULL;  // Pool full
}

static void release_pinned_buffer(void *ptr) {
    for (int i = 0; i < MAX_PINNED_BUFFERS; i++) {
        if (g_pinned[i].ptr == ptr) {
            g_pinned[i].in_use = 0;
            return;
        }
    }
}

extern "C" void flux_cuda_sgemm(int transA, int transB,
                                 int M, int N, int K,
                                 float alpha,
                                 const float *A, int lda,
                                 const float *B, int ldb,
                                 float beta,
                                 float *C, int ldc) {
    if (!g_initialized) return;

    size_t size_A = (size_t)(transA ? K * M : M * K) * sizeof(float);
    size_t size_B = (size_t)(transB ? N * K : K * N) * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    // Get/upload B (weights - cached)
    float *d_B = (float*)get_weight(B, size_B);

    // Get activation buffers
    float *d_A = (float*)get_buffer(size_A);
    float *d_C = (float*)get_buffer(size_C);

    // Use pinned memory for faster transfer if available
    void *pinned_A = get_pinned_buffer(size_A);
    void *pinned_C = get_pinned_buffer(size_C);

    if (pinned_A && pinned_C) {
        // Fast path with pinned memory
        memcpy(pinned_A, A, size_A);
        CUDA_CHECK(cudaMemcpyAsync(d_A, pinned_A, size_A, cudaMemcpyHostToDevice, g_stream));
    } else {
        // Fallback
        CUDA_CHECK(cudaMemcpyAsync(d_A, A, size_A, cudaMemcpyHostToDevice, g_stream));
    }

    if (beta != 0.0f) {
        CUDA_CHECK(cudaMemcpyAsync(d_C, C, size_C, cudaMemcpyHostToDevice, g_stream));
    }

    // cuBLAS (column-major, so swap A/B for row-major)
    cublasOperation_t opA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(g_cublas, opA, opB,
                             N, M, K,
                             &alpha,
                             d_B, transB ? K : N,
                             d_A, transA ? M : K,
                             &beta,
                             d_C, N));

    // Copy result back
    if (pinned_C) {
        CUDA_CHECK(cudaMemcpyAsync(pinned_C, d_C, size_C, cudaMemcpyDeviceToHost, g_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_stream));
        memcpy(C, pinned_C, size_C);
        release_pinned_buffer(pinned_C);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(C, d_C, size_C, cudaMemcpyDeviceToHost, g_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_stream));
    }

    if (pinned_A) release_pinned_buffer(pinned_A);
    release_buffer(d_A);
    release_buffer(d_C);
}

/* ============================================================================
 * Async SGEMM - No sync, for batching multiple operations
 * ============================================================================ */

/* Persistent activation buffers for async operations */
static float *g_async_in = NULL;
static float *g_async_out = NULL;
static size_t g_async_in_size = 0;
static size_t g_async_out_size = 0;

static void ensure_async_buffers(size_t in_size, size_t out_size) {
    if (g_async_in_size < in_size) {
        if (g_async_in) cudaFree(g_async_in);
        cudaMalloc(&g_async_in, in_size);
        g_async_in_size = in_size;
    }
    if (g_async_out_size < out_size) {
        if (g_async_out) cudaFree(g_async_out);
        cudaMalloc(&g_async_out, out_size);
        g_async_out_size = out_size;
    }
}

/* Async SGEMM - queues operation without waiting */
extern "C" void flux_cuda_sgemm_async(int transA, int transB,
                                       int M, int N, int K,
                                       float alpha,
                                       const float *A, int lda,
                                       const float *B, int ldb,
                                       float beta,
                                       float *C, int ldc) {
    if (!g_initialized) return;

    size_t size_A = (size_t)(transA ? K * M : M * K) * sizeof(float);
    size_t size_B = (size_t)(transB ? N * K : K * N) * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    ensure_async_buffers(size_A, size_C);

    float *d_B = (float*)get_weight(B, size_B);

    cudaMemcpyAsync(g_async_in, A, size_A, cudaMemcpyHostToDevice, g_stream);
    if (beta != 0.0f) {
        cudaMemcpyAsync(g_async_out, C, size_C, cudaMemcpyHostToDevice, g_stream);
    }

    cublasOperation_t opA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasSgemm(g_cublas, opA, opB,
                N, M, K,
                &alpha,
                d_B, transB ? K : N,
                g_async_in, transA ? M : K,
                &beta,
                g_async_out, N);

    cudaMemcpyAsync(C, g_async_out, size_C, cudaMemcpyDeviceToHost, g_stream);
    // NOTE: No sync here - caller must sync when needed
}

/* ============================================================================
 * GPU-Only Operations (no CPU transfer)
 * ============================================================================ */

/* Linear layer entirely on GPU */
extern "C" void flux_cuda_linear_gpu(float *d_out, const float *d_in,
                                      const float *d_weight, const float *d_bias,
                                      int seq_len, int in_dim, int out_dim) {
    if (!g_initialized) return;

    float alpha = 1.0f, beta = 0.0f;

    // out = in @ weight^T
    CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             out_dim, seq_len, in_dim,
                             &alpha,
                             d_weight, in_dim,
                             d_in, in_dim,
                             &beta,
                             d_out, out_dim));

    // Add bias if present
    if (d_bias) {
        int n = seq_len * out_dim;
        int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        kernel_add_bias<<<blocks, CUDA_BLOCK_SIZE, 0, g_stream>>>(
            d_out, d_bias, seq_len, out_dim);
    }
}

/* Softmax on GPU */
extern "C" void flux_cuda_softmax_gpu(float *d_x, int rows, int cols) {
    if (cols <= 1024) {
        // Use optimized kernel for reasonable sizes
        int threads = min(cols, 256);
        kernel_softmax_optimized<<<rows, threads, threads * sizeof(float), g_stream>>>(
            d_x, rows, cols);
    } else {
        // Fall back to simple kernel
        kernel_softmax_rows<<<rows, 1, 0, g_stream>>>(d_x, rows, cols);
    }
}

/* RMSNorm on GPU */
extern "C" void flux_cuda_rms_norm_gpu(float *d_out, const float *d_x,
                                        const float *d_weight,
                                        int seq_len, int hidden, float eps) {
    kernel_rms_norm<<<seq_len, 1, 0, g_stream>>>(
        d_out, d_x, d_weight, seq_len, hidden, eps);
}

/* LayerNorm on GPU */
extern "C" void flux_cuda_layer_norm_gpu(float *d_out, const float *d_x,
                                          const float *d_gamma, const float *d_beta,
                                          int seq_len, int hidden, float eps) {
    kernel_layer_norm<<<seq_len, 1, 0, g_stream>>>(
        d_out, d_x, d_gamma, d_beta, seq_len, hidden, eps);
}

/* SiLU on GPU */
extern "C" void flux_cuda_silu_gpu(float *d_x, int n) {
    int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernel_silu<<<blocks, CUDA_BLOCK_SIZE, 0, g_stream>>>(d_x, n);
}

/* SwiGLU on GPU */
extern "C" void flux_cuda_swiglu_gpu(float *d_out, const float *d_x,
                                      const float *d_gate, int n) {
    int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernel_swiglu<<<blocks, CUDA_BLOCK_SIZE, 0, g_stream>>>(d_out, d_x, d_gate, n);
}

/* Element-wise add on GPU */
extern "C" void flux_cuda_add_gpu(float *d_out, const float *d_a,
                                   const float *d_b, int n) {
    int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernel_add<<<blocks, CUDA_BLOCK_SIZE, 0, g_stream>>>(d_out, d_a, d_b, n);
}

/* Element-wise add inplace on GPU */
extern "C" void flux_cuda_add_inplace_gpu(float *d_a, const float *d_b, int n) {
    int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernel_add_inplace<<<blocks, CUDA_BLOCK_SIZE, 0, g_stream>>>(d_a, d_b, n);
}

/* Element-wise multiply on GPU */
extern "C" void flux_cuda_mul_gpu(float *d_out, const float *d_a,
                                   const float *d_b, int n) {
    int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernel_mul<<<blocks, CUDA_BLOCK_SIZE, 0, g_stream>>>(d_out, d_a, d_b, n);
}

/* Scale on GPU */
extern "C" void flux_cuda_scale_gpu(float *d_x, float s, int n) {
    int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernel_scale<<<blocks, CUDA_BLOCK_SIZE, 0, g_stream>>>(d_x, s, n);
}

/* ============================================================================
 * Attention - Full GPU Implementation
 * ============================================================================ */

extern "C" void flux_cuda_attention_gpu(float *d_out,
                                         const float *d_Q, const float *d_K, const float *d_V,
                                         int seq_q, int seq_k, int heads, int head_dim,
                                         float scale) {
    if (!g_initialized) return;

    // Allocate scores buffer: [heads, seq_q, seq_k]
    size_t scores_size = (size_t)heads * seq_q * seq_k * sizeof(float);
    float *d_scores = (float*)get_buffer(scores_size);

    float alpha = scale;
    float beta = 0.0f;

    // For each head: scores = Q @ K^T
    for (int h = 0; h < heads; h++) {
        const float *d_Q_h = d_Q + h * seq_q * head_dim;
        const float *d_K_h = d_K + h * seq_k * head_dim;
        float *d_scores_h = d_scores + h * seq_q * seq_k;

        // scores[seq_q, seq_k] = Q[seq_q, head_dim] @ K[seq_k, head_dim]^T
        CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 seq_k, seq_q, head_dim,
                                 &alpha,
                                 d_K_h, head_dim,
                                 d_Q_h, head_dim,
                                 &beta,
                                 d_scores_h, seq_k));
    }

    // Softmax over last dimension
    for (int h = 0; h < heads; h++) {
        float *d_scores_h = d_scores + h * seq_q * seq_k;
        flux_cuda_softmax_gpu(d_scores_h, seq_q, seq_k);
    }

    alpha = 1.0f;

    // For each head: out = scores @ V
    for (int h = 0; h < heads; h++) {
        float *d_scores_h = d_scores + h * seq_q * seq_k;
        const float *d_V_h = d_V + h * seq_k * head_dim;
        float *d_out_h = d_out + h * seq_q * head_dim;

        // out[seq_q, head_dim] = scores[seq_q, seq_k] @ V[seq_k, head_dim]
        CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 head_dim, seq_q, seq_k,
                                 &alpha,
                                 d_V_h, head_dim,
                                 d_scores_h, seq_k,
                                 &beta,
                                 d_out_h, head_dim));
    }

    release_buffer(d_scores);
}

/* ============================================================================
 * Memory Transfer Helpers
 * ============================================================================ */

extern "C" float *flux_cuda_malloc(size_t bytes) {
    float *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

extern "C" void flux_cuda_free(float *ptr) {
    if (ptr) cudaFree(ptr);
}

extern "C" void flux_cuda_memcpy_to_device(float *dst, const float *src, size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, g_stream));
}

extern "C" void flux_cuda_memcpy_to_host(float *dst, const float *src, size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

/* Upload weight to GPU cache */
extern "C" float *flux_cuda_upload_weight(const float *host_ptr, size_t bytes) {
    return (float*)get_weight(host_ptr, bytes);
}

/* Get activation buffer */
extern "C" float *flux_cuda_get_buffer(size_t bytes) {
    return (float*)get_buffer(bytes);
}

/* Release activation buffer */
extern "C" void flux_cuda_release_buffer(float *ptr) {
    release_buffer(ptr);
}

/* ============================================================================
 * Batched SGEMM
 * ============================================================================ */

extern "C" void flux_cuda_sgemm_batched(int transA, int transB,
                                         int M, int N, int K,
                                         float alpha,
                                         const float *A, int lda, int strideA,
                                         const float *B, int ldb, int strideB,
                                         float beta,
                                         float *C, int ldc, int strideC,
                                         int batch_count) {
    if (!g_initialized) return;

    size_t size_A = (size_t)strideA * batch_count * sizeof(float);
    size_t size_B = (size_t)strideB * batch_count * sizeof(float);
    size_t size_C = (size_t)strideC * batch_count * sizeof(float);

    float *d_A = (float*)get_buffer(size_A);
    float *d_B = (float*)get_buffer(size_B);
    float *d_C = (float*)get_buffer(size_C);

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, size_A, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, size_B, cudaMemcpyHostToDevice, g_stream));
    if (beta != 0.0f) {
        CUDA_CHECK(cudaMemcpyAsync(d_C, C, size_C, cudaMemcpyHostToDevice, g_stream));
    }

    cublasOperation_t opA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas,
                                           opA, opB,
                                           N, M, K,
                                           &alpha,
                                           d_B, transB ? K : N, strideB,
                                           d_A, transA ? M : K, strideA,
                                           &beta,
                                           d_C, N, strideC,
                                           batch_count));

    CUDA_CHECK(cudaMemcpyAsync(C, d_C, size_C, cudaMemcpyDeviceToHost, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    release_buffer(d_A);
    release_buffer(d_B);
    release_buffer(d_C);
}

/* ============================================================================
 * Host-callable Batched Attention
 * Handles all memory transfers automatically
 * Q: [heads, seq_q, head_dim] (transposed layout)
 * K: [heads, seq_k, head_dim]
 * V: [heads, seq_k, head_dim]
 * out: [heads, seq_q, head_dim]
 * ============================================================================ */

extern "C" void flux_cuda_attention(float *out,
                                     const float *Q, const float *K, const float *V,
                                     int heads, int seq_q, int seq_k, int head_dim,
                                     float scale) {
    if (!g_initialized) return;

    size_t size_Q = (size_t)heads * seq_q * head_dim * sizeof(float);
    size_t size_K = (size_t)heads * seq_k * head_dim * sizeof(float);
    size_t size_V = (size_t)heads * seq_k * head_dim * sizeof(float);
    size_t size_out = (size_t)heads * seq_q * head_dim * sizeof(float);
    size_t size_scores = (size_t)heads * seq_q * seq_k * sizeof(float);

    // Allocate GPU buffers
    float *d_Q = (float*)get_buffer(size_Q);
    float *d_K = (float*)get_buffer(size_K);
    float *d_V = (float*)get_buffer(size_V);
    float *d_out = (float*)get_buffer(size_out);
    float *d_scores = (float*)get_buffer(size_scores);

    // Upload inputs
    CUDA_CHECK(cudaMemcpyAsync(d_Q, Q, size_Q, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_K, K, size_K, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_V, V, size_V, cudaMemcpyHostToDevice, g_stream));

    float alpha_scale = scale;
    float alpha_one = 1.0f;
    float beta = 0.0f;

    // Use cuBLAS batched SGEMM for all heads at once
    // scores = Q @ K^T * scale
    // Q: [heads, seq_q, head_dim] -> treat as batch of [seq_q, head_dim]
    // K: [heads, seq_k, head_dim] -> treat as batch of [seq_k, head_dim]
    // scores: [heads, seq_q, seq_k]
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           seq_k, seq_q, head_dim,
                                           &alpha_scale,
                                           d_K, head_dim, seq_k * head_dim,
                                           d_Q, head_dim, seq_q * head_dim,
                                           &beta,
                                           d_scores, seq_k, seq_q * seq_k,
                                           heads));

    // Softmax each head's scores
    flux_cuda_softmax_gpu(d_scores, heads * seq_q, seq_k);

    // out = scores @ V
    // scores: [heads, seq_q, seq_k]
    // V: [heads, seq_k, head_dim]
    // out: [heads, seq_q, head_dim]
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           head_dim, seq_q, seq_k,
                                           &alpha_one,
                                           d_V, head_dim, seq_k * head_dim,
                                           d_scores, seq_k, seq_q * seq_k,
                                           &beta,
                                           d_out, head_dim, seq_q * head_dim,
                                           heads));

    // Download result
    CUDA_CHECK(cudaMemcpyAsync(out, d_out, size_out, cudaMemcpyDeviceToHost, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    // Release buffers
    release_buffer(d_Q);
    release_buffer(d_K);
    release_buffer(d_V);
    release_buffer(d_out);
    release_buffer(d_scores);
}
