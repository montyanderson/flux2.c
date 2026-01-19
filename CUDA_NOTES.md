# CUDA Implementation Notes for FLUX.2-klein

## Current Status

A basic cuBLAS integration has been implemented in `flux_cuda.cu`. However, performance testing reveals that the simple "copy-compute-copy" approach is significantly slower than CPU BLAS due to memory transfer overhead.

### Benchmark Results (256x256, 2 steps, NVIDIA B200)

| Backend | Total Time | Text Encoding | Denoising | VAE Decode |
|---------|------------|---------------|-----------|------------|
| **BLAS (OpenBLAS)** | **19s** | 4.7s | 5.6s | 2.4s |
| CUDA (simple) | 555s | 263s | 12.5s | 274s |

The CUDA version is **29x slower** due to cudaMemcpy overhead.

## Why Current CUDA is Slow

The fundamental problem is that we're calling `cudaMemcpy` for every matrix multiply:

1. Copy input activations to GPU
2. Execute cuBLAS SGEMM
3. Copy output activations back to CPU

For the text encoder (36 transformer layers with many small matrices), this overhead dominates:
- ~300 small matmuls per layer
- Each matmul has 2 memcpy calls
- PCIe latency (~10μs) × thousands of calls = seconds of overhead

## What's Needed for Fast GPU Inference

To achieve real GPU acceleration, we need a **full GPU pipeline** like the Metal backend (`flux_metal.m`):

### 1. Pre-upload All Weights at Load Time
```c
// At model load
for (each weight tensor W) {
    cuda_weight_cache[W] = cudaMalloc + cudaMemcpy(W)
}
```

### 2. Keep Activations on GPU Throughout Forward Pass
```c
// Instead of:
flux_linear(cpu_output, cpu_input, cpu_weight)  // 3 memcpy per call

// Do:
flux_cuda_linear_gpu_to_gpu(gpu_output, gpu_input, gpu_weight)  // 0 memcpy
```

### 3. Only Transfer at Boundaries
```c
// At inference start
cudaMemcpy(gpu_text_tokens, cpu_text_tokens)

// Entire forward pass stays on GPU
// ... text_encoder -> transformer -> vae_decoder ...

// At inference end
cudaMemcpy(cpu_image, gpu_image)
```

### 4. Required CUDA Kernels

Beyond cuBLAS SGEMM, we need GPU kernels for:
- RMSNorm / LayerNorm
- SiLU / SwiGLU activations
- Softmax
- RoPE (Rotary Position Embedding)
- Attention (ideally Flash Attention)
- Element-wise operations (add, mul, scale)
- Patchify / Unpatchify
- Convolution (transposed)

### 5. Memory Management

The B200 has 178GB VRAM - more than enough for the model (~15GB) plus activations (~2GB for 256x256):
- Model weights: ~15GB (cached)
- Text encoder activations: ~500MB peak
- Transformer activations: ~1GB peak
- VAE activations: ~500MB peak

## Implementation Options

### Option A: Write Custom CUDA Backend (1-2 weeks)
- Port all operations from `flux_kernels.c` to CUDA
- Follow Metal backend architecture
- Maximum performance, full control

### Option B: Use NVIDIA CUTLASS/cuDNN (2-3 days)
- cuDNN for optimized attention and convolutions
- CUTLASS for fused GEMM operations
- Less code, good performance

### Option C: Use TensorRT (1 day)
- Export model to ONNX, optimize with TensorRT
- Automatic kernel fusion, Tensor Core utilization
- Limited flexibility, but fastest option

### Option D: Stay with BLAS for Now (0 days)
- OpenBLAS with multi-threading is already 19s
- Good enough for many use cases
- No additional complexity

## Recommended Path Forward

For this codebase, **Option D (BLAS)** is currently the most practical:
- The server's CPU is powerful (high-core Xeon)
- OpenBLAS automatically uses all cores
- 19s for 256x256 is acceptable

For production GPU deployment, **Option C (TensorRT)** would be ideal:
- NVIDIA's optimized inference runtime
- Automatic Tensor Core utilization
- Handles all the complexity

## Files Added

- `flux_cuda.h` - CUDA backend header
- `flux_cuda.cu` - cuBLAS implementation with weight caching
- `Makefile` - Added `cuda` target

Build with:
```bash
export PATH=/usr/local/cuda/bin:$PATH
make cuda
```

The CUDA code works correctly but is not performant. It serves as a starting point for a proper GPU pipeline implementation.
