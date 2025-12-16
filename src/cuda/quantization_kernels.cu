#include "quantization_kernels.cuh"
#include <cub/cub.cuh>

namespace jalapeno {

// ==================== QUANTIZATION KERNELS ====================

// FP16/FP32 -> INT8 quantization
template<typename T>
__global__ void quantize_int8_kernel(
    const T* input,
    int8_t* output,
    const QuantizationParams params,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    float val = static_cast<float>(input[idx]);
    
    // Quantize: (val - min) / scale
    float quantized = (val - params.min_val) / params.scale;
    
    // Clamp to 8-bit range and round
    int8_t qval = static_cast<int8_t>(fminf(fmaxf(quantized + 0.5f, 0.0f), 255.0f));
    
    output[idx] = qval;
}

// INT8 -> FP16/FP32 dequantization
template<typename T>
__global__ void dequantize_int8_kernel(
    const int8_t* input,
    T* output,
    const QuantizationParams params,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    float qval = static_cast<float>(input[idx]);
    float val = qval * params.scale + params.min_val;
    
    output[idx] = static_cast<T>(val);
}

// FP16/FP32 -> INT4 quantization (packed 2 values per byte)
template<typename T>
__global__ void quantize_int4_kernel(
    const T* input,
    uint8_t* output,  // Packed: 2 int4 values per byte
    const QuantizationParams params,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_idx = idx / 2;  // 2 elements per output byte
    
    if (idx >= num_elements) return;
    
    float val = static_cast<float>(input[idx]);
    
    // Quantize to 4-bit
    float quantized = (val - params.min_val) / params.scale;
    uint8_t qval = static_cast<uint8_t>(fminf(fmaxf(quantized + 0.5f, 0.0f), 15.0f));
    
    // Pack into byte (even idx in lower 4 bits, odd in upper 4 bits)
    if (idx % 2 == 0) {
        // Even index: lower 4 bits
        output[output_idx] = (output[output_idx] & 0xF0) | (qval & 0x0F);
    } else {
        // Odd index: upper 4 bits
        output[output_idx] = (output[output_idx] & 0x0F) | ((qval & 0x0F) << 4);
    }
}

// INT4 -> FP16/FP32 dequantization (unpacking)
template<typename T>
__global__ void dequantize_int4_kernel(
    const uint8_t* input,  // Packed: 2 int4 values per byte
    T* output,
    const QuantizationParams params,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_elements) return;
    
    size_t input_idx = idx / 2;
    uint8_t packed = input[input_idx];
    
    // Extract 4-bit value
    uint8_t qval;
    if (idx % 2 == 0) {
        // Even index: lower 4 bits
        qval = packed & 0x0F;
    } else {
        // Odd index: upper 4 bits
        qval = (packed >> 4) & 0x0F;
    }
    
    // Dequantize
    float val = static_cast<float>(qval) * params.scale + params.min_val;
    output[idx] = static_cast<T>(val);
}

// NF4 quantization (4-bit NormalFloat)
template<typename T>
__global__ void quantize_nf4_kernel(
    const T* input,
    uint8_t* output,  // Packed: 2 nf4 values per byte
    const NF4Params nf4_params,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_idx = idx / 2;
    
    if (idx >= num_elements) return;
    
    float val = static_cast<float>(input[idx]);
    
    // Quantize using NF4 levels
    uint8_t qval = nf4_params.quantize(val);
    
    // Pack into byte
    if (idx % 2 == 0) {
        output[output_idx] = (output[output_idx] & 0xF0) | (qval & 0x0F);
    } else {
        output[output_idx] = (output[output_idx] & 0x0F) | ((qval & 0x0F) << 4);
    }
}

template<typename T>
__global__ void dequantize_nf4_kernel(
    const uint8_t* input,
    T* output,
    const NF4Params nf4_params,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_elements) return;
    
    size_t input_idx = idx / 2;
    uint8_t packed = input[input_idx];
    
    // Extract 4-bit value
    uint8_t qval;
    if (idx % 2 == 0) {
        qval = packed & 0x0F;
    } else {
        qval = (packed >> 4) & 0x0F;
    }
    
    // Dequantize using NF4 levels
    float val = nf4_params.dequantize(qval);
    output[idx] = static_cast<T>(val);
}

// FP8 quantization
template<typename T>
__global__ void quantize_fp8_kernel(
    const T* input,
    uint8_t* output,
    const FP8Params fp8_params,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    float val = static_cast<float>(input[idx]);
    output[idx] = fp8_params.quantize(val);
}

template<typename T>
__global__ void dequantize_fp8_kernel(
    const uint8_t* input,
    T* output,
    const FP8Params fp8_params,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    uint8_t qval = input[idx];
    float val = fp8_params.dequantize(qval);
    output[idx] = static_cast<T>(val);
}

// Group-wise quantization (per-tensor is too coarse)
template<typename T>
__global__ void quantize_groupwise_int8_kernel(
    const T* input,
    int8_t* output,
    const QuantizationParams* group_params,  // One per group
    size_t num_elements,
    size_t group_size,
    size_t num_groups
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    size_t group_idx = idx / group_size;
    if (group_idx >= num_groups) return;
    
    const QuantizationParams& params = group_params[group_idx];
    float val = static_cast<float>(input[idx]);
    
    float quantized = (val - params.min_val) / params.scale;
    int8_t qval = static_cast<int8_t>(fminf(fmaxf(quantized + 0.5f, 0.0f), 255.0f));
    
    output[idx] = qval;
}

// Asymmetric quantization with zero point
template<typename T>
__global__ void quantize_asymmetric_int8_kernel(
    const T* input,
    uint8_t* output,  // Unsigned 8-bit
    const QuantizationParams params,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    float val = static_cast<float>(input[idx]);
    
    // Apply zero point
    float quantized = val / params.scale + params.zero_point;
    uint8_t qval = static_cast<uint8_t>(fminf(fmaxf(quantized + 0.5f, 0.0f), 255.0f));
    
    output[idx] = qval;
}

// ==================== RANGE FINDING KERNELS ====================

// Find min and max in a tensor (for quantization parameters)
template<typename T>
__global__ void find_min_max_kernel(
    const T* input,
    float* min_out,
    float* max_out,
    size_t num_elements
) {
    extern __shared__ float shared_mem[];
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;
    
    // Initialize thread values
    float thread_min = INFINITY;
    float thread_max = -INFINITY;
    
    // Process elements
    for (size_t i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float val = static_cast<float>(input[i]);
        thread_min = fminf(thread_min, val);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Store in shared memory
    shared_mem[tid] = thread_min;
    shared_mem[tid + blockDim.x] = thread_max;
    __syncthreads();
    
    // Reduce in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = fminf(shared_mem[tid], shared_mem[tid + s]);
            shared_mem[tid + blockDim.x] = fmaxf(shared_mem[tid + blockDim.x], 
                                                shared_mem[tid + blockDim.x + s]);
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (tid == 0) {
        min_out[blockIdx.x] = shared_mem[0];
        max_out[blockIdx.x] = shared_mem[blockDim.x];
    }
}

// Find min/max per group
template<typename T>
__global__ void find_min_max_per_group_kernel(
    const T* input,
    float* group_mins,
    float* group_maxs,
    size_t num_elements,
    size_t group_size,
    size_t num_groups
) {
    size_t group_idx = blockIdx.x;
    if (group_idx >= num_groups) return;
    
    size_t start_idx = group_idx * group_size;
    size_t end_idx = min(start_idx + group_size, num_elements);
    
    float group_min = INFINITY;
    float group_max = -INFINITY;
    
    for (size_t i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
        float val = static_cast<float>(input[i]);
        group_min = fminf(group_min, val);
        group_max = fmaxf(group_max, val);
    }
    
    // Reduce within block
    __shared__ float shared_mins[256];
    __shared__ float shared_maxs[256];
    
    shared_mins[threadIdx.x] = group_min;
    shared_maxs[threadIdx.x] = group_max;
    __syncthreads();
    
    // Parallel reduction
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mins[threadIdx.x] = fminf(shared_mins[threadIdx.x], 
                                            shared_mins[threadIdx.x + s]);
            shared_maxs[threadIdx.x] = fmaxf(shared_maxs[threadIdx.x], 
                                            shared_maxs[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        group_mins[group_idx] = shared_mins[0];
        group_maxs[group_idx] = shared_maxs[0];
    }
}

// ==================== QUANTIZED MATRIX MULTIPLICATION ====================

// Quantized GEMM: A (int8) * B (int8) -> C (float)
__global__ void gemm_int8_kernel(
    const int8_t* A,  // M x K, row-major
    const int8_t* B,  // K x N, row-major
    float* C,         // M x N, row-major
    const QuantizationParams params_A,
    const QuantizationParams params_B,
    int M, int N, int K
) {
    // Block size for tiling
    constexpr int TILE_SIZE = 16;
    
    // Shared memory tiles
    __shared__ int8_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < K; t += TILE_SIZE) {
        // Load tile A
        if (row < M && (t + threadIdx.x) < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + (t + threadIdx.x)];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }
        
        // Load tile B
        if ((t + threadIdx.y) < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            int8_t a_val = tile_A[threadIdx.y][k];
            int8_t b_val = tile_B[k][threadIdx.x];
            
            // Dequantize and multiply
            float a_f32 = static_cast<float>(a_val) * params_A.scale + params_A.min_val;
            float b_f32 = static_cast<float>(b_val) * params_B.scale + params_B.min_val;
            acc += a_f32 * b_f32;
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Mixed precision GEMM: A (int4) * B (float16) -> C (float16)
__global__ void gemm_mixed_int4_fp16_kernel(
    const uint8_t* A_packed,  // M x K, int4 packed
    const half* B,            // K x N, fp16
    half* C,                  // M x N, fp16
    const QuantizationParams params_A,
    int M, int N, int K
) {
    constexpr int TILE_SIZE = 32;
    constexpr int INT4_TILE = TILE_SIZE * 2;  // 2 int4 per byte
    
    __shared__ uint8_t tile_A_packed[TILE_SIZE][INT4_TILE / 8];
    __shared__ half tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    half2 acc = __float2half2_rn(0.0f);
    
    for (int t = 0; t < K; t += TILE_SIZE) {
        // Load packed int4 tile A
        if (row < M) {
            for (int i = 0; i < INT4_TILE / 8; i += blockDim.x) {
                int idx = i + threadIdx.x;
                if (idx < INT4_TILE / 8 && (t * 2 + idx * 8) < K * 2) {
                    tile_A_packed[threadIdx.y][idx] = 
                        A_packed[row * (K * 2 / 8) + t * 2 / 8 + idx];
                }
            }
        }
        
        // Load fp16 tile B
        if ((t + threadIdx.y) < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        }
        
        __syncthreads();
        
        // Compute
        for (int k = 0; k < TILE_SIZE; ++k) {
            // Unpack 2 int4 values from packed byte
            uint8_t packed = tile_A_packed[threadIdx.y][k / 2];
            uint8_t a_val1, a_val2;
            
            if (k % 2 == 0) {
                a_val1 = packed & 0x0F;
                a_val2 = (packed >> 4) & 0x0F;
            } else {
                a_val1 = (packed >> 4) & 0x0F;
                // Get next byte for second value if available
                if (k + 1 < TILE_SIZE) {
                    uint8_t next_packed = tile_A_packed[threadIdx.y][(k + 1) / 2];
                    a_val2 = next_packed & 0x0F;
                } else {
                    a_val2 = 0;
                }
            }
            
            // Dequantize int4 to fp16
            half a1_f16 = __float2half(static_cast<float>(a_val1) * params_A.scale + params_A.min_val);
            half a2_f16 = __float2half(static_cast<float>(a_val2) * params_A.scale + params_A.min_val);
            
            half b_val = tile_B[k][threadIdx.x];
            
            // Fused multiply-add
            acc.x = __hfma(a1_f16, b_val, acc.x);
            acc.y = __hfma(a2_f16, b_val, acc.y);
        }
        
        __syncthreads();
    }
    
    // Store accumulated results
    if (row < M && col < N) {
        C[row * N + col] = __hadd(acc.x, acc.y);
    }
}

// ==================== QUANTIZATION UTILITIES ====================

// Compute quantization parameters from min/max
__host__ QuantizationParams compute_quantization_params(
    float min_val, float max_val, int bits, bool symmetric = false
) {
    QuantizationParams params;
    params.min_val = min_val;
    params.max_val = max_val;
    params.range = max_val - min_val;
    
    if (symmetric) {
        // Symmetric quantization around zero
        float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
        params.min_val = -abs_max;
        params.max_val = abs_max;
        params.range = 2 * abs_max;
    }
    
    if (bits == 8) {
        params.scale = params.range / 255.0f;
        params.zero_point = -params.min_val / params.scale;
    } else if (bits == 4) {
        params.scale = params.range / 15.0f;
        params.zero_point = -params.min_val / params.scale;
    } else {
        params.scale = 1.0f;
        params.zero_point = 0.0f;
    }
    
    return params;
}

// Find optimal quantization parameters for a tensor
template<typename T>
__host__ QuantizationParams find_optimal_quantization_params(
    const T* data, size_t num_elements, int bits, 
    bool symmetric = false, float percentile = 0.999f
) {
    // Copy to host if needed (simplified)
    std::vector<float> host_data(num_elements);
    cudaMemcpy(host_data.data(), data, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Sort to find percentile
    std::vector<float> sorted = host_data;
    std::sort(sorted.begin(), sorted.end());
    
    size_t idx = static_cast<size_t>(sorted.size() * percentile);
    float max_val = sorted[idx];
    float min_val = sorted[sorted.size() - idx - 1];
    
    // Clamp outliers
    if (symmetric) {
        float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
        min_val = -abs_max;
        max_val = abs_max;
    }
    
    return compute_quantization_params(min_val, max_val, bits, symmetric);
}

// ==================== WRAPPER FUNCTIONS ====================

// Quantize tensor (wrapper)
template<typename T>
void quantize_tensor(
    const T* input,
    void* output,  // Type depends on quantization method
    size_t num_elements,
    QuantizationMethod method,
    const QuantizationParams& params,
    cudaStream_t stream = 0
) {
    KernelConfig config(num_elements);
    
    switch (method) {
        case QuantizationMethod::INT8: {
            auto* out_int8 = static_cast<int8_t*>(output);
            quantize_int8_kernel<<<config.grid_size, config.block_size, 0, stream>>>(
                input, out_int8, params, num_elements
            );
            break;
        }
        case QuantizationMethod::INT4: {
            auto* out_int4 = static_cast<uint8_t*>(output);
            quantize_int4_kernel<<<config.grid_size, config.block_size, 0, stream>>>(
                input, out_int4, params, num_elements
            );
            break;
        }
        case QuantizationMethod::NF4: {
            auto* out_nf4 = static_cast<uint8_t*>(output);
            NF4Params nf4_params;
            quantize_nf4_kernel<<<config.grid_size, config.block_size, 0, stream>>>(
                input, out_nf4, nf4_params, num_elements
            );
            break;
        }
        case QuantizationMethod::FP8_E4M3:
        case QuantizationMethod::FP8_E5M2: {
            auto* out_fp8 = static_cast<uint8_t*>(output);
            FP8Params fp8_params;
            fp8_params.format = (method == QuantizationMethod::FP8_E4M3) ? 
                               FP8Params::E4M3 : FP8Params::E5M2;
            quantize_fp8_kernel<<<config.grid_size, config.block_size, 0, stream>>>(
                input, out_fp8, fp8_params, num_elements
            );
            break;
        }
        default:
            throw std::runtime_error("Unsupported quantization method");
    }
    
    cudaStreamSynchronize(stream);
}

// Dequantize tensor (wrapper)
template<typename T>
void dequantize_tensor(
    const void* input,  // Type depends on quantization method
    T* output,
    size_t num_elements,
    QuantizationMethod method,
    const QuantizationParams& params,
    cudaStream_t stream = 0
) {
    KernelConfig config(num_elements);
    
    switch (method) {
        case QuantizationMethod::INT8: {
            auto* in_int8 = static_cast<const int8_t*>(input);
            dequantize_int8_kernel<<<config.grid_size, config.block_size, 0, stream>>>(
                in_int8, output, params, num_elements
            );
            break;
        }
        case QuantizationMethod::INT4: {
            auto* in_int4 = static_cast<const uint8_t*>(input);
            dequantize_int4_kernel<<<config.grid_size, config.block_size, 0, stream>>>(
                in_int4, output, params, num_elements
            );
            break;
        }
        case QuantizationMethod::NF4: {
            auto* in_nf4 = static_cast<const uint8_t*>(input);
            NF4Params nf4_params;
            dequantize_nf4_kernel<<<config.grid_size, config.block_size, 0, stream>>>(
                in_nf4, output, nf4_params, num_elements
            );
            break;
        }
        case QuantizationMethod::FP8_E4M3:
        case QuantizationMethod::FP8_E5M2: {
            auto* in_fp8 = static_cast<const uint8_t*>(input);
            FP8Params fp8_params;
            fp8_params.format = (method == QuantizationMethod::FP8_E4M3) ? 
                               FP8Params::E4M3 : FP8Params::E5M2;
            dequantize_fp8_kernel<<<config.grid_size, config.block_size, 0, stream>>>(
                in_fp8, output, fp8_params, num_elements
            );
            break;
        }
        default:
            throw std::runtime_error("Unsupported quantization method");
    }
    
    cudaStreamSynchronize(stream);
}

// Find min/max of tensor
template<typename T>
std::pair<float, float> find_min_max(
    const T* data, size_t num_elements, cudaStream_t stream = 0
) {
    size_t num_blocks = 256;
    size_t block_size = 256;
    size_t shared_mem = 2 * block_size * sizeof(float);
    
    // Allocate temporary memory
    float* d_block_mins, *d_block_maxs;
    cudaMalloc(&d_block_mins, num_blocks * sizeof(float));
    cudaMalloc(&d_block_maxs, num_blocks * sizeof(float));
    
    // Launch kernel
    find_min_max_kernel<<<num_blocks, block_size, shared_mem, stream>>>(
        data, d_block_mins, d_block_maxs, num_elements
    );
    
    // Copy to host
    std::vector<float> block_mins(num_blocks), block_maxs(num_blocks);
    cudaMemcpyAsync(block_mins.data(), d_block_mins, num_blocks * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(block_maxs.data(), d_block_maxs, num_blocks * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // Reduce on host
    float global_min = INFINITY;
    float global_max = -INFINITY;
    
    for (size_t i = 0; i < num_blocks; ++i) {
        global_min = fminf(global_min, block_mins[i]);
        global_max = fmaxf(global_max, block_maxs[i]);
    }
    
    // Cleanup
    cudaFree(d_block_mins);
    cudaFree(d_block_maxs);
    
    return {global_min, global_max};
}

// Perform quantized matrix multiplication
void quantized_gemm(
    const void* A, QuantizationMethod method_A, const QuantizationParams& params_A,
    const void* B, QuantizationMethod method_B, const QuantizationParams& params_B,
    void* C, DataType dtype_C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    // This is a simplified wrapper
    // In practice, would dispatch to appropriate kernel based on quantization methods
    
    if (method_A == QuantizationMethod::INT8 && method_B == QuantizationMethod::INT8) {
        auto* A_int8 = static_cast<const int8_t*>(A);
        auto* B_int8 = static_cast<const int8_t*>(B);
        auto* C_f32 = static_cast<float*>(C);
        
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        
        gemm_int8_kernel<<<grid, block, 0, stream>>>(
            A_int8, B_int8, C_f32, params_A, params_B, M, N, K
        );
    } else if (method_A == QuantizationMethod::INT4 && method_B == QuantizationMethod::FP16) {
        auto* A_int4 = static_cast<const uint8_t*>(A);
        auto* B_fp16 = static_cast<const half*>(B);
        auto* C_fp16 = static_cast<half*>(C);
        
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        
        gemm_mixed_int4_fp16_kernel<<<grid, block, 0, stream>>>(
            A_int4, B_fp16, C_fp16, params_A, M, N, K
        );
    } else {
        throw std::runtime_error("Unsupported quantized GEMM combination");
    }
    
    cudaStreamSynchronize(stream);
}

} // namespace jalapeno
