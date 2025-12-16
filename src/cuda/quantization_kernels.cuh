#ifndef JALAPENO_QUANTIZATION_KERNELS_CUH
#define JALAPENO_QUANTIZATION_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace jalapeno {

// Quantization structures
struct QuantizationParams {
    float scale;
    float zero_point;
    float min_val;
    float max_val;
    float range;
    
    __host__ __device__ QuantizationParams() 
        : scale(1.0f), zero_point(0.0f), 
          min_val(0.0f), max_val(0.0f), range(0.0f) {}
    
    __host__ __device__ QuantizationParams(float min, float max, int bits)
        : min_val(min), max_val(max) {
        range = max - min;
        if (bits == 8) {
            scale = range / 255.0f;
            zero_point = -min / scale;
        } else if (bits == 4) {
            scale = range / 15.0f;
            zero_point = -min / scale;
        } else {
            scale = 1.0f;
            zero_point = 0.0f;
        }
    }
};

// NF4 quantization (4-bit NormalFloat)
struct NF4Params {
    // Precomputed quantization levels for normal distribution
    static constexpr int NUM_LEVELS = 16;
    float levels[NUM_LEVELS];
    
    __host__ NF4Params() {
        // NF4 quantization levels (from QLoRA paper)
        const float qlevels[NUM_LEVELS] = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };
        
        for (int i = 0; i < NUM_LEVELS; ++i) {
            levels[i] = qlevels[i];
        }
    }
    
    __device__ float dequantize(uint8_t qval) const {
        int idx = qval & 0x0F;  // Lower 4 bits
        return levels[idx];
    }
    
    __device__ uint8_t quantize(float val) const {
        // Find closest level
        uint8_t best_idx = 0;
        float best_diff = fabsf(val - levels[0]);
        
        for (uint8_t i = 1; i < NUM_LEVELS; ++i) {
            float diff = fabsf(val - levels[i]);
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = i;
            }
        }
        
        return best_idx;
    }
};

// FP8 (E4M3 and E5M2 formats)
struct FP8Params {
    enum Format { E4M3, E5M2 };
    
    Format format;
    
    __device__ float dequantize(uint8_t fp8) const {
        if (format == E4M3) {
            return dequantize_e4m3(fp8);
        } else {
            return dequantize_e5m2(fp8);
        }
    }
    
    __device__ uint8_t quantize(float val) const {
        if (format == E4M3) {
            return quantize_e4m3(val);
        } else {
            return quantize_e5m2(val);
        }
    }
    
private:
    __device__ float dequantize_e4m3(uint8_t fp8) const {
        // Extract sign, exponent, mantissa
        uint8_t sign = (fp8 >> 7) & 0x1;
        uint8_t exponent = (fp8 >> 3) & 0xF;
        uint8_t mantissa = fp8 & 0x7;
        
        if (exponent == 0) {
            // Subnormal or zero
            if (mantissa == 0) return 0.0f;
            float val = mantissa / 8.0f * powf(2.0f, -6.0f);
            return sign ? -val : val;
        } else if (exponent == 0xF) {
            // NaN or Inf (E4M3 doesn't have Inf, only NaN)
            return NAN;
        } else {
            // Normal number
            float val = (1.0f + mantissa / 8.0f) * powf(2.0f, exponent - 7);
            return sign ? -val : val;
        }
    }
    
    __device__ uint8_t quantize_e4m3(float val) const {
        // Simplified quantization - in practice would use hardware instructions
        if (val == 0.0f) return 0;
        
        uint8_t sign = val < 0 ? 0x80 : 0;
        val = fabsf(val);
        
        // Find exponent
        int exponent = 0;
        float abs_val = val;
        
        while (abs_val >= 2.0f && exponent < 7) {
            abs_val /= 2.0f;
            exponent++;
        }
        while (abs_val < 1.0f && exponent > -8) {
            abs_val *= 2.0f;
            exponent--;
        }
        
        exponent += 7;  // Bias
        
        if (exponent < 0) {
            // Subnormal
            exponent = 0;
            abs_val *= powf(2.0f, 7);
        } else if (exponent > 0xF) {
            // Clamp to max
            exponent = 0xF;
            abs_val = 1.875f;  // Max mantissa
        }
        
        // Quantize mantissa
        uint8_t mantissa = static_cast<uint8_t>((abs_val - 1.0f) * 8.0f);
        mantissa = mantissa & 0x7;  // 3 bits
        
        return sign | (exponent << 3) | mantissa;
    }
    
    __device__ float dequantize_e5m2(uint8_t fp8) const {
        uint8_t sign = (fp8 >> 7) & 0x1;
        uint8_t exponent = (fp8 >> 2) & 0x1F;
        uint8_t mantissa = fp8 & 0x3;
        
        if (exponent == 0) {
            if (mantissa == 0) return 0.0f;
            float val = mantissa / 4.0f * powf(2.0f, -14.0f);
            return sign ? -val : val;
        } else if (exponent == 0x1F) {
            // NaN or Inf
            return mantissa == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
        } else {
            float val = (1.0f + mantissa / 4.0f) * powf(2.0f, exponent - 15);
            return sign ? -val : val;
        }
    }
    
    __device__ uint8_t quantize_e5m2(float val) const {
        // Similar to E4M3 but with different exponent range
        if (val == 0.0f) return 0;
        
        uint8_t sign = val < 0 ? 0x80 : 0;
        val = fabsf(val);
        
        int exponent = 0;
        float abs_val = val;
        
        while (abs_val >= 2.0f && exponent < 15) {
            abs_val /= 2.0f;
            exponent++;
        }
        while (abs_val < 1.0f && exponent > -16) {
            abs_val *= 2.0f;
            exponent--;
        }
        
        exponent += 15;  // Bias
        
        if (exponent < 0) {
            exponent = 0;
            abs_val *= powf(2.0f, 15);
        } else if (exponent > 0x1F) {
            exponent = 0x1F;
            abs_val = 1.75f;  // Max mantissa
        }
        
        uint8_t mantissa = static_cast<uint8_t>((abs_val - 1.0f) * 4.0f);
        mantissa = mantissa & 0x3;  // 2 bits
        
        return sign | (exponent << 2) | mantissa;
    }
};

// Kernel launch configurations
struct KernelConfig {
    dim3 block_size;
    dim3 grid_size;
    size_t shared_mem;
    
    KernelConfig(size_t num_elements) {
        block_size = dim3(256, 1, 1);
        grid_size = dim3((num_elements + block_size.x - 1) / block_size.x, 1, 1);
        shared_mem = 0;
    }
    
    KernelConfig(size_t num_elements, size_t block_size_dim) {
        block_size = dim3(block_size_dim, 1, 1);
        grid_size = dim3((num_elements + block_size.x - 1) / block_size.x, 1, 1);
        shared_mem = 0;
    }
};

} // namespace jalapeno

#endif // JALAPENO_QUANTIZATION_KERNELS_CUH
