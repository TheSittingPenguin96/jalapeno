#include "tensor.hpp"
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace jalapeno {

UnifiedTensor::UnifiedTensor(const std::vector<size_t>& shape, DataType dtype) {
    metadata_.shape = shape;
    metadata_.dtype = dtype;
    
    // Calculate size
    size_t element_count = 1;
    for (auto dim : shape) {
        element_count *= dim;
    }
    metadata_.element_count = element_count;
    
    // Calculate bytes based on dtype
    size_t element_size = 0;
    switch (dtype) {
        case DataType::FP32: element_size = 4; break;
        case DataType::FP16: element_size = 2; break;
        case DataType::BF16: element_size = 2; break;
        case DataType::INT8: element_size = 1; break;
        case DataType::INT4: element_size = 0.5; break;  // Special handling
        case DataType::NF4: element_size = 0.5; break;
        case DataType::FP8: element_size = 1; break;
    }
    
    metadata_.bytes = element_count * element_size;
    if (dtype == DataType::INT4 || dtype == DataType::NF4) {
        // Round up for 4-bit types
        metadata_.bytes = (element_count + 1) / 2;
    }
    
    metadata_.current_tier = MemoryTier::TIER_2_CPU_RAM;
    metadata_.preferred_tier = MemoryTier::TIER_0_GPU_HBM;
    metadata_.access_count = 0;
    metadata_.temperature = 0.0f;
    
    cudaEventCreate(&transfer_event_);
}

UnifiedTensor::~UnifiedTensor() {
    free_current_allocation();
    cudaEventDestroy(transfer_event_);
    
    if (compressed_data_) {
        free(compressed_data_);
    }
}

void UnifiedTensor::allocate(MemoryTier tier) {
    free_current_allocation();
    
    switch (tier) {
        case MemoryTier::TIER_0_GPU_HBM:
        case MemoryTier::TIER_1_GPU_VRAM:
            allocate_gpu();
            break;
        case MemoryTier::TIER_2_CPU_RAM:
            allocate_cpu();
            break;
        case MemoryTier::TIER_3_NVME_SSD:
            allocate_disk();
            break;
        default:
            throw std::runtime_error("Unsupported memory tier");
    }
    
    metadata_.current_tier = tier;
    metadata_.is_pinned = (tier == MemoryTier::TIER_0_GPU_HBM || 
                          tier == MemoryTier::TIER_1_GPU_VRAM);
}

void UnifiedTensor::allocate_gpu() {
    cudaError_t err = cudaMalloc(&gpu_ptr_, metadata_.bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA allocation failed: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // Optional: allocate as managed memory for easier migration
    // cudaMallocManaged(&gpu_ptr_, metadata_.bytes);
}

void UnifiedTensor::allocate_cpu() {
    // Use aligned allocation for better performance
    const size_t alignment = 4096;  // Page size
    cpu_ptr_ = aligned_alloc(alignment, metadata_.bytes);
    if (!cpu_ptr_) {
        throw std::runtime_error("CPU allocation failed");
    }
    
    // Optionally pin for faster GPU transfers
    if (metadata_.is_pinned) {
        cudaHostRegister(cpu_ptr_, metadata_.bytes, cudaHostRegisterDefault);
    }
}

void UnifiedTensor::allocate_disk() {
    // Create a temporary file for memory mapping
    char filename[] = "/tmp/jalapeno_XXXXXX";
    int fd = mkstemp(filename);
    if (fd == -1) {
        throw std::runtime_error("Failed to create temp file");
    }
    
    // Extend file to required size
    if (ftruncate(fd, metadata_.bytes) == -1) {
        close(fd);
        throw std::runtime_error("Failed to truncate file");
    }
    
    // Memory map the file
    disk_mmap_ptr_ = mmap(nullptr, metadata_.bytes, 
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
    
    close(fd);
    unlink(filename);  // File persists while mapped
    
    if (disk_mmap_ptr_ == MAP_FAILED) {
        throw std::runtime_error("Memory mapping failed");
    }
}

void UnifiedTensor::free_current_allocation() {
    switch (metadata_.current_tier) {
        case MemoryTier::TIER_0_GPU_HBM:
        case MemoryTier::TIER_1_GPU_VRAM:
            if (gpu_ptr_) {
                cudaFree(gpu_ptr_);
                gpu_ptr_ = nullptr;
            }
            break;
        case MemoryTier::TIER_2_CPU_RAM:
            if (cpu_ptr_) {
                if (metadata_.is_pinned) {
                    cudaHostUnregister(cpu_ptr_);
                }
                free(cpu_ptr_);
                cpu_ptr_ = nullptr;
            }
            break;
        case MemoryTier::TIER_3_NVME_SSD:
            if (disk_mmap_ptr_) {
                munmap(disk_mmap_ptr_, metadata_.bytes);
                disk_mmap_ptr_ = nullptr;
            }
            break;
        default:
            break;
    }
}

void UnifiedTensor::migrate(MemoryTier target_tier) {
    if (metadata_.current_tier == target_tier) {
        return;
    }
    
    // Allocate destination
    allocate(target_tier);
    
    // Copy data
    void* src = nullptr;
    switch (metadata_.current_tier) {
        case MemoryTier::TIER_0_GPU_HBM:
        case MemoryTier::TIER_1_GPU_VRAM:
            src = gpu_ptr_;
            break;
        case MemoryTier::TIER_2_CPU_RAM:
            src = cpu_ptr_;
            break;
        case MemoryTier::TIER_3_NVME_SSD:
            src = disk_mmap_ptr_;
            break;
    }
    
    void* dst = nullptr;
    switch (target_tier) {
        case MemoryTier::TIER_0_GPU_HBM:
        case MemoryTier::TIER_1_GPU_VRAM:
            dst = gpu_ptr_;
            break;
        case MemoryTier::TIER_2_CPU_RAM:
            dst = cpu_ptr_;
            break;
        case MemoryTier::TIER_3_NVME_SSD:
            dst = disk_mmap_ptr_;
            break;
    }
    
    // Perform copy based on direction
    if ((metadata_.current_tier == MemoryTier::TIER_2_CPU_RAM || 
         metadata_.current_tier == MemoryTier::TIER_3_NVME_SSD) &&
        (target_tier == MemoryTier::TIER_0_GPU_HBM || 
         target_tier == MemoryTier::TIER_1_GPU_VRAM)) {
        // CPU/SSD → GPU
        cudaMemcpy(dst, src, metadata_.bytes, cudaMemcpyHostToDevice);
    } else if ((metadata_.current_tier == MemoryTier::TIER_0_GPU_HBM || 
                metadata_.current_tier == MemoryTier::TIER_1_GPU_VRAM) &&
               (target_tier == MemoryTier::TIER_2_CPU_RAM || 
                target_tier == MemoryTier::TIER_3_NVME_SSD)) {
        // GPU → CPU/SSD
        cudaMemcpy(dst, src, metadata_.bytes, cudaMemcpyDeviceToHost);
    } else {
        // CPU ↔ SSD or within same type
        memcpy(dst, src, metadata_.bytes);
    }
    
    metadata_.current_tier = target_tier;
}

void UnifiedTensor::record_access() {
    metadata_.access_count++;
    metadata_.last_access_time = 
        std::chrono::steady_clock::now().time_since_epoch().count();
    
    // Update temperature (exponential moving average)
    const float alpha = 0.1f;
    metadata_.temperature = alpha + (1.0f - alpha) * metadata_.temperature;
}

float UnifiedTensor::access_temperature() const {
    // Cool down over time
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto time_since_access = now - metadata_.last_access_time;
    
    // Convert to seconds and apply exponential decay
    float seconds = time_since_access / 1e9f;
    float decay = expf(-seconds / 60.0f);  // Half-life of 60 seconds
    
    return metadata_.temperature * decay;
}

} // namespace jalapeno
