#ifndef JALAPENO_TENSOR_HPP
#define JALAPENO_TENSOR_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace jalapeno {

enum class MemoryTier : uint8_t {
    TIER_0_GPU_HBM = 0,  // GPU High-Bandwidth Memory (fastest)
    TIER_1_GPU_VRAM,     // Regular GPU memory
    TIER_2_CPU_RAM,      // System RAM
    TIER_3_NVME_SSD,     // Fast storage
    TIER_4_NETWORK       // Future: remote memory
};

enum class DataType : uint8_t {
    FP32 = 0,
    FP16,
    BF16,
    INT8,
    INT4,
    NF4,    // 4-bit NormalFloat
    FP8     // Future support
};

struct TensorMetadata {
    std::vector<size_t> shape;
    size_t element_count;
    size_t bytes;
    DataType dtype;
    MemoryTier current_tier;
    MemoryTier preferred_tier;
    
    // Access pattern tracking
    uint64_t last_access_time;
    uint32_t access_count;
    float temperature;  // How "hot" is this tensor (access frequency)
    
    // For paging
    bool is_pinned;
    bool is_compressed;
    size_t compressed_size;
};

class UnifiedTensor {
public:
    UnifiedTensor() = default;
    UnifiedTensor(const std::vector<size_t>& shape, DataType dtype);
    ~UnifiedTensor();
    
    // Memory management
    void allocate(MemoryTier tier);
    void migrate(MemoryTier target_tier);
    void prefetch(MemoryTier target_tier);
    void compress();
    void decompress();
    
    // Data access
    template<typename T>
    T* data();
    
    template<typename T>
    const T* data() const;
    
    // Metadata
    const TensorMetadata& metadata() const { return metadata_; }
    size_t size_bytes() const { return metadata_.bytes; }
    
    // Access tracking
    void record_access();
    float access_temperature() const;
    
private:
    TensorMetadata metadata_;
    
    // Raw data pointers for each tier (only one is valid at a time)
    void* gpu_ptr_ = nullptr;
    void* cpu_ptr_ = nullptr;
    void* disk_mmap_ptr_ = nullptr;  // For memory-mapped files
    
    // Compression buffer
    void* compressed_data_ = nullptr;
    
    // Async transfer state
    bool transfer_in_progress_ = false;
    cudaEvent_t transfer_event_;
    
    void allocate_gpu();
    void allocate_cpu();
    void allocate_disk();
    void free_current_allocation();
};

// Smart pointer for tensors
using TensorPtr = std::shared_ptr<UnifiedTensor>;

// Tensor slice for large models
class TensorSlice {
public:
    TensorSlice(TensorPtr parent, size_t offset, size_t size);
    
    // View operations (no copy)
    TensorPtr view() const;
    
private:
    TensorPtr parent_;
    size_t offset_;
    size_t size_;
};

} // namespace jalapeno

#endif // JALAPENO_TENSOR_HPP
