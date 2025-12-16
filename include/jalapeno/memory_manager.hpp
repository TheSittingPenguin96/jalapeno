#ifndef JALAPENO_MEMORY_MANAGER_HPP
#define JALAPENO_MEMORY_MANAGER_HPP

#include "tensor.hpp"
#include <unordered_map>
#include <list>
#include <mutex>
#include <atomic>
#include <queue>
#include <thread>

namespace jalapeno {

class MemoryPool {
public:
    MemoryPool(MemoryTier tier, size_t capacity);
    
    bool allocate(size_t size, void** ptr);
    bool free(void* ptr, size_t size);
    
    size_t used() const { return used_; }
    size_t capacity() const { return capacity_; }
    size_t available() const { return capacity_ - used_; }
    
private:
    MemoryTier tier_;
    size_t capacity_;
    size_t used_;
    std::mutex mutex_;
    
    // Buddy allocator or slab allocator would go here
    std::map<size_t, std::vector<void*>> free_blocks_;
};

class PagingPolicy {
public:
    virtual ~PagingPolicy() = default;
    
    virtual TensorPtr select_victim() = 0;
    virtual void record_access(TensorPtr tensor) = 0;
    virtual void record_prefetch(TensorPtr tensor) = 0;
    
    virtual float calculate_cost(TensorPtr tensor, MemoryTier target) = 0;
};

class LRUPagingPolicy : public PagingPolicy {
public:
    TensorPtr select_victim() override;
    void record_access(TensorPtr tensor) override;
    void record_prefetch(TensorPtr tensor) override;
    float calculate_cost(TensorPtr tensor, MemoryTier target) override;
    
private:
    std::list<TensorPtr> access_order_;
    std::unordered_map<TensorPtr, 
        std::list<TensorPtr>::iterator> access_map_;
    std::mutex mutex_;
};

class MemoryHierarchyManager {
public:
    static MemoryHierarchyManager& instance();
    
    // Configuration
    void initialize(const std::map<MemoryTier, size_t>& capacities);
    
    // Tensor lifecycle
    TensorPtr create_tensor(const std::vector<size_t>& shape, 
                           DataType dtype, 
                           MemoryTier initial_tier);
    
    bool migrate_tensor(TensorPtr tensor, MemoryTier target_tier);
    bool prefetch_tensor(TensorPtr tensor, MemoryTier target_tier);
    
    // Statistics
    struct MemoryStats {
        size_t gpu_used;
        size_t gpu_total;
        size_t cpu_used;
        size_t cpu_total;
        size_t disk_used;
        size_t hit_rate;
        size_t miss_rate;
        size_t swap_operations;
    };
    
    MemoryStats get_stats() const;
    
    // Async operations
    void start_async_transfer(TensorPtr tensor, MemoryTier target);
    bool is_transfer_complete(TensorPtr tensor);
    
    // Predictive prefetching
    void register_access_pattern(TensorPtr accessed, TensorPtr likely_next);
    std::vector<TensorPtr> predict_next_accesses(TensorPtr current);
    
private:
    MemoryHierarchyManager();
    ~MemoryHierarchyManager();
    
    std::map<MemoryTier, std::unique_ptr<MemoryPool>> pools_;
    std::unique_ptr<PagingPolicy> paging_policy_;
    
    // Access pattern learning
    std::map<TensorPtr, std::map<TensorPtr, uint32_t>> access_patterns_;
    
    // Async transfer queue
    struct TransferTask {
        TensorPtr tensor;
        MemoryTier source;
        MemoryTier target;
        cudaEvent_t event;
        bool completed;
    };
    
    std::queue<TransferTask> transfer_queue_;
    std::thread transfer_thread_;
    std::atomic<bool> transfer_running_;
    
    void transfer_worker();
    
    // Statistics
    mutable std::mutex stats_mutex_;
    size_t hits_ = 0;
    size_t misses_ = 0;
    size_t swaps_ = 0;
    
    // Singleton
    MemoryHierarchyManager(const MemoryHierarchyManager&) = delete;
    MemoryHierarchyManager& operator=(const MemoryHierarchyManager&) = delete;
};

} // namespace jalapeno

#endif // JALAPENO_MEMORY_MANAGER_HPP
