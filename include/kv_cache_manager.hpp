#ifndef JALAPENO_KV_CACHE_MANAGER_HPP
#define JALAPENO_KV_CACHE_MANAGER_HPP

#include "tensor.hpp"
#include "memory_manager.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <bitset>
#include <cmath>

namespace jalapeno {

// Forward declarations
class KVCompressor;
class CacheEvictionPolicy;

enum class KVCompressionMethod : uint8_t {
    NONE = 0,
    INT8_QUANTIZATION,
    INT4_QUANTIZATION,
    NF4_QUANTIZATION,
    PRUNING,           // Remove low-magnitude entries
    DCT_COMPRESSION,   // Frequency domain compression
    DICTIONARY,        // Dictionary coding
    DELTA_COMPRESSION, // Store differences
    ADAPTIVE_MIXED     // Mix of methods per layer
};

enum class CacheLocation : uint8_t {
    GPU_HBM = 0,    // GPU High-bandwidth memory (fastest)
    GPU_VRAM,       // Regular GPU memory
    CPU_RAM,        // System RAM
    CPU_PMEM,       // Persistent memory (Optane)
    NVME_SSD,       // Fast storage
    COMPRESSED_RAM  // Compressed in RAM
};

struct KVBlock {
    std::vector<int64_t> token_positions;  // Which tokens are stored here
    TensorPtr keys;                        // Key tensor block
    TensorPtr values;                      // Value tensor block
    size_t block_size;                     // Number of tokens in this block
    size_t layer_idx;                      // Which transformer layer
    size_t head_idx;                       // Which attention head
    
    // Metadata for cache management
    uint64_t last_access_time;
    uint32_t access_count;
    float importance_score;
    bool is_dirty;                         // Needs to be written back
    bool is_pinned;                        // Cannot be evicted
    bool is_compressed;
    KVCompressionMethod compression_method;
    
    // Statistics
    float average_attention_score;         // How important is this block
    float temporal_locality;               // How clustered in time
    float spatial_locality;                // How clustered in sequence
    
    KVBlock() 
        : block_size(0), layer_idx(0), head_idx(0),
          last_access_time(0), access_count(0),
          importance_score(0.0f), is_dirty(false),
          is_pinned(false), is_compressed(false),
          compression_method(KVCompressionMethod::NONE),
          average_attention_score(0.0f),
          temporal_locality(0.0f),
          spatial_locality(0.0f) {}
};

struct KVConfig {
    // Sizing
    size_t max_total_tokens = 32768;       // Maximum tokens in cache
    size_t block_size = 64;                // Tokens per block
    size_t max_blocks_per_layer = 512;     // Max blocks per layer
    
    // Compression
    KVCompressionMethod default_compression = KVCompressionMethod::INT8_QUANTIZATION;
    bool adaptive_compression = true;
    float compression_ratio_target = 0.5f; // Target 50% size reduction
    bool lossy_compression = true;         // Allow lossy compression
    
    // Eviction policy
    std::string eviction_policy = "importance_aware";
    float eviction_threshold = 0.8f;       // Start evicting at 80% full
    bool enable_prefetch = true;
    bool enable_async_eviction = true;
    
    // Multi-tier storage
    std::vector<CacheLocation> storage_hierarchy = {
        CacheLocation::GPU_HBM,
        CacheLocation::GPU_VRAM,
        CacheLocation::CPU_RAM,
        CacheLocation::NVME_SSD
    };
    
    // Monitoring
    bool collect_statistics = true;
    size_t statistics_interval_ms = 1000;
};

class KVCompressor {
public:
    virtual ~KVCompressor() = default;
    
    virtual std::pair<TensorPtr, TensorPtr> compress(
        const TensorPtr& keys,
        const TensorPtr& values,
        KVCompressionMethod method,
        float quality
    ) = 0;
    
    virtual std::pair<TensorPtr, TensorPtr> decompress(
        const TensorPtr& compressed_keys,
        const TensorPtr& compressed_values,
        KVCompressionMethod method
    ) = 0;
    
    virtual float estimate_compression_ratio(
        const TensorPtr& tensor,
        KVCompressionMethod method
    ) const = 0;
    
    virtual float estimate_error(
        const TensorPtr& original,
        const TensorPtr& decompressed
    ) const = 0;
};

class QuantizationCompressor : public KVCompressor {
private:
    struct QuantizationInfo {
        float scale;
        float zero_point;
        float min_val;
        float max_val;
    };
    
public:
    std::pair<TensorPtr, TensorPtr> compress(
        const TensorPtr& keys,
        const TensorPtr& values,
        KVCompressionMethod method,
        float quality
    ) override;
    
    std::pair<TensorPtr, TensorPtr> decompress(
        const TensorPtr& compressed_keys,
        const TensorPtr& compressed_values,
        KVCompressionMethod method
    ) override;
    
    float estimate_compression_ratio(
        const TensorPtr& tensor,
        KVCompressionMethod method
    ) const override;
    
    float estimate_error(
        const TensorPtr& original,
        const TensorPtr& decompressed
    ) const override;
    
private:
    QuantizationInfo compute_quantization_info(
        const TensorPtr& tensor,
        KVCompressionMethod method
    ) const;
    
    TensorPtr quantize_tensor(
        const TensorPtr& tensor,
        const QuantizationInfo& info,
        KVCompressionMethod method
    );
    
    TensorPtr dequantize_tensor(
        const TensorPtr& quantized,
        const QuantizationInfo& info,
        KVCompressionMethod method
    );
};

class CacheEvictionPolicy {
public:
    virtual ~CacheEvictionPolicy() = default;
    
    virtual std::vector<KVBlock*> select_eviction_candidates(
        std::vector<KVBlock*>& blocks,
        size_t required_space,
        const std::unordered_map<std::string, float>& context
    ) = 0;
    
    virtual void record_access(KVBlock* block) = 0;
    virtual void record_importance(KVBlock* block, float importance) = 0;
    virtual float calculate_priority(KVBlock* block) const = 0;
};

class ImportanceAwareEviction : public CacheEvictionPolicy {
private:
    struct BlockMetrics {
        float recency_score;
        float frequency_score;
        float importance_score;
        float attention_score;
        float size_penalty;
        float final_priority;
    };
    
public:
    std::vector<KVBlock*> select_eviction_candidates(
        std::vector<KVBlock*>& blocks,
        size_t required_space,
        const std::unordered_map<std::string, float>& context
    ) override;
    
    void record_access(KVBlock* block) override;
    void record_importance(KVBlock* block, float importance) override;
    float calculate_priority(KVBlock* block) const override;
    
private:
    BlockMetrics compute_metrics(KVBlock* block) const;
    void update_importance_decay();
    
    std::unordered_map<KVBlock*, uint64_t> last_access_times_;
    std::unordered_map<KVBlock*, uint32_t> access_counts_;
    std::unordered_map<KVBlock*, float> importance_scores_;
    
    mutable std::mutex metrics_mutex_;
    
    // Weighting factors (can be learned/adapted)
    float recency_weight_ = 0.3f;
    float frequency_weight_ = 0.2f;
    float importance_weight_ = 0.3f;
    float attention_weight_ = 0.2f;
};

class KVCacheManager {
public:
    KVCacheManager(
        size_t num_layers,
        size_t num_heads,
        size_t head_dim,
        size_t max_seq_len,
        const KVConfig& config
    );
    ~KVCacheManager();
    
    // Core operations
    std::pair<TensorPtr, TensorPtr> get_kv(
        size_t layer_idx,
        size_t head_idx,
        const std::vector<int64_t>& token_positions
    );
    
    void update_kv(
        size_t layer_idx,
        size_t head_idx,
        const std::vector<int64_t>& token_positions,
        const TensorPtr& new_keys,
        const TensorPtr& new_values
    );
    
    void prefetch_kv(
        size_t layer_idx,
        size_t head_idx,
        const std::vector<int64_t>& predicted_positions
    );
    
    // Memory management
    void compress_block(KVBlock* block, KVCompressionMethod method);
    void decompress_block(KVBlock* block);
    void migrate_block(KVBlock* block, CacheLocation target_location);
    void evict_blocks(size_t required_space);
    
    // State management
    void clear();
    void reset_layer(size_t layer_idx);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
    // Statistics and monitoring
    struct CacheStats {
        size_t total_blocks;
        size_t blocks_in_gpu;
        size_t blocks_in_cpu;
        size_t blocks_in_storage;
        size_t compressed_blocks;
        
        size_t total_tokens;
        size_t cache_hits;
        size_t cache_misses;
        size_t prefetch_hits;
        size_t evicted_blocks;
        
        float hit_rate;
        float average_latency_ms;
        float compression_ratio;
        float memory_utilization;
        
        // Per-layer stats
        std::vector<size_t> layer_block_counts;
        std::vector<float> layer_hit_rates;
    };
    
    CacheStats get_stats() const;
    void collect_detailed_statistics();
    
    // Configuration
    void update_config(const KVConfig& new_config);
    void set_compression_method(KVCompressionMethod method);
    void set_eviction_policy(const std::string& policy);
    
    // Advanced features
    void set_attention_mask(const TensorPtr& mask);
    void set_importance_scores(
        size_t layer_idx,
        const std::vector<float>& scores
    );
    
    void register_access_pattern(
        size_t layer_idx,
        size_t head_idx,
        const std::vector<int64_t>& pattern
    );
    
    std::vector<int64_t> predict_next_accesses(
        size_t layer_idx,
        size_t head_idx,
        int64_t current_position
    );
    
private:
    // Core data structures
    std::vector<std::vector<std::vector<KVBlock*>>> layer_head_blocks_;
    std::unordered_map<uint64_t, KVBlock*> block_map_;  // ID -> Block
    
    // Cache state
    size_t current_tokens_ = 0;
    size_t current_blocks_ = 0;
    std::bitset<128> active_layers_;  // Which layers have active cache
    
    // Components
    std::unique_ptr<MemoryHierarchyManager> memory_mgr_;
    std::unique_ptr<KVCompressor> compressor_;
    std::unique_ptr<CacheEvictionPolicy> eviction_policy_;
    
    // Configuration
    KVConfig config_;
    size_t num_layers_;
    size_t num_heads_;
    size_t head_dim_;
    size_t max_seq_len_;
    
    // Async operations
    std::thread eviction_thread_;
    std::thread compression_thread_;
    std::atomic<bool> running_;
    
    std::queue<std::function<void()>> async_queue_;
    std::mutex async_mutex_;
    std::condition_variable async_cv_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    CacheStats stats_;
    
    // Access pattern learning
    struct AccessPattern {
        std::vector<int64_t> recent_positions;
        std::vector<float> transition_probabilities;
        float locality_score;
    };
    
    std::vector<std::vector<AccessPattern>> access_patterns_;
    
    // Internal methods
    KVBlock* find_or_create_block(
        size_t layer_idx,
        size_t head_idx,
        const std::vector<int64_t>& token_positions
    );
    
    KVBlock* find_block(
        size_t layer_idx,
        size_t head_idx,
        const std::vector<int64_t>& token_positions
    );
    
    void create_block(
        size_t layer_idx,
        size_t head_idx,
        const std::vector<int64_t>& token_positions,
        const TensorPtr& keys,
        const TensorPtr& values
    );
    
    void merge_blocks(KVBlock* block1, KVBlock* block2);
    void split_block(KVBlock* block, size_t split_point);
    
    uint64_t generate_block_id(
        size_t layer_idx,
        size_t head_idx,
        int64_t start_pos
    ) const;
    
    void async_worker();
    void eviction_worker();
    void compression_worker();
    
    void update_statistics(bool hit, size_t block_size, float latency_ms);
    void adjust_compression_level();
    void adapt_eviction_policy();
    
    // Memory management helpers
    size_t calculate_block_size(const KVBlock* block) const;
    CacheLocation suggest_location(const KVBlock* block) const;
    bool should_compress(const KVBlock* block) const;
};

// Utility functions
std::unique_ptr<KVCacheManager> create_kv_cache_manager(
    size_t num_layers,
    size_t num_heads,
    size_t head_dim,
    size_t max_seq_len = 4096,
    const KVConfig& config = KVConfig()
);

} // namespace jalapeno

#endif // JALAPENO_KV_CACHE_MANAGER_HPP
