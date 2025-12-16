#ifndef JALAPENO_LAYER_STREAMER_HPP
#define JALAPENO_LAYER_STREAMER_HPP

#include "tensor.hpp"
#include "memory_manager.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

namespace jalapeno {

// Forward declarations
class ModelGraph;
class ExecutionPlan;

struct LayerProfile {
    std::string name;
    std::string type;  // Linear, Attention, MLP, Embedding, etc.
    
    // Performance characteristics
    size_t flops;
    size_t memory_footprint;
    size_t activation_size;
    
    // Memory access pattern
    float compute_intensity;  // FLOPs/byte
    float memory_bandwidth_requirement;
    float data_locality;
    
    // Dependencies
    std::vector<std::string> input_layers;
    std::vector<std::string> output_layers;
    
    // Device preferences
    std::vector<MemoryTier> preferred_tiers;
    float gpu_efficiency;  // 0-1 score of how well it runs on GPU
    float cpu_efficiency;
    
    // Quantization tolerance
    float quantization_sensitivity;  // Lower = more tolerant to quantization
};

struct ModelSegment {
    std::string id;
    std::vector<std::string> layer_names;
    size_t memory_footprint;
    float execution_time_ms;
    float activation_memory;
    
    // For dependency tracking
    std::vector<std::string> dependencies;
    std::vector<std::string> dependents;
    
    // Prefetch hints
    std::vector<std::string> likely_next_segments;
    float prefetch_priority;
};

class LayerCache {
private:
    struct CacheEntry {
        std::string segment_id;
        TensorPtr weights;
        TensorPtr biases;
        std::chrono::steady_clock::time_point last_access;
        uint32_t access_count;
        float importance_score;
        bool is_dirty;
        
        bool operator<(const CacheEntry& other) const {
            // Lower importance = more likely to evict
            return importance_score < other.importance_score;
        }
    };
    
    size_t capacity_bytes_;
    size_t current_usage_bytes_;
    
    std::unordered_map<std::string, CacheEntry> entries_;
    std::priority_queue<CacheEntry> eviction_queue_;
    
    std::mutex mutex_;
    
public:
    LayerCache(size_t capacity_bytes);
    
    bool contains(const std::string& segment_id);
    CacheEntry& get(const std::string& segment_id);
    void put(const std::string& segment_id, const CacheEntry& entry);
    void update_importance(const std::string& segment_id, float importance);
    void evict(size_t required_space);
    
    size_t usage() const { return current_usage_bytes_; }
    size_t capacity() const { return capacity_bytes_; }
    float occupancy() const { return (float)current_usage_bytes_ / capacity_bytes_; }
};

class PrefetchPredictor {
public:
    virtual ~PrefetchPredictor() = default;
    
    virtual std::vector<std::string> predict_next_segments(
        const std::string& current_segment,
        const std::vector<float>& context_features
    ) = 0;
    
    virtual void record_transition(
        const std::string& from_segment,
        const std::string& to_segment,
        float transition_time
    ) = 0;
    
    virtual void train() = 0;
};

class MarkovPrefetchPredictor : public PrefetchPredictor {
private:
    struct Transition {
        std::string target_segment;
        uint32_t count;
        float total_time;
        float avg_time;
    };
    
    std::unordered_map<std::string, std::vector<Transition>> transition_matrix_;
    std::unordered_map<std::string, uint32_t> segment_access_count_;
    
    size_t max_predictions_ = 3;
    float confidence_threshold_ = 0.3f;
    
public:
    std::vector<std::string> predict_next_segments(
        const std::string& current_segment,
        const std::vector<float>& context_features
    ) override;
    
    void record_transition(
        const std::string& from_segment,
        const std::string& to_segment,
        float transition_time
    ) override;
    
    void train() override;
    
private:
    void normalize_transitions();
    float calculate_confidence(const Transition& trans) const;
};

class ExecutionPlanner {
public:
    struct ExecutionStep {
        std::string segment_id;
        MemoryTier target_tier;
        float estimated_time_ms;
        std::vector<std::string> dependencies;
        bool can_prefetch;
    };
    
    std::vector<ExecutionStep> create_plan(
        const std::vector<std::string>& segment_ids,
        const std::unordered_map<std::string, ModelSegment>& segments,
        const MemoryHierarchyManager& memory_mgr,
        const DeviceConfig& device_config
    );
    
private:
    float estimate_transfer_time(
        size_t data_size,
        MemoryTier source,
        MemoryTier target,
        const DeviceConfig& config
    ) const;
    
    float estimate_compute_time(
        const ModelSegment& segment,
        MemoryTier tier,
        const DeviceConfig& config
    ) const;
};

class LayerStreamer {
public:
    LayerStreamer(
        const std::string& model_path,
        const DeviceConfig& device_config,
        const MemoryConfig& memory_config
    );
    ~LayerStreamer();
    
    // Model loading and initialization
    bool load_model();
    bool is_loaded() const { return model_loaded_; }
    
    // Execution
    TensorPtr forward(TensorPtr input);
    TensorPtr execute_segment(const std::string& segment_id, TensorPtr input);
    
    // Memory management
    void ensure_segment_resident(const std::string& segment_id);
    void prefetch_segments(const std::vector<std::string>& segment_ids);
    void evict_segment(const std::string& segment_id);
    
    // State management
    void set_checkpoint(const std::string& checkpoint_id);
    void save_checkpoint(const std::string& checkpoint_id);
    
    // Statistics and monitoring
    struct StreamerStats {
        size_t segments_loaded;
        size_t segments_evicted;
        size_t prefetch_hits;
        size_t prefetch_misses;
        size_t cache_hits;
        size_t cache_misses;
        float avg_load_time_ms;
        float gpu_utilization;
        float memory_efficiency;
    };
    
    StreamerStats get_stats() const;
    
    // Configuration
    void set_prefetch_strategy(const std::string& strategy);
    void set_cache_policy(const std::string& policy);
    void set_eviction_threshold(float threshold);
    
private:
    // Model data
    std::string model_path_;
    std::unique_ptr<ModelGraph> model_graph_;
    std::unordered_map<std::string, ModelSegment> segments_;
    std::unordered_map<std::string, LayerProfile> layer_profiles_;
    
    // Memory and compute
    std::unique_ptr<MemoryHierarchyManager> memory_mgr_;
    std::unique_ptr<LayerCache> layer_cache_;
    std::unique_ptr<PrefetchPredictor> prefetch_predictor_;
    std::unique_ptr<ExecutionPlanner> execution_planner_;
    
    // Device configuration
    DeviceConfig device_config_;
    MemoryConfig memory_config_;
    
    // Execution state
    std::string current_segment_;
    std::vector<std::string> execution_history_;
    std::unordered_map<std::string, TensorPtr> intermediate_activations_;
    
    // Async operations
    std::vector<std::future<void>> async_operations_;
    std::queue<std::string> prefetch_queue_;
    std::thread prefetch_thread_;
    std::atomic<bool> prefetch_running_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    StreamerStats stats_;
    
    // Internal methods
    void parse_model();
    void profile_layers();
    void partition_model();
    
    void load_segment_async(const std::string& segment_id);
    void unload_segment_async(const std::string& segment_id);
    
    void prefetch_worker();
    void start_prefetch_thread();
    void stop_prefetch_thread();
    
    std::vector<std::string> get_next_segments_predictive();
    std::vector<std::string> get_next_segments_linear();
    
    bool model_loaded_ = false;
    bool profiling_done_ = false;
};

// Utility functions
std::unique_ptr<LayerStreamer> create_layer_streamer(
    const std::string& model_path,
    const std::string& device = "auto"
);

} // namespace jalapeno

#endif // JALAPENO_LAYER_STREAMER_HPP
