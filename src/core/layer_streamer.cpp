#include "layer_streamer.hpp"
#include "model_parser.hpp"
#include "hardware_profiler.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvtx3.hpp>

namespace jalapeno {

// LayerCache implementation
LayerCache::LayerCache(size_t capacity_bytes) 
    : capacity_bytes_(capacity_bytes), current_usage_bytes_(0) {}

bool LayerCache::contains(const std::string& segment_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.find(segment_id) != entries_.end();
}

LayerCache::CacheEntry& LayerCache::get(const std::string& segment_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(segment_id);
    if (it == entries_.end()) {
        throw std::runtime_error("Segment not in cache: " + segment_id);
    }
    
    // Update access info
    it->second.last_access = std::chrono::steady_clock::now();
    it->second.access_count++;
    
    // Update importance based on recency and frequency
    auto now = std::chrono::steady_clock::now();
    auto seconds_since_access = std::chrono::duration_cast<std::chrono::seconds>(
        now - it->second.last_access).count();
    
    // Exponential decay of importance based on time
    float recency_factor = expf(-seconds_since_access / 60.0f);  // 60s half-life
    float frequency_factor = logf(1.0f + it->second.access_count);
    it->second.importance_score = recency_factor * frequency_factor;
    
    return it->second;
}

void LayerCache::put(const std::string& segment_id, const CacheEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if we need to evict
    if (current_usage_bytes_ + entry.weights->size_bytes() > capacity_bytes_) {
        evict(entry.weights->size_bytes());
    }
    
    entries_[segment_id] = entry;
    eviction_queue_.push(entry);
    current_usage_bytes_ += entry.weights->size_bytes();
    
    if (entry.biases) {
        current_usage_bytes_ += entry.biases->size_bytes();
    }
}

void LayerCache::update_importance(const std::string& segment_id, float importance) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(segment_id);
    if (it != entries_.end()) {
        it->second.importance_score = importance;
        
        // Rebuild priority queue (inefficient but simple for now)
        // TODO: Use a better data structure like Fibonacci heap
        std::priority_queue<CacheEntry> new_queue;
        for (const auto& pair : entries_) {
            new_queue.push(pair.second);
        }
        eviction_queue_ = std::move(new_queue);
    }
}

void LayerCache::evict(size_t required_space) {
    std::vector<CacheEntry> kept_entries;
    size_t freed_bytes = 0;
    
    while (!eviction_queue_.empty() && freed_bytes < required_space) {
        CacheEntry entry = eviction_queue_.top();
        eviction_queue_.pop();
        
        if (entries_.find(entry.segment_id) != entries_.end()) {
            // Actually evict
            freed_bytes += entry.weights->size_bytes();
            if (entry.biases) {
                freed_bytes += entry.biases->size_bytes();
            }
            
            entries_.erase(entry.segment_id);
            
            // If segment is dirty, save to storage
            if (entry.is_dirty) {
                // TODO: Save to disk
            }
        }
    }
    
    current_usage_bytes_ -= freed_bytes;
}

// MarkovPrefetchPredictor implementation
std::vector<std::string> MarkovPrefetchPredictor::predict_next_segments(
    const std::string& current_segment,
    const std::vector<float>& context_features
) {
    std::vector<std::string> predictions;
    
    auto it = transition_matrix_.find(current_segment);
    if (it == transition_matrix_.end()) {
        return predictions;
    }
    
    // Get transitions from current segment
    const auto& transitions = it->second;
    
    // Calculate scores for each possible transition
    std::vector<std::pair<std::string, float>> scored_transitions;
    for (const auto& trans : transitions) {
        float confidence = calculate_confidence(trans);
        if (confidence >= confidence_threshold_) {
            scored_transitions.emplace_back(trans.target_segment, confidence);
        }
    }
    
    // Sort by confidence
    std::sort(scored_transitions.begin(), scored_transitions.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
    
    // Take top K predictions
    for (size_t i = 0; i < std::min(max_predictions_, scored_transitions.size()); i++) {
        predictions.push_back(scored_transitions[i].first);
    }
    
    return predictions;
}

void MarkovPrefetchPredictor::record_transition(
    const std::string& from_segment,
    const std::string& to_segment,
    float transition_time
) {
    auto& transitions = transition_matrix_[from_segment];
    
    // Find existing transition
    bool found = false;
    for (auto& trans : transitions) {
        if (trans.target_segment == to_segment) {
            trans.count++;
            trans.total_time += transition_time;
            trans.avg_time = trans.total_time / trans.count;
            found = true;
            break;
        }
    }
    
    // Add new transition if not found
    if (!found) {
        transitions.push_back({to_segment, 1, transition_time, transition_time});
    }
    
    segment_access_count_[from_segment]++;
    
    // Periodically normalize
    if (segment_access_count_[from_segment] % 100 == 0) {
        normalize_transitions();
    }
}

void MarkovPrefetchPredictor::train() {
    normalize_transitions();
}

void MarkovPrefetchPredictor::normalize_transitions() {
    for (auto& pair : transition_matrix_) {
        auto& transitions = pair.second;
        uint32_t total_count = 0;
        
        for (const auto& trans : transitions) {
            total_count += trans.count;
        }
        
        // Normalize counts to probabilities (optional)
        // Could be used for better confidence calculation
    }
}

float MarkovPrefetchPredictor::calculate_confidence(const Transition& trans) const {
    // Confidence based on frequency and recency
    float frequency_confidence = 1.0f - expf(-trans.count / 10.0f);
    
    // Higher confidence for faster transitions (less likely to be blocked by compute)
    float speed_confidence = 1.0f / (1.0f + trans.avg_time / 100.0f);
    
    return frequency_confidence * speed_confidence;
}

// ExecutionPlanner implementation
std::vector<ExecutionPlanner::ExecutionStep> 
ExecutionPlanner::create_plan(
    const std::vector<std::string>& segment_ids,
    const std::unordered_map<std::string, ModelSegment>& segments,
    const MemoryHierarchyManager& memory_mgr,
    const DeviceConfig& device_config
) {
    std::vector<ExecutionStep> plan;
    
    for (const auto& seg_id : segment_ids) {
        auto seg_it = segments.find(seg_id);
        if (seg_it == segments.end()) {
            continue;
        }
        
        const auto& segment = seg_it->second;
        
        ExecutionStep step;
        step.segment_id = seg_id;
        step.dependencies = segment.dependencies;
        
        // Choose target tier based on segment characteristics
        // For now, simple heuristic: put compute-intensive segments on GPU
        if (segment.memory_footprint < device_config.gpu_memory_limit_mb * 1024 * 1024) {
            step.target_tier = MemoryTier::TIER_0_GPU_HBM;
        } else {
            step.target_tier = MemoryTier::TIER_2_CPU_RAM;
        }
        
        // Estimate times
        float transfer_time = estimate_transfer_time(
            segment.memory_footprint,
            MemoryTier::TIER_3_NVME_SSD,  // Assuming loading from disk
            step.target_tier,
            device_config
        );
        
        float compute_time = estimate_compute_time(
            segment,
            step.target_tier,
            device_config
        );
        
        step.estimated_time_ms = transfer_time + compute_time;
        step.can_prefetch = transfer_time > compute_time;  // Prefetch if transfer is bottleneck
        
        plan.push_back(step);
    }
    
    // Sort by dependencies and estimated time
    std::sort(plan.begin(), plan.end(), [](const ExecutionStep& a, const ExecutionStep& b) {
        // Check if b depends on a
        if (std::find(b.dependencies.begin(), b.dependencies.end(), a.segment_id) != b.dependencies.end()) {
            return true;  // a before b
        }
        // Check if a depends on b
        if (std::find(a.dependencies.begin(), a.dependencies.end(), b.segment_id) != a.dependencies.end()) {
            return false;  // b before a
        }
        // No dependency, sort by estimated time
        return a.estimated_time_ms < b.estimated_time_ms;
    });
    
    return plan;
}

float ExecutionPlanner::estimate_transfer_time(
    size_t data_size,
    MemoryTier source,
    MemoryTier target,
    const DeviceConfig& config
) const {
    // Simplified bandwidth model
    const std::map<std::pair<MemoryTier, MemoryTier>, float> bandwidths = {
        {{MemoryTier::TIER_3_NVME_SSD, MemoryTier::TIER_2_CPU_RAM}, 3.0f * 1024 * 1024},  // 3 GB/s NVMe
        {{MemoryTier::TIER_2_CPU_RAM, MemoryTier::TIER_0_GPU_HBM}, 16.0f * 1024 * 1024},  // 16 GB/s PCIe
        {{MemoryTier::TIER_3_NVME_SSD, MemoryTier::TIER_0_GPU_HBM}, 3.0f * 1024 * 1024},  // Limited by NVMe
    };
    
    auto key = std::make_pair(source, target);
    auto it = bandwidths.find(key);
    if (it == bandwidths.end()) {
        // Default to slowest
        return data_size / (1.0f * 1024 * 1024);  // 1 MB/s
    }
    
    float bandwidth = it->second;  // Bytes per second
    return (data_size / bandwidth) * 1000.0f;  // Convert to ms
}

float ExecutionPlanner::estimate_compute_time(
    const ModelSegment& segment,
    MemoryTier tier,
    const DeviceConfig& config
) const {
    // Simplified compute model
    // TODO: Use actual hardware profiling
    float flops_per_sec = 0.0f;
    
    switch (tier) {
        case MemoryTier::TIER_0_GPU_HBM:
            flops_per_sec = config.gpu_flops;  // Should be set in config
            break;
        case MemoryTier::TIER_2_CPU_RAM:
            flops_per_sec = config.cpu_flops;
            break;
        default:
            flops_per_sec = 1e9;  // 1 GFLOP/s default
    }
    
    return (segment.memory_footprint * 2) / flops_per_sec * 1000.0f;  // Rough estimate
}

// LayerStreamer implementation
LayerStreamer::LayerStreamer(
    const std::string& model_path,
    const DeviceConfig& device_config,
    const MemoryConfig& memory_config
) : model_path_(model_path),
    device_config_(device_config),
    memory_config_(memory_config),
    prefetch_running_(false) {
    
    // Initialize components
    memory_mgr_ = std::make_unique<MemoryHierarchyManager>();
    layer_cache_ = std::make_unique<LayerCache>(
        memory_config.gpu_memory_limit_mb * 1024 * 1024
    );
    prefetch_predictor_ = std::make_unique<MarkovPrefetchPredictor>();
    execution_planner_ = std::make_unique<ExecutionPlanner>();
}

LayerStreamer::~LayerStreamer() {
    stop_prefetch_thread();
    
    // Wait for async operations
    for (auto& future : async_operations_) {
        if (future.valid()) {
            future.wait();
        }
    }
}

bool LayerStreamer::load_model() {
    nvtxRangePushA("LayerStreamer::load_model");
    
    try {
        // 1. Parse model structure
        parse_model();
        
        // 2. Profile layers (can be async)
        profile_layers();
        
        // 3. Partition model into segments
        partition_model();
        
        // 4. Initialize memory manager
        std::map<MemoryTier, size_t> capacities = {
            {MemoryTier::TIER_0_GPU_HBM, device_config_.gpu_memory_limit_mb * 1024 * 1024},
            {MemoryTier::TIER_2_CPU_RAM, device_config_.cpu_memory_limit_mb * 1024 * 1024},
        };
        memory_mgr_->initialize(capacities);
        
        // 5. Start prefetch thread
        start_prefetch_thread();
        
        model_loaded_ = true;
        profiling_done_ = true;
        
        nvtxRangePop();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        nvtxRangePop();
        return false;
    }
}

TensorPtr LayerStreamer::forward(TensorPtr input) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    nvtxRangePushA("LayerStreamer::forward");
    
    TensorPtr current_output = input;
    
    // Execute segments in order
    for (const auto& segment_pair : segments_) {
        const std::string& segment_id = segment_pair.first;
        
        // Ensure segment is resident
        ensure_segment_resident(segment_id);
        
        // Execute segment
        nvtxRangePushA(("Segment: " + segment_id).c_str());
        current_output = execute_segment(segment_id, current_output);
        nvtxRangePop();
        
        // Record transition for prefetch predictor
        if (!current_segment_.empty()) {
            prefetch_predictor_->record_transition(
                current_segment_,
                segment_id,
                0.0f  // TODO: Measure actual transition time
            );
        }
        current_segment_ = segment_id;
        execution_history_.push_back(segment_id);
        
        // Update intermediate activations
        intermediate_activations_[segment_id] = current_output;
        
        // Trigger prefetch for next likely segments
        auto next_segments = get_next_segments_predictive();
        prefetch_segments(next_segments);
        
        // Check memory pressure and evict if needed
        if (layer_cache_->occupancy() > 0.9f) {  // 90% full
            // Evict least important segments not in immediate use
            for (const auto& seg : segments_) {
                if (seg.first != segment_id && 
                    !seg.second.dependents.empty() &&  // Has dependents
                    layer_cache_->contains(seg.first)) {
                    // Check if this segment will be needed soon
                    bool needed_soon = false;
                    for (const auto& next_seg : next_segments) {
                        if (next_seg == seg.first) {
                            needed_soon = true;
                            break;
                        }
                    }
                    
                    if (!needed_soon) {
                        evict_segment(seg.first);
                        break;
                    }
                }
            }
        }
    }
    
    nvtxRangePop();
    return current_output;
}

TensorPtr LayerStreamer::execute_segment(const std::string& segment_id, TensorPtr input) {
    // Get segment from cache
    auto& cache_entry = layer_cache_->get(segment_id);
    
    // Update cache importance
    auto now = std::chrono::steady_clock::now();
    auto seconds_since_access = std::chrono::duration_cast<std::chrono::seconds>(
        now - cache_entry.last_access).count();
    float importance = expf(-seconds_since_access / 30.0f);  // 30s half-life
    layer_cache_->update_importance(segment_id, importance);
    
    // Execute the segment (placeholder - actual execution depends on model type)
    // This would integrate with actual model execution framework
    
    // For now, return input as placeholder
    return input;
}

void LayerStreamer::ensure_segment_resident(const std::string& segment_id) {
    nvtxRangePushA(("ensure_resident: " + segment_id).c_str());
    
    if (layer_cache_->contains(segment_id)) {
        // Cache hit
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.cache_hits++;
        }
        nvtxRangePop();
        return;
    }
    
    // Cache miss - need to load
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.cache_misses++;
    }
    
    // Load segment synchronously (could be async in background)
    load_segment_async(segment_id);
    
    // Wait for load to complete
    for (auto& future : async_operations_) {
        if (future.valid()) {
            future.wait();
        }
    }
    
    nvtxRangePop();
}

void LayerStreamer::prefetch_segments(const std::vector<std::string>& segment_ids) {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    
    for (const auto& seg_id : segment_ids) {
        // Check if already in cache or already queued
        if (!layer_cache_->contains(seg_id)) {
            bool already_queued = false;
            std::queue<std::string> temp_queue = prefetch_queue_;
            while (!temp_queue.empty()) {
                if (temp_queue.front() == seg_id) {
                    already_queued = true;
                    break;
                }
                temp_queue.pop();
            }
            
            if (!already_queued) {
                prefetch_queue_.push(seg_id);
                
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.prefetch_misses++;
                }
            } else {
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.prefetch_hits++;
                }
            }
        }
    }
    
    // Notify prefetch thread
    prefetch_cv_.notify_one();
}

void LayerStreamer::evict_segment(const std::string& segment_id) {
    if (layer_cache_->contains(segment_id)) {
        unload_segment_async(segment_id);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.segments_evicted++;
        }
    }
}

void LayerStreamer::load_segment_async(const std::string& segment_id) {
    // Launch async load operation
    auto future = std::async(std::launch::async, [this, segment_id]() {
        nvtxRangePushA(("load_segment: " + segment_id).c_str());
        
        // Simulate loading from disk
        auto start = std::chrono::steady_clock::now();
        
        // TODO: Actual loading logic
        // 1. Load weights from disk
        // 2. Decompress if needed
        // 3. Transfer to GPU
        
        auto end = std::chrono::steady_clock::now();
        float load_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.segments_loaded++;
            stats_.avg_load_time_ms = 
                (stats_.avg_load_time_ms * (stats_.segments_loaded - 1) + load_time_ms) / stats_.segments_loaded;
        }
        
        nvtxRangePop();
    });
    
    async_operations_.push_back(std::move(future));
}

void LayerStreamer::unload_segment_async(const std::string& segment_id) {
    auto future = std::async(std::launch::async, [this, segment_id]() {
        // Remove from cache
        // If dirty, save to disk first
        
        // For now, just remove from our tracking
        // Actual cache eviction happens in LayerCache::evict
    });
    
    async_operations_.push_back(std::move(future));
}

void LayerStreamer::prefetch_worker() {
    while (prefetch_running_) {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        
        // Wait for work or shutdown
        prefetch_cv_.wait(lock, [this]() {
            return !prefetch_queue_.empty() || !prefetch_running_;
        });
        
        if (!prefetch_running_) {
            break;
        }
        
        // Process prefetch queue
        while (!prefetch_queue_.empty()) {
            std::string segment_id = prefetch_queue_.front();
            prefetch_queue_.pop();
            
            // Release lock during loading
            lock.unlock();
            
            // Load segment if not already cached
            if (!layer_cache_->contains(segment_id)) {
                load_segment_async(segment_id);
            }
            
            lock.lock();
        }
    }
}

void LayerStreamer::start_prefetch_thread() {
    prefetch_running_ = true;
    prefetch_thread_ = std::thread(&LayerStreamer::prefetch_worker, this);
}

void LayerStreamer::stop_prefetch_thread() {
    prefetch_running_ = false;
    prefetch_cv_.notify_all();
    
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
}

std::vector<std::string> LayerStreamer::get_next_segments_predictive() {
    std::vector<float> context_features;
    // Extract features from current context
    // (e.g., current segment, recent history, input characteristics)
    
    return prefetch_predictor_->predict_next_segments(
        current_segment_,
        context_features
    );
}

std::vector<std::string> LayerStreamer::get_next_segments_linear() {
    // Simple linear prediction: next segment in order
    auto current_it = segments_.find(current_segment_);
    if (current_it == segments_.end()) {
        return {};
    }
    
    // Find next segment in dependency order
    std::vector<std::string> next_segments;
    
    for (const auto& dependent : current_it->second.dependents) {
        next_segments.push_back(dependent);
    }
    
    return next_segments;
}

void LayerStreamer::parse_model() {
    // TODO: Implement model parsing
    // This would use ONNX, PyTorch, or custom format
    // For now, create dummy segments for testing
    
    // Example for a 13B parameter model with 40 layers
    size_t num_layers = 40;
    size_t segment_size = 4;  // Layers per segment
    
    for (size_t i = 0; i < num_layers; i += segment_size) {
        std::string seg_id = "segment_" + std::to_string(i / segment_size);
        
        ModelSegment segment;
        segment.id = seg_id;
        segment.memory_footprint = 1024 * 1024 * 1024;  // 1GB per segment
        segment.execution_time_ms = 10.0f;
        
        // Create layer names
        for (size_t j = 0; j < segment_size && (i + j) < num_layers; j++) {
            segment.layer_names.push_back("layer_" + std::to_string(i + j));
        }
        
        // Set dependencies (linear chain)
        if (i > 0) {
            std::string prev_seg = "segment_" + std::to_string((i / segment_size) - 1);
            segment.dependencies.push_back(prev_seg);
            
            // Update previous segment's dependents
            auto prev_it = segments_.find(prev_seg);
            if (prev_it != segments_.end()) {
                prev_it->second.dependents.push_back(seg_id);
            }
        }
        
        segments_[seg_id] = segment;
    }
}

void LayerStreamer::profile_layers() {
    // TODO: Implement actual profiling
    // This would run each layer on different hardware and measure performance
    
    for (auto& segment_pair : segments_) {
        auto& segment = segment_pair.second;
        
        // Create dummy profile
        LayerProfile profile;
        profile.name = segment.id;
        profile.type = "TransformerBlock";
        profile.flops = segment.memory_footprint * 2;  // Rough estimate
        profile.memory_footprint = segment.memory_footprint;
        profile.compute_intensity = 10.0f;  // FLOPs/byte
        profile.gpu_efficiency = 0.9f;
        profile.cpu_efficiency = 0.3f;
        
        layer_profiles_[segment.id] = profile;
    }
    
    profiling_done_ = true;
}

void LayerStreamer::partition_model() {
    // Partitioning already done in parse_model for now
    // Could implement more sophisticated partitioning here
}

LayerStreamer::StreamerStats LayerStreamer::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void LayerStreamer::set_prefetch_strategy(const std::string& strategy) {
    // TODO: Implement different strategies
}

void LayerStreamer::set_cache_policy(const std::string& policy) {
    // TODO: Implement different cache policies
}

void LayerStreamer::set_eviction_threshold(float threshold) {
    // TODO: Implement threshold adjustment
}

} // namespace jalapeno
