#include "kv_cache_manager.hpp"
#include "hardware_profiler.hpp"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvtx3/nvtx3.hpp>

namespace jalapeno {

// QuantizationCompressor implementation
std::pair<TensorPtr, TensorPtr> QuantizationCompressor::compress(
    const TensorPtr& keys,
    const TensorPtr& values,
    KVCompressionMethod method,
    float quality
) {
    nvtxRangePushA("QuantizationCompressor::compress");
    
    if (method == KVCompressionMethod::NONE) {
        nvtxRangePop();
        return {keys, values};
    }
    
    try {
        // Compute quantization parameters
        auto key_info = compute_quantization_info(keys, method);
        auto value_info = compute_quantization_info(values, method);
        
        // Quantize tensors
        auto quantized_keys = quantize_tensor(keys, key_info, method);
        auto quantized_values = quantize_tensor(values, value_info, method);
        
        // Store quantization info in tensor metadata (simplified)
        // In production, this would be stored separately
        
        nvtxRangePop();
        return {quantized_keys, quantized_values};
        
    } catch (const std::exception& e) {
        std::cerr << "Compression failed: " << e.what() << std::endl;
        nvtxRangePop();
        return {keys, values};  // Fallback to no compression
    }
}

std::pair<TensorPtr, TensorPtr> QuantizationCompressor::decompress(
    const TensorPtr& compressed_keys,
    const TensorPtr& compressed_values,
    KVCompressionMethod method
) {
    nvtxRangePushA("QuantizationCompressor::decompress");
    
    if (method == KVCompressionMethod::NONE) {
        nvtxRangePop();
        return {compressed_keys, compressed_values};
    }
    
    // In production, we would retrieve quantization info from metadata
    // For now, create dummy info
    QuantizationInfo key_info, value_info;
    key_info.scale = 1.0f;
    key_info.zero_point = 0.0f;
    value_info.scale = 1.0f;
    value_info.zero_point = 0.0f;
    
    auto decompressed_keys = dequantize_tensor(compressed_keys, key_info, method);
    auto decompressed_values = dequantize_tensor(compressed_values, value_info, method);
    
    nvtxRangePop();
    return {decompressed_keys, decompressed_values};
}

float QuantizationCompressor::estimate_compression_ratio(
    const TensorPtr& tensor,
    KVCompressionMethod method
) const {
    if (method == KVCompressionMethod::NONE) {
        return 1.0f;
    }
    
    // Estimate based on method
    switch (method) {
        case KVCompressionMethod::INT8_QUANTIZATION:
            return 0.5f;  // 16-bit -> 8-bit = 50%
        case KVCompressionMethod::INT4_QUANTIZATION:
            return 0.25f; // 16-bit -> 4-bit = 25%
        case KVCompressionMethod::NF4_QUANTIZATION:
            return 0.25f;
        default:
            return 0.75f; // Conservative estimate
    }
}

float QuantizationCompressor::estimate_error(
    const TensorPtr& original,
    const TensorPtr& decompressed
) const {
    // Simple MSE estimation
    // In production, would compute actual error
    size_t num_elements = original->metadata().element_count;
    
    if (num_elements == 0) {
        return 0.0f;
    }
    
    // This is a placeholder - actual implementation would compare tensors
    return 0.01f;  // Assume 1% error
}

QuantizationCompressor::QuantizationInfo 
QuantizationCompressor::compute_quantization_info(
    const TensorPtr& tensor,
    KVCompressionMethod method
) const {
    QuantizationInfo info;
    
    // Get tensor data (simplified - assumes float)
    // In production, would compute actual min/max
    float* data = static_cast<float*>(tensor->data());
    size_t num_elements = tensor->metadata().element_count;
    
    if (num_elements == 0) {
        info.min_val = 0.0f;
        info.max_val = 0.0f;
        info.scale = 1.0f;
        info.zero_point = 0.0f;
        return info;
    }
    
    // Find min/max (simplified - in production would use GPU)
    info.min_val = data[0];
    info.max_val = data[0];
    
    for (size_t i = 1; i < num_elements; ++i) {
        if (data[i] < info.min_val) info.min_val = data[i];
        if (data[i] > info.max_val) info.max_val = data[i];
    }
    
    // Compute scale and zero point based on quantization method
    if (method == KVCompressionMethod::INT8_QUANTIZATION) {
        float range = info.max_val - info.min_val;
        info.scale = range / 255.0f;  // 8-bit range
        info.zero_point = -info.min_val / info.scale;
    } else if (method == KVCompressionMethod::INT4_QUANTIZATION) {
        float range = info.max_val - info.min_val;
        info.scale = range / 15.0f;   // 4-bit range
        info.zero_point = -info.min_val / info.scale;
    }
    
    return info;
}

TensorPtr QuantizationCompressor::quantize_tensor(
    const TensorPtr& tensor,
    const QuantizationInfo& info,
    KVCompressionMethod method
) {
    size_t num_elements = tensor->metadata().element_count;
    
    // Create output tensor with appropriate dtype
    DataType output_dtype;
    switch (method) {
        case KVCompressionMethod::INT8_QUANTIZATION:
            output_dtype = DataType::INT8;
            break;
        case KVCompressionMethod::INT4_QUANTIZATION:
            output_dtype = DataType::INT4;
            break;
        case KVCompressionMethod::NF4_QUANTIZATION:
            output_dtype = DataType::NF4;
            break;
        default:
            throw std::runtime_error("Unsupported quantization method");
    }
    
    auto quantized = std::make_shared<UnifiedTensor>(
        tensor->metadata().shape,
        output_dtype
    );
    
    // Quantize data (simplified CPU implementation)
    float* input = static_cast<float*>(tensor->data());
    
    if (method == KVCompressionMethod::INT8_QUANTIZATION) {
        int8_t* output = static_cast<int8_t*>(quantized->data());
        for (size_t i = 0; i < num_elements; ++i) {
            float val = (input[i] - info.min_val) / info.scale;
            output[i] = static_cast<int8_t>(std::round(val));
        }
    }
    // TODO: Implement other quantization methods
    
    return quantized;
}

TensorPtr QuantizationCompressor::dequantize_tensor(
    const TensorPtr& quantized,
    const QuantizationInfo& info,
    KVCompressionMethod method
) {
    size_t num_elements = quantized->metadata().element_count;
    
    // Create output tensor with FP32
    auto decompressed = std::make_shared<UnifiedTensor>(
        quantized->metadata().shape,
        DataType::FP32
    );
    
    float* output = static_cast<float*>(decompressed->data());
    
    if (method == KVCompressionMethod::INT8_QUANTIZATION) {
        int8_t* input = static_cast<int8_t*>(quantized->data());
        for (size_t i = 0; i < num_elements; ++i) {
            output[i] = input[i] * info.scale + info.min_val;
        }
    }
    // TODO: Implement other dequantization methods
    
    return decompressed;
}

// ImportanceAwareEviction implementation
std::vector<KVBlock*> ImportanceAwareEviction::select_eviction_candidates(
    std::vector<KVBlock*>& blocks,
    size_t required_space,
    const std::unordered_map<std::string, float>& context
) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Compute priorities for all blocks
    std::vector<std::pair<KVBlock*, float>> block_priorities;
    for (auto* block : blocks) {
        if (block && !block->is_pinned) {
            float priority = calculate_priority(block);
            block_priorities.emplace_back(block, priority);
        }
    }
    
    // Sort by priority (lower = more evictable)
    std::sort(block_priorities.begin(), block_priorities.end(),
              [](const auto& a, const auto& b) {
                  return a.second < b.second;
              });
    
    // Select candidates until we have enough space
    std::vector<KVBlock*> candidates;
    size_t freed_space = 0;
    
    for (const auto& [block, priority] : block_priorities) {
        size_t block_size = block->keys->size_bytes() + 
                           (block->values ? block->values->size_bytes() : 0);
        
        candidates.push_back(block);
        freed_space += block_size;
        
        if (freed_space >= required_space) {
            break;
        }
    }
    
    return candidates;
}

void ImportanceAwareEviction::record_access(KVBlock* block) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    last_access_times_[block] = now;
    access_counts_[block]++;
    
    // Apply exponential decay to old accesses
    update_importance_decay();
}

void ImportanceAwareEviction::record_importance(KVBlock* block, float importance) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    importance_scores_[block] = importance;
}

float ImportanceAwareEviction::calculate_priority(KVBlock* block) const {
    auto metrics = compute_metrics(block);
    
    // Weighted sum of metrics
    float priority = 
        metrics.recency_score * recency_weight_ +
        metrics.frequency_score * frequency_weight_ +
        metrics.importance_score * importance_weight_ +
        metrics.attention_score * attention_weight_ +
        metrics.size_penalty;
    
    return priority;
}

ImportanceAwareEviction::BlockMetrics 
ImportanceAwareEviction::compute_metrics(KVBlock* block) const {
    BlockMetrics metrics;
    
    // Recency: time since last access (normalized)
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last_access = last_access_times_.count(block) ? 
                      last_access_times_.at(block) : 0;
    
    float seconds_since_access = (now - last_access) / 1e9f;
    metrics.recency_score = expf(-seconds_since_access / 60.0f);  // 60s half-life
    
    // Frequency: log of access count
    uint32_t count = access_counts_.count(block) ? access_counts_.at(block) : 1;
    metrics.frequency_score = logf(1.0f + count) / logf(100.0f);  // Normalized to [0,1]
    
    // Importance: externally provided score
    metrics.importance_score = importance_scores_.count(block) ? 
                              importance_scores_.at(block) : 0.5f;
    
    // Attention score: from block metadata
    metrics.attention_score = block ? block->average_attention_score : 0.5f;
    
    // Size penalty: larger blocks are slightly more evictable
    size_t block_size = block ? (block->keys->size_bytes() + 
                               (block->values ? block->values->size_bytes() : 0)) : 0;
    metrics.size_penalty = block_size / (1024.0f * 1024.0f * 100.0f);  // 100MB = +1 priority
    
    return metrics;
}

void ImportanceAwareEviction::update_importance_decay() {
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    
    // Apply exponential decay to importance scores
    for (auto& [block, score] : importance_scores_) {
        auto last_access = last_access_times_[block];
        float seconds_since_access = (now - last_access) / 1e9f;
        float decay = expf(-seconds_since_access / 300.0f);  // 5-minute half-life
        score *= decay;
    }
}

// KVCacheManager implementation
KVCacheManager::KVCacheManager(
    size_t num_layers,
    size_t num_heads,
    size_t head_dim,
    size_t max_seq_len,
    const KVConfig& config
) : config_(config),
    num_layers_(num_layers),
    num_heads_(num_heads),
    head_dim_(head_dim),
    max_seq_len_(max_seq_len),
    running_(true) {
    
    // Initialize data structures
    layer_head_blocks_.resize(num_layers_);
    for (auto& layer : layer_head_blocks_) {
        layer.resize(num_heads_);
    }
    
    access_patterns_.resize(num_layers_);
    for (auto& layer : access_patterns_) {
        layer.resize(num_heads_);
    }
    
    // Initialize components
    memory_mgr_ = std::make_unique<MemoryHierarchyManager>();
    compressor_ = std::make_unique<QuantizationCompressor>();
    eviction_policy_ = std::make_unique<ImportanceAwareEviction>();
    
    // Initialize memory manager
    std::map<MemoryTier, size_t> capacities;
    for (auto location : config_.storage_hierarchy) {
        MemoryTier tier;
        switch (location) {
            case CacheLocation::GPU_HBM:
            case CacheLocation::GPU_VRAM:
                tier = MemoryTier::TIER_0_GPU_HBM;
                capacities[tier] = 8ULL * 1024 * 1024 * 1024;  // 8GB
                break;
            case CacheLocation::CPU_RAM:
            case CacheLocation::COMPRESSED_RAM:
                tier = MemoryTier::TIER_2_CPU_RAM;
                capacities[tier] = 32ULL * 1024 * 1024 * 1024; // 32GB
                break;
            case CacheLocation::NVME_SSD:
                tier = MemoryTier::TIER_3_NVME_SSD;
                capacities[tier] = 128ULL * 1024 * 1024 * 1024; // 128GB
                break;
            default:
                continue;
        }
    }
    memory_mgr_->initialize(capacities);
    
    // Start worker threads
    eviction_thread_ = std::thread(&KVCacheManager::eviction_worker, this);
    compression_thread_ = std::thread(&KVCacheManager::compression_worker, this);
    
    // Initialize statistics
    stats_.total_blocks = 0;
    stats_.cache_hits = 0;
    stats_.cache_misses = 0;
    stats_.hit_rate = 0.0f;
    stats_.layer_block_counts.resize(num_layers_, 0);
    stats_.layer_hit_rates.resize(num_layers_, 0.0f);
}

KVCacheManager::~KVCacheManager() {
    running_ = false;
    
    // Signal worker threads
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_cv_.notify_all();
    }
    
    // Join threads
    if (eviction_thread_.joinable()) {
        eviction_thread_.join();
    }
    if (compression_thread_.joinable()) {
        compression_thread_.join();
    }
    
    // Clear cache
    clear();
}

std::pair<TensorPtr, TensorPtr> KVCacheManager::get_kv(
    size_t layer_idx,
    size_t head_idx,
    const std::vector<int64_t>& token_positions
) {
    nvtxRangePushA("KVCacheManager::get_kv");
    
    if (layer_idx >= num_layers_ || head_idx >= num_heads_) {
        throw std::out_of_range("Layer or head index out of range");
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Try to find existing block
    KVBlock* block = find_block(layer_idx, head_idx, token_positions);
    
    if (block) {
        // Cache hit
        update_statistics(true, calculate_block_size(block), 0.0f);
        
        // Update access tracking
        eviction_policy_->record_access(block);
        block->last_access_time = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        block->access_count++;
        
        // Decompress if needed
        if (block->is_compressed) {
            decompress_block(block);
        }
        
        auto end_time = std::chrono::steady_clock::now();
        float latency_ms = std::chrono::duration<float, std::milli>(
            end_time - start_time).count();
        
        update_statistics(true, calculate_block_size(block), latency_ms);
        
        nvtxRangePop();
        return {block->keys, block->values};
    }
    
    // Cache miss
    update_statistics(false, 0, 0.0f);
    
    // Create new block (keys/values will be filled by caller)
    block = find_or_create_block(layer_idx, head_idx, token_positions);
    
    auto end_time = std::chrono::steady_clock::now();
    float latency_ms = std::chrono::duration<float, std::milli>(
        end_time - start_time).count();
    
    update_statistics(false, calculate_block_size(block), latency_ms);
    
    nvtxRangePop();
    return {block->keys, block->values};
}

void KVCacheManager::update_kv(
    size_t layer_idx,
    size_t head_idx,
    const std::vector<int64_t>& token_positions,
    const TensorPtr& new_keys,
    const TensorPtr& new_values
) {
    nvtxRangePushA("KVCacheManager::update_kv");
    
    // Find or create block
    KVBlock* block = find_or_create_block(layer_idx, head_idx, token_positions);
    
    // Update block contents
    block->keys = new_keys;
    block->values = new_values;
    block->is_dirty = true;
    block->last_access_time = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    
    // Update importance based on recent access
    float importance = 1.0f / (1.0f + block->access_count);
    eviction_policy_->record_importance(block, importance);
    
    // Check if we need compression
    if (should_compress(block)) {
        compress_block(block, config_.default_compression);
    }
    
    // Check memory pressure
    size_t block_size = calculate_block_size(block);
    current_tokens_ += token_positions.size();
    current_blocks_++;
    
    if (current_tokens_ > config_.max_total_tokens * config_.eviction_threshold) {
        evict_blocks(block_size * 2);  // Evict double what we just added
    }
    
    nvtxRangePop();
}

void KVCacheManager::prefetch_kv(
    size_t layer_idx,
    size_t head_idx,
    const std::vector<int64_t>& predicted_positions
) {
    if (!config_.enable_prefetch || predicted_positions.empty()) {
        return;
    }
    
    // Queue async prefetch operation
    std::lock_guard<std::mutex> lock(async_mutex_);
    async_queue_.push([this, layer_idx, head_idx, predicted_positions]() {
        // Prefetch predicted blocks
        for (size_t i = 0; i < predicted_positions.size(); i += config_.block_size) {
            std::vector<int64_t> positions;
            for (size_t j = 0; j < config_.block_size && i + j < predicted_positions.size(); ++j) {
                positions.push_back(predicted_positions[i + j]);
            }
            
            // Check if block exists
            if (!find_block(layer_idx, head_idx, positions)) {
                // Create empty block to prefetch
                auto block = std::make_unique<KVBlock>();
                block->layer_idx = layer_idx;
                block->head_idx = head_idx;
                block->token_positions = positions;
                block->block_size = positions.size();
                
                // Mark as prefetched (low importance)
                eviction_policy_->record_importance(block.get(), 0.1f);
            }
        }
        
        stats_.prefetch_hits++;
    });
    
    async_cv_.notify_one();
}

void KVCacheManager::compress_block(KVBlock* block, KVCompressionMethod method) {
    if (!block || block->is_compressed || method == KVCompressionMethod::NONE) {
        return;
    }
    
    nvtxRangePushA("KVCacheManager::compress_block");
    
    try {
        // Estimate compression ratio
        float estimated_ratio = compressor_->estimate_compression_ratio(
            block->keys, method);
        
        if (estimated_ratio > config_.compression_ratio_target) {
            // Compression would be effective
            
            auto [compressed_keys, compressed_values] = compressor_->compress(
                block->keys, block->values, method, 0.8f);  // 80% quality
            
            // Update block
            block->keys = compressed_keys;
            block->values = compressed_values;
            block->is_compressed = true;
            block->compression_method = method;
            
            stats_.compressed_blocks++;
            stats_.compression_ratio = 
                (stats_.compression_ratio * (stats_.compressed_blocks - 1) + estimated_ratio) 
                / stats_.compressed_blocks;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Block compression failed: " << e.what() << std::endl;
    }
    
    nvtxRangePop();
}

void KVCacheManager::decompress_block(KVBlock* block) {
    if (!block || !block->is_compressed) {
        return;
    }
    
    nvtxRangePushA("KVCacheManager::decompress_block");
    
    try {
        auto [decompressed_keys, decompressed_values] = compressor_->decompress(
            block->keys, block->values, block->compression_method);
        
        block->keys = decompressed_keys;
        block->values = decompressed_values;
        block->is_compressed = false;
        
    } catch (const std::exception& e) {
        std::cerr << "Block decompression failed: " << e.what() << std::endl;
    }
    
    nvtxRangePop();
}

void KVCacheManager::migrate_block(KVBlock* block, CacheLocation target_location) {
    if (!block) return;
    
    nvtxRangePushA("KVCacheManager::migrate_block");
    
    MemoryTier target_tier;
    switch (target_location) {
        case CacheLocation::GPU_HBM:
        case CacheLocation::GPU_VRAM:
            target_tier = MemoryTier::TIER_0_GPU_HBM;
            break;
        case CacheLocation::CPU_RAM:
        case CacheLocation::COMPRESSED_RAM:
            target_tier = MemoryTier::TIER_2_CPU_RAM;
            break;
        case CacheLocation::NVME_SSD:
            target_tier = MemoryTier::TIER_3_NVME_SSD;
            break;
        default:
            nvtxRangePop();
            return;
    }
    
    // Migrate tensors
    if (block->keys) {
        memory_mgr_->migrate_tensor(block->keys, target_tier);
    }
    if (block->values) {
        memory_mgr_->migrate_tensor(block->values, target_tier);
    }
    
    nvtxRangePop();
}

void KVCacheManager::evict_blocks(size_t required_space) {
    if (!config_.enable_async_eviction) {
        // Synchronous eviction
        std::vector<KVBlock*> all_blocks;
        for (const auto& layer : layer_head_blocks_) {
            for (const auto& head : layer) {
                for (auto* block : head) {
                    if (block) all_blocks.push_back(block);
                }
            }
        }
        
        std::unordered_map<std::string, float> context;
        auto candidates = eviction_policy_->select_eviction_candidates(
            all_blocks, required_space, context);
        
        for (auto* block : candidates) {
            // Remove from data structures
            auto& head_blocks = layer_head_blocks_[block->layer_idx][block->head_idx];
            head_blocks.erase(std::remove(head_blocks.begin(), head_blocks.end(), block),
                            head_blocks.end());
            
            // Remove from block map
            uint64_t block_id = generate_block_id(
                block->layer_idx, block->head_idx, block->token_positions.front());
            block_map_.erase(block_id);
            
            // Update statistics
            current_tokens_ -= block->block_size;
            current_blocks_--;
            stats_.evicted_blocks++;
            
            delete block;
        }
    } else {
        // Queue async eviction
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_queue_.push([this, required_space]() {
            this->evict_blocks(required_space);
        });
        async_cv_.notify_one();
    }
}

void KVCacheManager::clear() {
    // Clear all blocks
    for (auto& layer : layer_head_blocks_) {
        for (auto& head : layer) {
            for (auto* block : head) {
                delete block;
            }
            head.clear();
        }
    }
    
    block_map_.clear();
    current_tokens_ = 0;
    current_blocks_ = 0;
    
    // Reset statistics
    stats_.total_blocks = 0;
    stats_.cache_hits = 0;
    stats_.cache_misses = 0;
    std::fill(stats_.layer_block_counts.begin(), 
             stats_.layer_block_counts.end(), 0);
}

KVBlock* KVCacheManager::find_or_create_block(
    size_t layer_idx,
    size_t head_idx,
    const std::vector<int64_t>& token_positions
) {
    // First, try to find existing block
    KVBlock* block = find_block(layer_idx, head_idx, token_positions);
    if (block) {
        return block;
    }
    
    // Create new block
    auto new_block = std::make_unique<KVBlock>();
    new_block->layer_idx = layer_idx;
    new_block->head_idx = head_idx;
    new_block->token_positions = token_positions;
    new_block->block_size = token_positions.size();
    new_block->last_access_time = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    
    // Create empty tensors for keys and values
    // Shape: [block_size, head_dim]
    std::vector<size_t> shape = {token_positions.size(), head_dim_};
    new_block->keys = std::make_shared<UnifiedTensor>(shape, DataType::FP16);
    new_block->values = std::make_shared<UnifiedTensor>(shape, DataType::FP16);
    
    // Allocate in suggested location
    CacheLocation location = suggest_location(new_block.get());
    MemoryTier tier;
    switch (location) {
        case CacheLocation::GPU_HBM:
        case CacheLocation::GPU_VRAM:
            tier = MemoryTier::TIER_0_GPU_HBM;
            break;
        default:
            tier = MemoryTier::TIER_2_CPU_RAM;
    }
    new_block->keys->allocate(tier);
    new_block->values->allocate(tier);
    
    // Add to data structures
    KVBlock* block_ptr = new_block.release();
    layer_head_blocks_[layer_idx][head_idx].push_back(block_ptr);
    
    uint64_t block_id = generate_block_id(layer_idx, head_idx, token_positions.front());
    block_map_[block_id] = block_ptr;
    
    // Update statistics
    current_tokens_ += token_positions.size();
    current_blocks_++;
    stats_.total_blocks++;
    stats_.layer_block_counts[layer_idx]++;
    
    return block_ptr;
}

KVBlock* KVCacheManager::find_block(
    size_t layer_idx,
    size_t head_idx,
    const std::vector<int64_t>& token_positions
) {
    if (token_positions.empty()) {
        return nullptr;
    }
    
    // For simplicity, check if a block starts at the first token position
    // In production, would need more sophisticated block matching
    uint64_t block_id = generate_block_id(layer_idx, head_idx, token_positions.front());
    
    auto it = block_map_.find(block_id);
    if (it != block_map_.end()) {
        KVBlock* block = it->second;
        // Check if block contains all requested positions
        // (Simplified - assumes block matches exactly)
        return block;
    }
    
    return nullptr;
}

uint64_t KVCacheManager::generate_block_id(
    size_t layer_idx,
    size_t head_idx,
    int64_t start_pos
) const {
    // Combine indices into a single 64-bit ID
    uint64_t id = (static_cast<uint64_t>(layer_idx) << 48) |
                  (static_cast<uint64_t>(head_idx) << 32) |
                  (static_cast<uint64_t>(start_pos) & 0xFFFFFFFF);
    return id;
}

void KVCacheManager::async_worker() {
    while (running_) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(async_mutex_);
            async_cv_.wait(lock, [this]() {
                return !async_queue_.empty() || !running_;
            });
            
            if (!running_ && async_queue_.empty()) {
                break;
            }
            
            if (!async_queue_.empty()) {
                task = std::move(async_queue_.front());
                async_queue_.pop();
            }
        }
        
        if (task) {
            try {
                task();
            } catch (const std::exception& e) {
                std::cerr << "Async task failed: " << e.what() << std::endl;
            }
        }
    }
}

void KVCacheManager::eviction_worker() {
    while (running_) {
        // Periodically check for eviction
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Check memory pressure
        float utilization = static_cast<float>(current_tokens_) / config_.max_total_tokens;
        if (utilization > config_.eviction_threshold) {
            size_t to_free = current_tokens_ - 
                           static_cast<size_t>(config_.max_total_tokens * config_.eviction_threshold * 0.9f);
            evict_blocks(to_free * head_dim_ * sizeof(float) * 2);  // Approximate size
        }
    }
}

void KVCacheManager::compression_worker() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Find blocks that should be compressed
        for (auto& layer : layer_head_blocks_) {
            for (auto& head : layer) {
                for (auto* block : head) {
                    if (block && !block->is_compressed && should_compress(block)) {
                        compress_block(block, config_.default_compression);
                    }
                }
            }
        }
    }
}

void KVCacheManager::update_statistics(bool hit, size_t block_size, float latency_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (hit) {
        stats_.cache_hits++;
    } else {
        stats_.cache_misses++;
    }
    
    size_t total_accesses = stats_.cache_hits + stats_.cache_misses;
    if (total_accesses > 0) {
        stats_.hit_rate = static_cast<float>(stats_.cache_hits) / total_accesses;
    }
    
    if (latency_ms > 0) {
        if (stats_.average_latency_ms == 0) {
            stats_.average_latency_ms = latency_ms;
        } else {
            stats_.average_latency_ms = 
                (stats_.average_latency_ms * 0.9f) + (latency_ms * 0.1f);
        }
    }
}

bool KVCacheManager::should_compress(const KVBlock* block) const {
    if (!block || block->is_compressed) {
        return false;
    }
    
    // Don't compress recently accessed blocks
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    float seconds_since_access = (now - block->last_access_time) / 1e9f;
    
    if (seconds_since_access < 60.0f) {  // Within last minute
        return false;
    }
    
    // Check block importance
    float priority = eviction_policy_->calculate_priority(const_cast<KVBlock*>(block));
    if (priority > 0.7f) {  // High priority = don't compress
        return false;
    }
    
    // Check if block is large enough to benefit from compression
    size_t block_size = calculate_block_size(block);
    if (block_size < 1024 * 1024) {  // Less than 1MB
        return false;
    }
    
    return true;
}

size_t KVCacheManager::calculate_block_size(const KVBlock* block) const {
    if (!block) return 0;
    
    size_t size = 0;
    if (block->keys) {
        size += block->keys->size_bytes();
    }
    if (block->values) {
        size += block->values->size_bytes();
    }
    return size;
}

CacheLocation KVCacheManager::suggest_location(const KVBlock* block) const {
    if (!block) return CacheLocation::CPU_RAM;
    
    // Simple heuristic: put frequently accessed blocks in GPU
    float priority = eviction_policy_->calculate_priority(const_cast<KVBlock*>(block));
    
    if (priority > 0.8f) {  // High priority
        return CacheLocation::GPU_HBM;
    } else if (priority > 0.5f) {  // Medium priority
        return CacheLocation::GPU_VRAM;
    } else if (priority > 0.2f) {  // Low priority
        return CacheLocation::CPU_RAM;
    } else {  // Very low priority
        return CacheLocation::NVME_SSD;
    }
}

KVCacheManager::CacheStats KVCacheManager::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Update counts from current state
    auto mutable_this = const_cast<KVCacheManager*>(this);
    mutable_this->stats_.blocks_in_gpu = 0;
    mutable_this->stats_.blocks_in_cpu = 0;
    mutable_this->stats_.blocks_in_storage = 0;
    mutable_this->stats_.total_tokens = current_tokens_;
    
    for (const auto& layer : layer_head_blocks_) {
        for (const auto& head : layer) {
            for (const auto* block : head) {
                if (block) {
                    if (block->keys->metadata().current_tier == MemoryTier::TIER_0_GPU_HBM) {
                        mutable_this->stats_.blocks_in_gpu++;
                    } else if (block->keys->metadata().current_tier == MemoryTier::TIER_2_CPU_RAM) {
                        mutable_this->stats_.blocks_in_cpu++;
                    } else {
                        mutable_this->stats_.blocks_in_storage++;
                    }
                }
            }
        }
    }
    
    mutable_this->stats_.memory_utilization = 
        static_cast<float>(current_tokens_) / config_.max_total_tokens;
    
    return stats_;
}

} // namespace jalapeno
