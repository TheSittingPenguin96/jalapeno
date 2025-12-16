#ifndef JALAPENO_MARKOV_PREFETCHER_HPP
#define JALAPENO_MARKOV_PREFETCHER_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "tensor.hpp"

namespace jalapeno {

struct AccessPattern {
    std::string from_segment;
    std::string to_segment;
    float transition_probability;
    uint32_t transition_count;
    float average_latency;
    float confidence;
    
    // For adaptive learning
    float recent_boost;  // Recently observed transitions get a boost
    uint32_t last_observed;
    
    bool operator<(const AccessPattern& other) const {
        // Higher probability/confidence first
        return (transition_probability * confidence) > 
               (other.transition_probability * other.confidence);
    }
};

struct ContextFeatures {
    // Input characteristics
    float sequence_length;
    float attention_pattern_entropy;
    float layer_diversity;  // How many different layers accessed
    float temporal_locality;  // Time between accesses
    
    // Model characteristics
    float model_size;  // Relative size
    std::string model_type;  // "llama", "mistral", "phi", etc.
    
    // Hardware state
    float memory_pressure;
    float gpu_utilization;
    
    // Derived features
    std::vector<float> to_vector() const {
        return {
            sequence_length,
            attention_pattern_entropy,
            layer_diversity,
            temporal_locality,
            model_size,
            memory_pressure,
            gpu_utilization
        };
    }
};

class MarkovChain {
private:
    struct State {
        std::string name;
        uint32_t visit_count;
        std::map<std::string, AccessPattern> transitions;
        std::vector<float> feature_vector;  // For similarity matching
        float stationary_probability;
        
        // For temporal patterns
        std::vector<uint64_t> access_timestamps;
        float average_interarrival_time;
    };
    
    std::unordered_map<std::string, State> states_;
    std::unordered_map<std::string, std::string> aliases_;  // For similar states
    std::vector<std::string> state_sequence_;  // History of state visits
    
    // Learning parameters
    float learning_rate_ = 0.1f;
    float decay_factor_ = 0.99f;
    float exploration_rate_ = 0.1f;
    size_t max_history_length_ = 1000;
    
    // Similarity matching
    float similarity_threshold_ = 0.7f;
    
    mutable std::mutex mutex_;
    
public:
    MarkovChain() = default;
    
    void record_transition(const std::string& from_state, 
                          const std::string& to_state,
                          float latency_ms = 0.0f,
                          const ContextFeatures& context = {});
    
    std::vector<std::string> predict_next_states(
        const std::string& current_state,
        const ContextFeatures& context,
        size_t k = 3
    );
    
    std::vector<std::string> predict_sequence(
        const std::string& start_state,
        size_t length,
        const ContextFeatures& context
    );
    
    void update_parameters(const ContextFeatures& context);
    void prune_infrequent_transitions(uint32_t min_count = 2);
    void merge_similar_states(float threshold = 0.8f);
    
    void save_to_file(const std::string& filename) const;
    void load_from_file(const std::string& filename);
    
    // Analysis
    float calculate_entropy(const std::string& state) const;
    float calculate_stationary_distribution() const;
    std::vector<std::string> get_most_likely_path(
        const std::string& start,
        const std::string& end,
        size_t max_steps = 10
    ) const;
    
    // Statistics
    struct Statistics {
        size_t total_states;
        size_t total_transitions;
        float average_branching_factor;
        float average_transition_confidence;
        float prediction_accuracy;  // Tracked externally
    };
    
    Statistics get_statistics() const;
    
private:
    void update_transition_probabilities(State& state);
    float calculate_state_similarity(const State& s1, const State& s2) const;
    float calculate_context_similarity(
        const ContextFeatures& c1,
        const ContextFeatures& c2
    ) const;
    
    void decay_transitions();
    void normalize_probabilities();
    
    std::string find_alias(const std::string& state_name) const;
    void create_alias(const std::string& state1, const std::string& state2);
};

class NeuralMarkovPrefetcher {
private:
    struct NeuralLayer {
        std::vector<float> weights;
        std::vector<float> biases;
        size_t input_size;
        size_t output_size;
        
        std::vector<float> forward(const std::vector<float>& input) const;
        void update(const std::vector<float>& input, 
                   const std::vector<float>& error,
                   float learning_rate);
    };
    
    MarkovChain markov_chain_;
    std::vector<NeuralLayer> neural_layers_;
    
    // Hybrid prediction: combine Markov and neural predictions
    float markov_weight_ = 0.7f;
    float neural_weight_ = 0.3f;
    
    // Training
    bool is_training_ = false;
    size_t training_steps_ = 0;
    float average_prediction_error_ = 0.0f;
    
    mutable std::mutex mutex_;
    
public:
    NeuralMarkovPrefetcher(size_t input_size = 64, size_t hidden_size = 128);
    
    void record_pattern(
        const std::string& current_state,
        const std::string& next_state,
        const ContextFeatures& context,
        float latency_ms = 0.0f
    );
    
    std::vector<std::string> predict(
        const std::string& current_state,
        const ContextFeatures& context,
        size_t k = 3
    );
    
    void train(const std::vector<std::vector<float>>& training_data,
               size_t epochs = 10);
    
    void update_weights(const std::vector<float>& features,
                       const std::vector<std::string>& actual_next,
                       float learning_rate = 0.01f);
    
    float get_prediction_confidence(
        const std::string& current_state,
        const std::string& predicted_state,
        const ContextFeatures& context
    ) const;
    
    void save_model(const std::string& path) const;
    void load_model(const std::string& path);
    
private:
    std::vector<float> extract_features(
        const std::string& state,
        const ContextFeatures& context
    ) const;
    
    std::vector<float> neural_predict(const std::vector<float>& features) const;
    void update_neural_network(
        const std::vector<float>& input,
        const std::vector<float>& target
    );
};

class HierarchicalMarkovPrefetcher {
private:
    struct HierarchyLevel {
        std::map<std::string, MarkovChain> chains;  // One per context type
        size_t abstraction_level;  // 0: fine-grained, higher: more abstract
        float confidence_threshold;
    };
    
    std::vector<HierarchyLevel> levels_;
    std::unordered_map<std::string, std::string> state_to_abstract_;
    
    // Context clustering
    std::vector<ContextFeatures> context_clusters_;
    std::unordered_map<std::string, size_t> state_to_cluster_;
    
public:
    HierarchicalMarkovPrefetcher(size_t num_levels = 3);
    
    void add_state(const std::string& state, const ContextFeatures& context);
    void record_transition(const std::string& from, const std::string& to,
                          const ContextFeatures& context);
    
    std::vector<std::string> predict(
        const std::string& current_state,
        const ContextFeatures& context,
        size_t k = 3
    );
    
    void update_hierarchy();
    void recluster_contexts(size_t num_clusters = 10);
    
private:
    std::string abstract_state(const std::string& state, size_t level) const;
    size_t find_context_cluster(const ContextFeatures& context) const;
    float calculate_context_distance(const ContextFeatures& c1,
                                   const ContextFeatures& c2) const;
};

class PrefetchOrchestrator {
public:
    struct PrefetchDecision {
        std::string segment_id;
        MemoryTier target_tier;
        float priority;
        float estimated_benefit;
        size_t estimated_size;
        bool immediate;  // Load now vs background
    };
    
    PrefetchOrchestrator(
        std::shared_ptr<MarkovChain> markov_chain,
        std::shared_ptr<MemoryHierarchyManager> memory_mgr
    );
    
    std::vector<PrefetchDecision> make_decisions(
        const std::string& current_segment,
        const ContextFeatures& context,
        const std::map<std::string, size_t>& segment_sizes,
        const MemoryStats& memory_stats
    );
    
    void update_feedback(
        const std::string& predicted_segment,
        bool was_used,
        float latency_saved
    );
    
    void adapt_parameters(const MemoryStats& stats);
    
private:
    std::shared_ptr<MarkovChain> markov_chain_;
    std::shared_ptr<MemoryHierarchyManager> memory_mgr_;
    
    // Adaptive parameters
    float aggressiveness_ = 0.5f;  // 0: conservative, 1: aggressive
    float accuracy_threshold_ = 0.6f;
    float memory_pressure_weight_ = 0.3f;
    
    // Feedback learning
    struct Feedback {
        uint32_t total_predictions;
        uint32_t correct_predictions;
        float total_latency_saved;
        float average_confidence;
    };
    
    std::unordered_map<std::string, Feedback> feedback_;
    
    float calculate_prefetch_benefit(
        const std::string& segment,
        float probability,
        size_t segment_size,
        const MemoryStats& stats
    ) const;
    
    MemoryTier select_target_tier(
        float priority,
        size_t size,
        const MemoryStats& stats
    ) const;
};

// Factory function
std::unique_ptr<MarkovChain> create_markov_prefetcher();
std::unique_ptr<NeuralMarkovPrefetcher> create_neural_prefetcher();
std::unique_ptr<HierarchicalMarkovPrefetcher> create_hierarchical_prefetcher();

} // namespace jalapeno

#endif // JALAPENO_MARKOV_PREFETCHER_HPP
