#include "markov_prefetcher.hpp"
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

namespace jalapeno {

// MarkovChain implementation
void MarkovChain::record_transition(const std::string& from_state, 
                                   const std::string& to_state,
                                   float latency_ms,
                                   const ContextFeatures& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Get or create states
    State& from_state_obj = states_[from_state];
    from_state_obj.name = from_state;
    from_state_obj.visit_count++;
    from_state_obj.access_timestamps.push_back(
        std::chrono::steady_clock::now().time_since_epoch().count()
    );
    
    // Keep only recent timestamps
    if (from_state_obj.access_timestamps.size() > 100) {
        from_state_obj.access_timestamps.erase(
            from_state_obj.access_timestamps.begin()
        );
    }
    
    // Update interarrival time
    if (from_state_obj.access_timestamps.size() >= 2) {
        uint64_t total_interval = 0;
        for (size_t i = 1; i < from_state_obj.access_timestamps.size(); ++i) {
            total_interval += from_state_obj.access_timestamps[i] - 
                            from_state_obj.access_timestamps[i-1];
        }
        from_state_obj.average_interarrival_time = 
            total_interval / (from_state_obj.access_timestamps.size() - 1);
    }
    
    // Update transition
    auto& transition = from_state_obj.transitions[to_state];
    if (transition.transition_count == 0) {
        transition.from_segment = from_state;
        transition.to_segment = to_state;
        transition.average_latency = latency_ms;
        transition.last_observed = 0;
        transition.recent_boost = 1.0f;
    }
    
    transition.transition_count++;
    transition.average_latency = 
        (transition.average_latency * (transition.transition_count - 1) + latency_ms) 
        / transition.transition_count;
    
    // Apply recency boost
    transition.recent_boost = std::min(2.0f, transition.recent_boost * 1.1f);
    transition.last_observed = from_state_obj.access_timestamps.size();
    
    // Update transition probabilities
    update_transition_probabilities(from_state_obj);
    
    // Add to state sequence
    state_sequence_.push_back(from_state);
    if (state_sequence_.size() > max_history_length_) {
        state_sequence_.erase(state_sequence_.begin());
    }
    
    // Apply decay to all transitions periodically
    if (from_state_obj.visit_count % 100 == 0) {
        decay_transitions();
    }
}

std::vector<std::string> MarkovChain::predict_next_states(
    const std::string& current_state,
    const ContextFeatures& context,
    size_t k
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> predictions;
    
    auto it = states_.find(current_state);
    if (it == states_.end()) {
        // State not seen before, use similar states
        std::string alias = find_alias(current_state);
        if (!alias.empty()) {
            it = states_.find(alias);
        }
        
        if (it == states_.end()) {
            return predictions;  // No predictions possible
        }
    }
    
    const State& state = it->second;
    
    // Collect transitions with their scores
    std::vector<std::pair<std::string, float>> scored_transitions;
    
    for (const auto& [to_state, pattern] : state.transitions) {
        if (pattern.transition_count == 0) continue;
        
        // Calculate score based on probability and confidence
        float probability = pattern.transition_probability;
        float confidence = pattern.confidence;
        
        // Apply context similarity if available
        float context_score = 1.0f;
        if (!state.feature_vector.empty()) {
            // Simplified context matching
            context_score = 0.8f + 0.2f * (pattern.recent_boost / 2.0f);
        }
        
        float score = probability * confidence * context_score;
        scored_transitions.emplace_back(to_state, score);
    }
    
    // Sort by score (descending)
    std::sort(scored_transitions.begin(), scored_transitions.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
    
    // Take top k
    for (size_t i = 0; i < std::min(k, scored_transitions.size()); ++i) {
        predictions.push_back(scored_transitions[i].first);
    }
    
    return predictions;
}

void MarkovChain::update_transition_probabilities(State& state) {
    uint32_t total_transitions = 0;
    
    // Calculate total transition count
    for (const auto& [to_state, pattern] : state.transitions) {
        total_transitions += pattern.transition_count;
    }
    
    if (total_transitions == 0) return;
    
    // Update probabilities
    for (auto& [to_state, pattern] : state.transitions) {
        float base_probability = static_cast<float>(pattern.transition_count) 
                               / total_transitions;
        
        // Apply recency boost
        pattern.transition_probability = base_probability * pattern.recent_boost;
        
        // Calculate confidence based on observation count
        pattern.confidence = 1.0f - expf(-pattern.transition_count / 10.0f);
    }
    
    // Normalize probabilities
    normalize_probabilities();
}

float MarkovChain::calculate_state_similarity(const State& s1, const State& s2) const {
    // Simple similarity based on common transitions
    if (s1.transitions.empty() || s2.transitions.empty()) {
        return 0.0f;
    }
    
    size_t common = 0;
    for (const auto& [to_state, pattern] : s1.transitions) {
        if (s2.transitions.find(to_state) != s2.transitions.end()) {
            common++;
        }
    }
    
    size_t total = s1.transitions.size() + s2.transitions.size() - common;
    if (total == 0) return 0.0f;
    
    return static_cast<float>(common) / total;
}

void MarkovChain::decay_transitions() {
    // Apply exponential decay to transition counts
    for (auto& [state_name, state] : states_) {
        for (auto& [to_state, pattern] : state.transitions) {
            pattern.transition_count = 
                static_cast<uint32_t>(pattern.transition_count * decay_factor_);
            
            // Decay recency boost
            pattern.recent_boost *= 0.95f;
            if (pattern.recent_boost < 1.0f) {
                pattern.recent_boost = 1.0f;
            }
            
            // Remove infrequent transitions
            if (pattern.transition_count < 1) {
                pattern.transition_count = 0;
            }
        }
        
        // Clean up zero-count transitions
        std::vector<std::string> to_remove;
        for (const auto& [to_state, pattern] : state.transitions) {
            if (pattern.transition_count == 0) {
                to_remove.push_back(to_state);
            }
        }
        
        for (const auto& to_state : to_remove) {
            state.transitions.erase(to_state);
        }
    }
}

void MarkovChain::normalize_probabilities() {
    for (auto& [state_name, state] : states_) {
        float total = 0.0f;
        
        // Calculate total probability
        for (auto& [to_state, pattern] : state.transitions) {
            total += pattern.transition_probability;
        }
        
        if (total > 0.0f) {
            // Normalize
            for (auto& [to_state, pattern] : state.transitions) {
                pattern.transition_probability /= total;
            }
        }
    }
}

std::string MarkovChain::find_alias(const std::string& state_name) const {
    auto it = aliases_.find(state_name);
    if (it != aliases_.end()) {
        return it->second;
    }
    return "";
}

MarkovChain::Statistics MarkovChain::get_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    Statistics stats;
    stats.total_states = states_.size();
    
    size_t total_transitions = 0;
    float total_confidence = 0.0f;
    size_t confidence_count = 0;
    
    for (const auto& [state_name, state] : states_) {
        total_transitions += state.transitions.size();
        
        for (const auto& [to_state, pattern] : state.transitions) {
            total_confidence += pattern.confidence;
            confidence_count++;
        }
    }
    
    stats.total_transitions = total_transitions;
    stats.average_branching_factor = 
        stats.total_states > 0 ? 
        static_cast<float>(total_transitions) / stats.total_states : 0.0f;
    stats.average_transition_confidence = 
        confidence_count > 0 ? total_confidence / confidence_count : 0.0f;
    
    return stats;
}

// NeuralMarkovPrefetcher implementation
std::vector<float> NeuralMarkovPrefetcher::NeuralLayer::forward(
    const std::vector<float>& input
) const {
    std::vector<float> output(output_size, 0.0f);
    
    for (size_t i = 0; i < output_size; ++i) {
        float sum = biases[i];
        for (size_t j = 0; j < input_size; ++j) {
            sum += input[j] * weights[i * input_size + j];
        }
        output[i] = std::tanh(sum);  // Activation
    }
    
    return output;
}

NeuralMarkovPrefetcher::NeuralMarkovPrefetcher(
    size_t input_size, size_t hidden_size
) {
    // Create a simple 2-layer neural network
    NeuralLayer layer1;
    layer1.input_size = input_size;
    layer1.output_size = hidden_size;
    layer1.weights.resize(input_size * hidden_size, 0.01f);
    layer1.biases.resize(hidden_size, 0.0f);
    
    NeuralLayer layer2;
    layer2.input_size = hidden_size;
    layer2.output_size = input_size;  // Predict next state embedding
    layer2.weights.resize(hidden_size * input_size, 0.01f);
    layer2.biases.resize(input_size, 0.0f);
    
    neural_layers_.push_back(layer1);
    neural_layers_.push_back(layer2);
    
    // Initialize with small random weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.01f);
    
    for (auto& layer : neural_layers_) {
        for (auto& w : layer.weights) w = dist(gen);
        for (auto& b : layer.biases) b = dist(gen);
    }
}

void NeuralMarkovPrefetcher::record_pattern(
    const std::string& current_state,
    const std::string& next_state,
    const ContextFeatures& context,
    float latency_ms
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Record in Markov chain
    markov_chain_.record_transition(current_state, next_state, latency_ms, context);
    
    // Also train neural network
    if (is_training_) {
        auto features = extract_features(current_state, context);
        // Simplified: target is one-hot of next state
        // In practice, we'd need a mapping from state to embedding
        
        training_steps_++;
        
        // Update average error
        if (training_steps_ % 100 == 0) {
            // Evaluate prediction accuracy
            auto predictions = predict(current_state, context, 3);
            bool correct = false;
            for (const auto& pred : predictions) {
                if (pred == next_state) {
                    correct = true;
                    break;
                }
            }
            
            float error = correct ? 0.0f : 1.0f;
            average_prediction_error_ = 
                0.9f * average_prediction_error_ + 0.1f * error;
        }
    }
}

std::vector<std::string> NeuralMarkovPrefetcher::predict(
    const std::string& current_state,
    const ContextFeatures& context,
    size_t k
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Get Markov predictions
    auto markov_preds = markov_chain_.predict_next_states(current_state, context, k);
    
    // Get neural predictions
    auto features = extract_features(current_state, context);
    auto neural_output = neural_predict(features);
    
    // Convert neural output to state predictions
    // This is simplified - in practice we'd need a decoder
    std::vector<std::string> neural_preds;
    
    // Hybrid: combine both predictions
    std::vector<std::string> final_predictions;
    
    // Start with Markov predictions
    for (const auto& pred : markov_preds) {
        if (std::find(final_predictions.begin(), final_predictions.end(), pred) == 
            final_predictions.end()) {
            final_predictions.push_back(pred);
        }
    }
    
    // Add neural predictions if different
    // (In real implementation, neural output would be decoded to states)
    
    // Limit to k predictions
    if (final_predictions.size() > k) {
        final_predictions.resize(k);
    }
    
    return final_predictions;
}

std::vector<float> NeuralMarkovPrefetcher::extract_features(
    const std::string& state,
    const ContextFeatures& context
) const {
    // Combine state and context features
    std::vector<float> features;
    
    // State features (simplified hash)
    size_t state_hash = std::hash<std::string>{}(state);
    features.push_back(static_cast<float>(state_hash % 1000) / 1000.0f);
    
    // Context features
    auto context_vec = context.to_vector();
    features.insert(features.end(), context_vec.begin(), context_vec.end());
    
    // Add Markov chain statistics for this state
    auto stats = markov_chain_.get_statistics();
    features.push_back(stats.average_transition_confidence);
    features.push_back(static_cast<float>(stats.total_states) / 1000.0f);
    
    return features;
}

std::vector<float> NeuralMarkovPrefetcher::neural_predict(
    const std::vector<float>& features
) const {
    std::vector<float> current = features;
    
    for (const auto& layer : neural_layers_) {
        current = layer.forward(current);
    }
    
    return current;  // Returns embedding of predicted next state
}

} // namespace jalapeno
