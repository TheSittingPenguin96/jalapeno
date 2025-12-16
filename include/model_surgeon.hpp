#ifndef JALAPENO_MODEL_SURGEON_HPP
#define JALAPENO_MODEL_SURGEON_HPP

#include "tensor.hpp"
#include "quantization_kernels.cuh"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <functional>

namespace jalapeno {

// Hardware capabilities
struct HardwareCapabilities {
    std::string name;
    std::string architecture;  // "ampere", "ampere_rtx", "orins", "xavier"
    
    // Compute capabilities
    float peak_tflops_fp32;
    float peak_tflops_fp16;
    float peak_tflops_int8;
    float peak_tflops_int4;
    
    // Memory hierarchy
    size_t gpu_hbm_size;      // High-bandwidth memory
    size_t gpu_vram_size;     // Regular GPU memory
    size_t cpu_ram_size;      // System RAM
    size_t l2_cache_size;
    size_t l1_cache_size;
    size_t shared_memory_size;
    
    // Bandwidths (GB/s)
    float gpu_hbm_bandwidth;
    float gpu_vram_bandwidth;
    float cpu_ram_bandwidth;
    float pcie_bandwidth;     // CPU-GPU
    float nvlink_bandwidth;   // If available
    
    // Specialized units
    bool has_tensor_cores;
    bool has_rt_cores;
    bool has_dla;             // Deep Learning Accelerator
    bool has_nvenc;           // Video encoder
    bool has_nvdec;           // Video decoder
    
    // Power constraints
    float tdp_watts;          // Thermal Design Power
    float typical_power_watts;
    float max_power_watts;
    
    // Supported operations
    std::set<std::string> supported_ops;
    std::map<std::string, float> op_efficiency;  // 0-1 score per operation type
    
    HardwareCapabilities() 
        : peak_tflops_fp32(0), peak_tflops_fp16(0),
          peak_tflops_int8(0), peak_tflops_int4(0),
          gpu_hbm_size(0), gpu_vram_size(0), cpu_ram_size(0),
          l2_cache_size(0), l1_cache_size(0), shared_memory_size(0),
          gpu_hbm_bandwidth(0), gpu_vram_bandwidth(0),
          cpu_ram_bandwidth(0), pcie_bandwidth(0), nvlink_bandwidth(0),
          has_tensor_cores(false), has_rt_cores(false),
          has_dla(false), has_nvenc(false), has_nvdec(false),
          tdp_watts(0), typical_power_watts(0), max_power_watts(0) {}
};

// Layer characteristics
struct LayerProfile {
    std::string name;
    std::string type;  // "linear", "conv2d", "attention", "layernorm", "activation"
    
    // Computational characteristics
    size_t total_flops;
    size_t total_memory_bytes;
    size_t parameters_count;
    size_t activation_size;
    
    // Memory access pattern
    float compute_intensity;     // FLOPs/byte (arithmetic intensity)
    float memory_footprint;      // Bytes accessed
    float data_locality;         // 0-1, higher = more reuse
    float parallelism;           // 0-1, higher = more parallelizable
    
    // Hardware-specific performance
    struct DevicePerformance {
        std::string device_name;
        float execution_time_ms;
        float power_consumption_w;
        float memory_bandwidth_utilization;
        float compute_utilization;
        float efficiency_score;  // Overall score 0-1
        
        bool operator<(const DevicePerformance& other) const {
            return efficiency_score < other.efficiency_score;
        }
    };
    
    std::vector<DevicePerformance> device_performances;
    
    // Dependencies
    std::vector<std::string> input_dependencies;
    std::vector<std::string> output_dependencies;
    
    // Quantization sensitivity
    float quantization_sensitivity;  // 0-1, higher = more sensitive
    std::vector<float> per_tensor_sensitivity;
    
    // Transformations applicable
    std::set<std::string> applicable_transforms;
};

// Surgery operation types
enum class SurgeryOperation {
    NONE = 0,
    QUANTIZE,           // Change precision
    PRUNE,              // Remove weights
    FUSE,               // Fuse layers
    SPLIT,              // Split across devices
    REPLACE_KERNEL,     // Use different implementation
    REORDER,            // Change execution order
    CACHE_OPTIMIZE,     // Optimize memory access
    TILE,               // Tiling for better cache usage
    VECTORIZE,          // Use vector instructions
    STREAM              // Pipeline across devices
};

// Surgery plan
struct SurgeryPlan {
    struct Operation {
        SurgeryOperation type;
        std::string layer_name;
        std::map<std::string, std::string> parameters;
        float estimated_benefit;  // Speedup or memory reduction
        float estimated_cost;     // Time or complexity cost
        int priority;
        
        bool operator<(const Operation& other) const {
            // Higher benefit/cost ratio first
            return (estimated_benefit / (estimated_cost + 1e-6)) < 
                   (other.estimated_benefit / (other.estimated_cost + 1e-6));
        }
    };
    
    std::vector<Operation> operations;
    float total_estimated_speedup;
    float total_memory_reduction;
    float total_power_reduction;
    
    // Execution schedule
    std::map<std::string, std::string> layer_to_device;
    std::map<std::string, DataType> layer_to_precision;
    std::vector<std::vector<std::string>> execution_stages;
    
    void add_operation(const Operation& op) {
        operations.push_back(op);
        std::push_heap(operations.begin(), operations.end());
    }
    
    Operation get_next_operation() {
        if (operations.empty()) {
            return Operation{};
        }
        std::pop_heap(operations.begin(), operations.end());
        Operation op = operations.back();
        operations.pop_back();
        return op;
    }
};

// Cost model for operations
struct CostModel {
    // Operation costs (normalized)
    float quantize_cost;
    float prune_cost;
    float fuse_cost;
    float split_cost;
    float kernel_replace_cost;
    
    // Benefit models
    struct BenefitModel {
        float speedup_per_flop_reduction;
        float memory_saving_benefit;
        float power_saving_benefit;
        float latency_improvement_benefit;
    };
    
    BenefitModel benefits;
    
    // Constraints
    float max_memory_usage;
    float max_power_budget;
    float max_latency;
    float min_accuracy;  // As fraction of original (0-1)
    
    CostModel() 
        : quantize_cost(1.0f), prune_cost(2.0f), fuse_cost(0.5f),
          split_cost(3.0f), kernel_replace_cost(1.5f),
          max_memory_usage(1e9), max_power_budget(100.0f),
          max_latency(1000.0f), min_accuracy(0.95f) {
        benefits.speedup_per_flop_reduction = 0.1f;
        benefits.memory_saving_benefit = 0.05f;
        benefits.power_saving_benefit = 0.02f;
        benefits.latency_improvement_benefit = 0.15f;
    }
};

class HardwareProfiler {
private:
    struct BenchmarkResult {
        std::string kernel_name;
        std::string device_name;
        float execution_time_ms;
        float gflops;
        float bandwidth_gbs;
        float power_w;
        float temperature_c;
        
        // Cache behavior
        size_t l1_hits;
        size_t l1_misses;
        size_t l2_hits;
        size_t l2_misses;
        
        // Instruction mix
        float fp32_ratio;
        float fp16_ratio;
        float int8_ratio;
        float other_ratio;
    };
    
public:
    HardwareProfiler(const std::string& device_name);
    
    HardwareCapabilities profile_capabilities();
    BenchmarkResult benchmark_kernel(
        const std::string& kernel_name,
        const std::vector<size_t>& problem_size,
        DataType precision
    );
    
    std::map<std::string, float> profile_operation(
        const std::string& op_type,
        const std::vector<size_t>& shape,
        DataType precision
    );
    
    float estimate_execution_time(
        const LayerProfile& layer,
        const std::string& device_name,
        DataType precision
    );
    
private:
    std::string device_name_;
    HardwareCapabilities capabilities_;
    
    // Cache for benchmark results
    std::map<std::string, BenchmarkResult> benchmark_cache_;
    
    void run_microbenchmarks();
    void detect_hardware_features();
    std::string get_device_architecture() const;
};

class LayerAnalyzer {
public:
    LayerAnalyzer(
        std::shared_ptr<HardwareProfiler> hardware_profiler,
        const CostModel& cost_model
    );
    
    LayerProfile analyze_layer(
        const std::string& layer_name,
        const std::string& layer_type,
        const std::vector<size_t>& input_shape,
        const std::vector<size_t>& output_shape,
        const std::vector<size_t>& weight_shape,
        const std::map<std::string, std::string>& attributes
    );
    
    float calculate_quantization_sensitivity(
        const LayerProfile& layer,
        const std::vector<float>& weight_data,
        const std::vector<float>& activation_data
    );
    
    std::vector<std::string> get_applicable_transforms(
        const LayerProfile& layer
    );
    
    std::vector<DevicePerformance> rank_devices_for_layer(
        const LayerProfile& layer,
        const std::vector<std::string>& available_devices
    );
    
private:
    std::shared_ptr<HardwareProfiler> hardware_profiler_;
    CostModel cost_model_;
    
    size_t calculate_flops(
        const std::string& layer_type,
        const std::vector<size_t>& input_shape,
        const std::vector<size_t>& weight_shape
    ) const;
    
    size_t calculate_memory_footprint(
        const std::vector<size_t>& input_shape,
        const std::vector<size_t>& weight_shape,
        const std::vector<size_t>& output_shape,
        DataType precision
    ) const;
    
    float estimate_data_locality(
        const std::string& layer_type,
        const std::vector<size_t>& shape
    ) const;
    
    float estimate_parallelism(
        const std::string& layer_type,
        const std::vector<size_t>& shape
    ) const;
};

class ModelSurgeon {
public:
    ModelSurgeon(
        const std::string& model_path,
        std::shared_ptr<HardwareProfiler> hardware_profiler,
        const CostModel& cost_model = CostModel()
    );
    
    // Main surgery interface
    SurgeryPlan create_surgery_plan(
        const std::map<std::string, float>& constraints
    );
    
    bool apply_surgery_plan(const SurgeryPlan& plan);
    
    // Individual transformations
    bool quantize_layer(
        const std::string& layer_name,
        DataType new_precision,
        QuantizationMethod method = QuantizationMethod::INT8
    );
    
    bool prune_layer(
        const std::string& layer_name,
        float sparsity_target,
        PruningMethod method = PruningMethod::MAGNITUDE
    );
    
    bool fuse_layers(
        const std::vector<std::string>& layer_names,
        const std::string& fused_layer_name
    );
    
    bool split_layer(
        const std::string& layer_name,
        const std::vector<std::string>& new_layer_names,
        const std::vector<std::vector<size_t>>& split_points
    );
    
    bool replace_kernel(
        const std::string& layer_name,
        const std::string& new_kernel_name,
        const std::map<std::string, std::string>& kernel_params
    );
    
    // Optimization passes
    void run_optimization_pass(
        const std::string& pass_name,
        const std::map<std::string, std::string>& parameters
    );
    
    // Analysis and reporting
    std::map<std::string, LayerProfile> get_layer_profiles() const;
    SurgeryPlan get_current_plan() const;
    std::map<std::string, float> get_performance_metrics() const;
    
    // Save/load transformed model
    bool save_transformed_model(const std::string& path);
    bool load_model(const std::string& path);
    
private:
    std::string model_path_;
    std::shared_ptr<HardwareProfiler> hardware_profiler_;
    CostModel cost_model_;
    std::unique_ptr<LayerAnalyzer> layer_analyzer_;
    
    // Model state
    std::map<std::string, LayerProfile> layer_profiles_;
    SurgeryPlan current_plan_;
    std::map<std::string, std::string> layer_to_device_;
    std::map<std::string, DataType> layer_to_precision_;
    
    // Transformation history
    std::vector<std::tuple<std::string, SurgeryOperation, float>> transformation_history_;
    
    // Internal methods
    void analyze_model();
    void generate_initial_placement();
    
    SurgeryPlan generate_quantization_plan(float accuracy_constraint);
    SurgeryPlan generate_pruning_plan(float sparsity_target);
    SurgeryPlan generate_fusion_plan();
    SurgeryPlan generate_device_placement_plan();
    
    bool validate_transformation(
        const std::string& layer_name,
        SurgeryOperation operation
    ) const;
    
    float estimate_transformation_benefit(
        const LayerProfile& layer,
        SurgeryOperation operation,
        const std::map<std::string, std::string>& params
    ) const;
    
    float estimate_transformation_cost(
        const LayerProfile& layer,
        SurgeryOperation operation
    ) const;
};

// Specialized transformers
class LayerFuser {
public:
    struct FusionPattern {
        std::vector<std::string> layer_types;
        std::string fused_type;
        std::function<bool(const LayerProfile&, const LayerProfile&)> condition;
        float expected_speedup;
    };
    
    LayerFuser(std::shared_ptr<HardwareProfiler> profiler);
    
    std::vector<std::vector<std::string>> find_fusion_candidates(
        const std::map<std::string, LayerProfile>& layers
    );
    
    bool can_fuse(
        const LayerProfile& layer1,
        const LayerProfile& layer2
    );
    
    LayerProfile create_fused_profile(
        const LayerProfile& layer1,
        const LayerProfile& layer2
    );
    
private:
    std::shared_ptr<HardwareProfiler> profiler_;
    std::vector<FusionPattern> patterns_;
    
    void initialize_patterns();
};

class DevicePlacer {
public:
    struct PlacementSolution {
        std::map<std::string, std::string> layer_to_device;
        float total_cost;
        float total_latency;
        float memory_usage;
        bool valid;
        
        bool operator<(const PlacementSolution& other) const {
            return total_cost < other.total_cost;
        }
    };
    
    DevicePlacer(
        const std::vector<std::string>& available_devices,
        std::shared_ptr<HardwareProfiler> profiler
    );
    
    PlacementSolution find_optimal_placement(
        const std::map<std::string, LayerProfile>& layers,
        const std::map<std::string, float>& constraints
    );
    
    std::vector<PlacementSolution> generate_pareto_front(
        const std::map<std::string, LayerProfile>& layers,
        size_t max_solutions = 10
    );
    
private:
    std::vector<std::string> available_devices_;
    std::shared_ptr<HardwareProfiler> profiler_;
    
    PlacementSolution solve_with_dp(
        const std::map<std::string, LayerProfile>& layers,
        const std::map<std::string, float>& constraints
    );
    
    PlacementSolution solve_with_ilp(
        const std::map<std::string, LayerProfile>& layers,
        const std::map<std::string, float>& constraints
    );
    
    PlacementSolution solve_with_heuristic(
        const std::map<std::string, LayerProfile>& layers,
        const std::map<std::string, float>& constraints
    );
    
    float calculate_communication_cost(
        const std::string& device1,
        const std::string& device2,
        size_t data_size
    ) const;
};

class AutoTuner {
public:
    struct TuningConfiguration {
        std::string kernel_name;
        std::map<std::string, size_t> parameters;  // block_size, tile_size, etc.
        float execution_time;
        float utilization;
        
        bool operator<(const TuningConfiguration& other) const {
            return execution_time < other.execution_time;
        }
    };
    
    AutoTuner(std::shared_ptr<HardwareProfiler> profiler);
    
    TuningConfiguration tune_kernel(
        const std::string& kernel_template,
        const std::vector<size_t>& problem_size,
        DataType precision,
        const std::map<std::string, std::vector<size_t>>& parameter_space
    );
    
    std::map<std::string, TuningConfiguration> tune_model(
        const std::map<std::string, LayerProfile>& layers
    );
    
private:
    std::shared_ptr<HardwareProfiler> profiler_;
    
    TuningConfiguration search_parameter_space(
        const std::string& kernel_template,
        const std::vector<size_t>& problem_size,
        DataType precision,
        const std::map<std::string, std::vector<size_t>>& parameter_space,
        const std::string& search_strategy = "bayesian"
    );
    
    bool validate_configuration(
        const std::string& kernel_template,
        const std::map<std::string, size_t>& parameters,
        const std::vector<size_t>& problem_size
    ) const;
};

// Factory function
std::unique_ptr<ModelSurgeon> create_model_surgeon(
    const std::string& model_path,
    const std::string& hardware_config = "auto"
);

} // namespace jalapeno

#endif // JALAPENO_MODEL_SURGEON_HPP
