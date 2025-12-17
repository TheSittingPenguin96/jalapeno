#include "model_surgeon.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvtx3.hpp>

namespace jalapeno {

// HardwareProfiler implementation
HardwareProfiler::HardwareProfiler(const std::string& device_name) 
    : device_name_(device_name) {
    detect_hardware_features();
    run_microbenchmarks();
}

HardwareCapabilities HardwareProfiler::profile_capabilities() {
    nvtxRangePushA("HardwareProfiler::profile_capabilities");
    
    HardwareCapabilities caps;
    caps.name = device_name_;
    caps.architecture = get_device_architecture();
    
    // Get CUDA device properties
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        caps.gpu_vram_size = prop.totalGlobalMem;
        caps.shared_memory_size = prop.sharedMemPerBlock;
        caps.l2_cache_size = prop.l2CacheSize;
        
        // Estimate capabilities based on architecture
        if (caps.architecture.find("ampere") != std::string::npos) {
            caps.has_tensor_cores = true;
            caps.peak_tflops_fp32 = prop.multiProcessorCount * 192 * prop.clockRate * 1e-6 * 2 / 1e3;  // Approximate
            caps.peak_tflops_fp16 = caps.peak_tflops_fp32 * 8;  // Tensor cores
            caps.peak_tflops_int8 = caps.peak_tflops_fp16 * 2;
            caps.peak_tflops_int4 = caps.peak_tflops_int8 * 2;
        } else if (caps.architecture.find("turing") != std::string::npos) {
            caps.has_tensor_cores = true;
            caps.peak_tflops_fp32 = prop.multiProcessorCount * 64 * prop.clockRate * 1e-6 * 2 / 1e3;
            caps.peak_tflops_fp16 = caps.peak_tflops_fp32 * 4;
            caps.peak_tflops_int8 = caps.peak_tflops_fp16 * 2;
        }
        
        // Memory bandwidth
        caps.gpu_vram_bandwidth = prop.memoryBusWidth * prop.memoryClockRate * 2.0 * 1e-6 / 8.0;
        
        // Supported operations
        caps.supported_ops = {
            "conv2d", "linear", "matmul", "attention",
            "layernorm", "softmax", "gelu", "relu"
        };
        
        // Operation efficiencies (estimated)
        caps.op_efficiency["conv2d"] = 0.8f;
        caps.op_efficiency["linear"] = 0.9f;
        caps.op_efficiency["matmul"] = 0.95f;
        caps.op_efficiency["attention"] = 0.7f;
        caps.op_efficiency["layernorm"] = 0.6f;
    }
    
    // CPU capabilities (simplified)
    caps.cpu_ram_size = 32ULL * 1024 * 1024 * 1024;  // Assume 32GB
    caps.cpu_ram_bandwidth = 50.0f;  // GB/s typical
    
    // Power constraints (Jetson specific)
    if (device_name_.find("jetson") != std::string::npos) {
        if (device_name_.find("orin") != std::string::npos) {
            caps.tdp_watts = 15.0f;  // Orin Nano/NX
            caps.max_power_watts = 25.0f;
        } else if (device_name_.find("xavier") != std::string::npos) {
            caps.tdp_watts = 10.0f;
            caps.max_power_watts = 15.0f;
        }
    } else {
        caps.tdp_watts = 100.0f;  // Desktop GPU
        caps.max_power_watts = 200.0f;
    }
    
    caps.typical_power_watts = caps.tdp_watts * 0.7f;
    
    nvtxRangePop();
    return caps;
}

HardwareProfiler::BenchmarkResult HardwareProfiler::benchmark_kernel(
    const std::string& kernel_name,
    const std::vector<size_t>& problem_size,
    DataType precision
) {
    nvtxRangePushA(("benchmark_kernel: " + kernel_name).c_str());
    
    // Check cache
    std::string cache_key = kernel_name + "_" + 
                           std::to_string(problem_size[0]) + "_" +
                           std::to_string(precision);
    
    auto it = benchmark_cache_.find(cache_key);
    if (it != benchmark_cache_.end()) {
        nvtxRangePop();
        return it->second;
    }
    
    BenchmarkResult result;
    result.kernel_name = kernel_name;
    result.device_name = device_name_;
    
    // Simplified benchmarking
    // In production, would run actual kernels
    
    // Estimate based on problem size and hardware
    size_t total_elements = 1;
    for (auto dim : problem_size) {
        total_elements *= dim;
    }
    
    // Rough estimation formula
    float bytes_per_element = (precision == DataType::FP32) ? 4.0f :
                             (precision == DataType::FP16) ? 2.0f :
                             (precision == DataType::INT8) ? 1.0f : 0.5f;
    
    size_t total_bytes = total_elements * bytes_per_element;
    
    // Estimate execution time (simplified)
    auto& caps = capabilities_;
    float compute_capability = (precision == DataType::FP32) ? caps.peak_tflops_fp32 :
                              (precision == DataType::FP16) ? caps.peak_tflops_fp16 :
                              (precision == DataType::INT8) ? caps.peak_tflops_int8 :
                              caps.peak_tflops_int4;
    
    // FLOPs estimation (rough)
    size_t flops = total_elements * 2;  // 2 FLOPs per element (multiply-add)
    
    result.execution_time_ms = (flops / (compute_capability * 1e9)) * 1000.0f;
    result.gflops = flops / (result.execution_time_ms * 1e6);
    result.bandwidth_gbs = total_bytes / (result.execution_time_ms * 1e6);
    
    // Power estimation
    result.power_w = caps.typical_power_watts * 
                    (result.gflops / (compute_capability * 1e3));
    
    // Cache behavior estimation
    result.l1_hits = total_elements * 0.8f;  // 80% hit rate
    result.l1_misses = total_elements * 0.2f;
    result.l2_hits = result.l1_misses * 0.6f;
    result.l2_misses = result.l1_misses * 0.4f;
    
    // Cache result
    benchmark_cache_[cache_key] = result;
    
    nvtxRangePop();
    return result;
}

float HardwareProfiler::estimate_execution_time(
    const LayerProfile& layer,
    const std::string& device_name,
    DataType precision
) {
    // Find device performance for this layer
    for (const auto& perf : layer.device_performances) {
        if (perf.device_name == device_name) {
            return perf.execution_time_ms;
        }
    }
    
    // Estimate if not profiled
    size_t flops = layer.total_flops;
    
    float compute_capability;
    if (precision == DataType::FP32) {
        compute_capability = capabilities_.peak_tflops_fp32;
    } else if (precision == DataType::FP16) {
        compute_capability = capabilities_.peak_tflops_fp16;
    } else if (precision == DataType::INT8) {
        compute_capability = capabilities_.peak_tflops_int8;
    } else {
        compute_capability = capabilities_.peak_tflops_int4;
    }
    
    // Account for operation efficiency
    float efficiency = 0.7f;  // Default
    auto it = capabilities_.op_efficiency.find(layer.type);
    if (it != capabilities_.op_efficiency.end()) {
        efficiency = it->second;
    }
    
    float time_ms = (flops / (compute_capability * 1e9 * efficiency)) * 1000.0f;
    
    // Add memory transfer time if needed
    float memory_time_ms = layer.total_memory_bytes / 
                          (capabilities_.gpu_vram_bandwidth * 1e6);
    
    return time_ms + memory_time_ms;
}

void HardwareProfiler::run_microbenchmarks() {
    // Run a set of microbenchmarks to characterize hardware
    std::vector<std::string> benchmark_kernels = {
        "matmul_fp32", "matmul_fp16", "conv2d_fp32", "attention_fp16"
    };
    
    for (const auto& kernel : benchmark_kernels) {
        std::vector<size_t> problem_size = {1024, 1024, 1024};
        DataType precision = (kernel.find("fp32") != std::string::npos) ? 
                            DataType::FP32 : DataType::FP16;
        
        auto result = benchmark_kernel(kernel, problem_size, precision);
        benchmark_cache_[kernel] = result;
    }
}

void HardwareProfiler::detect_hardware_features() {
    // Detect Jetson-specific features
    if (device_name_.find("jetson") != std::string::npos) {
        // Check for Orin features
        capabilities_.has_dla = (device_name_.find("orin") != std::string::npos);
        capabilities_.has_nvenc = true;
        capabilities_.has_nvdec = true;
        
        // Unified memory
        capabilities_.gpu_hbm_size = capabilities_.gpu_vram_size;
        capabilities_.gpu_hbm_bandwidth = capabilities_.gpu_vram_bandwidth;
    }
}

std::string HardwareProfiler::get_device_architecture() const {
    // Simplified detection
    if (device_name_.find("orin") != std::string::npos) {
        return "ampere_orin";
    } else if (device_name_.find("ampere") != std::string::npos) {
        return "ampere";
    } else if (device_name_.find("turing") != std::string::npos) {
        return "turing";
    } else if (device_name_.find("pascal") != std::string::npos) {
        return "pascal";
    }
    return "unknown";
}

// LayerAnalyzer implementation
LayerAnalyzer::LayerAnalyzer(
    std::shared_ptr<HardwareProfiler> hardware_profiler,
    const CostModel& cost_model
) : hardware_profiler_(hardware_profiler), cost_model_(cost_model) {}

LayerProfile LayerAnalyzer::analyze_layer(
    const std::string& layer_name,
    const std::string& layer_type,
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& output_shape,
    const std::vector<size_t>& weight_shape,
    const std::map<std::string, std::string>& attributes
) {
    nvtxRangePushA(("analyze_layer: " + layer_name).c_str());
    
    LayerProfile profile;
    profile.name = layer_name;
    profile.type = layer_type;
    
    // Calculate FLOPs
    profile.total_flops = calculate_flops(layer_type, input_shape, weight_shape);
    
    // Calculate memory footprint
    profile.total_memory_bytes = calculate_memory_footprint(
        input_shape, weight_shape, output_shape, DataType::FP16
    );
    
    // Calculate compute intensity
    if (profile.total_memory_bytes > 0) {
        profile.compute_intensity = static_cast<float>(profile.total_flops) / 
                                   profile.total_memory_bytes;
    } else {
        profile.compute_intensity = 0.0f;
    }
    
    // Estimate data locality
    profile.data_locality = estimate_data_locality(layer_type, input_shape);
    
    // Estimate parallelism
    profile.parallelism = estimate_parallelism(layer_type, output_shape);
    
    // Analyze dependencies (simplified)
    if (!attributes.empty()) {
        auto it = attributes.find("inputs");
        if (it != attributes.end()) {
            // Parse input dependencies
            std::stringstream ss(it->second);
            std::string dep;
            while (std::getline(ss, dep, ',')) {
                profile.input_dependencies.push_back(dep);
            }
        }
    }
    
    // Device performance ranking
    std::vector<std::string> devices = {"gpu", "cpu"};
    profile.device_performances = rank_devices_for_layer(profile, devices);
    
    // Default quantization sensitivity
    profile.quantization_sensitivity = 0.5f;
    
    // Applicable transforms
    profile.applicable_transforms = get_applicable_transforms(profile);
    
    nvtxRangePop();
    return profile;
}

size_t LayerAnalyzer::calculate_flops(
    const std::string& layer_type,
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& weight_shape
) const {
    if (layer_type == "linear" || layer_type == "matmul") {
        // FLOPs = 2 * M * N * K
        if (input_shape.size() >= 2 && weight_shape.size() >= 2) {
            size_t M = input_shape[0];
            size_t K = input_shape[1];
            size_t N = weight_shape[1];
            return 2 * M * N * K;
        }
    } else if (layer_type == "conv2d") {
        // FLOPs = 2 * H_out * W_out * C_in * C_out * K_h * K_w
        if (input_shape.size() == 4 && weight_shape.size() == 4) {
            size_t H_out = input_shape[2] - weight_shape[2] + 1;
            size_t W_out = input_shape[3] - weight_shape[3] + 1;
            size_t C_in = input_shape[1];
            size_t C_out = weight_shape[0];
            size_t K_h = weight_shape[2];
            size_t K_w = weight_shape[3];
            return 2 * H_out * W_out * C_in * C_out * K_h * K_w;
        }
    } else if (layer_type == "attention") {
        // Simplified: 4 * seq_len^2 * d_model
        if (input_shape.size() >= 2) {
            size_t seq_len = input_shape[0];
            size_t d_model = input_shape[1];
            return 4 * seq_len * seq_len * d_model;
        }
    }
    
    return 0;
}

size_t LayerAnalyzer::calculate_memory_footprint(
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& weight_shape,
    const std::vector<size_t>& output_shape,
    DataType precision
) const {
    size_t bytes_per_element = (precision == DataType::FP32) ? 4 :
                              (precision == DataType::FP16) ? 2 :
                              (precision == DataType::INT8) ? 1 : 1;
    
    size_t total_bytes = 0;
    
    // Input
    size_t input_elements = 1;
    for (auto dim : input_shape) input_elements *= dim;
    total_bytes += input_elements * bytes_per_element;
    
    // Weights
    size_t weight_elements = 1;
    for (auto dim : weight_shape) weight_elements *= dim;
    total_bytes += weight_elements * bytes_per_element;
    
    // Output
    size_t output_elements = 1;
    for (auto dim : output_shape) output_elements *= dim;
    total_bytes += output_elements * bytes_per_element;
    
    return total_bytes;
}

float LayerAnalyzer::estimate_data_locality(
    const std::string& layer_type,
    const std::vector<size_t>& shape
) const {
    // Higher value = more data reuse
    if (layer_type == "conv2d") {
        return 0.8f;  // High locality due to sliding window
    } else if (layer_type == "matmul") {
        return 0.7f;  // Good locality with tiling
    } else if (layer_type == "attention") {
        return 0.4f;  // Lower locality due to large matrices
    } else if (layer_type == "layernorm") {
        return 0.3f;  // Low locality
    }
    return 0.5f;
}

float LayerAnalyzer::estimate_parallelism(
    const std::string& layer_type,
    const std::vector<size_t>& shape
) const {
    // Higher value = more parallelizable
    if (layer_type == "matmul") {
        return 0.9f;  // Highly parallel
    } else if (layer_type == "conv2d") {
        return 0.8f;  // Good parallelism
    } else if (layer_type == "attention") {
        return 0.6f;  // Moderate parallelism
    } else if (layer_type == "layernorm") {
        return 0.4f;  // Limited parallelism
    }
    return 0.5f;
}

std::vector<LayerProfile::DevicePerformance> LayerAnalyzer::rank_devices_for_layer(
    const LayerProfile& layer,
    const std::vector<std::string>& available_devices
) {
    std::vector<LayerProfile::DevicePerformance> rankings;
    
    for (const auto& device : available_devices) {
        LayerProfile::DevicePerformance perf;
        perf.device_name = device;
        
        // Estimate execution time
        perf.execution_time_ms = hardware_profiler_->estimate_execution_time(
            layer, device, DataType::FP16
        );
        
        // Estimate power consumption
        auto caps = hardware_profiler_->profile_capabilities();
        perf.power_consumption_w = caps.typical_power_watts * 0.5f;  // Simplified
        
        // Calculate efficiency score
        // Higher score = better
        perf.efficiency_score = 1.0f / (perf.execution_time_ms + 1e-6);
        
        rankings.push_back(perf);
    }
    
    // Sort by efficiency
    std::sort(rankings.begin(), rankings.end());
    std::reverse(rankings.begin(), rankings.end());  // Highest first
    
    return rankings;
}

// ModelSurgeon implementation
ModelSurgeon::ModelSurgeon(
    const std::string& model_path,
    std::shared_ptr<HardwareProfiler> hardware_profiler,
    const CostModel& cost_model
) : model_path_(model_path),
    hardware_profiler_(hardware_profiler),
    cost_model_(cost_model),
    layer_analyzer_(std::make_unique<LayerAnalyzer>(hardware_profiler, cost_model)) {
    
    // Initialize
    analyze_model();
    generate_initial_placement();
}

void ModelSurgeon::analyze_model() {
    nvtxRangePushA("ModelSurgeon::analyze_model");
    
    // This would parse the actual model file
    // For now, create a sample model structure
    
    // Sample LLaMA-like model
    std::vector<std::tuple<std::string, std::string, std::vector<size_t>>> layers = {
        {"input_embedding", "embedding", {1, 4096, 5120}},
        {"ln_1", "layernorm", {1, 4096, 5120}},
        {"attn_q", "linear", {1, 4096, 5120, 5120}},
        {"attn_k", "linear", {1, 4096, 5120, 5120}},
        {"attn_v", "linear", {1, 4096, 5120, 5120}},
        {"attention", "attention", {1, 4096, 5120}},
        {"attn_out", "linear", {1, 4096, 5120, 5120}},
        {"mlp_1", "linear", {1, 4096, 5120, 13824}},
        {"gelu", "activation", {1, 4096, 13824}},
        {"mlp_2", "linear", {1, 4096, 13824, 5120}},
        {"residual", "add", {1, 4096, 5120}},
        {"ln_2", "layernorm", {1, 4096, 5120}},
    };
    
    // Repeat for 40 layers
    for (int layer_idx = 0; layer_idx < 40; ++layer_idx) {
        for (const auto& [base_name, layer_type, shape] : layers) {
            std::string layer_name = base_name + "_" + std::to_string(layer_idx);
            
            // Create weight shapes based on layer type
            std::vector<size_t> weight_shape;
            std::vector<size_t> input_shape = shape;
            std::vector<size_t> output_shape = shape;
            
            if (layer_type == "linear") {
                // Linear: [in_features, out_features]
                weight_shape = {shape[2], shape[2]};  // Simplified
                if (base_name == "mlp_1") weight_shape = {5120, 13824};
                else if (base_name == "mlp_2") weight_shape = {13824, 5120};
            } else if (layer_type == "attention") {
                // Self-attention
                weight_shape = {shape[2], shape[2]};
            }
            
            std::map<std::string, std::string> attributes;
            if (layer_idx > 0) {
                attributes["inputs"] = "ln_2_" + std::to_string(layer_idx - 1);
            }
            
            auto profile = layer_analyzer_->analyze_layer(
                layer_name, layer_type, input_shape, output_shape, weight_shape, attributes
            );
            
            layer_profiles_[layer_name] = profile;
        }
    }
    
    nvtxRangePop();
}

void ModelSurgeon::generate_initial_placement() {
    // Simple heuristic: put compute-intensive layers on GPU
    for (auto& [layer_name, profile] : layer_profiles_) {
        if (profile.compute_intensity > 10.0f) {
            layer_to_device_[layer_name] = "gpu";
            layer_to_precision_[layer_name] = DataType::FP16;
        } else if (profile.total_memory_bytes > 100 * 1024 * 1024) {
            // Large memory footprint, keep on CPU for now
            layer_to_device_[layer_name] = "cpu";
            layer_to_precision_[layer_name] = DataType::FP16;
        } else {
            layer_to_device_[layer_name] = "gpu";
            layer_to_precision_[layer_name] = DataType::FP16;
        }
    }
}

SurgeryPlan ModelSurgeon::create_surgery_plan(
    const std::map<std::string, float>& constraints
) {
    nvtxRangePushA("ModelSurgeon::create_surgery_plan");
    
    SurgeryPlan plan;
    
    // Extract constraints
    float max_memory = constraints.count("max_memory_mb") ? 
                      constraints.at("max_memory_mb") * 1024 * 1024 : 
                      cost_model_.max_memory_usage;
    
    float max_power = constraints.count("max_power_w") ? 
                     constraints.at("max_power_w") : 
                     cost_model_.max_power_budget;
    
    float min_accuracy = constraints.count("min_accuracy") ? 
                        constraints.at("min_accuracy") : 
                        cost_model_.min_accuracy;
    
    float target_speedup = constraints.count("target_speedup") ? 
                          constraints.at("target_speedup") : 2.0f;
    
    // Generate different optimization plans
    auto quantization_plan = generate_quantization_plan(min_accuracy);
    auto pruning_plan = generate_pruning_plan(0.5f);  // 50% sparsity target
    auto fusion_plan = generate_fusion_plan();
    auto placement_plan = generate_device_placement_plan();
    
    // Merge plans based on constraints
    // Start with quantization (usually highest benefit/cost)
    for (const auto& op : quantization_plan.operations) {
        if (validate_transformation(op.layer_name, op.type)) {
            plan.add_operation(op);
        }
    }
    
    // Add fusion
    for (const auto& op : fusion_plan.operations) {
        if (validate_transformation(op.layer_name, op.type)) {
            plan.add_operation(op);
        }
    }
    
    // Add pruning if memory constrained
    size_t total_memory = 0;
    for (const auto& [name, profile] : layer_profiles_) {
        total_memory += profile.total_memory_bytes;
    }
    
    if (total_memory > max_memory) {
        for (const auto& op : pruning_plan.operations) {
            if (validate_transformation(op.layer_name, op.type)) {
                plan.add_operation(op);
            }
        }
    }
    
    // Update placement
    plan.layer_to_device = placement_plan.layer_to_device;
    plan.layer_to_precision = placement_plan.layer_to_precision;
    
    // Estimate total benefits
    plan.total_estimated_speedup = 1.0f;
    plan.total_memory_reduction = 1.0f;
    
    for (const auto& op : plan.operations) {
        plan.total_estimated_speedup *= (1.0f + op.estimated_benefit);
        plan.total_memory_reduction *= (1.0f - op.estimated_benefit * 0.5f);  // Approximate
    }
    
    // Generate execution stages
    std::vector<std::string> current_stage;
    std::set<std::string> processed_layers;
    
    for (const auto& [layer_name, profile] : layer_profiles_) {
        if (processed_layers.find(layer_name) != processed_layers.end()) {
            continue;
        }
        
        current_stage.push_back(layer_name);
        processed_layers.insert(layer_name);
        
        // Check if we should start new stage
        if (current_stage.size() >= 5) {  // Batch of 5 layers per stage
            plan.execution_stages.push_back(current_stage);
            current_stage.clear();
        }
    }
    
    if (!current_stage.empty()) {
        plan.execution_stages.push_back(current_stage);
    }
    
    nvtxRangePop();
    return plan;
}

SurgeryPlan ModelSurgeon::generate_quantization_plan(float accuracy_constraint) {
    SurgeryPlan plan;
    
    for (const auto& [layer_name, profile] : layer_profiles_) {
        // Skip attention layers (more sensitive)
        if (profile.type == "attention") {
            continue;
        }
        
        // Check if quantization is beneficial
        if (profile.quantization_sensitivity < 0.3f) {
            // Low sensitivity, good candidate for quantization
            
            SurgeryPlan::Operation op;
            op.type = SurgeryOperation::QUANTIZE;
            op.layer_name = layer_name;
            op.priority = 10 - static_cast<int>(profile.quantization_sensitivity * 10);
            
            // Choose precision based on sensitivity
            DataType target_precision;
            if (profile.quantization_sensitivity < 0.1f) {
                target_precision = DataType::INT8;
                op.estimated_benefit = 2.0f;  // 2x speedup
            } else if (profile.quantization_sensitivity < 0.2f) {
                target_precision = DataType::FP16;
                op.estimated_benefit = 1.5f;
            } else {
                target_precision = DataType::FP16;
                op.estimated_benefit = 1.2f;
            }
            
            op.parameters = {
                {"precision", std::to_string(static_cast<int>(target_precision))},
                {"method", "per_tensor"},
                {"calibration", "min_max"}
            };
            
            op.estimated_cost = estimate_transformation_cost(profile, op.type);
            
            plan.add_operation(op);
        }
    }
    
    return plan;
}

SurgeryPlan ModelSurgeon::generate_pruning_plan(float sparsity_target) {
    SurgeryPlan plan;
    
    // Focus on large linear layers
    for (const auto& [layer_name, profile] : layer_profiles_) {
        if (profile.type == "linear" && profile.parameters_count > 1000000) {
            SurgeryPlan::Operation op;
            op.type = SurgeryOperation::PRUNE;
            op.layer_name = layer_name;
            op.priority = static_cast<int>(profile.parameters_count / 1000000);
            
            op.parameters = {
                {"sparsity", std::to_string(sparsity_target)},
                {"method", "magnitude"},
                {"granularity", "structured"}
            };
            
            // Benefit: memory reduction and some speedup
            op.estimated_benefit = 0.3f * sparsity_target;  // 30% of sparsity as benefit
            
            op.estimated_cost = estimate_transformation_cost(profile, op.type);
            
            plan.add_operation(op);
        }
    }
    
    return plan;
}

SurgeryPlan ModelSurgeon::generate_fusion_plan() {
    SurgeryPlan plan;
    
    // Look for fusion patterns
    std::vector<std::string> layer_names;
    for (const auto& [name, _] : layer_profiles_) {
        layer_names.push_back(name);
    }
    
    std::sort(layer_names.begin(), layer_names.end());
    
    // Pattern: Linear -> Activation -> Linear
    for (size_t i = 0; i < layer_names.size() - 2; ++i) {
        std::string layer1 = layer_names[i];
        std::string layer2 = layer_names[i + 1];
        std::string layer3 = layer_names[i + 2];
        
        auto& prof1 = layer_profiles_[layer1];
        auto& prof2 = layer_profiles_[layer2];
        auto& prof3 = layer_profiles_[layer3];
        
        if (prof1.type == "linear" && 
            prof2.type == "activation" && 
            prof3.type == "linear") {
            
            SurgeryPlan::Operation op;
            op.type = SurgeryOperation::FUSE;
            op.layer_name = layer1;  // Fuse all three
            op.priority = 5;
            
            op.parameters = {
                {"layers", layer1 + "," + layer2 + "," + layer3},
                {"fused_type", "fused_linear_activation_linear"}
            };
            
            // Fusing reduces kernel launch overhead and memory transfers
            op.estimated_benefit = 0.4f;
            op.estimated_cost = 1.0f;
            
            plan.add_operation(op);
            
            // Skip the next two layers since they'll be fused
            i += 2;
        }
    }
    
    return plan;
}

SurgeryPlan ModelSurgeon::generate_device_placement_plan() {
    SurgeryPlan plan;
    
    // Use device placer to find optimal placement
    std::vector<std::string> available_devices = {"gpu", "cpu"};
    
    DevicePlacer placer(available_devices, hardware_profiler_);
    auto constraints = std::map<std::string, float>{
        {"max_memory_mb", 8192.0f},  // 8GB
        {"max_power_w", 15.0f},      // 15W
        {"max_latency_ms", 1000.0f}
    };
    
    auto solution = placer.find_optimal_placement(layer_profiles_, constraints);
    
    plan.layer_to_device = solution.layer_to_device;
    
    // Set precision based on device
    for (auto& [layer_name, device] : plan.layer_to_device) {
        if (device == "gpu") {
            plan.layer_to_precision[layer_name] = DataType::FP16;
        } else {
            plan.layer_to_precision[layer_name] = DataType::FP32;
        }
    }
    
    return plan;
}

bool ModelSurgeon::apply_surgery_plan(const SurgeryPlan& plan) {
    nvtxRangePushA("ModelSurgeon::apply_surgery_plan");
    
    bool success = true;
    
    // Apply operations in priority order
    while (true) {
        auto op = plan.get_next_operation();
        if (op.type == SurgeryOperation::NONE) {
            break;
        }
        
        bool op_success = false;
        
        switch (op.type) {
            case SurgeryOperation::QUANTIZE:
                op_success = quantize_layer(
                    op.layer_name,
                    static_cast<DataType>(std::stoi(op.parameters.at("precision"))),
                    QuantizationMethod::INT8
                );
                break;
                
            case SurgeryOperation::PRUNE:
                op_success = prune_layer(
                    op.layer_name,
                    std::stof(op.parameters.at("sparsity"))
                );
                break;
                
            case SurgeryOperation::FUSE:
                // Parse layers to fuse
                std::string layers_str = op.parameters.at("layers");
                std::vector<std::string> layers_to_fuse;
                std::stringstream ss(layers_str);
                std::string layer;
                while (std::getline(ss, layer, ',')) {
                    layers_to_fuse.push_back(layer);
                }
                
                op_success = fuse_layers(
                    layers_to_fuse,
                    op.parameters.at("fused_type")
                );
                break;
        }
        
        if (!op_success) {
            success = false;
            std::cerr << "Failed to apply operation: " << 
                static_cast<int>(op.type) << " on " << op.layer_name << std::endl;
        } else {
            // Record transformation
            transformation_history_.push_back(
                std::make_tuple(op.layer_name, op.type, op.estimated_benefit)
            );
        }
    }
    
    // Update device placement
    layer_to_device_ = plan.layer_to_device;
    layer_to_precision_ = plan.layer_to_precision;
    
    current_plan_ = plan;
    
    nvtxRangePop();
    return success;
}

bool ModelSurgeon::quantize_layer(
    const std::string& layer_name,
    DataType new_precision,
    QuantizationMethod method
) {
    // Update layer precision
    if (layer_to_precision_.find(layer_name) != layer_to_precision_.end()) {
        layer_to_precision_[layer_name] = new_precision;
        return true;
    }
    return false;
}

float ModelSurgeon::estimate_transformation_cost(
    const LayerProfile& layer,
    SurgeryOperation operation
) const {
    // Simplified cost model
    switch (operation) {
        case SurgeryOperation::QUANTIZE:
            return 1.0f + layer.total_memory_bytes / (1024.0f * 1024.0f) * 0.01f;
        case SurgeryOperation::PRUNE:
            return 2.0f + layer.parameters_count / 1e6f * 0.1f;
        case SurgeryOperation::FUSE:
            return 0.5f;
        case SurgeryOperation::SPLIT:
            return 3.0f;
        default:
            return 1.0f;
    }
}

bool ModelSurgeon::validate_transformation(
    const std::string& layer_name,
    SurgeryOperation operation
) const {
    // Basic validation
    if (layer_profiles_.find(layer_name) == layer_profiles_.end()) {
        return false;
    }
    
    const auto& profile = layer_profiles_.at(layer_name);
    
    switch (operation) {
        case SurgeryOperation::QUANTIZE:
            return profile.quantization_sensitivity < 0.5f;
        case SurgeryOperation::PRUNE:
            return profile.parameters_count > 1000;
        case SurgeryOperation::FUSE:
            return !profile.input_dependencies.empty();
        default:
            return true;
    }
}

// Factory function
std::unique_ptr<ModelSurgeon> create_model_surgeon(
    const std::string& model_path,
    const std::string& hardware_config
) {
    auto profiler = std::make_shared<HardwareProfiler>(hardware_config);
    return std::make_unique<ModelSurgeon>(model_path, profiler);
}

} // namespace jalapeno
