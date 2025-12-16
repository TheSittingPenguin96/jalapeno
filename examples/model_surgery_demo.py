# examples/model_surgery_demo.py
#!/usr/bin/env python3
"""
Hardware-Aware Model Surgery Demo
"""

import torch
import torch.nn as nn
import time
import numpy as np
from jalapeno import ModelSurgeon, AutoSurgeon, HardwareCapabilities

def create_test_model():
    """Create a test model similar to LLaMA layers"""
    class TestModel(nn.Module):
        def __init__(self, num_layers=4):
            super().__init__()
            self.num_layers = num_layers
            
            # Embedding
            self.embedding = nn.Embedding(32000, 5120)
            
            # Transformer layers
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                layer = nn.ModuleDict({
                    'attention': nn.ModuleDict({
                        'q_proj': nn.Linear(5120, 5120),
                        'k_proj': nn.Linear(5120, 5120),
                        'v_proj': nn.Linear(5120, 5120),
                        'out_proj': nn.Linear(5120, 5120),
                    }),
                    'mlp': nn.ModuleDict({
                        'gate_proj': nn.Linear(5120, 13824),
                        'up_proj': nn.Linear(5120, 13824),
                        'down_proj': nn.Linear(13824, 5120),
                    }),
                    'input_layernorm': nn.LayerNorm(5120),
                    'post_attention_layernorm': nn.LayerNorm(5120),
                })
                self.layers.append(layer)
            
            # Output
            self.norm = nn.LayerNorm(5120)
            self.lm_head = nn.Linear(5120, 32000)
        
        def forward(self, x):
            # Simplified forward
            x = self.embedding(x)
            
            for layer in self.layers:
                # Attention
                attn_out = layer['attention']['out_proj'](
                    layer['attention']['q_proj'](x)
                )
                x = x + attn_out
                
                # MLP
                mlp_out = layer['mlp']['down_proj'](
                    nn.functional.silu(layer['mlp']['gate_proj'](x)) *
                    layer['mlp']['up_proj'](x)
                )
                x = x + mlp_out
            
            x = self.norm(x)
            return self.lm_head(x)
    
    return TestModel(num_layers=4)

def benchmark_model(model, input_shape=(1, 512)):
    """Benchmark model performance"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            x = torch.randint(0, 32000, input_shape)
            _ = model(x)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(10):
            x = torch.randint(0, 32000, input_shape)
            
            start = time.time()
            _ = model(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            
            times.append((end - start) * 1000)  # ms
    
    return {
        'average_latency_ms': np.mean(times),
        'p95_latency_ms': np.percentile(times, 95),
        'min_latency_ms': np.min(times),
        'max_latency_ms': np.max(times),
    }

def main():
    print("Hardware-Aware Model Surgery Demo")
    print("=" * 70)
    
    # Create model
    print("\n1. Creating test model...")
    model = create_test_model()
    
    print(f"   Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Model has {len(list(model.modules()))} modules")
    
    # Benchmark baseline
    print("\n2. Benchmarking baseline performance...")
    baseline_stats = benchmark_model(model)
    print(f"   Average latency: {baseline_stats['average_latency_ms']:.1f} ms")
    print(f"   P95 latency: {baseline_stats['p95_latency_ms']:.1f} ms")
    
    # Create model surgeon
    print("\n3. Initializing Model Surgeon...")
    
    # Get hardware capabilities
    caps = HardwareCapabilities.from_nvidia_smi()
    print(f"   Hardware: {caps.name}")
    print(f"   Architecture: {caps.architecture}")
    print(f"   FP16 TFLOPS: {caps.peak_tflops_fp16:.1f}")
    print(f"   GPU Memory: {caps.gpu_vram_size / (1024**3):.1f} GB")
    
    surgeon = ModelSurgeon(model, hardware_caps=caps)
    
    # Analyze hardware affinity
    print("\n4. Analyzing hardware affinity...")
    affinity = surgeon.analyze_hardware_affinity()
    
    print("   Top layers for GPU:")
    gpu_layers = sorted(
        [(name, scores['gpu']) for name, scores in affinity.items()],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    for layer_name, score in gpu_layers:
        print(f"     {layer_name}: {score:.3f}")
    
    print("\n   Top layers for CPU:")
    cpu_layers = sorted(
        [(name, scores['cpu']) for name, scores in affinity.items()],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    for layer_name, score in cpu_layers:
        print(f"     {layer_name}: {score:.3f}")
    
    # Create optimization plan
    print("\n5. Creating optimization plan...")
    
    constraints = {
        "max_memory_mb": 4096,  # 4GB constraint
        "max_power_w": 10.0,    # 10W power budget
        "target_speedup": 2.0,  # 2x speedup target
    }
    
    plan = surgeon.create_optimization_plan(constraints)
    
    print(f"   Generated {len(plan['operations'])} optimization operations")
    
    # Show top operations
    print("\n   Top optimization operations:")
    for op in plan['operations'][:5]:
        print(f"     {op['type']:20} {op.get('layer', 'multiple'):30} benefit: {op.get('estimated_benefit', 0):.2f}")
    
    # Apply optimizations
    print("\n6. Applying optimizations...")
    success = surgeon.apply_optimization_plan(plan)
    
    if success:
        print("   Successfully applied optimizations!")
    else:
        print("   Some optimizations failed")
    
    # Show placement visualization
    print("\n7. Model placement visualization:")
    print(surgeon.visualize_placement())
    
    # Estimate performance
    print("\n8. Performance estimation:")
    estimates = surgeon.estimate_performance()
    
    print(f"   Estimated latency: {estimates['estimated_latency_ms']:.1f} ms")
    print(f"   Estimated memory: {estimates['estimated_memory_mb']:.1f} MB")
    print(f"   Estimated speedup: {estimates['estimated_speedup']:.2f}x")
    
    # Apply quantization
    print("\n9. Applying quantization...")
    quantized_model = surgeon.apply_quantization({
        "dtype": "int8",
        "per_channel": True,
        "symmetric": True,
    })
    
    # Apply pruning
    print("   Applying pruning...")
    pruned_model = surgeon.apply_pruning(
        sparsity=0.3,
        pruning_method="l1_unstructured"
    )
    
    # Split model across devices
    print("   Splitting model across devices...")
    memory_constraints = {
        "gpu": 2048,  # 2GB GPU
        "cpu": 4096,  # 4GB CPU
    }
    
    split_assignments = surgeon.split_model_across_devices(memory_constraints)
    
    print(f"   GPU layers: {len(split_assignments['gpu'])}")
    print(f"   CPU layers: {len(split_assignments['cpu'])}")
    
    # AutoSurgeon with RL
    print("\n10. Running AutoSurgeon (Reinforcement Learning)...")
    auto_surgeon = AutoSurgeon(surgeon)
    
    best_config = auto_surgeon.optimize_with_rl(
        objective="latency",
        constraints=constraints,
        episodes=5
    )
    
    if best_config:
        print(f"   Best latency: {best_config['metrics']['estimated_latency_ms']:.1f} ms")
        print(f"   Best speedup: {baseline_stats['average_latency_ms'] / best_config['metrics']['estimated_latency_ms']:.2f}x")
    
    # Learn policy
    print("\n11. Learning transformation policy...")
    policy = auto_surgeon.learn_policy()
    
    if policy.get('learned_rules'):
        print(f"   Learned {len(policy['learned_rules'])} rules:")
        for rule in policy['learned_rules']:
            print(f"     - {rule['rule']}: {len(rule['layers'])} layers")
    
    # Generate comprehensive report
    print("\n12. Final Optimization Report:")
    print("=" * 70)
    
    report = surgeon.get_optimization_report()
    
    print(f"\nHardware: {report['hardware_capabilities']['name']}")
    print(f"Architecture: {report['hardware_capabilities']['architecture']}")
    print(f"GPU Memory: {report['hardware_capabilities']['gpu_memory_mb']:.1f} MB")
    print(f"TDP: {report['hardware_capabilities']['tdp_watts']:.1f} W")
    
    print(f"\nModel Statistics:")
    print(f"  Total layers: {report['model_statistics']['total_layers']}")
    print(f"  Total parameters: {report['model_statistics']['total_parameters']:,}")
    print(f"  Total FLOPs: {report['model_statistics']['total_flops'] / 1e9:.1f} G")
    print(f"  Estimated memory: {report['model_statistics']['estimated_memory_mb']:.1f} MB")
    
    print(f"\nOptimization Results:")
    print(f"  Transformations applied: {report['optimization_state']['transformations_applied']}")
    
    placement_summary = {}
    for device in report['optimization_state']['layer_placement'].values():
        placement_summary[device] = placement_summary.get(device, 0) + 1
    
    for device, count in placement_summary.items():
        print(f"  Layers on {device}: {count}")
    
    print(f"\nPerformance Estimates:")
    print(f"  Latency: {report['performance_estimates']['estimated_latency_ms']:.1f} ms")
    print(f"  Memory: {report['performance_estimates']['estimated_memory_mb']:.1f} MB")
    print(f"  Power: {report['performance_estimates']['estimated_power_w']:.1f} W")
    print(f"  Speedup: {report['performance_estimates']['estimated_speedup']:.2f}x")
    
    # Compare with baseline
    print(f"\nComparison with Baseline:")
    print(f"  Baseline latency: {baseline_stats['average_latency_ms']:.1f} ms")
    print(f"  Optimized latency: {report['performance_estimates']['estimated_latency_ms']:.1f} ms")
    print(f"  Improvement: {baseline_stats['average_latency_ms'] / report['performance_estimates']['estimated_latency_ms']:.2f}x")
    
    # Memory savings
    baseline_memory = report['model_statistics']['estimated_memory_mb']
    optimized_memory = report['performance_estimates']['estimated_memory_mb']
    memory_savings = (baseline_memory - optimized_memory) / baseline_memory * 100
    
    print(f"  Memory savings: {memory_savings:.1f}%")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")

def test_specific_optimizations():
    """Test specific optimization techniques"""
    print("\n" + "=" * 70)
    print("Testing Specific Optimization Techniques")
    print("=" * 70)
    
    model = create_test_model()
    caps = HardwareCapabilities.from_nvidia_smi()
    surgeon = ModelSurgeon(model, hardware_caps=caps)
    
    # Test 1: Precision selection
    print("\n1. Precision Selection Analysis:")
    for layer_name, profile in list(surgeon.layer_profiles.items())[:3]:
        print(f"   {layer_name}:")
        print(f"     Type: {profile.type}")
        print(f"     Parameters: {profile.parameters_count:,}")
        print(f"     FLOPs: {profile.total_flops / 1e9:.1f} G")
        print(f"     Quantization sensitivity: {profile.quantization_sensitivity:.3f}")
        
        if profile.quantization_sensitivity < 0.3:
            print(f"     → Good candidate for INT8 quantization")
        elif profile.quantization_sensitivity < 0.5:
            print(f"     → Can use FP16")
        else:
            print(f"     → Should keep FP32")
    
    # Test 2: Device placement analysis
    print("\n2. Device Placement Analysis:")
    affinity = surgeon.analyze_hardware_affinity()
    
    # Find layers that should move
    for layer_name, scores in list(affinity.items())[:5]:
        gpu_score = scores['gpu']
        cpu_score = scores['cpu']
        
        if gpu_score > cpu_score + 0.2:
            print(f"   {layer_name}: Should be on GPU (GPU: {gpu_score:.3f}, CPU: {cpu_score:.3f})")
        elif cpu_score > gpu_score + 0.2:
            print(f"   {layer_name}: Should be on CPU (GPU: {gpu_score:.3f}, CPU: {cpu_score:.3f})")
        else:
            print(f"   {layer_name}: Either device is fine (GPU: {gpu_score:.3f}, CPU: {cpu_score:.3f})")
    
    # Test 3: Fusion opportunities
    print("\n3. Fusion Opportunities:")
    layer_names = list(surgeon.layer_profiles.keys())
    
    # Look for Linear -> Activation -> Linear patterns
    fusion_candidates = []
    for i in range(len(layer_names) - 2):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        layer3 = layer_names[i + 2]
        
        prof1 = surgeon.layer_profiles[layer1]
        prof2 = surgeon.layer_profiles[layer2]
        prof3 = surgeon.layer_profiles[layer3]
        
        if (prof1.type == "Linear" and 
            prof2.type in ["ReLU", "GELU"] and
            prof3.type == "Linear"):
            
            fusion_candidates.append((layer1, layer2, layer3))
    
    if fusion_candidates:
        print(f"   Found {len(fusion_candidates)} fusion candidates")
        for i, (l1, l2, l3) in enumerate(fusion_candidates[:3]):
            print(f"   Candidate {i+1}: {l1} -> {l2} -> {l3}")
            print(f"     Estimated speedup: 30-50% from reduced kernel launches")
    else:
        print("   No fusion candidates found")
    
    # Test 4: Memory optimization
    print("\n4. Memory Optimization Analysis:")
    
    # Sort layers by memory usage
    memory_intensive = sorted(
        surgeon.layer_profiles.items(),
        key=lambda x: x[1].total_memory_bytes,
        reverse=True
    )[:5]
    
    print("   Most memory-intensive layers:")
    for layer_name, profile in memory_intensive:
        memory_mb = profile.total_memory_bytes / (1024 * 1024)
        print(f"     {layer_name}: {memory_mb:.1f} MB")
        
        if memory_mb > 100:
            print(f"       → Consider CPU placement or streaming")
        elif memory_mb > 10:
            print(f"       → May benefit from tiling")
    
    # Test 5: Power optimization
    print("\n5. Power Optimization Suggestions:")
    
    # Identify power-hungry layers
    power_hungry = []
    for layer_name, profile in surgeon.layer_profiles.items():
        if profile.compute_intensity > 50:  # Very compute-intensive
            power_hungry.append((layer_name, profile.compute_intensity))
    
    if power_hungry:
        print("   Power-hungry layers (high compute intensity):")
        for layer_name, intensity in sorted(power_hungry, key=lambda x: x[1], reverse=True)[:3]:
            print(f"     {layer_name}: CI={intensity:.1f} FLOPs/byte")
            print(f"       → Consider lower precision or CPU offload")
    else:
        print("   No extremely power-hungry layers detected")

if __name__ == "__main__":
    main()
    test_specific_optimizations()
