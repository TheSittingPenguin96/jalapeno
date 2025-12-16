# examples/adaptive_prefetch_quantize.py
#!/usr/bin/env python3
"""
Example: Integrated Markov Prefetcher + Adaptive Quantization
"""

import torch
import time
import numpy as np
from jalapeno import (
    MarkovPrefetcher, AdaptivePrefetcher, ContextFeatures,
    AdaptiveQuantizer, QuantizationMethod, QuantizationParams
)

def main():
    print("Adaptive Prefetching + Quantization Demo")
    print("=" * 60)
    
    # ========== Markov Prefetcher Demo ==========
    print("\n1. Markov Prefetcher Learning:")
    print("-" * 40)
    
    # Create prefetcher with neural network
    prefetcher = MarkovPrefetcher(use_neural=True)
    
    # Simulate layer access patterns
    layers = [f"layer_{i}" for i in range(20)]
    
    # Common patterns
    patterns = [
        # Sequential access
        [0, 1, 2, 3, 4, 5],
        # Skip connections
        [0, 5, 10, 15],
        # Attention pattern
        [0, 1, 0, 2, 0, 3],
        # Back-and-forth
        [0, 1, 0, 2, 1, 3],
    ]
    
    # Create context features
    context = ContextFeatures(
        sequence_length=128.0,
        attention_pattern_entropy=0.7,
        layer_diversity=0.5,
        model_size=7.0,  # 7B model
        model_type="llama",
        memory_pressure=0.3,
        gpu_utilization=0.6,
    )
    
    # Train prefetcher
    print("Training prefetcher with patterns...")
    num_transitions = 1000
    
    for i in range(num_transitions):
        # Choose a pattern
        pattern = patterns[i % len(patterns)]
        
        # Simulate transitions within pattern
        for j in range(len(pattern) - 1):
            from_layer = layers[pattern[j]]
            to_layer = layers[pattern[j + 1]]
            
            # Simulate latency (lower for sequential, higher for jumps)
            latency = 10.0 if abs(pattern[j] - pattern[j + 1]) == 1 else 30.0
            
            prefetcher.record_transition(
                from_layer, to_layer, latency, context
            )
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} transitions")
    
    # Test predictions
    print("\nTesting predictions:")
    test_layer = "layer_0"
    predictions, confidences = prefetcher.predict(test_layer, context, 3)
    
    print(f"From {test_layer}, predicted next layers:")
    for pred, conf in zip(predictions, confidences):
        print(f"  {pred} (confidence: {conf:.3f})")
    
    # Get statistics
    stats = prefetcher.get_statistics()
    print(f"\nPrefetcher Statistics:")
    print(f"  Total transitions: {stats['total_transitions']:,}")
    print(f"  States learned: {stats['states_learned']}")
    print(f"  Average confidence: {stats['average_transition_confidence']:.3f}")
    
    # ========== Adaptive Prefetcher Demo ==========
    print("\n\n2. Adaptive Prefetcher Strategy Switching:")
    print("-" * 40)
    
    adaptive_prefetcher = AdaptivePrefetcher()
    
    # Simulate different workloads
    workloads = [
        ("sequential", list(range(10))),
        ("random", np.random.permutation(10).tolist()),
        ("strided", list(range(0, 10, 2))),
    ]
    
    for workload_name, pattern in workloads:
        print(f"\nWorkload: {workload_name}")
        
        # Update context for this workload
        if workload_name == "sequential":
            context.temporal_locality = 0.9
        elif workload_name == "random":
            context.temporal_locality = 0.1
        else:
            context.temporal_locality = 0.5
        
        # Run through pattern
        for i in range(len(pattern) - 1):
            from_layer = layers[pattern[i]]
            to_layer = layers[pattern[i + 1]]
            
            adaptive_prefetcher.record_transition(
                from_layer, to_layer, context, 15.0
            )
        
        # Get strategy info
        strategy_info = adaptive_prefetcher.get_strategy_info()
        print(f"  Current strategy: {strategy_info['current_strategy']}")
        print(f"  Strategy weights: {strategy_info['strategy_weights']}")
    
    # ========== Adaptive Quantization Demo ==========
    print("\n\n3. Adaptive Quantization:")
    print("-" * 40)
    
    quantizer = AdaptiveQuantizer(target_compression=0.25)  # 4x compression
    
    # Create different types of tensors
    print("\nTesting different tensor types:")
    
    test_tensors = {
        "Dense weights": torch.randn(1024, 1024, dtype=torch.float16).cuda(),
        "Sparse weights": torch.randn(1024, 1024, dtype=torch.float16).cuda() * 
                         (torch.rand(1024, 1024).cuda() > 0.8).float(),
        "Normally distributed": torch.randn(512, 512, dtype=torch.float16).cuda(),
        "Small range": torch.rand(256, 256, dtype=torch.float16).cuda() * 0.1,
    }
    
    results = {}
    
    for name, tensor in test_tensors.items():
        print(f"\n{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Size: {tensor.numel() * tensor.element_size() / 1024**2:.2f} MB")
        
        # Analyze tensor
        characteristics = quantizer.analyze_tensor(tensor)
        print(f"  Characteristics:")
        print(f"    Range: [{characteristics['min']:.3f}, {characteristics['max']:.3f}]")
        print(f"    Sparsity: {characteristics['sparsity']:.3f}")
        print(f"    Normality: {characteristics['normality']:.3f}")
        
        # Select method
        method = quantizer.select_method(tensor)
        print(f"  Selected method: {method.name}")
        
        # Quantize adaptively
        start = time.time()
        quantized, used_method, params, metrics = quantizer.quantize_adaptive(
            tensor, target_compression=0.25, max_error=0.01
        )
        quant_time = time.time() - start
        
        # Dequantize
        start = time.time()
        dequantized = quantizer.quantizer.dequantize(
            quantized, used_method, tensor.shape, params, tensor.dtype
        )
        dequant_time = time.time() - start
        
        # Calculate compression
        orig_size = tensor.numel() * tensor.element_size()
        quant_size = quantized.numel() * quantized.element_size()
        compression = quant_size / orig_size
        
        print(f"  Results:")
        print(f"    Actual method used: {used_method.name}")
        print(f"    Compression: {compression:.3f} ({1/compression:.1f}x)")
        print(f"    RMSE: {metrics['rmse']:.6f}")
        print(f"    Max relative error: {metrics['max_relative_error']:.6f}")
        print(f"    Quantization time: {quant_time*1000:.2f} ms")
        print(f"    Dequantization time: {dequant_time*1000:.2f} ms")
        
        results[name] = {
            'method': used_method.name,
            'compression': compression,
            'rmse': metrics['rmse'],
            'quant_time_ms': quant_time * 1000,
        }
    
    # ========== Combined Demo ==========
    print("\n\n4. Combined Prefetching + Quantization Workflow:")
    print("-" * 40)
    
    # Simulate running a large model with both optimizations
    print("Simulating LLM inference with optimizations...")
    
    # Layer access pattern for transformer
    num_layers = 40
    batch_size = 1
    seq_len = 512
    hidden_size = 5120
    
    # Create layer weights (simulated)
    print(f"\nModel: {num_layers} layers, hidden_size={hidden_size}")
    print(f"Total parameters: ~{num_layers * hidden_size * hidden_size * 4 * 3 / 1e9:.1f}B")
    
    # Track statistics
    total_latency = 0.0
    cache_hits = 0
    cache_misses = 0
    
    # Simulate forward pass
    for layer_idx in range(num_layers):
        layer_name = f"transformer_layer_{layer_idx}"
        
        # Predict next layers for prefetching
        predictions, _ = prefetcher.predict(layer_name, context, 2)
        
        # Simulate layer execution
        start = time.time()
        
        # Would load weights, compute, etc.
        # Here we simulate quantization benefits
        
        if layer_idx < 5:
            # First few layers: cache misses
            cache_misses += 1
            time.sleep(0.02)  # Simulate loading
        else:
            # Later layers: cache hits (due to prefetching)
            cache_hits += 1
            time.sleep(0.005)  # Simulate cached access
        
        # Update prefetcher
        if layer_idx < num_layers - 1:
            next_layer = f"transformer_layer_{layer_idx + 1}"
            prefetcher.record_transition(layer_name, next_layer, 15.0, context)
        
        layer_time = time.time() - start
        total_latency += layer_time
        
        # Print progress
        if (layer_idx + 1) % 10 == 0:
            print(f"  Processed layer {layer_idx + 1}/{num_layers}")
    
    print(f"\nSimulation Results:")
    print(f"  Total latency: {total_latency:.2f}s")
    print(f"  Average layer latency: {total_latency/num_layers*1000:.1f}ms")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Cache misses: {cache_misses}")
    print(f"  Hit rate: {cache_hits/(cache_hits+cache_misses):.1%}")
    
    # Show quantization recommendations
    print(f"\nQuantization Recommendations:")
    recommendations = quantizer.get_method_recommendations()
    for method, info in recommendations.items():
        print(f"  {method}:")
        print(f"    Average error: {info['average_error']:.6f}")
        print(f"    Usage count: {info['usage_count']}")
        print(f"    Recommended for: {', '.join(info['recommended_for'][:2])}")
    
    # Memory savings estimate
    print(f"\nEstimated Memory Savings:")
    print("  Without optimizations:")
    print(f"    Model weights: {num_layers * hidden_size * hidden_size * 4 * 3 / 1024**3:.2f} GB")
    print(f"    KV cache (seq_len={seq_len}): {num_layers * 2 * seq_len * hidden_size * 2 / 1024**3:.2f} GB")
    
    print("  With optimizations:")
    # Assume 4x compression for weights, 2x for KV cache
    print(f"    Model weights (4x compression): {num_layers * hidden_size * hidden_size * 4 * 3 / 1024**3 / 4:.2f} GB")
    print(f"    KV cache (2x compression): {num_layers * 2 * seq_len * hidden_size * 2 / 1024**3 / 2:.2f} GB")
    
    total_savings = (num_layers * hidden_size * hidden_size * 4 * 3 / 1024**3) * 0.75 + \
                   (num_layers * 2 * seq_len * hidden_size * 2 / 1024**3) * 0.5
    print(f"  Total memory savings: {total_savings:.2f} GB")

if __name__ == "__main__":
    main()