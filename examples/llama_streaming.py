#examples/llama_streaming.py
#!/usr/bin/env python3
"""
Example: Running LLaMA 13B on Jetson Orin with layer streaming
"""

import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jalapeno import Runtime, LayerStreamer, StreamerConfig

def main():
    print("Jalapeño Layer Streaming Demo")
    print("=" * 50)
    
    # Initialize runtime for Jetson Orin
    runtime = Runtime(
        device_config={
            "name": "jetson_orin_nx",
            "gpu_memory_mb": 8192,  # 8GB GPU
            "cpu_memory_mb": 32768,  # 32GB RAM
            "enable_unified_memory": True,
        },
        memory_config={
            "gpu_memory_limit_mb": 6144,  # Use 6GB of GPU, leave 2GB for system
            "cpu_memory_limit_mb": 24576,  # Use 24GB of RAM
            "swap_directory": "/tmp/jalapeno_swap",
            "swap_size_gb": 64,
            "cache_policy": "adaptive_lru",
            "prefetch_strategy": "markov",
        }
    )
    
    print(f"Runtime initialized on: {runtime.device_config.name}")
    print(f"GPU memory: {runtime.memory_stats()['gpu_total_mb']:.1f} MB")
    print(f"CPU memory: {runtime.memory_stats()['cpu_total_mb']:.1f} MB")
    
    # Load model (this would be from disk in real scenario)
    print("\nLoading model...")
    
    # For demo, create a mock model with similar structure to LLaMA 13B
    # In real usage, you would load an actual model
    class MockLLaMA(torch.nn.Module):
        def __init__(self, num_layers=40, hidden_size=5120):
            super().__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            
            # Create layers (in reality these would be loaded from disk)
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(hidden_size, hidden_size * 4)
                for _ in range(num_layers)
            ])
            
            self.norm = torch.nn.LayerNorm(hidden_size)
            self.lm_head = torch.nn.Linear(hidden_size, 32000)  # vocab size
            
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
                x = torch.relu(x)  # Simulate activation
            x = self.norm(x)
            return self.lm_head(x)
    
    model = MockLLaMA(num_layers=40, hidden_size=5120)
    print(f"Model created: {model.num_layers} layers, {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Configure layer streaming
    streamer_config = StreamerConfig(
        prefetch_strategy="markov",
        cache_policy="adaptive_lru",
        eviction_threshold=0.85,
        max_prefetch_depth=2,
        enable_async_loading=True,
        enable_overlap_transfer=True,
        compression_level=1,
    )
    
    # Create layer streamer
    print("\nInitializing layer streamer...")
    streamer = LayerStreamer(model, runtime, streamer_config)
    
    # Generate some input
    batch_size = 1
    seq_len = 128
    hidden_size = 5120
    
    print(f"\nGenerating input: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = streamer.forward(input_tensor)
    
    # Benchmark
    print("\nBenchmarking...")
    num_runs = 10
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}", end="\r")
            output = streamer.forward(input_tensor)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    
    print(f"\n\nBenchmark Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per forward pass: {avg_time:.3f}s")
    print(f"  Throughput: {num_runs / total_time:.2f} passes/sec")
    
    # Show streaming statistics
    stats = streamer.get_stats()
    print(f"\nStreaming Statistics:")
    print(f"  Segments: {stats['segments']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    # Show memory statistics
    mem_stats = runtime.memory_stats()
    print(f"\nMemory Statistics:")
    print(f"  GPU used: {mem_stats['gpu_used_mb']:.1f} / {mem_stats['gpu_total_mb']:.1f} MB")
    print(f"  CPU used: {mem_stats['cpu_used_mb']:.1f} / {mem_stats['cpu_total_mb']:.1f} MB")
    print(f"  Swap operations: {mem_stats['swap_operations']}")
    
    # Demonstration of interactive usage
    print("\n" + "=" * 50)
    print("Interactive Demo: Streaming individual segments")
    
    # Execute segments one by one
    for seg_id in range(min(5, stats['segments'])):
        print(f"\nExecuting segment {seg_id}...")
        
        seg_start = time.time()
        output = streamer.forward(input_tensor, segment_id=seg_id)
        seg_time = time.time() - seg_start
        
        print(f"  Segment {seg_id} time: {seg_time:.3f}s")
        print(f"  Output shape: {output.shape}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
