# examples/llama_with_kv_cache.py
#!/usr/bin/env python3
"""
Example: Running LLaMA with adaptive KV cache management
"""

import sys
import time
import torch
import numpy as np
from jalapeno import Runtime, KVCacheManager, KVConfig, AdaptiveKVCache

def benchmark_kv_cache():
    print("Jalapeño KV Cache Benchmark")
    print("=" * 50)
    
    # Model parameters (LLaMA 7B-like)
    NUM_LAYERS = 32
    NUM_HEADS = 32
    HEAD_DIM = 128
    MAX_SEQ_LEN = 4096
    BATCH_SIZE = 1
    
    # Configure KV cache
    config = KVConfig(
        max_total_tokens=32768,
        block_size=64,
        default_compression=CompressionMethod.INT8_QUANTIZATION,
        adaptive_compression=True,
        eviction_policy="importance_aware",
        eviction_threshold=0.8,
        enable_prefetch=True,
        storage_hierarchy=[
            CacheLocation.GPU_HBM,
            CacheLocation.CPU_RAM,
            CacheLocation.NVME_SSD
        ]
    )
    
    # Create adaptive cache
    print("Initializing adaptive KV cache...")
    cache = AdaptiveKVCache(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        config=config
    )
    
    print(f"Cache configured for {NUM_LAYERS} layers, {NUM_HEADS} heads")
    print(f"Head dimension: {HEAD_DIM}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")
    print(f"Max cache tokens: {config.max_total_tokens:,}")
    print(f"Block size: {config.block_size}")
    
    # Simulate generation
    print("\nSimulating text generation...")
    
    total_tokens = 1024
    warmup_tokens = 100
    
    latencies = []
    hit_rates = []
    
    for token_idx in range(total_tokens):
        if token_idx % 100 == 0:
            print(f"  Token {token_idx}/{total_tokens}", end="\r")
        
        start_time = time.time()
        
        # Simulate attention across all layers and heads
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                # Current position and context
                positions = list(range(max(0, token_idx - 63), token_idx + 1))
                
                # Get from cache (some will be misses initially)
                keys, values = cache.get(layer, head, positions)
                
                # Simulate attention computation
                # (In real model, this would compute attention scores)
                
                # Generate new keys/values for this token
                new_keys = torch.randn(BATCH_SIZE, 1, HEAD_DIM, dtype=torch.float16)
                new_values = torch.randn(BATCH_SIZE, 1, HEAD_DIM, dtype=torch.float16)
                
                # Update cache with new token
                cache.update(layer, head, [token_idx], new_keys, new_values)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # ms
        
        if token_idx >= warmup_tokens:
            latencies.append(latency)
        
        # Get stats every 100 tokens
        if token_idx % 100 == 99:
            stats = cache.get_stats()
            hit_rates.append(stats['hit_rate'])
    
    print("\n\nBenchmark Results:")
    print("-" * 30)
    
    if latencies:
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Average latency per token: {avg_latency:.2f} ms")
        print(f"P95 latency: {p95_latency:.2f} ms")
        print(f"P99 latency: {p99_latency:.2f} ms")
        print(f"Tokens per second: {1000 / avg_latency:.1f}")
    
    if hit_rates:
        final_hit_rate = hit_rates[-1]
        avg_hit_rate = np.mean(hit_rates)
        
        print(f"\nCache Performance:")
        print(f"Final hit rate: {final_hit_rate:.1%}")
        print(f"Average hit rate: {avg_hit_rate:.1%}")
    
    # Final statistics
    stats = cache.get_stats()
    print(f"\nFinal Cache Statistics:")
    print(f"Total blocks: {stats['total_blocks']:,}")
    print(f"Blocks in GPU: {stats['blocks_in_gpu']:,}")
    print(f"Blocks in CPU: {stats['blocks_in_cpu']:,}")
    print(f"Compressed blocks: {stats['compressed_blocks']:,}")
    print(f"Total tokens cached: {stats['total_tokens']:,}")
    print(f"Cache hits: {stats['cache_hits']:,}")
    print(f"Cache misses: {stats['cache_misses']:,}")
    print(f"Prefetch hits: {stats['prefetch_hits']:,}")
    print(f"Evicted blocks: {stats['evicted_blocks']:,}")
    print(f"Memory utilization: {stats['memory_utilization']:.1%}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}")
    
    # Memory savings estimate
    uncompressed_size = stats['total_tokens'] * HEAD_DIM * 2 * 2  # 2 matrices * 2 bytes per fp16
    compressed_size = uncompressed_size * stats['compression_ratio']
    savings = uncompressed_size - compressed_size
    
    print(f"\nMemory Savings:")
    print(f"Uncompressed: {uncompressed_size / (1024**3):.2f} GB")
    print(f"Compressed: {compressed_size / (1024**3):.2f} GB")
    print(f"Savings: {savings / (1024**3):.2f} GB ({savings/uncompressed_size:.1%})")
    
    # Test different scenarios
    print("\n" + "=" * 50)
    print("Scenario Testing:")
    
    # Test 1: Long sequence (simulating chat context)
    print("\n1. Long sequence simulation (4096 tokens):")
    cache.clear()
    
    long_seq_positions = list(range(4096))
    start = time.time()
    
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            # Get all positions (will trigger eviction/compression)
            keys, values = cache.get(layer, head, long_seq_positions[:1000])  # First 1000
    
    long_seq_time = time.time() - start
    stats = cache.get_stats()
    
    print(f"   Time: {long_seq_time:.2f}s")
    print(f"   Memory utilization: {stats['memory_utilization']:.1%}")
    print(f"   Compressed blocks: {stats['compressed_blocks']}")
    
    # Test 2: Random access pattern
    print("\n2. Random access pattern:")
    cache.clear()
    
    random_positions = np.random.permutation(4096)[:500].tolist()
    start = time.time()
    
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            for pos in random_positions:
                keys, values = cache.get(layer, head, [pos])
    
    random_time = time.time() - start
    stats = cache.get_stats()
    
    print(f"   Time: {random_time:.2f}s")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    
    print("\nBenchmark completed successfully!")

def test_compression_methods():
    """Test different compression methods"""
    print("\n" + "=" * 50)
    print("Testing Compression Methods")
    print("=" * 50)
    
    NUM_LAYERS = 4
    NUM_HEADS = 8
    HEAD_DIM = 128
    
    methods = [
        (CompressionMethod.NONE, "No compression"),
        (CompressionMethod.INT8_QUANTIZATION, "INT8 Quantization"),
        (CompressionMethod.INT4_QUANTIZATION, "INT4 Quantization"),
        (CompressionMethod.PRUNING, "Pruning (50%)"),
    ]
    
    results = []
    
    for method, name in methods:
        print(f"\nTesting: {name}")
        
        config = KVConfig(
            max_total_tokens=8192,
            default_compression=method,
            adaptive_compression=False
        )
        
        cache = KVCacheManager(
            NUM_LAYERS, NUM_HEADS, HEAD_DIM,
            config=config
        )
        
        # Fill cache
        positions = list(range(1024))
        keys = torch.randn(1, 1024, HEAD_DIM, dtype=torch.float16)
        values = torch.randn(1, 1024, HEAD_DIM, dtype=torch.float16)
        
        start_fill = time.time()
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                cache.update_kv(layer, head, positions, keys, values)
        
        fill_time = time.time() - start_fill
        
        # Read back
        start_read = time.time()
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                read_keys, read_values = cache.get_kv(layer, head, positions[:100])
        
        read_time = time.time() - start_read
        
        stats = cache.get_statistics()
        
        results.append({
            'method': name,
            'fill_time': fill_time,
            'read_time': read_time,
            'compressed_blocks': stats['compressed_blocks'],
            'memory_utilization': stats['memory_utilization']
        })
        
        print(f"  Fill time: {fill_time:.3f}s")
        print(f"  Read time: {read_time:.3f}s")
        print(f"  Compressed blocks: {stats['compressed_blocks']}")
        print(f"  Memory utilization: {stats['memory_utilization']:.1%}")
    
    # Print comparison
    print("\n" + "=" * 50)
    print("Compression Method Comparison:")
    print("-" * 50)
    print(f"{'Method':<20} {'Fill Time':<10} {'Read Time':<10} {'Memory Util':<12}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['method']:<20} {result['fill_time']:<10.3f} {result['read_time']:<10.3f} {result['memory_utilization']:<12.1%}")

if __name__ == "__main__":
    benchmark_kv_cache()
    test_compression_methods()