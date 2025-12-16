#!/usr/bin/env python3
"""
Integrated example showing LayerStreamer + KVCacheManager working together
"""

import torch
import time
from jalapeno import Runtime, LayerStreamer, AdaptiveKVCache, KVConfig

class IntegratedModel:
    """Model that uses both layer streaming and adaptive KV cache"""
    
    def __init__(self, model_path, runtime_config=None, cache_config=None):
        # Initialize runtime
        self.runtime = Runtime(**(runtime_config or {}))
        
        # Load model with layer streaming
        self.streamer = LayerStreamer(
            model_path=model_path,
            runtime=self.runtime,
            config=LayerStreamerConfig(
                prefetch_strategy="markov",
                cache_policy="adaptive_lru",
                eviction_threshold=0.85,
            )
        )
        
        # Get model info for KV cache
        model_info = self.streamer.get_model_info()
        
        # Initialize adaptive KV cache
        self.kv_cache = AdaptiveKVCache(
            num_layers=model_info['num_layers'],
            num_heads=model_info['num_heads'],
            head_dim=model_info['head_dim'],
            max_seq_len=model_info['max_seq_len'],
            config=cache_config or KVConfig()
        )
        
        # State
        self.generated_tokens = 0
        self.context_length = 0
        
    def generate(self, prompt, max_tokens=100, temperature=0.7):
        """Generate text using both optimizations"""
        start_time = time.time()
        
        # Tokenize prompt
        tokens = self._tokenize(prompt)
        self.context_length = len(tokens)
        
        # Generate tokens
        generated = []
        
        for i in range(max_tokens):
            token_start = time.time()
            
            # Forward pass with streaming
            logits = self._forward_with_streaming(tokens)
            
            # Sample next token
            next_token = self._sample(logits, temperature)
            generated.append(next_token)
            tokens.append(next_token)
            
            # Update KV cache with new token
            self._update_kv_cache(next_token, i)
            
            token_time = time.time() - token_start
            self.generated_tokens += 1
            
            # Print progress
            if (i + 1) % 10 == 0:
                token_text = self._detokenize([next_token])
                print(f"  Token {i+1}/{max_tokens}: '{token_text}' ({token_time*1000:.1f}ms)")
        
        # Decode generated text
        generated_text = self._detokenize(generated)
        
        total_time = time.time() - start_time
        tokens_per_second = self.generated_tokens / total_time
        
        # Print statistics
        print(f"\nGeneration complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        
        # Print optimization statistics
        self._print_statistics()
        
        return generated_text
    
    def _forward_with_streaming(self, tokens):
        """Forward pass using layer streaming"""
        # Convert tokens to embeddings
        embeddings = self._get_embeddings(tokens)
        
        # Forward through streaming layers
        hidden_states = embeddings
        
        for segment_id in range(self.streamer.num_segments):
            # Get KV cache for this segment
            layer_start = segment_id * self.streamer.layers_per_segment
            layer_end = min((segment_id + 1) * self.streamer.layers_per_segment, 
                          self.streamer.model_info['num_layers'])
            
            # Retrieve cached keys/values for these layers
            kv_cache_inputs = []
            for layer in range(layer_start, layer_end):
                layer_kv = self.kv_cache.get_for_layers(layer, self.context_length)
                kv_cache_inputs.append(layer_kv)
            
            # Execute segment with cached KV
            hidden_states = self.streamer.execute_segment(
                segment_id, hidden_states, kv_cache_inputs
            )
        
        # Final layer norm and LM head
        logits = self._final_projection(hidden_states)
        
        return logits
    
    def _update_kv_cache(self, token, position):
        """Update KV cache with new token"""
        # This would update the cache with the new token's keys/values
        # from all attention layers
        
        for layer in range(self.streamer.model_info['num_layers']):
            # Get the new keys/values for this layer
            # (In real implementation, these would come from the attention computation)
            new_keys = torch.randn(1, 1, self.kv_cache.head_dim, dtype=torch.float16)
            new_values = torch.randn(1, 1, self.kv_cache.head_dim, dtype=torch.float16)
            
            # Update cache
            self.kv_cache.manager.update_kv(
                layer, 0, [position], new_keys, new_values
            )
    
    def _print_statistics(self):
        """Print optimization statistics"""
        print("\n" + "=" * 50)
        print("Optimization Statistics:")
        print("-" * 50)
        
        # Layer streaming stats
        streamer_stats = self.streamer.get_stats()
        print(f"Layer Streaming:")
        print(f"  Cache hits: {streamer_stats['cache_hits']}")
        print(f"  Cache misses: {streamer_stats['cache_misses']}")
        print(f"  Hit rate: {streamer_stats['cache_hit_rate']:.1%}")
        
        # KV cache stats
        cache_stats = self.kv_cache.get_stats()
        print(f"\nKV Cache:")
        print(f"  Total tokens cached: {cache_stats['total_tokens']:,}")
        print(f"  Cache hits: {cache_stats['cache_hits']:,}")
        print(f"  Cache misses: {cache_stats['cache_misses']:,}")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"  Memory utilization: {cache_stats['memory_utilization']:.1%}")
        print(f"  Compression ratio: {cache_stats['compression_ratio']:.2f}")
        
        # Memory statistics
        mem_stats = self.runtime.memory_stats()
        print(f"\nMemory Usage:")
        print(f"  GPU: {mem_stats['gpu_used_mb']:.1f} / {mem_stats['gpu_total_mb']:.1f} MB")
        print(f"  CPU: {mem_stats['cpu_used_mb']:.1f} / {mem_stats['cpu_total_mb']:.1f} MB")
    
    # Placeholder methods for tokenization/embeddings
    def _tokenize(self, text):
        return [hash(c) % 10000 for c in text[:10]]  # Simplified
    
    def _detokenize(self, tokens):
        return ''.join(chr(t % 256) for t in tokens)
    
    def _get_embeddings(self, tokens):
        return torch.randn(1, len(tokens), 5120, dtype=torch.float16)
    
    def _sample(self, logits, temperature):
        # Simplified sampling
        return torch.argmax(logits[0, -1]).item()
    
    def _final_projection(self, hidden_states):
        return torch.randn(1, hidden_states.shape[1], 32000, dtype=torch.float16)

def main():
    print("Integrated Layer Streaming + KV Cache Demo")
    print("=" * 60)
    
    # Configuration
    runtime_config = {
        "device_config": {
            "name": "jetson_orin_nx",
            "gpu_memory_mb": 8192,
            "cpu_memory_mb": 32768,
        },
        "memory_config": {
            "gpu_memory_limit_mb": 6144,
            "cpu_memory_limit_mb": 24576,
            "swap_directory": "/tmp/jalapeno_swap",
        }
    }
    
    cache_config = KVConfig(
        max_total_tokens=32768,
        block_size=64,
        default_compression=CompressionMethod.INT8_QUANTIZATION,
        adaptive_compression=True,
        eviction_threshold=0.8,
        enable_prefetch=True,
    )
    
    # Create integrated model
    print("Initializing integrated model...")
    model = IntegratedModel(
        model_path="llama-13b",  # Placeholder
        runtime_config=runtime_config,
        cache_config=cache_config
    )
    
    # Generate text
    print("\nGenerating text with optimizations...")
    prompt = "The future of artificial intelligence"
    
    generated = model.generate(
        prompt=prompt,
        max_tokens=50,
        temperature=0.7
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()