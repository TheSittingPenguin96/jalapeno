#python/kv_cache.py

"""
Python interface for Jalapeño KV Cache Manager
"""

import ctypes
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from dataclasses import dataclass
from enum import IntEnum

from .tensor import Tensor
from .runtime import Runtime

class CompressionMethod(IntEnum):
    NONE = 0
    INT8_QUANTIZATION = 1
    INT4_QUANTIZATION = 2
    NF4_QUANTIZATION = 3
    PRUNING = 4
    DCT_COMPRESSION = 5
    DICTIONARY = 6
    DELTA_COMPRESSION = 7
    ADAPTIVE_MIXED = 8

class CacheLocation(IntEnum):
    GPU_HBM = 0
    GPU_VRAM = 1
    CPU_RAM = 2
    CPU_PMEM = 3
    NVME_SSD = 4
    COMPRESSED_RAM = 5

@dataclass
class KVConfig:
    """Configuration for KV Cache Manager"""
    max_total_tokens: int = 32768
    block_size: int = 64
    max_blocks_per_layer: int = 512
    
    # Compression settings
    default_compression: CompressionMethod = CompressionMethod.INT8_QUANTIZATION
    adaptive_compression: bool = True
    compression_ratio_target: float = 0.5
    lossy_compression: bool = True
    
    # Eviction settings
    eviction_policy: str = "importance_aware"
    eviction_threshold: float = 0.8
    enable_prefetch: bool = True
    enable_async_eviction: bool = True
    
    # Storage hierarchy (fastest to slowest)
    storage_hierarchy: List[CacheLocation] = None
    
    # Monitoring
    collect_statistics: bool = True
    statistics_interval_ms: int = 1000
    
    def __post_init__(self):
        if self.storage_hierarchy is None:
            self.storage_hierarchy = [
                CacheLocation.GPU_HBM,
                CacheLocation.GPU_VRAM,
                CacheLocation.CPU_RAM,
                CacheLocation.NVME_SSD
            ]

class KVCacheManager:
    """
    Python wrapper for the C++ KV Cache Manager.
    
    Manages key-value cache for transformer models with:
    - Adaptive compression
    - Multi-tier storage
    - Intelligent eviction
    - Prefetching
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        config: Optional[KVConfig] = None,
        runtime: Optional[Runtime] = None
    ):
        """
        Initialize KV Cache Manager.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            config: Cache configuration
            runtime: Jalapeño runtime (optional)
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.config = config or KVConfig()
        self.runtime = runtime
        
        # Load native library
        self._load_native_lib()
        
        # Initialize native cache manager
        self._init_native_manager()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'prefetch_hits': 0,
            'evicted_blocks': 0,
            'compressed_blocks': 0,
            'total_tokens': 0,
        }
        
        # State tracking
        self.current_position = 0
        self.active_blocks = {}
        
    def _load_native_lib(self):
        """Load the native C++ library"""
        lib_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "build", "libjalapeno.so"
        )
        self._native = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self._native.jalapeno_create_kv_cache.argtypes = [
            ctypes.c_size_t,  # num_layers
            ctypes.c_size_t,  # num_heads
            ctypes.c_size_t,  # head_dim
            ctypes.c_size_t,  # max_seq_len
            ctypes.c_void_p,  # config
            ctypes.c_void_p,  # runtime handle (optional)
        ]
        self._native.jalapeno_create_kv_cache.restype = ctypes.c_void_p
        
        self._native.jalapeno_kv_cache_get.argtypes = [
            ctypes.c_void_p,  # cache handle
            ctypes.c_size_t,  # layer_idx
            ctypes.c_size_t,  # head_idx
            ctypes.POINTER(ctypes.c_int64),  # positions
            ctypes.c_size_t,  # num_positions
            ctypes.POINTER(ctypes.c_void_p),  # keys output
            ctypes.POINTER(ctypes.c_void_p),  # values output
        ]
        
        self._native.jalapeno_kv_cache_update.argtypes = [
            ctypes.c_void_p,  # cache handle
            ctypes.c_size_t,  # layer_idx
            ctypes.c_size_t,  # head_idx
            ctypes.POINTER(ctypes.c_int64),  # positions
            ctypes.c_size_t,  # num_positions
            ctypes.c_void_p,  # keys tensor
            ctypes.c_void_p,  # values tensor
        ]
        
        self._native.jalapeno_kv_cache_get_stats.argtypes = [
            ctypes.c_void_p,  # cache handle
            ctypes.c_void_p,  # stats struct
        ]
        
    def _init_native_manager(self):
        """Initialize the native cache manager"""
        # Convert config to C struct
        class NativeConfig(ctypes.Structure):
            _fields_ = [
                ("max_total_tokens", ctypes.c_size_t),
                ("block_size", ctypes.c_size_t),
                ("max_blocks_per_layer", ctypes.c_size_t),
                ("default_compression", ctypes.c_int),
                ("adaptive_compression", ctypes.c_bool),
                ("compression_ratio_target", ctypes.c_float),
                ("lossy_compression", ctypes.c_bool),
                ("eviction_threshold", ctypes.c_float),
                ("enable_prefetch", ctypes.c_bool),
                ("enable_async_eviction", ctypes.c_bool),
            ]
        
        config = NativeConfig()
        config.max_total_tokens = self.config.max_total_tokens
        config.block_size = self.config.block_size
        config.max_blocks_per_layer = self.config.max_blocks_per_layer
        config.default_compression = self.config.default_compression.value
        config.adaptive_compression = self.config.adaptive_compression
        config.compression_ratio_target = self.config.compression_ratio_target
        config.lossy_compression = self.config.lossy_compression
        config.eviction_threshold = self.config.eviction_threshold
        config.enable_prefetch = self.config.enable_prefetch
        config.enable_async_eviction = self.config.enable_async_eviction
        
        # Create cache manager
        runtime_handle = self.runtime._native_handle if self.runtime else None
        
        self._native_handle = self._native.jalapeno_create_kv_cache(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.max_seq_len,
            ctypes.byref(config),
            runtime_handle
        )
        
        if not self._native_handle:
            raise RuntimeError("Failed to create native KV cache manager")
    
    def get_kv(
        self,
        layer_idx: int,
        head_idx: int,
        positions: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get key-value cache for specific positions.
        
        Args:
            layer_idx: Transformer layer index
            head_idx: Attention head index
            positions: Token positions to retrieve
            
        Returns:
            Tuple of (keys, values) tensors
        """
        if layer_idx >= self.num_layers or head_idx >= self.num_heads:
            raise ValueError("Layer or head index out of range")
        
        # Convert positions to C array
        positions_array = (ctypes.c_int64 * len(positions))(*positions)
        
        # Prepare output pointers
        keys_ptr = ctypes.c_void_p()
        values_ptr = ctypes.c_void_p()
        
        # Call native function
        self._native.jalapeno_kv_cache_get(
            self._native_handle,
            layer_idx,
            head_idx,
            positions_array,
            len(positions),
            ctypes.byref(keys_ptr),
            ctypes.byref(values_ptr)
        )
        
        # Convert to PyTorch tensors (placeholder)
        # In production, would convert from native tensors
        batch_size = 1  # Assuming
        shape = (batch_size, len(positions), self.head_dim)
        
        # Create dummy tensors for now
        keys = torch.zeros(shape, dtype=torch.float16)
        values = torch.zeros(shape, dtype=torch.float16)
        
        # Update statistics
        cache_hit = True  # Would be determined by native code
        if cache_hit:
            self.stats['hits'] += 1
        else:
            self.stats['misses'] += 1
        
        self.stats['total_tokens'] += len(positions)
        
        return keys, values
    
    def update_kv(
        self,
        layer_idx: int,
        head_idx: int,
        positions: List[int],
        keys: torch.Tensor,
        values: torch.Tensor
    ):
        """
        Update key-value cache with new values.
        
        Args:
            layer_idx: Transformer layer index
            head_idx: Attention head index
            positions: Token positions to update
            keys: New key tensor
            values: New value tensor
        """
        if layer_idx >= self.num_layers or head_idx >= self.num_heads:
            raise ValueError("Layer or head index out of range")
        
        if len(positions) != keys.shape[1]:
            raise ValueError("Number of positions must match sequence dimension of keys")
        
        # Convert positions to C array
        positions_array = (ctypes.c_int64 * len(positions))(*positions)
        
        # Convert tensors to native format (placeholder)
        # In production, would convert PyTorch tensors to native tensors
        keys_native = self._tensor_to_native(keys)
        values_native = self._tensor_to_native(values)
        
        # Call native function
        self._native.jalapeno_kv_cache_update(
            self._native_handle,
            layer_idx,
            head_idx,
            positions_array,
            len(positions),
            keys_native,
            values_native
        )
        
        # Update current position
        self.current_position = max(self.current_position, max(positions))
        
        # Update active blocks tracking
        block_key = (layer_idx, head_idx, positions[0] // self.config.block_size)
        self.active_blocks[block_key] = {
            'last_access': self.current_position,
            'size': len(positions),
        }
    
    def prefetch(
        self,
        layer_idx: int,
        head_idx: int,
        predicted_positions: List[int]
    ):
        """
        Prefetch predicted positions into cache.
        
        Args:
            layer_idx: Transformer layer index
            head_idx: Attention head index
            predicted_positions: Predicted future token positions
        """
        if not self.config.enable_prefetch or not predicted_positions:
            return
        
        # Group positions into blocks
        block_size = self.config.block_size
        for i in range(0, len(predicted_positions), block_size):
            block_positions = predicted_positions[i:i + block_size]
            
            # Check if block is already cached
            block_key = (layer_idx, head_idx, block_positions[0] // block_size)
            if block_key not in self.active_blocks:
                # Prefetch this block
                # In production, would call native prefetch function
                self.stats['prefetch_hits'] += 1
    
    def compress(self, method: Optional[CompressionMethod] = None):
        """
        Compress cache using specified method.
        
        Args:
            method: Compression method (uses default if None)
        """
        # This would trigger native compression
        # For now, just update statistics
        if method is None:
            method = self.config.default_compression
        
        print(f"Compressing cache with method: {method.name}")
        
        # Simulate compression
        self.stats['compressed_blocks'] = len(self.active_blocks) // 2
    
    def evict(self, num_tokens: Optional[int] = None):
        """
        Evict tokens from cache.
        
        Args:
            num_tokens: Number of tokens to evict (evicts based on policy if None)
        """
        if num_tokens is None:
            # Evict based on threshold
            utilization = self.stats['total_tokens'] / self.config.max_total_tokens
            if utilization > self.config.eviction_threshold:
                to_evict = int(self.stats['total_tokens'] * 0.1)  # 10%
                self._evict_tokens(to_evict)
        else:
            self._evict_tokens(num_tokens)
    
    def _evict_tokens(self, num_tokens: int):
        """Evict specific number of tokens"""
        # Simple LRU eviction
        blocks_to_evict = []
        tokens_evicted = 0
        
        # Sort blocks by last access
        sorted_blocks = sorted(
            self.active_blocks.items(),
            key=lambda x: x[1]['last_access']
        )
        
        for block_key, block_info in sorted_blocks:
            if tokens_evicted >= num_tokens:
                break
            
            blocks_to_evict.append(block_key)
            tokens_evicted += block_info['size']
        
        # Evict blocks
        for block_key in blocks_to_evict:
            del self.active_blocks[block_key]
        
        self.stats['evicted_blocks'] += len(blocks_to_evict)
        self.stats['total_tokens'] -= tokens_evicted
        
        print(f"Evicted {len(blocks_to_evict)} blocks, {tokens_evicted} tokens")
    
    def clear(self):
        """Clear entire cache"""
        self.active_blocks.clear()
        self.stats['total_tokens'] = 0
        self.current_position = 0
        
        # Call native clear
        # self._native.jalapeno_kv_cache_clear(self._native_handle)
        
        print("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Get native stats (placeholder)
        class NativeStats(ctypes.Structure):
            _fields_ = [
                ("total_blocks", ctypes.c_size_t),
                ("blocks_in_gpu", ctypes.c_size_t),
                ("blocks_in_cpu", ctypes.c_size_t),
                ("blocks_in_storage", ctypes.c_size_t),
                ("compressed_blocks", ctypes.c_size_t),
                ("total_tokens", ctypes.c_size_t),
                ("cache_hits", ctypes.c_size_t),
                ("cache_misses", ctypes.c_size_t),
                ("prefetch_hits", ctypes.c_size_t),
                ("evicted_blocks", ctypes.c_size_t),
                ("hit_rate", ctypes.c_float),
                ("average_latency_ms", ctypes.c_float),
                ("compression_ratio", ctypes.c_float),
                ("memory_utilization", ctypes.c_float),
            ]
        
        stats = NativeStats()
        # self._native.jalapeno_kv_cache_get_stats(self._native_handle, ctypes.byref(stats))
        
        # Combine with Python stats
        result = {
            'total_blocks': len(self.active_blocks),
            'blocks_in_gpu': len(self.active_blocks),  # Placeholder
            'blocks_in_cpu': 0,
            'blocks_in_storage': 0,
            'compressed_blocks': self.stats['compressed_blocks'],
            'total_tokens': self.stats['total_tokens'],
            'cache_hits': self.stats['hits'],
            'cache_misses': self.stats['misses'],
            'prefetch_hits': self.stats['prefetch_hits'],
            'evicted_blocks': self.stats['evicted_blocks'],
            'hit_rate': self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses']),
            'average_latency_ms': 0.0,  # Would be from native
            'compression_ratio': 1.0,   # Would be from native
            'memory_utilization': self.stats['total_tokens'] / self.config.max_total_tokens,
            
            # Additional derived stats
            'active_blocks': len(self.active_blocks),
            'current_position': self.current_position,
            'config': {
                'max_total_tokens': self.config.max_total_tokens,
                'block_size': self.config.block_size,
                'eviction_threshold': self.config.eviction_threshold,
            }
        }
        
        return result
    
    def set_importance_scores(
        self,
        layer_idx: int,
        scores: List[float]
    ):
        """
        Set importance scores for cache blocks.
        
        Args:
            layer_idx: Layer index
            scores: Importance scores for each block
        """
        # This would update native importance scores
        # For now, just store for reference
        self._importance_scores = scores
    
    def register_access_pattern(
        self,
        layer_idx: int,
        head_idx: int,
        pattern: List[int]
    ):
        """
        Register access pattern for learning.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            pattern: Sequence of accessed positions
        """
        # This would update native access pattern learning
        pass
    
    def predict_next_accesses(
        self,
        layer_idx: int,
        head_idx: int,
        current_position: int
    ) -> List[int]:
        """
        Predict next accesses for prefetching.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            current_position: Current token position
            
        Returns:
            Predicted future positions
        """
        # Simple linear prediction
        block_size = self.config.block_size
        next_block_start = ((current_position // block_size) + 1) * block_size
        
        return list(range(next_block_start, next_block_start + block_size))
    
    def _tensor_to_native(self, tensor: torch.Tensor) -> ctypes.c_void_p:
        """Convert PyTorch tensor to native tensor (placeholder)"""
        # In production, would create native tensor from PyTorch data
        return ctypes.c_void_p()
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, '_native_handle') and self._native_handle:
            # Destroy native cache manager
            # self._native.jalapeno_destroy_kv_cache(self._native_handle)
            pass

class AdaptiveKVCache:
    """
    Higher-level KV cache with automatic adaptation.
    
    This class provides a more Pythonic interface and
    automatically adapts compression and eviction based on usage.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        config: Optional[KVConfig] = None
    ):
        self.manager = KVCacheManager(
            num_layers, num_heads, head_dim, max_seq_len, config
        )
        
        # Adaptation state
        self.adaptation_interval = 1000  # steps
        self.step_count = 0
        self.performance_history = []
        
        # Auto-tuning
        self.auto_tune = True
        self.target_hit_rate = 0.95
        self.target_memory_utilization = 0.7
        
    def get(self, layer_idx: int, head_idx: int, positions: List[int]):
        """Get keys/values with auto-adaptation"""
        # Get from cache
        keys, values = self.manager.get_kv(layer_idx, head_idx, positions)
        
        # Adapt if needed
        self.step_count += 1
        if self.step_count % self.adaptation_interval == 0:
            self._adapt()
        
        return keys, values
    
    def update(self, layer_idx: int, head_idx: int, positions: List[int], keys, values):
        """Update cache with auto-eviction"""
        # Check memory pressure
        stats = self.manager.get_statistics()
        if stats['memory_utilization'] > self.target_memory_utilization:
            self.manager.evict()
        
        # Update cache
        self.manager.update_kv(layer_idx, head_idx, positions, keys, values)
    
    def _adapt(self):
        """Adapt cache parameters based on performance"""
        if not self.auto_tune:
            return
        
        stats = self.manager.get_statistics()
        self.performance_history.append(stats['hit_rate'])
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        # Adjust compression if hit rate is low
        avg_hit_rate = sum(self.performance_history) / len(self.performance_history)
        
        if avg_hit_rate < self.target_hit_rate:
            # Increase compression to fit more in cache
            print(f"Low hit rate ({avg_hit_rate:.2f}), increasing compression")
            self.manager.compress()
        
        # Adjust eviction threshold based on memory utilization
        if stats['memory_utilization'] > 0.9:
            # More aggressive eviction
            self.manager.config.eviction_threshold = 0.7
            print("High memory utilization, lowering eviction threshold")
        elif stats['memory_utilization'] < 0.5:
            # Less aggressive eviction
            self.manager.config.eviction_threshold = 0.85
            print("Low memory utilization, raising eviction threshold")
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return self.manager.get_statistics()
