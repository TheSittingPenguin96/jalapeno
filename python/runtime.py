# python/runtime.py

import ctypes
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

# Load native library
_lib_path = os.path.join(os.path.dirname(__file__), "..", "build", "libjalapeno.so")
_native = ctypes.CDLL(_lib_path)

@dataclass
class DeviceConfig:
    """Configuration for target device"""
    name: str = "auto"  # auto, jetson_orin_nano, jetson_orin_nx, xavier_nx
    gpu_memory_mb: int = 0  # 0 = auto-detect
    cpu_memory_mb: int = 0
    enable_unified_memory: bool = True
    enable_async_transfers: bool = True
    max_concurrent_transfers: int = 4
    
@dataclass  
class MemoryConfig:
    """Memory hierarchy configuration"""
    gpu_memory_limit_mb: int = 0  # 0 = use all available
    cpu_memory_limit_mb: int = 0
    swap_directory: str = "/tmp/jalapeno_swap"
    swap_size_gb: int = 32
    cache_policy: str = "adaptive_lru"  # lru, lfu, arc, adaptive
    prefetch_strategy: str = "markov"  # none, linear, markov, neural
    compression_level: int = 1  # 0-3, higher = more compression
    
class Runtime:
    """Main Jalapeño runtime interface"""
    
    def __init__(
        self,
        device_config: Optional[DeviceConfig] = None,
        memory_config: Optional[MemoryConfig] = None
    ):
        self.device_config = device_config or DeviceConfig()
        self.memory_config = memory_config or MemoryConfig()
        
        # Detect device if auto
        if self.device_config.name == "auto":
            self.device_config.name = self._detect_device()
            
        # Initialize native runtime
        self._native_handle = self._initialize_native()
        
        # Memory manager
        self.memory_manager = None
        self.kv_cache_manager = None
        self.layer_streamer = None
        
        # Statistics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "swap_operations": 0,
            "gpu_utilization": 0.0,
            "cpu_utilization": 0.0,
        }
        
    def _detect_device(self) -> str:
        """Auto-detect Jetson device"""
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                
            if "orin nano" in model:
                return "jetson_orin_nano"
            elif "orin nx" in model:
                return "jetson_orin_nx" 
            elif "xavier nx" in model:
                return "jetson_xavier_nx"
            elif "agx orin" in model:
                return "jetson_agx_orin"
            else:
                return "generic_linux"
        except:
            return "generic_linux"
            
    def _initialize_native(self):
        """Initialize C++ runtime"""
        # Configure memory tiers based on device
        tier_capacities = {
            "tier_0_gpu_hbm": self._get_gpu_memory(),
            "tier_2_cpu_ram": self._get_cpu_memory(),
        }
        
        # Initialize native library
        _native.jalapeno_initialize(
            ctypes.c_char_p(self.device_config.name.encode()),
            tier_capacities
        )
        
        return _native.jalapeno_create_runtime()
        
    def load_model(
        self,
        model_path: str,
        model_type: str = "auto",  # llama, mistral, phi, qwen, etc.
        streaming: bool = True,
        quantization: Optional[str] = None,  # "int8", "int4", "nf4"
        offload_strategy: str = "auto",  # auto, minimal, balanced, aggressive
        context_size: int = 4096,
    ):
        """Load a model with Jalapeño optimizations"""
        from .model_loader import ModelLoader
        
        loader = ModelLoader(self)
        model = loader.load(
            model_path=model_path,
            model_type=model_type,
            streaming=streaming,
            quantization=quantization,
            offload_strategy=offload_strategy,
            context_size=context_size,
        )
        
        # Initialize layer streaming if enabled
        if streaming:
            from .layer_streamer import LayerStreamer
            self.layer_streamer = LayerStreamer(model, self)
            
        # Initialize KV cache manager
        from .kv_cache import KVCacheManager
        self.kv_cache_manager = KVCacheManager(
            model.config.hidden_size,
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            context_size,
            self
        )
        
        return model
        
    def generate(
        self,
        prompt: str,
        model: torch.nn.Module,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ):
        """Generate text with memory optimization"""
        if self.layer_streamer:
            return self.layer_streamer.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                kv_cache_manager=self.kv_cache_manager,
            )
        else:
            # Fallback to standard generation
            return self._standard_generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
    def memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        if self._native_handle:
            stats = _native.jalapeno_get_memory_stats(self._native_handle)
            return {
                "gpu_used_mb": stats.gpu_used / (1024 * 1024),
                "gpu_total_mb": stats.gpu_total / (1024 * 1024),
                "cpu_used_mb": stats.cpu_used / (1024 * 1024),
                "cpu_total_mb": stats.cpu_total / (1024 * 1024),
                "hit_rate": stats.hit_rate,
                "swap_operations": stats.swap_operations,
            }
        return {}
        
    def throughput(self) -> float:
        """Get current throughput in tokens/second"""
        if hasattr(self, '_last_generation_stats'):
            tokens = self._last_generation_stats.get('tokens_generated', 0)
            time = self._last_generation_stats.get('generation_time', 1.0)
            return tokens / time
        return 0.0
        
    def __del__(self):
        """Cleanup"""
        if hasattr(self, '_native_handle') and self._native_handle:
            _native.jalapeno_destroy_runtime(self._native_handle)
