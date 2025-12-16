#jalapeno/init.py

"""
Jalapeño: Memory-Centric AI Runtime for Edge Devices
"""

__version__ = "0.1.0-alpha"

from .runtime import Runtime, DeviceConfig, MemoryConfig
from .model_loader import ModelLoader
from .layer_streamer import LayerStreamer
from .kv_cache import KVCacheManager

# Export core functions
__all__ = [
    "Runtime",
    "DeviceConfig", 
    "MemoryConfig",
    "ModelLoader",
    "LayerStreamer",
    "KVCacheManager",
]