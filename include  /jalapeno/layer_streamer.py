# include/jalapeno/layer_streamer.py
class LayerStreamer:
    def __init__(self, model_path, device_config):
        self.model_graph = self._parse_model(model_path)
        self.layer_profiles = self._profile_layers()
        self.active_layers = LRUCache(capacity=GPU_MEMORY_CAPACITY)
        self.prefetch_queue = []
    
    def execute_layer(self, layer_id, inputs):
        # Check if layer is resident
        if layer_id not in self.active_layers:
            self._swap_in(layer_id)
            self._prefetch_next_layers(layer_id)
        
        # Execute
        return self._execute_on_optimal_device(layer_id, inputs)
    
    def _swap_in(self, layer_id):
        """Load layer from slow memory to fast memory"""
        # Choose source tier based on access frequency
        source_tier = self._select_source_tier(layer_id)
        
        # Async load with priority
        self.memory_mgr.transfer_async(
            src_tier=source_tier,
            dst_tier=MemoryTier.TIER_0,
            tensor_id=layer_id,
            priority=self._calculate_priority(layer_id)
        )
        
        # If GPU memory full, evict least recently used
        if self.memory_mgr.is_full(MemoryTier.TIER_0):
            victim = self.active_layers.evict()
            self._swap_out(victim)
