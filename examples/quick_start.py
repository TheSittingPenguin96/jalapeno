# examples/quick_start.py
import jalapeno as jl
import torch

# Initialize runtime
runtime = jl.Runtime(
    device="jetson_orin_nx",
    memory_config={
        "gpu_memory": "8GB",
        "cpu_memory": "16GB",
        "swap_path": "/path/to/ssd",
        "cache_policy": "adaptive"
    }
)

# Load a model larger than GPU memory
model = runtime.load_model(
    "meta-llama/Llama-2-13b-chat-hf",
    streaming=True,
    quantization="int4",
    kv_cache_optimization=True
)

# Generate with automatic memory management
output = runtime.generate(
    prompt="Explain quantum computing",
    max_length=512,
    temperature=0.7
)

print(f"Memory usage: {runtime.memory_stats()}")
print(f"Throughput: {runtime.throughput()} tokens/sec")
