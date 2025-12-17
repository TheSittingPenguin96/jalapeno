import jalapeno
import torch

# Initialize runtime
rt = jalapeno.Runtime()

# Check memory stats
print("Memory stats:", rt.memory_stats())

# Create a test tensor via Python
import numpy as np
from jalapeno.runtime import create_tensor

tensor = create_tensor((1024, 1024), "float32", device="cuda")
print("Tensor created on:", tensor.device)
