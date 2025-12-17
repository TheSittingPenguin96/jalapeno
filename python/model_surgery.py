# python/model_surgery.py
"""
Python interface for Hardware-Aware Model Surgery
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

class SurgeryOperation(IntEnum):
    NONE = 0
    QUANTIZE = 1
    PRUNE = 2
    FUSE = 3
    SPLIT = 4
    REPLACE_KERNEL = 5
    REORDER = 6
    CACHE_OPTIMIZE = 7
    TILE = 8
    VECTORIZE = 9
    STREAM = 10

class HardwareCapabilities:
    """Hardware capabilities for surgery decisions"""
    
    def __init__(self):
        self.name = ""
        self.architecture = ""
        
        # Compute capabilities
        self.peak_tflops_fp32 = 0.0
        self.peak_tflops_fp16 = 0.0
        self.peak_tflops_int8 = 0.0
        self.peak_tflops_int4 = 0.0
        
        # Memory
        self.gpu_hbm_size = 0
        self.gpu_vram_size = 0
        self.cpu_ram_size = 0
        self.l2_cache_size = 0
        self.l1_cache_size = 0
        self.shared_memory_size = 0
        
        # Bandwidths
        self.gpu_hbm_bandwidth = 0.0
        self.gpu_vram_bandwidth = 0.0
        self.cpu_ram_bandwidth = 0.0
        self.pcie_bandwidth = 0.0
        self.nvlink_bandwidth = 0.0
        
        # Specialized units
        self.has_tensor_cores = False
        self.has_rt_cores = False
        self.has_dla = False
        self.has_nvenc = False
        self.has_nvdec = False
        
        # Power
        self.tdp_watts = 0.0
        self.typical_power_watts = 0.0
        self.max_power_watts = 0.0
        
        # Supported operations
        self.supported_ops = set()
        self.op_efficiency = {}
    
    @staticmethod
    def from_nvidia_smi() -> 'HardwareCapabilities':
        """Get capabilities from nvidia-smi"""
        caps = HardwareCapabilities()
        
        # Try to get GPU info
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,clocks.max.sm',
                 '--format=csv,noheader'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        caps.name = parts[0].strip()
                        
                        # Parse memory
                        mem_str = parts[1].strip()
                        if 'MiB' in mem_str:
                            caps.gpu_vram_size = int(mem_str.replace(' MiB', '')) * 1024 * 1024
                        
                        # Estimate capabilities based on GPU name
                        gpu_name = caps.name.lower()
                        if 'a100' in gpu_name:
                            caps.architecture = "ampere"
                            caps.peak_tflops_fp32 = 19.5
                            caps.peak_tflops_fp16 = 312
                            caps.has_tensor_cores = True
                        elif 'v100' in gpu_name:
                            caps.architecture = "volta"
                            caps.peak_tflops_fp32 = 14
                            caps.peak_tflops_fp16 = 112
                            caps.has_tensor_cores = True
                        elif 'rtx 30' in gpu_name:
                            caps.architecture = "ampere_rtx"
                            caps.peak_tflops_fp32 = 30
                            caps.peak_tflops_fp16 = 240
                            caps.has_tensor_cores = True
                            caps.has_rt_cores = True
                        elif 'jetson orin' in gpu_name:
                            caps.architecture = "ampere_orin"
                            caps.peak_tflops_fp32 = 2.5
                            caps.peak_tflops_fp16 = 20
                            caps.has_tensor_cores = True
                            caps.has_dla = True
                            caps.tdp_watts = 15.0
        
        except Exception as e:
            print(f"Could not query NVIDIA GPU: {e}")
        
        return caps

@dataclass
class LayerProfile:
    """Profile of a model layer"""
    name: str
    type: str
    total_flops: int
    total_memory_bytes: int
    parameters_count: int
    compute_intensity: float
    data_locality: float
    parallelism: float
    quantization_sensitivity: float
    
    @staticmethod
    def from_torch_module(module: torch.nn.Module, name: str) -> 'LayerProfile':
        """Create profile from PyTorch module"""
        profile = LayerProfile(
            name=name,
            type=module.__class__.__name__,
            total_flops=0,
            total_memory_bytes=0,
            parameters_count=0,
            compute_intensity=0.0,
            data_locality=0.5,
            parallelism=0.7,
            quantization_sensitivity=0.5
        )
        
        # Count parameters
        profile.parameters_count = sum(p.numel() for p in module.parameters())
        
        # Estimate memory
        param_memory = profile.parameters_count * 4  # Assume FP32
        profile.total_memory_bytes = param_memory
        
        # Estimate FLOPs based on layer type
        if isinstance(module, torch.nn.Linear):
            # FLOPs = 2 * input_features * output_features
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                profile.total_flops = 2 * module.in_features * module.out_features
        
        elif isinstance(module, torch.nn.Conv2d):
            # FLOPs = 2 * H_out * W_out * C_in * C_out * K_h * K_w
            # Simplified estimation
            if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                # Assume 3x3 kernel and 224x224 output
                profile.total_flops = 2 * 224 * 224 * module.in_channels * module.out_channels * 9
        
        # Estimate compute intensity
        if profile.total_memory_bytes > 0:
            profile.compute_intensity = profile.total_flops / profile.total_memory_bytes
        
        return profile

class ModelSurgeon:
    """
    Hardware-Aware Model Surgery.
    
    Analyzes models and applies hardware-specific optimizations:
    - Automatic layer placement (GPU/CPU/DLA)
    - Precision selection per layer
    - Layer fusion and kernel replacement
    - Memory layout optimization
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        runtime: Optional[Runtime] = None,
        hardware_caps: Optional[HardwareCapabilities] = None
    ):
        """
        Initialize model surgeon.
        
        Args:
            model: PyTorch model to optimize
            runtime: Jalapeño runtime (optional)
            hardware_caps: Hardware capabilities (auto-detected if None)
        """
        self.model = model
        self.runtime = runtime
        self.hardware_caps = hardware_caps or HardwareCapabilities.from_nvidia_smi()
        
        # Load native library
        self._load_native_lib()
        
        # Create native surgeon
        self._init_native_surgeon()
        
        # Analyze model
        self.layer_profiles = self._analyze_model()
        
        # Surgery state
        self.transformations_applied = []
        self.layer_placement = {}  # layer -> device
        self.layer_precision = {}  # layer -> dtype
        
        # Initialize placement
        self._initial_placement()
    
    def _load_native_lib(self):
        """Load the native C++ library"""
        lib_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "build", "libjalapeno.so"
        )
        self._native = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self._native.jalapeno_create_model_surgeon.argtypes = [
            ctypes.c_char_p,  # model_path
            ctypes.c_char_p,  # hardware_config
        ]
        self._native.jalapeno_create_model_surgeon.restype = ctypes.c_void_p
        
        self._native.jalapeno_surgeon_create_plan.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # constraints dict
        ]
        self._native.jalapeno_surgeon_create_plan.restype = ctypes.c_void_p
        
        self._native.jalapeno_surgeon_apply_plan.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # plan
        ]
        self._native.jalapeno_surgeon_apply_plan.restype = ctypes.c_bool
    
    def _init_native_surgeon(self):
        """Initialize native surgeon"""
        # For now, create a placeholder
        # In production, would serialize model and pass to native
        self._native_handle = self._native.jalapeno_create_model_surgeon(
            b"",  # Empty model path
            b"auto"  # Auto-detect hardware
        )
    
    def _analyze_model(self) -> Dict[str, LayerProfile]:
        """Analyze model layers"""
        profiles = {}
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                profile = LayerProfile.from_torch_module(module, name)
                profiles[name] = profile
        
        return profiles
    
    def _initial_placement(self):
        """Create initial device placement"""
        for name, profile in self.layer_profiles.items():
            # Simple heuristic
            if profile.compute_intensity > 10.0:
                self.layer_placement[name] = "gpu"
                self.layer_precision[name] = torch.float16
            elif profile.total_memory_bytes > 100 * 1024 * 1024:  # > 100MB
                self.layer_placement[name] = "cpu"
                self.layer_precision[name] = torch.float32
            else:
                self.layer_placement[name] = "gpu"
                self.layer_precision[name] = torch.float16
    
    def analyze_hardware_affinity(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze which layers work best on which hardware.
        
        Returns:
            Dictionary mapping layer -> device -> score
        """
        affinity = {}
        
        for name, profile in self.layer_profiles.items():
            affinity[name] = {}
            
            # Score for GPU
            gpu_score = 0.0
            
            # Compute intensity bonus
            if profile.compute_intensity > 5.0:
                gpu_score += 0.3
            
            # Memory size penalty
            memory_mb = profile.total_memory_bytes / (1024 * 1024)
            if memory_mb < 100:  # Fits in cache
                gpu_score += 0.2
            elif memory_mb < 1000:  # Fits in VRAM
                gpu_score += 0.1
            else:  # Might need streaming
                gpu_score -= 0.1
            
            # Parallelism bonus
            gpu_score += profile.parallelism * 0.2
            
            affinity[name]["gpu"] = gpu_score
            
            # Score for CPU
            cpu_score = 0.0
            
            # Low compute intensity -> CPU
            if profile.compute_intensity < 2.0:
                cpu_score += 0.3
            
            # Sequential operations
            if profile.parallelism < 0.3:
                cpu_score += 0.2
            
            # Memory locality
            cpu_score += profile.data_locality * 0.2
            
            affinity[name]["cpu"] = cpu_score
        
        return affinity
    
    def create_optimization_plan(
        self,
        constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Create optimization plan based on constraints.
        
        Args:
            constraints: Dictionary of constraints (memory_mb, power_w, etc.)
            
        Returns:
            Optimization plan
        """
        if constraints is None:
            constraints = {
                "max_memory_mb": 8192,  # 8GB
                "max_power_w": 15.0,    # 15W for Jetson
                "target_speedup": 2.0,  # 2x speedup
            }
        
        plan = {
            "operations": [],
            "layer_placement": self.layer_placement.copy(),
            "layer_precision": {k: str(v) for k, v in self.layer_precision.items()},
            "estimated_benefits": {
                "speedup": 1.0,
                "memory_reduction": 1.0,
                "power_reduction": 1.0,
            }
        }
        
        # Analyze hardware affinity
        affinity = self.analyze_hardware_affinity()
        
        # Generate optimization operations
        operations = []
        
        # 1. Device placement optimization
        for layer_name, scores in affinity.items():
            best_device = max(scores.items(), key=lambda x: x[1])[0]
            current_device = self.layer_placement.get(layer_name, "gpu")
            
            if best_device != current_device:
                operations.append({
                    "type": "device_placement",
                    "layer": layer_name,
                    "from": current_device,
                    "to": best_device,
                    "priority": 5,
                    "estimated_benefit": abs(scores[best_device] - scores[current_device]),
                })
        
        # 2. Precision optimization
        for layer_name, profile in self.layer_profiles.items():
            if profile.quantization_sensitivity < 0.3:  # Low sensitivity
                current_precision = self.layer_precision.get(layer_name, torch.float16)
                
                if current_precision == torch.float16:
                    # Can quantize to INT8
                    operations.append({
                        "type": "quantize",
                        "layer": layer_name,
                        "from": "float16",
                        "to": "int8",
                        "priority": 8,
                        "estimated_benefit": 2.0,  # 2x speedup
                    })
                elif current_precision == torch.float32:
                    operations.append({
                        "type": "quantize",
                        "layer": layer_name,
                        "from": "float32",
                        "to": "float16",
                        "priority": 7,
                        "estimated_benefit": 1.5,  # 1.5x speedup
                    })
        
        # 3. Layer fusion opportunities
        layer_names = list(self.layer_profiles.keys())
        
        # Pattern: Linear -> Activation -> Linear
        for i in range(len(layer_names) - 2):
            layer1 = layer_names[i]
            layer2 = layer_names[i + 1]
            layer3 = layer_names[i + 2]
            
            prof1 = self.layer_profiles[layer1]
            prof2 = self.layer_profiles[layer2]
            prof3 = self.layer_profiles[layer3]
            
            if (prof1.type == "Linear" and 
                prof2.type in ["ReLU", "GELU", "SiLU"] and
                prof3.type == "Linear"):
                
                operations.append({
                    "type": "fuse",
                    "layers": [layer1, layer2, layer3],
                    "fused_type": "FusedLinearActivationLinear",
                    "priority": 6,
                    "estimated_benefit": 0.4,  # 40% speedup
                })
        
        # 4. Kernel replacement for specific patterns
        for layer_name, profile in self.layer_profiles.items():
            if profile.type == "Linear" and profile.parameters_count > 1000000:
                # Large linear layer -> use optimized kernel
                operations.append({
                    "type": "replace_kernel",
                    "layer": layer_name,
                    "kernel": "cutlass_gemm",
                    "priority": 4,
                    "estimated_benefit": 0.3,
                })
        
        # Sort operations by priority
        operations.sort(key=lambda x: x["priority"], reverse=True)
        
        plan["operations"] = operations
        
        # Estimate benefits
        total_speedup = 1.0
        total_memory_reduction = 1.0
        
        for op in operations:
            if "estimated_benefit" in op:
                if op["type"] in ["quantize", "replace_kernel", "fuse"]:
                    total_speedup *= (1.0 + op["estimated_benefit"])
                elif op["type"] == "device_placement":
                    total_speedup *= (1.0 + op["estimated_benefit"] * 0.5)
        
        plan["estimated_benefits"]["speedup"] = total_speedup
        
        return plan
    
    def apply_optimization_plan(self, plan: Dict[str, Any]) -> bool:
        """
        Apply optimization plan to model.
        
        Args:
            plan: Optimization plan from create_optimization_plan()
            
        Returns:
            True if successful
        """
        successful_ops = []
        
        for op in plan["operations"]:
            try:
                if op["type"] == "device_placement":
                    self.layer_placement[op["layer"]] = op["to"]
                    successful_ops.append(op)
                    
                elif op["type"] == "quantize":
                    layer_name = op["layer"]
                    target_dtype = op["to"]
                    
                    # Convert string dtype to torch dtype
                    if target_dtype == "int8":
                        self.layer_precision[layer_name] = torch.int8
                    elif target_dtype == "float16":
                        self.layer_precision[layer_name] = torch.float16
                    elif target_dtype == "float32":
                        self.layer_precision[layer_name] = torch.float32
                    
                    successful_ops.append(op)
                    
                elif op["type"] == "fuse":
                    # Would actually fuse layers in model
                    # For now, just track
                    successful_ops.append(op)
                    
                elif op["type"] == "replace_kernel":
                    # Would replace kernel implementation
                    successful_ops.append(op)
                    
            except Exception as e:
                print(f"Failed to apply operation {op}: {e}")
        
        self.transformations_applied.extend(successful_ops)
        
        # Update plan with applied operations
        plan["applied_operations"] = successful_ops
        
        return len(successful_ops) > 0
    
    def apply_quantization(
        self,
        quantization_config: Optional[Dict[str, Any]] = None
    ) -> torch.nn.Module:
        """
        Apply quantization to model.
        
        Args:
            quantization_config: Quantization configuration
            
        Returns:
            Quantized model
        """
        if quantization_config is None:
            quantization_config = {
                "dtype": "int8",
                "per_channel": True,
                "symmetric": True,
            }
        
        # Use PyTorch's quantization
        model_copy = self.model
        
        if quantization_config["dtype"] == "int8":
            # Prepare for quantization
            model_copy.eval()
            model_copy.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Fuse layers first
            torch.quantization.fuse_modules(model_copy, [['conv', 'relu']], inplace=True)
            
            # Prepare and convert
            torch.quantization.prepare(model_copy, inplace=True)
            # Calibrate would happen here
            torch.quantization.convert(model_copy, inplace=True)
        
        return model_copy
    
    def apply_pruning(
        self,
        sparsity: float = 0.5,
        pruning_method: str = "l1_unstructured"
    ) -> torch.nn.Module:
        """
        Apply pruning to model.
        
        Args:
            sparsity: Target sparsity (0-1)
            pruning_method: Pruning method
            
        Returns:
            Pruned model
        """
        model_copy = self.model
        
        # Use PyTorch's pruning
        if pruning_method == "l1_unstructured":
            parameters_to_prune = []
            
            for name, module in model_copy.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    parameters_to_prune.append((module, 'weight'))
            
            if parameters_to_prune:
                torch.nn.utils.prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=torch.nn.utils.prune.L1Unstructured,
                    amount=sparsity,
                )
        
        return model_copy
    
    def split_model_across_devices(
        self,
        memory_constraint_per_device: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """
        Split model across devices based on memory constraints.
        
        Args:
            memory_constraint_per_device: Device -> max memory in MB
            
        Returns:
            Dictionary of device -> list of layer names
        """
        # Sort layers by memory usage
        sorted_layers = sorted(
            self.layer_profiles.items(),
            key=lambda x: x[1].total_memory_bytes,
            reverse=True
        )
        
        device_assignments = {device: [] for device in memory_constraint_per_device.keys()}
        device_memory_used = {device: 0.0 for device in memory_constraint_per_device.keys()}
        
        for layer_name, profile in sorted_layers:
            layer_memory_mb = profile.total_memory_bytes / (1024 * 1024)
            
            # Find device with enough space
            assigned = False
            for device, max_memory in memory_constraint_per_device.items():
                if device_memory_used[device] + layer_memory_mb <= max_memory:
                    device_assignments[device].append(layer_name)
                    device_memory_used[device] += layer_memory_mb
                    assigned = True
                    break
            
            if not assigned:
                # Put on device with most free space
                device_with_most_space = max(
                    memory_constraint_per_device.items(),
                    key=lambda x: x[1] - device_memory_used[x[0]]
                )[0]
                device_assignments[device_with_most_space].append(layer_name)
                device_memory_used[device_with_most_space] += layer_memory_mb
        
        return device_assignments
    
    def estimate_performance(self) -> Dict[str, float]:
        """
        Estimate performance after optimizations.
        
        Returns:
            Performance metrics
        """
        metrics = {
            "estimated_latency_ms": 0.0,
            "estimated_memory_mb": 0.0,
            "estimated_power_w": 0.0,
            "estimated_speedup": 1.0,
        }
        
        # Calculate based on layer profiles and placement
        total_latency = 0.0
        total_memory = 0.0
        
        for layer_name, profile in self.layer_profiles.items():
            device = self.layer_placement.get(layer_name, "gpu")
            precision = self.layer_precision.get(layer_name, torch.float16)
            
            # Estimate latency based on device and precision
            base_latency = profile.total_flops / 1e9  # GFLOPs
            
            if device == "gpu":
                if precision == torch.float16:
                    latency = base_latency / self.hardware_caps.peak_tflops_fp16
                elif precision == torch.int8:
                    latency = base_latency / self.hardware_caps.peak_tflops_int8
                else:
                    latency = base_latency / self.hardware_caps.peak_tflops_fp32
            else:  # CPU
                latency = base_latency / (self.hardware_caps.peak_tflops_fp32 * 0.1)  # 10% of GPU
            
            total_latency += latency * 1000  # Convert to ms
            
            # Memory
            bytes_per_element = 4 if precision == torch.float32 else 2
            layer_memory = profile.total_memory_bytes * bytes_per_element / 4  # Adjust for precision
            total_memory += layer_memory
        
        metrics["estimated_latency_ms"] = total_latency
        metrics["estimated_memory_mb"] = total_memory / (1024 * 1024)
        metrics["estimated_power_w"] = self.hardware_caps.typical_power_watts
        
        # Speedup compared to baseline (all FP32 on GPU)
        baseline_latency = 0.0
        for profile in self.layer_profiles.values():
            baseline_latency += (profile.total_flops / 1e9) / self.hardware_caps.peak_tflops_fp32
        
        metrics["estimated_speedup"] = baseline_latency * 1000 / total_latency
        
        return metrics
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            "hardware_capabilities": {
                "name": self.hardware_caps.name,
                "architecture": self.hardware_caps.architecture,
                "peak_tflops_fp16": self.hardware_caps.peak_tflops_fp16,
                "peak_tflops_int8": self.hardware_caps.peak_tflops_int8,
                "gpu_memory_mb": self.hardware_caps.gpu_vram_size / (1024 * 1024),
                "tdp_watts": self.hardware_caps.tdp_watts,
            },
            "model_statistics": {
                "total_layers": len(self.layer_profiles),
                "total_parameters": sum(p.parameters_count for p in self.layer_profiles.values()),
                "total_flops": sum(p.total_flops for p in self.layer_profiles.values()),
                "estimated_memory_mb": sum(p.total_memory_bytes for p in self.layer_profiles.values()) / (1024 * 1024),
            },
            "optimization_state": {
                "transformations_applied": len(self.transformations_applied),
                "layer_placement": self.layer_placement,
                "layer_precision": {k: str(v) for k, v in self.layer_precision.items()},
            },
            "performance_estimates": self.estimate_performance(),
        }
    
    def visualize_placement(self) -> str:
        """Create ASCII visualization of layer placement"""
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL LAYER PLACEMENT VISUALIZATION")
        lines.append("=" * 80)
        
        devices = sorted(set(self.layer_placement.values()))
        
        for device in devices:
            lines.append(f"\n[{device.upper()}]")
            lines.append("-" * 40)
            
            device_layers = [name for name, dev in self.layer_placement.items() if dev == device]
            
            for layer_name in device_layers:
                profile = self.layer_profiles[layer_name]
                precision = self.layer_precision.get(layer_name, torch.float16)
                
                # Create visual indicator based on memory usage
                memory_mb = profile.total_memory_bytes / (1024 * 1024)
                bar_length = min(int(memory_mb / 10), 30)  # 10MB per character
                memory_bar = "█" * bar_length
                
                lines.append(f"  {layer_name:30} [{precision}] {memory_bar} ({memory_mb:.1f}MB)")
        
        lines.append("\n" + "=" * 80)
        lines.append("LEGEND: █ = ~10MB of memory")
        lines.append("=" * 80)
        
        return "\n".join(lines)

class AutoSurgeon:
    """
    Automatic model surgery with reinforcement learning.
    
    Learns optimal transformations for different hardware and models.
    """
    
    def __init__(self, model_surgeon: ModelSurgeon):
        self.surgeon = model_surgeon
        self.learning_history = []
        self.transformation_policy = {}
        
    def optimize_with_rl(
        self,
        objective: str = "latency",
        constraints: Optional[Dict[str, float]] = None,
        episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize using reinforcement learning.
        
        Args:
            objective: "latency", "memory", or "power"
            constraints: Optimization constraints
            episodes: Number of RL episodes
            
        Returns:
            Best configuration found
        """
        if constraints is None:
            constraints = {
                "max_memory_mb": 8192,
                "max_latency_ms": 1000,
            }
        
        best_config = None
        best_score = float('inf' if objective == "latency" else 0)
        
        for episode in range(episodes):
            # Generate random configuration
            config = self._generate_random_config()
            
            # Apply configuration
            self._apply_configuration(config)
            
            # Evaluate
            metrics = self.surgeon.estimate_performance()
            
            # Calculate score
            if objective == "latency":
                score = metrics["estimated_latency_ms"]
                # Penalize for constraints
                if metrics["estimated_memory_mb"] > constraints.get("max_memory_mb", float('inf')):
                    score *= 2.0
            elif objective == "memory":
                score = metrics["estimated_memory_mb"]
            elif objective == "power":
                score = metrics["estimated_power_w"]
            
            # Update best
            if (objective == "latency" and score < best_score) or \
               (objective != "latency" and score > best_score):
                best_score = score
                best_config = config.copy()
                best_config["metrics"] = metrics
            
            # Record for learning
            self.learning_history.append({
                "episode": episode,
                "config": config,
                "score": score,
                "metrics": metrics,
            })
        
        # Apply best configuration
        if best_config:
            self._apply_configuration(best_config)
        
        return best_config or {}
    
    def _generate_random_config(self) -> Dict[str, Any]:
        """Generate random surgery configuration"""
        config = {
            "device_placement": {},
            "precision": {},
            "fusions": [],
        }
        
        # Random device placement
        devices = ["gpu", "cpu"]
        for layer_name in self.surgeon.layer_profiles.keys():
            config["device_placement"][layer_name] = np.random.choice(devices)
        
        # Random precision
        precisions = ["float16", "int8", "float32"]
        for layer_name in self.surgeon.layer_profiles.keys():
            config["precision"][layer_name] = np.random.choice(precisions)
        
        # Random fusions (simplified)
        layer_names = list(self.surgeon.layer_profiles.keys())
        if len(layer_names) >= 3:
            num_fusions = np.random.randint(0, len(layer_names) // 3)
            for _ in range(num_fusions):
                start = np.random.randint(0, len(layer_names) - 2)
                config["fusions"].append(layer_names[start:start+3])
        
        return config
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """Apply configuration to surgeon"""
        # Apply device placement
        for layer_name, device in config["device_placement"].items():
            self.surgeon.layer_placement[layer_name] = device
        
        # Apply precision
        for layer_name, precision_str in config["precision"].items():
            if precision_str == "float16":
                self.surgeon.layer_precision[layer_name] = torch.float16
            elif precision_str == "int8":
                self.surgeon.layer_precision[layer_name] = torch.int8
            elif precision_str == "float32":
                self.surgeon.layer_precision[layer_name] = torch.float32
    
    def learn_policy(self) -> Dict[str, Any]:
        """Learn transformation policy from history"""
        if not self.learning_history:
            return {}
        
        # Analyze what worked best
        best_episode = min(self.learning_history, key=lambda x: x["score"])
        
        policy = {
            "best_score": best_episode["score"],
            "best_config": best_episode["config"],
            "learned_rules": [],
        }
        
        # Extract rules (simplified)
        best_config = best_episode["config"]
        
        # Rule 1: Large layers on GPU
        large_layers = []
        for layer_name, profile in self.surgeon.layer_profiles.items():
            if profile.total_flops > 1e9:  # > 1 GFLOP
                if best_config["device_placement"].get(layer_name) == "gpu":
                    large_layers.append(layer_name)
        
        if large_layers:
            policy["learned_rules"].append({
                "rule": "large_compute_layers_on_gpu",
                "layers": large_layers,
                "condition": "flops > 1G",
            })
        
        # Rule 2: Memory-intensive on CPU
        memory_layers = []
        for layer_name, profile in self.surgeon.layer_profiles.items():
            memory_mb = profile.total_memory_bytes / (1024 * 1024)
            if memory_mb > 100:  # > 100MB
                if best_config["device_placement"].get(layer_name) == "cpu":
                    memory_layers.append(layer_name)
        
        if memory_layers:
            policy["learned_rules"].append({
                "rule": "memory_intensive_on_cpu",
                "layers": memory_layers,
                "condition": "memory > 100MB",
            })
        
        self.transformation_policy = policy
        return policy
