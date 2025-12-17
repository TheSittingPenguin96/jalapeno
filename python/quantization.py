# python/quantization.py
"""
Python interface for Quantization Kernels
"""

import ctypes
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from enum import IntEnum

class QuantizationMethod(IntEnum):
    NONE = 0
    INT8 = 1
    INT4 = 2
    NF4 = 3
    FP8_E4M3 = 4
    FP8_E5M2 = 5
    GROUPWISE_INT8 = 6
    ASYMMETRIC_INT8 = 7

class QuantizationParams:
    """Parameters for quantization"""
    
    def __init__(
        self,
        scale: float = 1.0,
        zero_point: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 0.0
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.min_val = min_val
        self.max_val = max_val
    
    @staticmethod
    def from_tensor(
        tensor: torch.Tensor,
        bits: int = 8,
        symmetric: bool = False,
        percentile: float = 0.999
    ) -> 'QuantizationParams':
        """
        Compute quantization parameters from tensor.
        
        Args:
            tensor: Input tensor
            bits: Number of bits for quantization
            symmetric: Use symmetric quantization
            percentile: Percentile for clipping outliers
            
        Returns:
            QuantizationParams
        """
        # Convert to numpy for computation
        data = tensor.detach().cpu().numpy().flatten()
        
        # Find range (with percentile clipping)
        if percentile < 1.0:
            sorted_data = np.sort(np.abs(data))
            idx = int(len(sorted_data) * percentile)
            max_abs = sorted_data[idx]
            
            if symmetric:
                min_val = -max_abs
                max_val = max_abs
            else:
                # Need to find both min and max percentiles
                sorted_data = np.sort(data)
                min_idx = int(len(sorted_data) * (1 - percentile) / 2)
                max_idx = int(len(sorted_data) * (1 + percentile) / 2)
                min_val = sorted_data[min_idx]
                max_val = sorted_data[max_idx]
        else:
            min_val = float(data.min())
            max_val = float(data.max())
            if symmetric:
                max_abs = max(abs(min_val), abs(max_val))
                min_val = -max_abs
                max_val = max_abs
        
        # Compute scale and zero point
        range_val = max_val - min_val
        
        if bits == 8:
            scale = range_val / 255.0
            zero_point = -min_val / scale if scale != 0 else 0.0
        elif bits == 4:
            scale = range_val / 15.0
            zero_point = -min_val / scale if scale != 0 else 0.0
        else:
            scale = 1.0
            zero_point = 0.0
        
        return QuantizationParams(scale, zero_point, min_val, max_val)
    
    def to_native(self):
        """Convert to native structure"""
        # This would create a C struct
        pass

class Quantizer:
    """
    Python wrapper for quantization kernels.
    
    Provides high-performance quantization/dequantization on GPU.
    """
    
    def __init__(self):
        """Initialize quantizer"""
        # Load native library
        self._load_native_lib()
        
        # Create native quantizer
        self._native_handle = self._create_native_quantizer()
        
        # Statistics
        self.stats = {
            'total_quantized': 0,
            'total_dequantized': 0,
            'average_compression_ratio': 0.0,
            'average_error': 0.0,
        }
    
    def _load_native_lib(self):
        """Load the native C++ library"""
        lib_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "build", "libjalapeno.so"
        )
        self._native = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self._native.jalapeno_create_quantizer.argtypes = []
        self._native.jalapeno_create_quantizer.restype = ctypes.c_void_p
        
        self._native.jalapeno_quantize_tensor.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # input tensor
            ctypes.c_void_p,  # output tensor
            ctypes.c_size_t,  # num_elements
            ctypes.c_int,     # method
            ctypes.c_void_p,  # params
            ctypes.c_void_p,  # stream
        ]
        
        self._native.jalapeno_dequantize_tensor.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # input tensor
            ctypes.c_void_p,  # output tensor
            ctypes.c_size_t,  # num_elements
            ctypes.c_int,     # method
            ctypes.c_void_p,  # params
            ctypes.c_void_p,  # stream
        ]
        
        self._native.jalapeno_find_min_max.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # tensor
            ctypes.c_size_t,  # num_elements
            ctypes.POINTER(ctypes.c_float),  # min output
            ctypes.POINTER(ctypes.c_float),  # max output
            ctypes.c_void_p,  # stream
        ]
    
    def _create_native_quantizer(self):
        """Create native quantizer instance"""
        handle = self._native.jalapeno_create_quantizer()
        if not handle:
            raise RuntimeError("Failed to create quantizer")
        return handle
    
    def quantize(
        self,
        tensor: torch.Tensor,
        method: QuantizationMethod,
        params: Optional[QuantizationParams] = None,
        stream: Any = None
    ) -> torch.Tensor:
        """
        Quantize a tensor.
        
        Args:
            tensor: Input tensor (FP16/FP32)
            method: Quantization method
            params: Quantization parameters (auto-computed if None)
            stream: CUDA stream
            
        Returns:
            Quantized tensor
        """
        if params is None:
            bits = 8 if method == QuantizationMethod.INT8 else 4
            params = QuantizationParams.from_tensor(tensor, bits=bits)
        
        # Determine output dtype and shape
        if method in [QuantizationMethod.INT4, QuantizationMethod.NF4]:
            # Packed: 2 values per byte
            output_shape = (tensor.numel() + 1) // 2
            dtype = torch.uint8
        elif method in [QuantizationMethod.INT8, QuantizationMethod.ASYMMETRIC_INT8]:
            output_shape = tensor.shape
            dtype = torch.int8 if method == QuantizationMethod.INT8 else torch.uint8
        elif method in [QuantizationMethod.FP8_E4M3, QuantizationMethod.FP8_E5M2]:
            output_shape = tensor.shape
            dtype = torch.uint8
        else:
            raise ValueError(f"Unsupported quantization method: {method}")
        
        # Create output tensor
        output = torch.empty(output_shape, dtype=dtype, device=tensor.device)
        
        # Convert to native
        input_ptr = self._tensor_to_ptr(tensor)
        output_ptr = self._tensor_to_ptr(output)
        params_ptr = self._params_to_ptr(params)
        stream_ptr = self._stream_to_ptr(stream)
        
        # Call native quantization
        self._native.jalapeno_quantize_tensor(
            self._native_handle,
            input_ptr,
            output_ptr,
            tensor.numel(),
            method.value,
            params_ptr,
            stream_ptr
        )
        
        # Update statistics
        self.stats['total_quantized'] += tensor.numel()
        
        # Calculate compression ratio
        input_size = tensor.numel() * tensor.element_size()
        output_size = output.numel() * output.element_size()
        ratio = output_size / input_size
        self.stats['average_compression_ratio'] = (
            self.stats['average_compression_ratio'] * 
            (self.stats['total_quantized'] - tensor.numel()) + ratio
        ) / self.stats['total_quantized']
        
        return output
    
    def dequantize(
        self,
        tensor: torch.Tensor,
        method: QuantizationMethod,
        original_shape: Tuple[int, ...],
        params: QuantizationParams,
        dtype: torch.dtype = torch.float16,
        stream: Any = None
    ) -> torch.Tensor:
        """
        Dequantize a tensor.
        
        Args:
            tensor: Quantized tensor
            method: Quantization method used
            original_shape: Shape of original tensor
            params: Quantization parameters used for quantization
            dtype: Output dtype (FP16/FP32)
            stream: CUDA stream
            
        Returns:
            Dequantized tensor
        """
        # Create output tensor
        output = torch.empty(original_shape, dtype=dtype, device=tensor.device)
        
        # Convert to native
        input_ptr = self._tensor_to_ptr(tensor)
        output_ptr = self._tensor_to_ptr(output)
        params_ptr = self._params_to_ptr(params)
        stream_ptr = self._stream_to_ptr(stream)
        
        # Call native dequantization
        num_elements = np.prod(original_shape)
        
        self._native.jalapeno_dequantize_tensor(
            self._native_handle,
            input_ptr,
            output_ptr,
            num_elements,
            method.value,
            params_ptr,
            stream_ptr
        )
        
        # Update statistics
        self.stats['total_dequantized'] += num_elements
        
        return output
    
    def find_min_max(
        self,
        tensor: torch.Tensor,
        stream: Any = None
    ) -> Tuple[float, float]:
        """
        Find min and max values in tensor.
        
        Args:
            tensor: Input tensor
            stream: CUDA stream
            
        Returns:
            Tuple of (min_value, max_value)
        """
        min_val = ctypes.c_float()
        max_val = ctypes.c_float()
        
        tensor_ptr = self._tensor_to_ptr(tensor)
        stream_ptr = self._stream_to_ptr(stream)
        
        self._native.jalapeno_find_min_max(
            self._native_handle,
            tensor_ptr,
            tensor.numel(),
            ctypes.byref(min_val),
            ctypes.byref(max_val),
            stream_ptr
        )
        
        return min_val.value, max_val.value
    
    def quantized_matmul(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        method_A: QuantizationMethod,
        method_B: QuantizationMethod,
        params_A: QuantizationParams,
        params_B: QuantizationParams,
        stream: Any = None
    ) -> torch.Tensor:
        """
        Perform quantized matrix multiplication.
        
        Args:
            A: Left matrix (quantized)
            B: Right matrix (quantized or FP16)
            method_A: Quantization method for A
            method_B: Quantization method for B
            params_A: Quantization parameters for A
            params_B: Quantization parameters for B
            stream: CUDA stream
            
        Returns:
            Result matrix (FP16/FP32)
        """
        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2
        assert A.shape[1] == B.shape[0]
        
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        # Create output tensor
        output = torch.empty((M, N), dtype=torch.float16, device=A.device)
        
        # Convert to native
        A_ptr = self._tensor_to_ptr(A)
        B_ptr = self._tensor_to_ptr(B)
        output_ptr = self._tensor_to_ptr(output)
        params_A_ptr = self._params_to_ptr(params_A)
        params_B_ptr = self._params_to_ptr(params_B)
        stream_ptr = self._stream_to_ptr(stream)
        
        # Call native quantized GEMM
        self._native.jalapeno_quantized_gemm(
            self._native_handle,
            A_ptr, method_A.value, params_A_ptr,
            B_ptr, method_B.value, params_B_ptr,
            output_ptr,  # FP16 output
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
            stream_ptr
        )
        
        return output
    
    def _tensor_to_ptr(self, tensor: torch.Tensor) -> ctypes.c_void_p:
        """Convert PyTorch tensor to void pointer"""
        return ctypes.c_void_p(tensor.data_ptr())
    
    def _params_to_ptr(self, params: QuantizationParams) -> ctypes.c_void_p:
        """Convert QuantizationParams to native pointer (placeholder)"""
        return ctypes.c_void_p()
    
    def _stream_to_ptr(self, stream: Any) -> ctypes.c_void_p:
        """Convert CUDA stream to pointer (0 for default)"""
        if stream is None:
            return ctypes.c_void_p(0)
        # Convert PyTorch stream to CUDA stream pointer
        return ctypes.c_void_p(stream.cuda_stream)
    
    def measure_error(
        self,
        original: torch.Tensor,
        quantized: torch.Tensor,
        method: QuantizationMethod,
        params: QuantizationParams
    ) -> Dict[str, float]:
        """
        Measure quantization error.
        
        Args:
            original: Original tensor
            quantized: Quantized tensor
            method: Quantization method used
            params: Quantization parameters used
            
        Returns:
            Dictionary with error metrics
        """
        # Dequantize
        dequantized = self.dequantize(
            quantized, method, original.shape, params, original.dtype
        )
        
        # Calculate errors
        mse = torch.mean((original - dequantized) ** 2).item()
        rmse = np.sqrt(mse)
        
        abs_error = torch.abs(original - dequantized)
        max_abs_error = torch.max(abs_error).item()
        mean_abs_error = torch.mean(abs_error).item()
        
        # Relative error
        with torch.no_grad():
            rel_error = abs_error / (torch.abs(original) + 1e-8)
            max_rel_error = torch.max(rel_error).item()
            mean_rel_error = torch.mean(rel_error).item()
        
        # Update average error
        self.stats['average_error'] = (
            self.stats['average_error'] * 
            (self.stats['total_quantized'] - original.numel()) + mse
        ) / self.stats['total_quantized']
        
        return {
            'mse': mse,
            'rmse': rmse,
            'max_absolute_error': max_abs_error,
            'mean_absolute_error': mean_abs_error,
            'max_relative_error': max_rel_error,
            'mean_relative_error': mean_rel_error,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quantization statistics"""
        return self.stats.copy()
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, '_native_handle') and self._native_handle:
            # Destroy native quantizer
            # self._native.jalapeno_destroy_quantizer(self._native_handle)
            pass

class AdaptiveQuantizer:
    """
    Adaptive quantizer that selects optimal quantization method
    based on tensor characteristics and target compression.
    """
    
    def __init__(self, target_compression: float = 0.25):
        """
        Initialize adaptive quantizer.
        
        Args:
            target_compression: Target compression ratio (0.25 = 4x compression)
        """
        self.quantizer = Quantizer()
        self.target_compression = target_compression
        
        # Method performance tracking
        self.method_performance = {
            QuantizationMethod.INT8: {'error': 0.0, 'speed': 0.0, 'usage': 0},
            QuantizationMethod.INT4: {'error': 0.0, 'speed': 0.0, 'usage': 0},
            QuantizationMethod.NF4: {'error': 0.0, 'speed': 0.0, 'usage': 0},
            QuantizationMethod.FP8_E4M3: {'error': 0.0, 'speed': 0.0, 'usage': 0},
        }
        
        # Tensor characteristics -> method mapping
        self.characteristic_rules = []
        self._init_rules()
    
    def _init_rules(self):
        """Initialize rules for method selection"""
        # Rule: High sparsity -> INT4
        self.characteristic_rules.append({
            'condition': lambda stats: stats['sparsity'] > 0.8,
            'method': QuantizationMethod.INT4,
            'priority': 1.0
        })
        
        # Rule: Normal distribution -> NF4
        self.characteristic_rules.append({
            'condition': lambda stats: 0.2 < stats['normality'] < 0.8,
            'method': QuantizationMethod.NF4,
            'priority': 0.8
        })
        
        # Rule: Small range -> INT8
        self.characteristic_rules.append({
            'condition': lambda stats: stats['range_ratio'] < 0.1,
            'method': QuantizationMethod.INT8,
            'priority': 0.9
        })
        
        # Rule: Need high precision -> FP8
        self.characteristic_rules.append({
            'condition': lambda stats: stats['required_precision'] > 0.95,
            'method': QuantizationMethod.FP8_E4M3,
            'priority': 0.7
        })
    
    def analyze_tensor(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Analyze tensor characteristics.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Dictionary of characteristics
        """
        # Convert to numpy for analysis
        data = tensor.detach().cpu().numpy().flatten()
        
        # Basic statistics
        min_val = float(data.min())
        max_val = float(data.max())
        mean_val = float(data.mean())
        std_val = float(data.std())
        
        # Range characteristics
        range_val = max_val - min_val
        abs_max = max(abs(min_val), abs(max_val))
        range_ratio = range_val / (abs_max + 1e-8) if abs_max > 0 else 0.0
        
        # Sparsity (near-zero values)
        threshold = abs_max * 0.01  # 1% of max magnitude
        sparsity = np.sum(np.abs(data) < threshold) / len(data)
        
        # Normality (how close to normal distribution)
        # Simplified: check if values are mostly within 2 std of mean
        within_2std = np.sum(np.abs(data - mean_val) < 2 * std_val) / len(data)
        normality = within_2std
        
        # Entropy (information content)
        hist, _ = np.histogram(data, bins=50)
        hist = hist / len(data)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(50)  # 50 bins
        normalized_entropy = entropy / max_entropy
        
        return {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'range': range_val,
            'range_ratio': range_ratio,
            'sparsity': sparsity,
            'normality': normality,
            'entropy': normalized_entropy,
            'required_precision': 0.8,  # Could be passed as parameter
        }
    
    def select_method(
        self,
        tensor: torch.Tensor,
        target_bits: Optional[int] = None
    ) -> QuantizationMethod:
        """
        Select optimal quantization method for tensor.
        
        Args:
            tensor: Input tensor
            target_bits: Target bits per value (None for auto)
            
        Returns:
            Selected quantization method
        """
        # Analyze tensor
        characteristics = self.analyze_tensor(tensor)
        
        # Apply rules
        method_scores = {}
        
        for rule in self.characteristic_rules:
            if rule['condition'](characteristics):
                method = rule['method']
                method_scores[method] = method_scores.get(method, 0.0) + rule['priority']
        
        # Add performance-based scoring
        for method, perf in self.method_performance.items():
            if perf['usage'] > 0:
                # Higher score for better performance (lower error, higher speed)
                error_score = 1.0 - perf['error']
                speed_score = perf['speed']
                perf_score = 0.7 * error_score + 0.3 * speed_score
                method_scores[method] = method_scores.get(method, 0.0) + perf_score
        
        # If target bits specified, filter methods
        if target_bits is not None:
            valid_methods = []
            if target_bits == 8:
                valid_methods = [QuantizationMethod.INT8, QuantizationMethod.FP8_E4M3]
            elif target_bits == 4:
                valid_methods = [QuantizationMethod.INT4, QuantizationMethod.NF4]
            
            # Remove invalid methods
            for method in list(method_scores.keys()):
                if method not in valid_methods:
                    del method_scores[method]
        
        # Select best method
        if method_scores:
            best_method = max(method_scores.items(), key=lambda x: x[1])[0]
        else:
            # Default to INT8
            best_method = QuantizationMethod.INT8
        
        return best_method
    
    def quantize_adaptive(
        self,
        tensor: torch.Tensor,
        target_compression: Optional[float] = None,
        max_error: Optional[float] = None
    ) -> Tuple[torch.Tensor, QuantizationMethod, QuantizationParams, Dict[str, float]]:
        """
        Quantize tensor with adaptive method selection.
        
        Args:
            tensor: Input tensor
            target_compression: Target compression ratio
            max_error: Maximum allowed error
            
        Returns:
            Tuple of (quantized_tensor, method, params, metrics)
        """
        if target_compression is None:
            target_compression = self.target_compression
        
        # Select method based on target compression
        target_bits = None
        if target_compression <= 0.25:  # 4x compression
            target_bits = 4
        else:
            target_bits = 8
        
        method = self.select_method(tensor, target_bits)
        
        # Compute quantization parameters
        bits = 4 if method in [QuantizationMethod.INT4, QuantizationMethod.NF4] else 8
        symmetric = method == QuantizationMethod.INT8
        
        params = QuantizationParams.from_tensor(
            tensor, bits=bits, symmetric=symmetric
        )
        
        # Quantize
        quantized = self.quantizer.quantize(tensor, method, params)
        
        # Measure error
        error_metrics = self.quantizer.measure_error(tensor, quantized, method, params)
        
        # Check if error is acceptable
        if max_error is not None and error_metrics['rmse'] > max_error:
            # Try with higher precision
            if method in [QuantizationMethod.INT4, QuantizationMethod.NF4]:
                # Switch to 8-bit
                method = QuantizationMethod.INT8
                params = QuantizationParams.from_tensor(tensor, bits=8, symmetric=True)
                quantized = self.quantizer.quantize(tensor, method, params)
                error_metrics = self.quantizer.measure_error(tensor, quantized, method, params)
        
        # Update performance tracking
        self._update_performance(method, error_metrics, tensor.numel())
        
        # Calculate actual compression
        input_size = tensor.numel() * tensor.element_size()
        output_size = quantized.numel() * quantized.element_size()
        actual_compression = output_size / input_size
        
        # Add compression to metrics
        error_metrics['target_compression'] = target_compression
        error_metrics['actual_compression'] = actual_compression
        error_metrics['compression_ratio'] = 1.0 / actual_compression
        
        return quantized, method, params, error_metrics
    
    def _update_performance(
        self,
        method: QuantizationMethod,
        error_metrics: Dict[str, float],
        num_elements: int
    ):
        """Update performance tracking for method"""
        perf = self.method_performance[method]
        
        # Update error (exponential moving average)
        alpha = 0.1
        perf['error'] = alpha * error_metrics['rmse'] + (1 - alpha) * perf['error']
        
        # Update usage count
        perf['usage'] += 1
        
        # Speed would be measured from timing
    
    def get_method_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for quantization methods"""
        recommendations = {}
        
        for method, perf in self.method_performance.items():
            if perf['usage'] > 0:
                recommendations[method.name] = {
                    'average_error': perf['error'],
                    'usage_count': perf['usage'],
                    'recommended_for': self._get_recommended_use_cases(method),
                }
        
        return recommendations
    
    def _get_recommended_use_cases(self, method: QuantizationMethod) -> List[str]:
        """Get recommended use cases for method"""
        if method == QuantizationMethod.INT8:
            return ["weights", "activations", "general purpose"]
        elif method == QuantizationMethod.INT4:
            return ["weights", "sparse tensors", "memory-constrained"]
        elif method == QuantizationMethod.NF4:
            return ["normally distributed weights", "LLM parameters"]
        elif method == QuantizationMethod.FP8_E4M3:
            return ["high precision required", "training", "sensitive layers"]
        else:
            return []
