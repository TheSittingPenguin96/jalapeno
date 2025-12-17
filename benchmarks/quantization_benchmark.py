#benchmarks/quantization_benchmark.py
#!/usr/bin/env python3
"""
Benchmark quantization kernels
"""

import torch
import time
import numpy as np
from jalapeno import Quantizer, QuantizationMethod, QuantizationParams

def benchmark_quantization():
    print("Quantization Kernel Benchmark")
    print("=" * 60)
    
    quantizer = Quantizer()
    
    # Test sizes (elements)
    sizes = [1024, 8192, 65536, 524288, 4194304, 33554432]
    
    methods = [
        (QuantizationMethod.INT8, "INT8"),
        (QuantizationMethod.INT4, "INT4"),
        (QuantizationMethod.NF4, "NF4"),
        (QuantizationMethod.FP8_E4M3, "FP8_E4M3"),
    ]
    
    results = {}
    
    for size in sizes:
        print(f"\nSize: {size:,} elements ({size * 2 / 1024**2:.2f} MB FP16)")
        
        # Create random tensor
        torch.manual_seed(42)
        tensor = torch.randn(size, dtype=torch.float16).cuda()
        
        # Compute quantization params
        params = QuantizationParams.from_tensor(tensor, bits=8)
        
        for method, name in methods:
            # Warmup
            for _ in range(3):
                quantized = quantizer.quantize(tensor, method, params)
                _ = quantizer.dequantize(quantized, method, tensor.shape, params)
            
            # Benchmark quantization
            torch.cuda.synchronize()
            start = time.time()
            
            num_iterations = 100 if size <= 65536 else 10
            for _ in range(num_iterations):
                quantized = quantizer.quantize(tensor, method, params)
            
            torch.cuda.synchronize()
            quant_time = (time.time() - start) / num_iterations
            
            # Benchmark dequantization
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(num_iterations):
                dequantized = quantizer.dequantize(
                    quantized, method, tensor.shape, params
                )
            
            torch.cuda.synchronize()
            dequant_time = (time.time() - start) / num_iterations
            
            # Calculate throughput
            quant_throughput = size / quant_time / 1e9  # G elements/sec
            dequant_throughput = size / dequant_time / 1e9
            
            # Calculate compression ratio
            if method == QuantizationMethod.INT4 or method == QuantizationMethod.NF4:
                # Packed: 2 values per byte
                compressed_size = (size + 1) // 2
            else:
                compressed_size = size
            
            compression_ratio = (compressed_size * 1) / (size * 2)  # FP16 = 2 bytes
            
            # Store results
            if size not in results:
                results[size] = {}
            
            results[size][name] = {
                'quant_time_ms': quant_time * 1000,
                'dequant_time_ms': dequant_time * 1000,
                'quant_throughput_gps': quant_throughput,
                'dequant_throughput_gps': dequant_throughput,
                'compression_ratio': compression_ratio,
            }
            
            print(f"  {name}:")
            print(f"    Quant: {quant_time*1000:.3f} ms ({quant_throughput:.2f} G elem/s)")
            print(f"    Dequant: {dequant_time*1000:.3f} ms ({dequant_throughput:.2f} G elem/s)")
            print(f"    Compression: {compression_ratio:.3f} ({1/compression_ratio:.1f}x)")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Size':>12} {'Method':>10} {'Quant (ms)':>12} {'Dequant (ms)':>12} {'Throughput (G/s)':>16} {'Compression':>12}")
    print("-" * 80)
    
    for size in sizes:
        for method_name in ["INT8", "INT4", "NF4", "FP8_E4M3"]:
            if method_name in results[size]:
                r = results[size][method_name]
                print(f"{size:12,} {method_name:>10} {r['quant_time_ms']:12.3f} {r['dequant_time_ms']:12.3f} "
                      f"{r['quant_throughput_gps']:8.2f}/{r['dequant_throughput_gps']:8.2f} {1/r['compression_ratio']:12.1f}x")

def benchmark_matmul():
    print("\n\nQuantized Matrix Multiplication Benchmark")
    print("=" * 60)
    
    quantizer = Quantizer()
    
    # Matrix sizes
    sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    
    # Methods to test
    configs = [
        ("FP16", None, None),
        ("INT8xINT8", QuantizationMethod.INT8, QuantizationMethod.INT8),
        ("INT4xFP16", QuantizationMethod.INT4, None),
    ]
    
    results = {}
    
    for M, N, K in sizes:
        print(f"\nMatrix: {M}x{K} * {K}x{N} = {M}x{N}")
        print(f"Memory: {M*K*2/1024**2:.1f}MB * {K*N*2/1024**2:.1f}MB")
        
        # Create random matrices
        A = torch.randn(M, K, dtype=torch.float16).cuda()
        B = torch.randn(K, N, dtype=torch.float16).cuda()
        
        # Compute quantization params
        params_A = QuantizationParams.from_tensor(A, bits=8)
        params_B = QuantizationParams.from_tensor(B, bits=8)
        
        for config_name, method_A, method_B in configs:
            # Prepare matrices
            if method_A:
                A_quant = quantizer.quantize(A, method_A, params_A)
                A_for_mult = A_quant
            else:
                A_for_mult = A
            
            if method_B:
                B_quant = quantizer.quantize(B, method_B, params_B)
                B_for_mult = B_quant
            else:
                B_for_mult = B
            
            # Warmup
            for _ in range(3):
                if method_A or method_B:
                    C = quantizer.quantized_matmul(
                        A_for_mult, B_for_mult, 
                        method_A or QuantizationMethod.NONE,
                        method_B or QuantizationMethod.NONE,
                        params_A, params_B
                    )
                else:
                    C = torch.matmul(A, B)
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            
            num_iterations = 10 if M <= 1024 else 3
            for _ in range(num_iterations):
                if method_A or method_B:
                    C = quantizer.quantized_matmul(
                        A_for_mult, B_for_mult,
                        method_A or QuantizationMethod.NONE,
                        method_B or QuantizationMethod.NONE,
                        params_A, params_B
                    )
                else:
                    C = torch.matmul(A, B)
            
            torch.cuda.synchronize()
            avg_time = (time.time() - start) / num_iterations
            
            # Calculate FLOPs (2 * M * N * K for matmul)
            flops = 2 * M * N * K
            tflops = flops / avg_time / 1e12
            
            # Calculate memory savings
            if method_A == QuantizationMethod.INT4:
                a_size = (M * K + 1) // 2  # Packed
            elif method_A == QuantizationMethod.INT8:
                a_size = M * K
            else:
                a_size = M * K * 2  # FP16
            
            if method_B == QuantizationMethod.INT8:
                b_size = K * N
            else:
                b_size = K * N * 2  # FP16
            
            total_size = a_size + b_size
            fp16_size = M * K * 2 + K * N * 2
            compression = total_size / fp16_size
            
            # Store results
            if (M, N, K) not in results:
                results[(M, N, K)] = {}
            
            results[(M, N, K)][config_name] = {
                'time_ms': avg_time * 1000,
                'tflops': tflops,
                'compression': compression,
            }
            
            print(f"  {config_name}:")
            print(f"    Time: {avg_time*1000:.2f} ms")
            print(f"    TFLOPS: {tflops:.2f}")
            print(f"    Memory: {compression:.3f} ({1/compression:.1f}x less)")
            
            # Verify correctness
            if method_A or method_B:
                # Dequantize result for comparison
                if method_A:
                    A_dequant = quantizer.dequantize(A_quant, method_A, A.shape, params_A)
                else:
                    A_dequant = A
                
                if method_B:
                    B_dequant = quantizer.dequantize(B_quant, method_B, B.shape, params_B)
                else:
                    B_dequant = B
                
                # Reference FP16 matmul
                C_ref = torch.matmul(A_dequant, B_dequant)
                
                # Calculate error
                error = torch.max(torch.abs(C - C_ref)).item()
                print(f"    Max error vs dequantized: {error:.6f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MATMUL SUMMARY")
    print("=" * 60)
    
    for (M, N, K) in sizes:
        print(f"\n{M}x{K} * {K}x{N}:")
        for config_name in ["FP16", "INT8xINT8", "INT4xFP16"]:
            if config_name in results[(M, N, K)]:
                r = results[(M, N, K)][config_name]
                print(f"  {config_name:10} {r['time_ms']:8.2f} ms {r['tflops']:8.2f} TFLOPS {1/r['compression']:8.1f}x memory save")

if __name__ == "__main__":
    benchmark_quantization()
    benchmark_matmul()
