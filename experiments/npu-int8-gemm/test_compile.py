#!/usr/bin/env python3
import numpy as np
import aie.iron as iron
from aie.iron.kernels import mm

TILE_M, TILE_K, TILE_N = 1, 32, 16
matmul_kernel = mm(
    dim_m=TILE_M,
    dim_k=TILE_K,
    dim_n=TILE_N,
    input_dtype=np.int8,
    output_dtype=np.int32,
    vectorized=False,
)

print(f"matmul_kernel: {matmul_kernel}")
print(f"matmul_kernel object_file_name: {matmul_kernel.object_file_name}")
print(f"matmul_kernel compile_flags: {matmul_kernel.compile_flags}")
print(f"matmul_kernel symbol_prefix: {matmul_kernel.symbol_prefix}")
print(f"matmul_kernel shared_object_file_name: {matmul_kernel.shared_object_file_name}")
print(f"matmul_kernel source_path: {matmul_kernel.source_path}")
