#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')
import aie.iron as iron
from aie.iron.kernels import mm
import numpy as np

matmul_kernel = iron.kernels.mm(
    dim_m=1,
    dim_k=32,
    dim_n=16,
    input_dtype=np.int8,
    output_dtype=np.int32,
    vectorized=False,
)

print(f"matmul_kernel: {matmul_kernel}")
print(f"dir: {[x for x in dir(matmul_kernel) if not x.startswith('_')]}")
