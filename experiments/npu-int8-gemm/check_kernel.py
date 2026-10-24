import numpy as np
import aie.iron as iron
from aie.iron.kernels import mm

M, K, N = 1, 32, 16
matmul_kernel = mm(dim_m=M, dim_k=K, dim_n=N, input_dtype=np.int8, output_dtype=np.int32, vectorized=False)
print(f"Kernel: {matmul_kernel}")
print(f"Dir: {[x for x in dir(matmul_kernel) if not x.startswith('_')]}")