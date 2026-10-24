import sys
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase40")
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase39")
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase20g")

import numpy as np
import os
import time
from ref_q8_gemm import ref_q8_0_gemm
from bench_q8_c4 import pack_A_vec, pack_B_vec
from phase40_bench import run_kernel_generic

M, K, N = 512, 1024, 3584
cache_dir = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

q8v_xc = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.xclbin"
q8v_ins = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.insts.bin"

rng = np.random.default_rng(42)
A = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
B = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
nb = K // 32
As = np.ones((M, nb), np.float32)
Bs = np.ones((nb, N), np.float32)

C_ref = ref_q8_0_gemm(M, K, N, A, As, B, Bs)
print(f"C_ref: min={C_ref.min():.3f} max={C_ref.max():.3f} mean={C_ref.mean():.3f}")

m_tile, n_tile, k_chunk = 64, 64, 256
Ap = pack_A_vec(A, As, M, K, m_tile, k_chunk)
Bp = pack_B_vec(B, Bs, K, N, n_tile, k_chunk)
print(f"Ap shape: {Ap.shape}, Bp shape: {Bp.shape}")

print("running kernel...")
C_dev, t = run_kernel_generic(q8v_xc, Ap, Bp, M, K, N)
print(f"kernel returned in {t:.2f}ms")
print(f"C_dev: min={C_dev.min():.3f} max={C_dev.max():.3f} mean={C_dev.mean():.3f}")
print(f"C_dev has NaN: {np.isnan(C_dev).any()}")
print(f"C_dev has inf: {np.isinf(C_dev).any()}")

diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
max_err = float(np.max(diff))
print(f"max_err={max_err:.3e}")
