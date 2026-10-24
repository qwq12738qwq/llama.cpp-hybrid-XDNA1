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

M, K, N = 64, 256, 64
cache_dir = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

# Use a smaller shape that might have a cached kernel
# Or just run with M64_K256_N64 even if no kernel exists, we'll see

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

# Write packed data to files for inspection
np.save("/tmp/Ap.npy", Ap)
np.save("/tmp/Bp.npy", Bp)
print("packed data saved")

# Check scales in packed data
a_scales = np.frombuffer(Ap[M*K:].tobytes(), dtype=np.float32).reshape(M, K//32)
b_scales = np.frombuffer(Bp[K*N:].tobytes(), dtype=np.float32).reshape(K//32, N)
print(f"A scales: min={a_scales.min():.6f} max={a_scales.max():.6f}")
print(f"B scales: min={b_scales.min():.6f} max={b_scales.max():.6f}")

# Try to find a kernel
q8v_xc = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.xclbin"
print(f"looking for: {q8v_xc}")
print(f"exists: {os.path.exists(q8v_xc)}")

# List available kernels
import glob
kernels = glob.glob(f"{cache_dir}/M*_K*_N*_C1_Q8SCALEV/design.xclbin")
print(f"available Q8SCALEV kernels: {len(kernels)}")
for k in sorted(kernels)[:5]:
    print(f"  {k}")
if len(kernels) > 5:
    print(f"  ... and {len(kernels)-5} more")
