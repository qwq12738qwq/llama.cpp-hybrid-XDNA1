#!/usr/bin/env python3
"""Detailed test for M=1 K=32 N=16 - check for 6x factor."""
import os, sys, time
import numpy as np
sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')
from aie.utils.npukernel import NPUKernel

XCLBIN = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_m1_k32_n16/design.xclbin"
INSTS = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_m1_k32_n16/design.insts.bin"
M, K, N = 1, 32, 16

if not os.path.exists(XCLBIN):
    print(f"XCLBIN not found: {XCLBIN}")
    sys.exit(1)

print(f"[TEST] Loading xclbin for M={M} K={K} N={N}...")
kernel = NPUKernel(
    xclbin_path=XCLBIN,
    insts_path=INSTS,
    kernel_name="MLIR_AIE",
)

print("[TEST] Loaded kernel")

# Test pattern: all ones - expected output is all 32s
print("\n=== Pattern: All ones (expected output = 32) ===")
A_np = np.ones((1, K), dtype=np.int8)
B_np = np.ones((K, N), dtype=np.int8)
C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)

import aie.iron as iron
A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
B = iron.tensor(B_np, dtype=np.int8, device="npu")
C = iron.tensor((1, N), dtype=np.int32, device="npu")

kernel(A, B, C)
C_out = C.numpy()

print(f"C_ref (first 8): {C_ref.flatten()[:8]}")
print(f"C_out (first 8): {C_out.flatten()[:8]}")
print(f"Ratio: {C_out.flatten()[:8] / C_ref.flatten()[:8]}")

# Test pattern: A[k] = k, B = 1
print("\n=== Pattern: A[k]=k (0..31), B=1 (expected: sum 0..31 = 496) ===")
A_np = np.arange(K, dtype=np.int8).reshape(1, K)
B_np = np.ones((K, N), dtype=np.int8)
C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)

A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
B = iron.tensor(B_np, dtype=np.int8, device="npu")
C = iron.tensor((1, N), dtype=np.int32, device="npu")

kernel(A, B, C)
C_out = C.numpy()

print(f"C_ref (first 8): {C_ref.flatten()[:8]}")
print(f"C_out (first 8): {C_out.flatten()[:8]}")
ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
print(f"Ratio: {ratio.flatten()[:8]}")

# Test pattern: A=1, B[k,n] = k*N+n
print("\n=== Pattern: A=1, B[k,n]=k*N+n ===")
A_np = np.ones((1, K), dtype=np.int8)
B_np = np.arange(K*N, dtype=np.int8).reshape(K, N)
C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)

A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
B = iron.tensor(B_np, dtype=np.int8, device="npu")
C = iron.tensor((1, N), dtype=np.int32, device="npu")

kernel(A, B, C)
C_out = C.numpy()

print(f"C_ref (first 8): {C_ref.flatten()[:8]}")
print(f"C_out (first 8): {C_out.flatten()[:8]}")
ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
print(f"Ratio: {ratio.flatten()[:8]}")

# Test random
print("\n=== Pattern: Random (5 iterations) ===")
for i in range(5):
    A_np = np.random.default_rng(i + 42).integers(-128, 127, size=(1, K), dtype=np.int8)
    B_np = np.random.default_rng(i + 100).integers(-128, 127, size=(K, N), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)

    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((1, N), dtype=np.int32, device="npu")

    kernel(A, B, C)
    C_out = C.numpy()

    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    
    if np.any(diff > 0):
        print(f"  Iter {i}: MISMATCH! max_diff={max_diff}")
        print(f"  C_ref[:8]={C_ref.flatten()[:8]}")
        print(f"  C_out[:8]={C_out.flatten()[:8]}")
        ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
        print(f"  Ratio[:8]={ratio.flatten()[:8]}")
    else:
        print(f"  Iter {i}: PASS")

