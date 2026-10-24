#!/usr/bin/env python3
"""Test M=1 K=1024 N=16 by chunking 32x M=1 K=32 N=16 using Phase 9H xclbin."""
import os, sys, time
import numpy as np
sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')
from aie.utils.npukernel import NPUKernel

XCLBIN = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_m1_k32_n16/design.xclbin"
INSTS = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_m1_k32_n16/design.insts.bin"
M, K_CHUNK, N = 1, 32, 16
K_TOTAL = 1024
NUM_CHUNKS = K_TOTAL // K_CHUNK  # 32

if not os.path.exists(XCLBIN):
    print(f"XCLBIN not found: {XCLBIN}")
    sys.exit(1)

print(f"[TEST] Loading xclbin for M={M} K={K_CHUNK} N={N} (will chunk {NUM_CHUNKS}x for K={K_TOTAL})...")
kernel = NPUKernel(
    xclbin_path=XCLBIN,
    insts_path=INSTS,
    kernel_name="MLIR_AIE",
)

print("[TEST] Loaded kernel")

# Test pattern 1: all ones
print("\n=== Pattern 1: All ones (expected output = 1024) ===")
A_np = np.ones((1, K_TOTAL), dtype=np.int8)
B_np = np.ones((K_TOTAL, N), dtype=np.int8)
C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)

import aie.iron as iron
C_out = np.zeros((1, N), dtype=np.int32)

for chunk in range(NUM_CHUNKS):
    A_chunk = A_np[:, chunk*K_CHUNK:(chunk+1)*K_CHUNK].copy()
    B_chunk = B_np[chunk*K_CHUNK:(chunk+1)*K_CHUNK, :].copy()
    
    A = iron.tensor(A_chunk, dtype=np.int8, device="npu")
    B = iron.tensor(B_chunk, dtype=np.int8, device="npu")
    C = iron.tensor((1, N), dtype=np.int32, device="npu")
    
    kernel(A, B, C)
    C_out += C.numpy()

print(f"C_ref[:8]={C_ref.flatten()[:8]}")
print(f"C_out[:8]={C_out.flatten()[:8]}")
ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
print(f"Ratio[:8]={ratio.flatten()[:8]}")

if np.array_equal(C_out, C_ref):
    print("PASS: Exact match!")
else:
    print("FAIL: Mismatch!")
    diff = np.abs(C_out - C_ref)
    print(f"Max diff: {np.max(diff)}")

# Test pattern 2: A[k] = k, B = 1
print("\n=== Pattern 2: A[k]=k (0..1023), B=1 (expected: sum 0..1023 = 523776) ===")
A_np = np.arange(K_TOTAL, dtype=np.int8).reshape(1, K_TOTAL)
B_np = np.ones((K_TOTAL, N), dtype=np.int8)
C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)

C_out = np.zeros((1, N), dtype=np.int32)

for chunk in range(NUM_CHUNKS):
    A_chunk = A_np[:, chunk*K_CHUNK:(chunk+1)*K_CHUNK].copy()
    B_chunk = B_np[chunk*K_CHUNK:(chunk+1)*K_CHUNK, :].copy()
    
    A = iron.tensor(A_chunk, dtype=np.int8, device="npu")
    B = iron.tensor(B_chunk, dtype=np.int8, device="npu")
    C = iron.tensor((1, N), dtype=np.int32, device="npu")
    
    kernel(A, B, C)
    C_out += C.numpy()

print(f"C_ref[:8]={C_ref.flatten()[:8]}")
print(f"C_out[:8]={C_out.flatten()[:8]}")
ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
print(f"Ratio[:8]={ratio.flatten()[:8]}")

if np.array_equal(C_out, C_ref):
    print("PASS: Exact match!")
else:
    print("FAIL: Mismatch!")
    diff = np.abs(C_out - C_ref)
    print(f"Max diff: {np.max(diff)}")

# Test pattern 3: A=1, B[k,n] = k*N+n
print("\n=== Pattern 3: A=1, B[k,n]=k*N+n ===")
A_np = np.ones((1, K_TOTAL), dtype=np.int8)
B_np = np.arange(K_TOTAL*N, dtype=np.int8).reshape(K_TOTAL, N)
C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)

C_out = np.zeros((1, N), dtype=np.int32)

for chunk in range(NUM_CHUNKS):
    A_chunk = A_np[:, chunk*K_CHUNK:(chunk+1)*K_CHUNK].copy()
    B_chunk = B_np[chunk*K_CHUNK:(chunk+1)*K_CHUNK, :].copy()
    
    A = iron.tensor(A_chunk, dtype=np.int8, device="npu")
    B = iron.tensor(B_chunk, dtype=np.int8, device="npu")
    C = iron.tensor((1, N), dtype=np.int32, device="npu")
    
    kernel(A, B, C)
    C_out += C.numpy()

print(f"C_ref[:8]={C_ref.flatten()[:8]}")
print(f"C_out[:8]={C_out.flatten()[:8]}")
ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
print(f"Ratio[:8]={ratio.flatten()[:8]}")

if np.array_equal(C_out, C_ref):
    print("PASS: Exact match!")
else:
    print("FAIL: Mismatch!")
    diff = np.abs(C_out - C_ref)
    print(f"Max diff: {np.max(diff)}")

# Test pattern 4: random
print("\n=== Pattern 4: Random (5 iterations) ===")
all_pass = True
for i in range(5):
    A_np = np.random.default_rng(i + 42).integers(-128, 127, size=(1, K_TOTAL), dtype=np.int8)
    B_np = np.random.default_rng(i + 100).integers(-128, 127, size=(K_TOTAL, N), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)

    C_out = np.zeros((1, N), dtype=np.int32)

    for chunk in range(NUM_CHUNKS):
        A_chunk = A_np[:, chunk*K_CHUNK:(chunk+1)*K_CHUNK].copy()
        B_chunk = B_np[chunk*K_CHUNK:(chunk+1)*K_CHUNK, :].copy()
        
        A = iron.tensor(A_chunk, dtype=np.int8, device="npu")
        B = iron.tensor(B_chunk, dtype=np.int8, device="npu")
        C = iron.tensor((1, N), dtype=np.int32, device="npu")
        
        kernel(A, B, C)
        C_out += C.numpy()

    if np.array_equal(C_out, C_ref):
        print(f"  Iter {i}: PASS")
    else:
        print(f"  Iter {i}: FAIL")
        diff = np.abs(C_out - C_ref)
        print(f"    Max diff: {np.max(diff)}")
        ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
        print(f"    Ratio[:8]={ratio.flatten()[:8]}")
        all_pass = False

print(f"\n=== SUMMARY ===")
if all_pass:
    print("ALL TESTS PASSED!")
    sys.exit(0)
else:
    print("SOME TESTS FAILED!")
    sys.exit(1)
