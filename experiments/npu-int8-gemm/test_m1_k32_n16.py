#!/usr/bin/env python3
"""Phase 9-H Step 1: Reproduce M=1 K=32 N=16 6x failure."""
import os, sys, time
import numpy as np
sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')
from aie.utils.npukernel import NPUKernel

XCLBIN = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_m1_k32_n16/design.xclbin"
INSTS = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_m1_k32_n16/design.insts.bin"
M, K, N = 1, 32, 16
NUM_ITERATIONS = 100

# Check if xclbin exists
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

mismatches = 0
latencies = []

# Test pattern 1: all ones
print("\n=== Pattern 1: All ones ===")
for i in range(10):
    A_np = np.ones((1, 32), dtype=np.int8)
    B_np = np.ones((32, 16), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
    
    import aie.iron as iron
    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((1, 16), dtype=np.int32, device="npu")
    
    start = time.perf_counter()
    kernel(A, B, C)
    elapsed = (time.perf_counter() - start) * 1e6
    
    C_out = C.numpy()
    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    
    if np.any(diff > 0):
        print(f"  Iter {i}: MISMATCH! max_diff={max_diff}")
        print(f"  C_ref={C_ref.flatten()[:8]}")
        print(f"  C_out={C_out.flatten()[:8]}")
        print(f"  Ratio={C_out.flatten()[:8] / C_ref.flatten()[:8]}")
        mismatches += 1
    
    if i % 5 == 0:
        print(f"  Iter {i}: latency={elapsed:.2f} us, max_diff={max_diff}")
    
    latencies.append(elapsed)

# Test pattern 2: A[k] = k, B[k,n] = 1
print("\n=== Pattern 2: A[k]=k, B=1 ===")
for i in range(10):
    A_np = np.arange(32, dtype=np.int8).reshape(1, 32)
    B_np = np.ones((32, 16), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
    
    import aie.iron as iron
    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((1, 16), dtype=np.int32, device="npu")
    
    start = time.perf_counter()
    kernel(A, B, C)
    elapsed = (time.perf_counter() - start) * 1e6
    
    C_out = C.numpy()
    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    
    if np.any(diff > 0):
        print(f"  Iter {i}: MISMATCH! max_diff={max_diff}")
        print(f"  C_ref={C_ref.flatten()[:8]}")
        print(f"  C_out={C_out.flatten()[:8]}")
        print(f"  Ratio={C_out.flatten()[:8] / C_ref.flatten()[:8]}")
        mismatches += 1

# Test pattern 3: A=1, B[k,n] = k*N+n
print("\n=== Pattern 3: A=1, B[k,n]=k*N+n ===")
for i in range(10):
    A_np = np.ones((1, 32), dtype=np.int8)
    B_np = np.arange(32*16, dtype=np.int8).reshape(32, 16)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
    
    import aie.iron as iron
    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((1, 16), dtype=np.int32, device="npu")
    
    start = time.perf_counter()
    kernel(A, B, C)
    elapsed = (time.perf_counter() - start) * 1e6
    
    C_out = C.numpy()
    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    
    if np.any(diff > 0):
        print(f"  Iter {i}: MISMATCH! max_diff={max_diff}")
        print(f"  C_ref={C_ref.flatten()[:8]}")
        print(f"  C_out={C_out.flatten()[:8]}")
        print(f"  Ratio={C_out.flatten()[:8] / C_ref.flatten()[:8]}")
        mismatches += 1

# Test pattern 4: deterministic random
print("\n=== Pattern 4: Deterministic random ===")
for i in range(20):
    A_np = np.random.default_rng(i + 42).integers(-128, 127, size=(1, 32), dtype=np.int8)
    B_np = np.random.default_rng(i + 100).integers(-128, 127, size=(32, 16), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
    
    import aie.iron as iron
    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((1, 16), dtype=np.int32, device="npu")
    
    start = time.perf_counter()
    kernel(A, B, C)
    elapsed = (time.perf_counter() - start) * 1e6
    
    C_out = C.numpy()
    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    
    if np.any(diff > 0):
        print(f"  Iter {i}: MISMATCH! max_diff={max_diff}")
        print(f"  C_ref={C_ref.flatten()[:8]}")
        print(f"  C_out={C_out.flatten()[:8]}")
        ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
        print(f"  Ratio={ratio.flatten()[:8]}")
        mismatches += 1

# Test pattern 5: positive-only
print("\n=== Pattern 5: Positive-only [0,127] ===")
for i in range(10):
    A_np = np.random.default_rng(i + 200).integers(0, 127, size=(1, 32), dtype=np.int8)
    B_np = np.random.default_rng(i + 300).integers(0, 127, size=(32, 16), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
    
    import aie.iron as iron
    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((1, 16), dtype=np.int32, device="npu")
    
    start = time.perf_counter()
    kernel(A, B, C)
    elapsed = (time.perf_counter() - start) * 1e6
    
    C_out = C.numpy()
    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    
    if np.any(diff > 0):
        print(f"  Iter {i}: MISMATCH! max_diff={max_diff}")
        print(f"  C_ref={C_ref.flatten()[:8]}")
        print(f"  C_out={C_out.flatten()[:8]}")
        ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
        print(f"  Ratio={ratio.flatten()[:8]}")
        mismatches += 1

# Test pattern 6: negative-only
print("\n=== Pattern 6: Negative-only [-128,-1] ===")
for i in range(10):
    A_np = np.random.default_rng(i + 400).integers(-128, -1, size=(1, 32), dtype=np.int8)
    B_np = np.random.default_rng(i + 500).integers(-128, -1, size=(32, 16), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
    
    import aie.iron as iron
    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((1, 16), dtype=np.int32, device="npu")
    
    start = time.perf_counter()
    kernel(A, B, C)
    elapsed = (time.perf_counter() - start) * 1e6
    
    C_out = C.numpy()
    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    
    if np.any(diff > 0):
        print(f"  Iter {i}: MISMATCH! max_diff={max_diff}")
        print(f"  C_ref={C_ref.flatten()[:8]}")
        print(f"  C_out={C_out.flatten()[:8]}")
        ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
        print(f"  Ratio={ratio.flatten()[:8]}")
        mismatches += 1

# Test pattern 7: mixed-sign
print("\n=== Pattern 7: Mixed-sign [-128,127] ===")
for i in range(10):
    A_np = np.random.default_rng(i + 600).integers(-128, 127, size=(1, 32), dtype=np.int8)
    B_np = np.random.default_rng(i + 700).integers(-128, 127, size=(32, 16), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
    
    import aie.iron as iron
    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((1, 16), dtype=np.int32, device="npu")
    
    start = time.perf_counter()
    kernel(A, B, C)
    elapsed = (time.perf_counter() - start) * 1e6
    
    C_out = C.numpy()
    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    
    if np.any(diff > 0):
        print(f"  Iter {i}: MISMATCH! max_diff={max_diff}")
        print(f"  C_ref={C_ref.flatten()[:8]}")
        print(f"  C_out={C_out.flatten()[:8]}")
        ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
        print(f"  Ratio={ratio.flatten()[:8]}")
        mismatches += 1

# Test pattern 8: small values [-2,2]
print("\n=== Pattern 8: Small values [-2,2] ===")
for i in range(10):
    A_np = np.random.default_rng(i + 800).integers(-2, 3, size=(1, 32), dtype=np.int8)
    B_np = np.random.default_rng(i + 900).integers(-2, 3, size=(32, 16), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
    
    import aie.iron as iron
    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((1, 16), dtype=np.int32, device="npu")
    
    start = time.perf_counter()
    kernel(A, B, C)
    elapsed = (time.perf_counter() - start) * 1e6
    
    C_out = C.numpy()
    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    
    if np.any(diff > 0):
        print(f"  Iter {i}: MISMATCH! max_diff={max_diff}")
        print(f"  C_ref={C_ref.flatten()[:8]}")
        print(f"  C_out={C_out.flatten()[:8]}")
        ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
        print(f"  Ratio={ratio.flatten()[:8]}")
        mismatches += 1

print(f"\n=== SUMMARY ===")
print(f"Total mismatches: {mismatches}")
if latencies:
    print(f"Latency: mean={np.mean(latencies):.2f} us, median={np.median(latencies):.2f} us")

if mismatches == 0:
    print("PASS: Zero numerical mismatches!")
    sys.exit(0)
else:
    print("FAIL: Numerical mismatches detected!")
    sys.exit(1)
