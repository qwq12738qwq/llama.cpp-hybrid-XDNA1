#!/usr/bin/env python3
"""
Stage 2: Microkernel tests to isolate arithmetic vs ABI issues.
Tests increasingly simple operations:
A: C[i] = A[i]
B: C[i] = A[i] + B[i]  
C: C[i] = int32(A[i]) * int32(B[i])
D: C[i] = sum(A[i] * B[i])
E: Full C[n] = sum_k A[k] * B[k,n]
"""
import os, sys
import numpy as np
sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')
from aie.utils.npukernel import NPUKernel
import aie.iron as iron

# Use the working xclbin
XCLBIN = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_test_new/design.xclbin"
INSTS = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_test_new/design.insts.bin"

if not os.path.exists(XCLBIN):
    print(f"XCLBIN not found: {XCLBIN}")
    sys.exit(1)

kernel = NPUKernel(xclbin_path=XCLBIN, insts_path=INSTS, kernel_name="MLIR_AIE")
print(f"Loaded kernel from {XCLBIN}")

M, K, N = 1, 32, 16

def run_test(name, A_np, B_np, C_ref, desc=""):
    A = iron.tensor(np.array(A_np, dtype=np.int8), dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")
    kernel(A, B, C)
    C_out = C.numpy()
    
    diff = np.abs(C_out - C_ref)
    max_diff = np.max(diff)
    mismatches = np.sum(diff > 0)
    
    if mismatches == 0:
        print(f"  {name}: PASS {desc}")
    else:
        print(f"  {name}: FAIL - {mismatches} mismatches, max_diff={max_diff} {desc}")
        print(f"    C_ref[:8]={C_ref.flatten()[:8]}")
        print(f"    C_out[:8]={C_out.flatten()[:8]}")
        ratio = np.where(C_ref != 0, C_out.astype(float) / C_ref.astype(float), 0)
        print(f"    Ratio[:8]={ratio.flatten()[:8]}")
    return mismatches == 0

print("\n=== Stage 2 Microkernel Tests ===\n")

# Test A: C[i] = A[i] - Not directly testable with matmul, skip
# Test B: C[i] = A[i] + B[i] - Not directly testable with matmul, skip

# Test C: Element-wise multiply C[i] = A[i] * B[i] for single element
# We can test this by using K=1, but our kernel is fixed at K=32
# Instead, we test patterns that isolate multiplication

print("Test C: Single element multiply (K=1 equivalent)")
# Use A with only first element non-zero, B with only first element non-zero
A_np = np.zeros((M, K), dtype=np.int8)
B_np = np.zeros((K, N), dtype=np.int8)
A_np[0, 0] = 5
B_np[0, 0] = 7
# Expected: C[0,0] = 5*7 = 35, all others 0
C_ref = np.zeros((M, N), dtype=np.int32)
C_ref[0, 0] = 35
run_test("Single multiply", A_np, B_np, C_ref, "(5*7=35)")

# Test with negative values
A_np = np.zeros((M, K), dtype=np.int8)
B_np = np.zeros((K, N), dtype=np.int8)
A_np[0, 0] = -5
B_np[0, 0] = 7
C_ref = np.zeros((M, N), dtype=np.int32)
C_ref[0, 0] = -35
run_test("Single multiply (neg)", A_np, B_np, C_ref, "(-5*7=-35)")

# Test D: Sum of products (accumulation)
print("\nTest D: Accumulation over K")
# A = [1,1,1,...], B = [1,1,1,...] -> each C[n] = sum(1*1) = K = 32
A_np = np.ones((M, K), dtype=np.int8)
B_np = np.ones((K, N), dtype=np.int8)
C_ref = np.full((M, N), K, dtype=np.int32)
run_test("Accumulation K=32", A_np, B_np, C_ref, "(sum 1*1 = 32)")

# Accumulation with varying values
A_np = np.arange(K, dtype=np.int8).reshape(1, K)  # 0,1,2,...,31
B_np = np.ones((K, N), dtype=np.int8)
C_ref = np.full((M, N), sum(range(K)), dtype=np.int32)  # 0+1+...+31 = 496
run_test("Accumulation 0..31", A_np, B_np, C_ref, "(sum 0..31 = 496)")

# Test with negative accumulation
A_np = np.full((M, K), -1, dtype=np.int8)
B_np = np.ones((K, N), dtype=np.int8)
C_ref = np.full((M, N), -K, dtype=np.int32)
run_test("Accumulation (-1)*1", A_np, B_np, C_ref, "(sum -1*1 = -32)")

# Test E: Full matmul (already tested but repeat for completeness)
print("\nTest E: Full M=1 K=32 N=16 matmul")
# Random test
np.random.seed(42)
A_np = np.random.randint(-128, 127, size=(M, K), dtype=np.int8)
B_np = np.random.randint(-128, 127, size=(K, N), dtype=np.int8)
C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
run_test("Random matmul", A_np, B_np, C_ref)

# Edge case: max int8 values
A_np = np.full((M, K), 127, dtype=np.int8)
B_np = np.full((K, N), 127, dtype=np.int8)
C_ref = np.full((M, N), K * 127 * 127, dtype=np.int32)
run_test("Max int8 values", A_np, B_np, C_ref, f"(32*127*127={32*127*127})")

# Edge case: min int8 values
A_np = np.full((M, K), -128, dtype=np.int8)
B_np = np.full((K, N), -128, dtype=np.int8)
C_ref = np.full((M, N), K * (-128) * (-128), dtype=np.int32)
run_test("Min int8 values", A_np, B_np, C_ref, f"(32*128*128={32*128*128})")

# Mixed signs
A_np = np.full((M, K), -128, dtype=np.int8)
B_np = np.full((K, N), 127, dtype=np.int8)
C_ref = np.full((M, N), K * (-128) * 127, dtype=np.int32)
run_test("Mixed signs", A_np, B_np, C_ref, f"(32*-128*127={32*(-128)*127})")

print("\n=== Stage 2 Complete ===")