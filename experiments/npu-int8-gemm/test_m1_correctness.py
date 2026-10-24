#!/usr/bin/env python3
"""Phase 9-F Step 3: Standalone XRT correctness test using pre-compiled xclbin."""
import os, sys, time
import numpy as np
sys.path.insert(0, '/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')
from aie.utils.npukernel import NPUKernel
import numpy as np

XCLBIN = "/workspace/llama.cpp/experiments/npu-int8-gemm/k1024_m1_n16/design.xclbin"
INSTS = "/workspace/llama.cpp/experiments/npu-int8-gemm/k1024_m1_n16/design.insts.bin"
M, K, N = 1, 1024, 16
NUM_ITERATIONS = 100

print("[TEST] Loading pre-compiled xclbin...")
kernel = NPUKernel(
    xclbin_path=XCLBIN,
    insts_path=INSTS,
    kernel_name="MLIR_AIE",
)

print("[TEST] Loaded kernel")

mismatches = 0
latencies = []

for i in range(100):
    A_np = np.random.default_rng(i + 42).integers(-128, 127, size=(1, 1024), dtype=np.int8)
    B_np = np.random.default_rng(i + 100).integers(-128, 127, size=(1024, 16), dtype=np.int8)
    C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
    
    # Create tensors using iron
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
        mismatches += 1
    
    if i % 20 == 0:
        print(f"  Iter {i}: latency={elapsed:.2f} us, max_diff={max_diff}")
    
    latencies.append(elapsed)

print(f"\nResults after 100 iterations:")
print(f"  Mismatches: {mismatches}")
print(f"  Latency: mean={np.mean(latencies):.2f} us, median={np.median(latencies):.2f} us, min={np.min(latencies):.2f}, max={np.max(latencies):.2f}")

if mismatches == 0:
    print("PASS: Zero numerical mismatches!")
    sys.exit(0)
else:
    print("FAIL: Numerical mismatches detected!")
    sys.exit(1)
