#!/usr/bin/env python3
"""Quick timing test for phase21-based multi-tile GEMM."""
import sys
import time
import numpy as np

sys.path.insert(0, "/workspace/llama.cpp/experiments/phase34/source")
import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.device import from_name
from gen_phase21 import build_design

M, K, N = 512, 2048, 2048
rows, cols = 4, 4
iron.set_current_device(from_name("npu", n_cols=4))

@iron.jit
def design(A: In, B: In, C: Out, *,
           M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
           mt: CompileTime[int] = 64, kt: CompileTime[int] = 64, nt: CompileTime[int] = 64,
           rows: CompileTime[int] = rows, cols: CompileTime[int] = cols,
           din: CompileTime[str] = "i8", dout: CompileTime[str] = "i32", act: CompileTime[str] = "none"):
    return build_design(M, K, N, 64, 64, 64, rows, cols, np.int8, np.int32, "none")

rng = np.random.default_rng(42)
A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)
A = iron.tensor(A_np, dtype=np.int8, device="npu")
B = iron.tensor(B_np, dtype=np.int8, device="npu")

# Warmup
C = iron.tensor((M, N), dtype=np.int32, device="npu")
design(A, B, C, M=M, K=K, N=N)

# Benchmark
host_times = []
npu_times = []
for i in range(10):
    C = iron.tensor((M, N), dtype=np.int32, device="npu")
    t0 = time.perf_counter()
    ret = design(A, B, C, M=M, K=K, N=N)
    t1 = time.perf_counter()
    host_times.append((t1 - t0) * 1e6)

    npu_ns = None
    if isinstance(ret, tuple):
        for item in ret:
            if hasattr(item, "npu_time"):
                npu_ns = item.npu_time
                break
    elif hasattr(ret, "npu_time"):
        npu_ns = ret.npu_time

    if npu_ns:
        npu_times.append(npu_ns / 1000.0)
        print(f"  iter {i}: host={host_times[-1]:.0f}us npu={npu_times[-1]:.0f}us")
    else:
        print(f"  iter {i}: host={host_times[-1]:.0f}us npu=N/A")

print(f"\nhost mean={np.mean(host_times):.0f}us median={np.median(host_times):.0f}us")
if npu_times:
    print(f"npu  mean={np.mean(npu_times):.0f}us median={np.median(npu_times):.0f}us")

got = C.numpy()
ref = A_np.astype(np.int32) @ B_np.astype(np.int32)
mm = int(np.sum(got != ref))
print(f"mismatches={mm}/{M*N}")
