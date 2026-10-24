#!/usr/bin/env python3
"""Test using Phase 34's build_design directly."""
import sys, os, time, numpy as np
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase34/source")
import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.device import from_name
from gen_phase21 import build_design

M, K, N = 512, 2048, 2048
n_rows, n_cols = 1, 4
iron.set_current_device(from_name("npu", n_cols=4))

@iron.jit
def design(A: In, B: In, C: Out, *,
           M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
           mt: CompileTime[int] = 64, kt: CompileTime[int] = 64, nt: CompileTime[int] = 64,
           rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
           din: CompileTime[str] = "i8", dout: CompileTime[str] = "i32", act: CompileTime[str] = "none"):
    return build_design(M, K, N, 64, 64, 64, rows, cols, np.int8, np.int32, "none")

os.makedirs("/tmp/p34_direct", exist_ok=True)
xp = "/tmp/p34_direct/design.xclbin"
ip = "/tmp/p34_direct/design.insts.bin"
compiled = design.specialize(M=M, K=K, N=N, mt=64, kt=64, nt=64,
                             rows=n_rows, cols=n_cols, din="i8", dout="i32", act="none").compile(xclbin_path=xp, inst_path=ip)
print("compiled:", compiled[0])

rng = np.random.default_rng(42)
A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)
A = iron.tensor(A_np, dtype=np.int8, device="npu")
B = iron.tensor(B_np, dtype=np.int8, device="npu")

for i in range(3):
    C = iron.tensor((M, N), dtype=np.int32, device="npu")
    t0 = time.perf_counter()
    ret = design(A, B, C, M=M, K=K, N=N)
    us = (time.perf_counter() - t0) * 1e6
    print(f"iter {i}: {us:.0f} us")
    npu_ns = getattr(ret, "npu_time", None)
    if npu_ns:
        print(f"  npu: {npu_ns/1000:.0f} us")
    got = C.numpy()
    ref = A_np.astype(np.int32) @ B_np.astype(np.int32)
    mm = int(np.sum(got != ref))
    print(f"  mismatches: {mm}")
