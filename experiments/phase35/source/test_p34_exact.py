#!/usr/bin/env python3
"""Exact copy of Phase 34 bench_scaling.py pattern."""
import sys, os, time, statistics, numpy as np
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase34/source")
import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.device import from_name
from gen_phase21 import build_design

M, K, N = 512, 2048, 2048
n_rows, n_cols = 1, 4
iron.set_current_device(from_name("npu", n_cols=4))

def main():
    @iron.jit
    def design(A: In, B: In, C: Out, *,
               M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
               mt: CompileTime[int] = 64, kt: CompileTime[int] = 64, nt: CompileTime[int] = 64,
               rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
               din: CompileTime[str] = "i8", dout: CompileTime[str] = "i32", act: CompileTime[str] = "none"):
        return build_design(M, K, N, 64, 64, 64, rows, cols, np.int8, np.int32, "none")

    work_dir = "/tmp/p34_exact"
    os.makedirs(work_dir, exist_ok=True)
    xp = os.path.join(work_dir, "design.xclbin")
    ip = os.path.join(work_dir, "design.insts.bin")

    if not os.path.exists(xp):
        print("ERROR: xclbin not found")
        sys.exit(1)

    compiled = design.specialize(
        M=M, K=K, N=N, mt=64, kt=64, nt=64,
        rows=n_rows, cols=n_cols, din="i8", dout="i32", act="none",
    ).compile(xclbin_path=xp, inst_path=ip)
    print(f"Loaded: {compiled[0]}")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)
    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    ref = A_np.astype(np.int32) @ B_np.astype(np.int32)

    host_times = []
    npu_times = []
    for i in range(5):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        t0 = time.perf_counter()
        ret = design(A, B, C, M=M, K=K, N=N)
        e2e_us = (time.perf_counter() - t0) * 1e6
        host_times.append(e2e_us)

        npu_ns = None
        if isinstance(ret, tuple):
            for item in ret:
                if hasattr(item, "npu_time"):
                    npu_ns = item.npu_time
                    break
        elif hasattr(ret, "npu_time"):
            npu_ns = ret.npu_time
        if npu_ns is not None:
            npu_times.append(npu_ns / 1000.0)

        got = C.numpy()
        mm = int(np.sum(got != ref))
        print(f"  iter {i}: host={e2e_us:.0f}us  npu={npu_times[-1] if npu_times else 'N/A':.0f}us  mm={mm}")

    host_times.sort()
    npu_times.sort()
    print(f"host median: {host_times[len(host_times)//2]:.0f}us")
    if npu_times:
        print(f"npu median: {npu_times[len(npu_times)//2]:.0f}us")

main()
