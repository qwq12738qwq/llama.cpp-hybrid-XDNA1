#!/usr/bin/env python3
"""Phase 43: Compile 3x4 mmul-v2 topology.

3x4 = 12 workers (3 rows x 4 cols).

Resource estimate: 46 ObjectFifos (likely exceeds MemTile budget).

Usage (inside npu-dev container):
  python3 experiments/phase43/compile_3x4.py --work-dir /tmp/q8c4mmul_3x4
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase40'))
from gen_q8_c4_mmul_v2_v4 import build

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out
    HAVE_AIE = True
except ImportError:
    HAVE_AIE = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--fifo-depth", type=int, default=1)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    if not HAVE_AIE:
        print("ERROR: aie.iron not available. Run inside npu-dev container.")
        sys.exit(1)

    M, K, N = args.M, args.K, args.N
    n_rows, n_cols = args.rows, args.cols

    print(f"[3x4 compile] M={M} K={K} N={N} rows={n_rows} cols={n_cols}")
    print(f"  Workers: {n_rows * n_cols}")
    print(f"  Fifo depth: {args.fifo_depth}")

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    iron.set_current_device(iron.device.from_name("npu", n_cols=n_cols))

    @iron.jit
    def design(A: In, B: In, C: Out, *,
               M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
               rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
               dummy: CompileTime[int] = 0):
        return build(M, K, N, cols, n_rows=rows, fifo_depth=args.fifo_depth)

    print("  Compiling...")
    t0 = time.time()
    try:
        compiled = design.specialize(
            M=M, K=K, N=N, rows=n_rows, cols=n_cols, dummy=0
        ).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        print(f"  Compiled in {time.time()-t0:.1f}s -> {compiled[0]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("SUCCESS")


if __name__ == "__main__":
    main()
