#!/usr/bin/env python3
"""Phase 21: correctness check for a fused (or plain) whole-array GEMM xclbin.

Runs the kernel once via pyxrt and compares C against the CPU reference;
with --relu it expects C = relu(A @ B).

Usage:
  python fused_check.py <xclbin> <insts> <M> <K> <N> [--relu]
"""
import argparse
import os
import time

import numpy as np
import pyxrt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xclbin")
    ap.add_argument("insts")
    ap.add_argument("M", type=int)
    ap.add_argument("K", type=int)
    ap.add_argument("N", type=int)
    ap.add_argument("--relu", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(7)
    A = rng.integers(-127, 128, (a.M, a.K), dtype=np.int8)
    B = rng.integers(-127, 128, (a.K, a.N), dtype=np.int8)

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(a.xclbin)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    gid3, gid1 = kern.group_id(3), kern.group_id(1)
    a_bo = pyxrt.bo(dev, A.nbytes, pyxrt.bo.flags.host_only, gid3)
    b_bo = pyxrt.bo(dev, B.nbytes, pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(dev, a.M * a.N * 4, pyxrt.bo.flags.host_only, gid3)
    inst = os.path.getsize(a.insts)
    i_bo = pyxrt.bo(dev, inst, pyxrt.bo.flags.cacheable, gid1)
    with open(a.insts, "rb") as f:
        i_bo.write(f.read(), 0)
    i_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    a_bo.write(A.tobytes(), 0)
    b_bo.write(B.tobytes(), 0)
    c_bo.write(b"\x00" * (a.M * a.N * 4), 0)
    for b in (a_bo, b_bo, c_bo):
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    r = pyxrt.run(kern)
    r.set_arg(0, 3)
    r.set_arg(1, i_bo)
    r.set_arg(2, inst)
    r.set_arg(3, a_bo)
    r.set_arg(4, b_bo)
    r.set_arg(5, c_bo)
    t0 = time.perf_counter()
    r.start()
    r.wait()
    ms = (time.perf_counter() - t0) * 1000.0
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    C = np.frombuffer(c_bo.read(a.M * a.N * 4, 0), np.int32).reshape(a.M, a.N)

    ref = (A.astype(np.int64) @ B.astype(np.int64))
    if a.relu:
        ref = np.maximum(ref, 0)
    ref = ref.astype(np.int32)
    bad = int((C != ref).sum())
    print(f"CHECK {'relu' if a.relu else 'plain'} M={a.M} K={a.K} N={a.N} "
          f"mismatches={bad}/{a.M*a.N} ms={ms:.3f} -> {'PASS' if bad == 0 else 'FAIL'}")
    if bad:
        idx = np.argwhere(C != ref)[:3]
        for i, j in idx:
            print(f"  [{i},{j}] got={C[i,j]} ref={ref[i,j]}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
