#!/usr/bin/env python3
"""Phase 40M: Hardware benchmark for Q8SCALEV-C4 mmul v2 4x4 kernel."""
import argparse
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from gen_q8_c4_mmul_v2_v2 import (
    build, pack_A, pack_B, ref_q8_gemm, simulate_kernel_cpu,
    M_TILE, N_TILE, KC, NBLK
)

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config
    HAVE_AIE = True
except ImportError:
    HAVE_AIE = False


def run_kernel_xrt(xclbin_path, A_buf, B_buf, M, K, N):
    """Run kernel on NPU hardware via pyxrt."""
    import pyxrt

    xclbin = pyxrt.xclbin(xclbin_path)
    uuid = xclbin.get_uuid()
    device = pyxrt.device(0)
    device.register_xclbin(xclbin)
    hwctx = pyxrt.hw_context(device, uuid)
    kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
    gid1 = kernel.group_id(1)
    gid3 = kernel.group_id(3)
    inst_bytes = os.path.getsize(xclbin_path.replace(".xclbin", ".insts.bin"))
    with open(xclbin_path.replace(".xclbin", ".insts.bin"), "rb") as f:
        insts = np.frombuffer(f.read(), dtype=np.uint32)
    inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
    inst_map = np.frombuffer(inst_bo.map(), dtype=np.uint32)
    inst_map[:] = insts
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    a_bo = pyxrt.bo(device, A_buf.nbytes, pyxrt.bo.flags.host_only, gid3)
    b_bo = pyxrt.bo(device, B_buf.nbytes, pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(device, M * N * 4, pyxrt.bo.flags.host_only, gid3)
    c_map = np.frombuffer(c_bo.map(), dtype=np.float32)
    c_map[:] = 0
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    a_bo.write(np.ascontiguousarray(A_buf).tobytes(), 0)
    b_bo.write(np.ascontiguousarray(B_buf).tobytes(), 0)
    a_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    b_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run = pyxrt.run(kernel)
    run.set_arg(0, np.uint64(3))
    run.set_arg(1, inst_bo)
    run.set_arg(2, np.uint32(inst_bytes))
    run.set_arg(3, a_bo)
    run.set_arg(4, b_bo)
    run.set_arg(5, c_bo)
    t0 = time.perf_counter()
    run.start()
    run.wait()
    t1 = time.perf_counter()
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    c_raw = np.frombuffer(c_bo.map(), dtype=np.float32)
    C = c_raw.reshape(M, N).copy()
    return C, (t1 - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="geometric",
                    choices=["uniform", "two_scale", "geometric", "realistic"])
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_rows, n_cols = args.rows, args.cols
    nb = K // 32

    print(f"Phase 40M v2 4x4: Q8SCALEV-C4 mmul hardware benchmark")
    print(f"  M={M} K={K} N={N} rows={n_rows} cols={n_cols} iters={args.iters}")
    print()

    # Generate test data
    rng = np.random.default_rng(42)
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)

    if args.pattern == "uniform":
        As = np.ones((M, nb), dtype=np.float32)
        Bs = np.ones((nb, N), dtype=np.float32)
    elif args.pattern == "two_scale":
        As = np.ones((M, nb), dtype=np.float32)
        Bs = np.ones((nb, N), dtype=np.float32)
        As[:, nb//2:] = 2.0
        Bs[nb//2:, :] = 2.0
    elif args.pattern == "geometric":
        factors = np.array([min(2.0 ** i, 65504.0) for i in range(nb)], dtype=np.float32)
        As = np.ones((M, nb), dtype=np.float32) * factors[np.newaxis, :]
        Bs = np.ones((nb, N), dtype=np.float32) * factors[:, np.newaxis]
    elif args.pattern == "realistic":
        As = rng.uniform(0.001, 0.02, size=(M, nb)).astype(np.float32)
        Bs = rng.uniform(0.001, 0.02, size=(nb, N)).astype(np.float32)

    C_ref = ref_q8_gemm(A_i8, As, B_i8, Bs)
    print(f"  Reference computed: C[0,0]={C_ref[0,0]:.4f}")

    # Pack data
    Ap = pack_A(A_i8, As, M, K, M_TILE, KC)
    Bp = pack_B(B_i8, Bs, K, N, N_TILE, KC)

    # CPU simulation for sanity check
    C_cpu = simulate_kernel_cpu(Ap, Bp, M, K, N, M_TILE, N_TILE, KC)
    diff_cpu = np.abs(C_cpu.astype(np.float64) - C_ref.astype(np.float64))
    bad_cpu = int(np.sum(diff_cpu > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))
    print(f"  CPU simulation: max_err={float(np.max(diff_cpu)):.4e} bad={bad_cpu}/{M*N}")
    if bad_cpu > 0:
        print("  WARNING: CPU simulation mismatch!")
        return

    # Compile or load cached kernel
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    if HAVE_AIE and not (os.path.exists(xp) and os.path.exists(ip)):
        print("  Compiling kernel...")
        os.makedirs(args.work_dir, exist_ok=True)

        @iron.jit
        def design(A: In, B: In, C: Out, *,
                   M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
                   rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
                   dummy: CompileTime[int] = 0):
            return build(M, K, N, n_cols, n_rows=n_rows)

        t0 = time.time()
        design.specialize(M=M, K=K, N=N, rows=n_rows, cols=n_cols, dummy=0).compile(
            xclbin_path=xp, inst_path=ip, pdi_path=pp)
        print(f"  Compiled in {time.time() - t0:.1f}s")
    elif not os.path.exists(xp) or not os.path.exists(ip):
        print(f"  ERROR: kernel not found at {xp}")
        return

    # Run hardware benchmark
    print(f"  Running hardware benchmark ({args.iters} iterations)...")
    times = []
    for it in range(args.iters):
        try:
            C_dev, t = run_kernel_xrt(xp, Ap, Bp, M, K, N)
            times.append(t)
            diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
            bad = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))
            print(f"    iter {it+1}: {t:.2f} ms  bad={bad}/{M*N}")
        except Exception as e:
            print(f"    iter {it+1} FAILED: {e}")
            break

    if times:
        dev_ms = np.median(times)
        gops = M * K * N / (dev_ms / 1000.0) / 1e9
        print(f"\n  Median device time: {dev_ms:.2f} ms")
        print(f"  GOPS: {gops:.1f}")
        print(f"  Pattern: {args.pattern}")


if __name__ == "__main__":
    main()
