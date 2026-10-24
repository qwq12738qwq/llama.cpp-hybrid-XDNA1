#!/usr/bin/env python3
"""Phase 43: Hardware benchmark for down projection (K=3584, N=1024)."""
import argparse
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase40'))
from gen_q8_c4_mmul_v2 import (
    pack_A, pack_B, ref_q8_gemm, simulate_kernel_cpu,
    M_TILE, N_TILE, KC
)


def run_kernel_xrt(xclbin_path, A_buf, B_buf, M, K, N):
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
    ap.add_argument("--K", type=int, default=3584)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="geometric",
                    choices=["uniform", "two_scale", "geometric", "realistic"])
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    nb = K // 32
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
    Ap = pack_A(A_i8, As, M, K, M_TILE, KC)
    Bp = pack_B(B_i8, Bs, K, N, N_TILE, KC)

    xp = os.path.join(args.work_dir, "design.xclbin")
    if not os.path.exists(xp):
        print(f"ERROR: kernel not found at {xp}")
        sys.exit(1)

    print(f"[down bench] M={M} K={K} N={N} iters={args.iters}")
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


if __name__ == "__main__":
    main()
