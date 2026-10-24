#!/usr/bin/env python3
"""Phase 40M: Hardware benchmark for Q8SCALEV-C4 mmul v2 kernel.

Run inside npu-dev container:
  python experiments/phase40/bench_q8_c4_mmul_v2.py \
    --M 512 --K 1024 --N 3584 --iters 5 \
    --work-dir /tmp/q8c4mmul_hwtest
"""
import argparse
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from gen_q8_c4_mmul_v2 import (
    build, pack_A, pack_B, ref_q8_gemm, simulate_kernel_cpu,
    M_TILE, N_TILE, KC, NBLK
)

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config, DefaultNPURuntime, HostRuntime, NPUKernel, TraceConfig, tensor
    HAVE_AIE = True
except ImportError:
    HAVE_AIE = False


def run_kernel_xrt(xclbin_path, A_buf, B_buf, M, K, N):
    """Run kernel on NPU hardware via pyxrt."""
    import pyxrt
    import numpy as np

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


def run_kernel_trace(xclbin_path, insts_path, A_buf, B_buf, M, K, N, trace_size, trace_file, trace_json):
    from aie.utils import DefaultNPURuntime, HostRuntime, NPUKernel, TraceConfig, tensor

    output = np.zeros(M * N, dtype=np.float32)
    buffers = [
        tensor(np.ascontiguousarray(A_buf), dtype=A_buf.dtype),
        tensor(np.ascontiguousarray(B_buf), dtype=B_buf.dtype),
        tensor(output, dtype=output.dtype),
    ]
    tcfg = TraceConfig(
        trace_size=trace_size,
        trace_file=trace_file,
        reuse_output_buffer=True,
        last_tensor_shape=output.shape,
        last_tensor_dtype=output.dtype,
    )
    HostRuntime.prepare_args_for_trace(buffers, tcfg)
    kern = NPUKernel(xclbin_path, insts_path, trace_config=tcfg)
    runtime = DefaultNPURuntime.load(kern)
    t0 = time.perf_counter()
    DefaultNPURuntime.run(runtime, buffers)
    t1 = time.perf_counter()
    trace_buffer, _ = HostRuntime.extract_trace_from_args(buffers, tcfg)
    tcfg.write_trace(trace_buffer.view(np.uint32))
    if tcfg.physical_mlir_path:
        tcfg.trace_to_json(tcfg.physical_mlir_path, trace_json)
    result = np.asarray(buffers[-1])
    if result.size > M * N:
        result = result.view(np.float32)[:M * N]
    return result.reshape(M, N), (t1 - t0) * 1000.0


def run_kernel_pair_xrt(xclbin_path, A_buf, B_buf, M, K, N, mode, stagger_us, iters):
    import pyxrt
    import numpy as np
    import time

    inst_path = xclbin_path.replace(".xclbin", ".insts.bin")
    xclbin = pyxrt.xclbin(xclbin_path)
    uuid = xclbin.get_uuid()
    device = pyxrt.device(0)
    device.register_xclbin(xclbin)
    hwctx = pyxrt.hw_context(device, uuid)
    kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
    gid1 = kernel.group_id(1)
    gid3 = kernel.group_id(3)
    inst_bytes = os.path.getsize(inst_path)
    with open(inst_path, "rb") as f:
        insts = np.frombuffer(f.read(), dtype=np.uint32)
    inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
    inst_map = np.frombuffer(inst_bo.map(), dtype=np.uint32)
    inst_map[:] = insts
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    a_bo = []
    b_bo = []
    c_bo = []
    for _ in range(2):
        a = pyxrt.bo(device, A_buf.nbytes, pyxrt.bo.flags.host_only, gid3)
        b = pyxrt.bo(device, B_buf.nbytes, pyxrt.bo.flags.host_only, gid3)
        c = pyxrt.bo(device, M * N * 4, pyxrt.bo.flags.host_only, gid3)
        np.frombuffer(a.map(), dtype=np.uint8)[:] = np.frombuffer(np.ascontiguousarray(A_buf), dtype=np.uint8)
        np.frombuffer(b.map(), dtype=np.uint8)[:] = np.frombuffer(np.ascontiguousarray(B_buf), dtype=np.uint8)
        c_map = np.frombuffer(c.map(), dtype=np.float32)
        c_map[:] = 0
        a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        a_bo.append(a)
        b_bo.append(b)
        c_bo.append(c)

    def start(slot):
        run = pyxrt.run(kernel)
        run.set_arg(0, np.uint64(3))
        run.set_arg(1, inst_bo)
        run.set_arg(2, np.uint32(inst_bytes))
        run.set_arg(3, a_bo[slot])
        run.set_arg(4, b_bo[slot])
        run.set_arg(5, c_bo[slot])
        submit0 = time.perf_counter_ns()
        run.start()
        submit1 = time.perf_counter_ns()
        return run, submit0, submit1

    def wait(run, submit1):
        wait0 = time.perf_counter_ns()
        run.wait()
        wait1 = time.perf_counter_ns()
        c_bo[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        c_bo[1].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return wait0, wait1

    rows = []
    for _ in range(iters):
        wall0 = time.perf_counter_ns()
        rg, g_submit0, g_submit1 = start(0)
        if mode == "sequential":
            g_wait0, g_wait1 = wait(rg, g_submit1)
            ru, u_submit0, u_submit1 = start(1)
            u_wait0, u_wait1 = wait(ru, u_submit1)
        else:
            if mode == "staggered":
                time.sleep(stagger_us / 1e6)
            ru, u_submit0, u_submit1 = start(1)
            g_wait0, g_wait1 = wait(rg, g_submit1)
            u_wait0, u_wait1 = wait(ru, u_submit1)
        wall1 = time.perf_counter_ns()
        rows.append((
            (wall1 - wall0) / 1e6,
            (g_wait1 - g_wait0) / 1e6,
            (u_wait1 - u_wait0) / 1e6,
            (g_wait0 - g_submit1) / 1e6,
            (u_wait0 - u_submit1) / 1e6,
            (g_submit1 - g_submit0) / 1e6,
            (u_submit1 - u_submit0) / 1e6,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trace-size", type=int, default=0)
    ap.add_argument("--trace-col", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--fifo-depth", type=int, default=2, choices=[1, 2, 3, 4])
    ap.add_argument("--b-fifo-depth", type=int, default=2, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--c-fifo-depth", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--kernel-variant", choices=["original", "p32"], default="p32")
    ap.add_argument("--stack-size", type=lambda value: int(value, 0), default=0x800)
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="geometric",
                    choices=["uniform", "two_scale", "geometric", "realistic"])
    ap.add_argument("--pair", action="store_true")
    ap.add_argument("--pair-mode", type=str, default="sequential",
                    choices=["sequential", "both", "staggered"])
    ap.add_argument("--stagger-us", type=int, default=1000)
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cols = args.cols
    nb = K // 32

    print(f"Phase 40M: Q8SCALEV-C4 mmul v2 hardware benchmark")
    print(f"  M={M} K={K} N={N} cols={n_cols} iters={args.iters}")
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

    if HAVE_AIE and (args.trace_size > 0 or not (os.path.exists(xp) and os.path.exists(ip))):
        print("  Compiling kernel...")
        os.makedirs(args.work_dir, exist_ok=True)

        @iron.jit
        def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
            return build(M, K, N, n_cols, trace_size=args.trace_size, trace_col=args.trace_col, fifo_depth=args.fifo_depth, stack_size=args.stack_size, b_fifo_depth=(args.b_fifo_depth if args.b_fifo_depth else None), c_fifo_depth=args.c_fifo_depth, kernel_variant=args.kernel_variant)

        t0 = time.time()
        design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        print(f"  Compiled in {time.time() - t0:.1f}s")
    elif not os.path.exists(xp) or not os.path.exists(ip):
        print(f"  ERROR: kernel not found at {xp}")
        return

    if args.pair:
        try:
            rows = run_kernel_pair_xrt(xp, Ap, Bp, M, K, N,
                                       args.pair_mode, args.stagger_us, args.iters)
        except Exception as e:
            print(f"  pair {args.pair_mode} FAILED: {e}")
            return
        print(f"  Pair mode={args.pair_mode} stagger_us={args.stagger_us}")
        for i, row in enumerate(rows, 1):
            print("  iter=%d wall_ms=%.3f gate_wait_ms=%.3f up_wait_ms=%.3f "
                  "gate_host_wait_delay_ms=%.3f up_host_wait_delay_ms=%.3f "
                  "gate_submit_ms=%.3f up_submit_ms=%.3f" %
                  (i, row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
        med = np.median(np.asarray(rows), axis=0)
        print("  median wall_ms=%.3f gate_wait_ms=%.3f up_wait_ms=%.3f "
              "gate_host_wait_delay_ms=%.3f up_host_wait_delay_ms=%.3f" %
              tuple(med[:5]))
        return

    # Run hardware benchmark
    trace_file = os.path.join(args.work_dir, "trace.txt")
    trace_json = os.path.join(args.work_dir, "trace.json")
    if args.trace_size > 0:
        def run_once():
            return run_kernel_trace(xp, ip, Ap, Bp, M, K, N, args.trace_size, trace_file, trace_json)
    else:
        def run_once():
            return run_kernel_xrt(xp, Ap, Bp, M, K, N)

    print(f"  Running hardware benchmark (warmup={args.warmup}, measurements={args.iters})...")
    for it in range(args.warmup):
        try:
            run_once()
        except Exception as e:
            print(f"    warmup {it+1} FAILED: {e}")
            return
    times = []
    for it in range(args.iters):
        try:
            C_dev, t = run_once()
            times.append(t)
            diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
            max_err = float(np.max(diff))
            rmse = float(np.sqrt(np.mean(diff * diff)))
            bad = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))
            finite = bool(np.all(np.isfinite(C_dev)))
            print(f"    iter {it+1}: {t:.2f} ms  max_err={max_err:.6e} rmse={rmse:.6e} bad={bad}/{M*N} finite={finite}")
        except Exception as e:
            print(f"    iter {it+1} FAILED: {e}")
            break

    if times:
        dev_ms = float(np.median(times))
        gops = M * K * N / (dev_ms / 1000.0) / 1e9
        print(f"\n  Median device time: {dev_ms:.2f} ms")
        print(f"  p90: {np.percentile(times, 90):.2f} ms")
        print(f"  min: {min(times):.2f} ms  max: {max(times):.2f} ms")
        print(f"  GOPS: {gops:.1f}")
        print(f"  Pattern: {args.pattern}")


if __name__ == "__main__":
    main()
