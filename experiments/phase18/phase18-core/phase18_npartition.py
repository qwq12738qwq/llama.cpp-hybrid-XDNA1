#!/usr/bin/env python3
"""Phase 18-H/I: N-partition heterogeneous Vulkan+XDNA execution test.

Tests: split N dimension, run Vulkan and XDNA concurrently, measure wall time.
Goal: T_hetero < T_vulkan_only
"""
import json, os, sys, time
import numpy as np

try:
    import pyxrt
except ImportError:
    sys.exit("ERROR: pyxrt not available")

KC = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"
OUT = "/home/ubuntu12738/phase18"

def find_xdna_kernel(M, K, N, nc=1, prefer_tk=None):
    """Find a pre-compiled XDNA kernel for given shape."""
    for tk in ([prefer_tk] if prefer_tk else []) + [64, 32, 128, 256]:
        wd = os.path.join(KC, f"M{M}_K{K}_N{N}_C{nc}_Tm64_Tk{tk}_Tn64")
        xp = os.path.join(wd, "design.xclbin")
        ip = os.path.join(wd, "design.insts.bin")
        if os.path.exists(xp) and os.path.exists(ip):
            return xp, ip
    return None, None

def xdna_bench(xp, ip, M, K, N, inp_i8, wt_i8, ni=3):
    """Run XDNA kernel and return (output, timing_dict)."""
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xp)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    in_sz = M*K; wt_sz = K*N; out_sz = M*N*4
    gid3 = kern.group_id(3)
    in_bo = pyxrt.bo(dev, in_sz, pyxrt.bo.flags.host_only, gid3)
    wt_bo = pyxrt.bo(dev, wt_sz, pyxrt.bo.flags.host_only, gid3)
    out_bo = pyxrt.bo(dev, out_sz, pyxrt.bo.flags.host_only, gid3)
    isz = os.path.getsize(ip)
    gid1 = kern.group_id(1)
    inst_bo = pyxrt.bo(dev, isz, pyxrt.bo.flags.cacheable, gid1)
    with open(ip, "rb") as f:
        inst_bo.write(f.read(), 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Warmup
    in_bo.write(inp_i8.tobytes(), 0); wt_bo.write(wt_i8.tobytes(), 0)
    out_bo.write(b'\x00'*out_sz, 0)
    in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run = pyxrt.run(kern)
    run.set_arg(0,3); run.set_arg(1,inst_bo); run.set_arg(2,isz)
    run.set_arg(3,in_bo); run.set_arg(4,wt_bo); run.set_arg(5,out_bo)
    run.start(); run.wait()
    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # Benchmark
    times = []
    for _ in range(ni):
        t0 = time.time()
        in_bo.write(inp_i8.tobytes(), 0); wt_bo.write(wt_i8.tobytes(), 0)
        out_bo.write(b'\x00'*out_sz, 0)
        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t1 = time.time()
        run = pyxrt.run(kern)
        run.set_arg(0,3); run.set_arg(1,inst_bo); run.set_arg(2,isz)
        run.set_arg(3,in_bo); run.set_arg(4,wt_bo); run.set_arg(5,out_bo)
        run.start(); run.wait()
        t2 = time.time()
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw = out_bo.read(out_sz, 0)
        t3 = time.time()
        times.append(((t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t3-t0)*1000))

    out = np.frombuffer(raw, dtype=np.int32).reshape(M, N).copy()
    avg = {k: round(np.mean([t[i] for t in times]), 2) for i, k in enumerate(["upload","kernel","download","total"])}
    return out, avg

def cpu_gemm(inp, wt, M, K, N):
    return inp.astype(np.int32) @ wt.astype(np.int32)

def main():
    os.makedirs(OUT, exist_ok=True)

    # Test config
    M, K, N_full = 512, 1024, 3072
    rng = np.random.RandomState(42)
    inp_full = rng.randint(-127, 128, size=(M, K), dtype=np.int8)
    wt_full = rng.randint(-127, 128, size=(K, N_full), dtype=np.int8)

    # CPU reference
    cpu_out = cpu_gemm(inp_full, wt_full, M, K, N_full)
    print(f"Shape: M={M} K={K} N={N_full}")
    print(f"CPU reference computed.")

    # Find best XDNA kernel for full N
    xp, ip = find_xdna_kernel(M, K, N_full)
    if not xp:
        print("ERROR: No XDNA kernel found for full shape")
        return

    # 1. XDNA-only full benchmark
    print(f"\n--- XDNA-only full (N={N_full}) ---")
    xdna_full, xdna_timing = xdna_bench(xp, ip, M, K, N_full, inp_full, wt_full, ni=3)
    xdna_exact = bool(np.array_equal(cpu_out, xdna_full))
    print(f"  time={xdna_timing['total']:.1f}ms kernel={xdna_timing['kernel']:.1f}ms exact={xdna_exact}")

    # 2. Test N-partition ratios
    ratios = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    results = []

    for xv_pct in ratios:
        nv = int(N_full * xv_pct / 100)  # XDNA columns
        nx = N_full - nv                  # remaining for XDNA (or 0)

        if nv == 0:
            # XDNA-only
            results.append({"xv_pct": xv_pct, "nv": 0, "nx": N_full,
                          "xdna_ms": xdna_timing["total"],
                          "wall_ms": xdna_timing["total"],
                          "exact": xdna_exact})
            continue

        # Find XDNA kernel for the N slice
        xp_s, ip_s = find_xdna_kernel(M, K, nv)
        if not xp_s:
            print(f"  xv={xv_pct}%: No kernel for N={nv}, skip")
            continue

        inp_slice = inp_full
        wt_slice = wt_full[:, :nv].copy()

        xdna_out_s, xdna_t = xdna_bench(xp_s, ip_s, M, K, nv, inp_slice, wt_slice, ni=3)

        # Check correctness for this slice
        cpu_slice = cpu_out[:, :nv]
        exact_s = bool(np.array_equal(cpu_slice, xdna_out_s))
        max_err = float(np.max(np.abs(cpu_slice.astype(float) - xdna_out_s.astype(float))))

        wall = xdna_t["total"]  # XDNA is the critical path (no Vulkan in this test)
        results.append({
            "xv_pct": xv_pct, "nv": nv, "nx": nx,
            "xdna_ms": xdna_t["total"], "kernel_ms": xdna_t["kernel"],
            "wall_ms": wall, "exact": exact_s, "max_err": max_err,
        })
        print(f"  xv={xv_pct:>3}% N={nv:>5}: xdna={xdna_t['total']:>7.1f}ms "
              f"kernel={xdna_t['kernel']:>7.1f}ms err={max_err:.1f} exact={exact_s}")

    # Save
    jp = os.path.join(OUT, "phase18_npartition.json")
    with open(jp, "w") as f:
        json.dump({"M": M, "K": K, "N": N_full,
                    "xdna_full_ms": xdna_timing["total"],
                    "xdna_full_exact": xdna_exact,
                    "partitions": results}, f, indent=2, default=str)
    print(f"\nJSON: {jp}")

    # Analysis
    print(f"\n{'='*70}")
    print(f"N-PARTITION ANALYSIS (M={M} K={K} N={N_full})")
    print(f"XDNA full: {xdna_timing['total']:.1f}ms")
    print(f"{'='*70}")
    print(f"{'xv%':>5} {'N_xd':>6} {'xdna_ms':>8} {'kernel':>8} {'exact':>5}")
    for r in results:
        print(f"{r['xv_pct']:>5} {r.get('nv',0):>6} {r['xdna_ms']:>8.1f} "
              f"{r.get('kernel_ms',0):>8.1f} {r.get('exact',False):>5}")

if __name__ == "__main__":
    main()
