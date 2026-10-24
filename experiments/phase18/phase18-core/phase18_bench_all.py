#!/usr/bin/env python3
"""Phase 18: Benchmark ALL pre-compiled kernels + compile missing key configs."""
import json, os, sys, time, subprocess, csv
import numpy as np

try:
    import pyxrt
except ImportError:
    sys.exit("ERROR: pyxrt not available")

KC = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"
OUT = "/home/ubuntu12738/phase18"

def find_all_kernels():
    """Find all pre-compiled kernels in cache."""
    kernels = []
    for entry in os.listdir(KC):
        xp = os.path.join(KC, entry, "design.xclbin")
        ip = os.path.join(KC, entry, "design.insts.bin")
        meta = os.path.join(KC, entry, "kernel_meta.json")
        if os.path.exists(xp) and os.path.exists(ip):
            # Parse M, K, N, C from directory name
            kernels.append({"dir": entry, "xclbin": xp, "insts": ip, "meta": meta})
    return kernels

def parse_kernel_shape(dirname):
    """Parse M, K, N, C from directory name like M512_K1024_N3072_C1_Tm64_Tk64_Tn64"""
    import re
    m = re.match(r".*M(\d+)_K(\d+)_N(\d+)_C(\d+)", dirname)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    # Also handle validation_ and test_ prefixes
    m = re.match(r"(?:validation|test)_M(\d+)_K(\d+)_N(\d+)_C(\d+)", dirname)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None

def xrt_exec(xp, ip, M, K, N, inp_i8, wt_i8, ni=3):
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
    for _ in range(1):
        in_bo.write(inp_i8.tobytes(), 0)
        wt_bo.write(wt_i8.tobytes(), 0)
        out_bo.write(b'\x00'*out_sz, 0)
        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = pyxrt.run(kern)
        run.set_arg(0, 3); run.set_arg(1, inst_bo); run.set_arg(2, isz)
        run.set_arg(3, in_bo); run.set_arg(4, wt_bo); run.set_arg(5, out_bo)
        run.start(); run.wait()
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    times = []
    for _ in range(ni):
        t0 = time.time()
        in_bo.write(inp_i8.tobytes(), 0)
        wt_bo.write(wt_i8.tobytes(), 0)
        out_bo.write(b'\x00'*out_sz, 0)
        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t1 = time.time()
        run = pyxrt.run(kern)
        run.set_arg(0, 3); run.set_arg(1, inst_bo); run.set_arg(2, isz)
        run.set_arg(3, in_bo); run.set_arg(4, wt_bo); run.set_arg(5, out_bo)
        run.start()
        run.wait()
        t2 = time.time()
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw = out_bo.read(out_sz, 0)
        t3 = time.time()
        out = np.frombuffer(raw, dtype=np.int32).reshape(M, N).copy()
        times.append({"upload": (t1-t0)*1000, "kernel": (t2-t1)*1000, "download": (t3-t2)*1000, "total": (t3-t0)*1000})

    avg = {k: round(np.mean([t[k] for t in times]), 2) for k in times[0]}
    out_avg = np.mean([np.frombuffer(out_bo.read(out_sz, 0), dtype=np.int32).reshape(M, N) for _ in range(1)], axis=0)
    return out, avg

def cpu_gemm(inp, wt, M, K, N):
    return inp.astype(np.int32) @ wt.astype(np.int32)

def main():
    os.makedirs(OUT, exist_ok=True)
    kernels = find_all_kernels()
    print(f"Found {len(kernels)} pre-compiled kernels")

    results = []
    for i, k in enumerate(kernels):
        shape = parse_kernel_shape(k["dir"])
        if not shape:
            print(f"  [{i+1}] SKIP {k['dir']} (can't parse shape)")
            continue
        M, K, N, C = shape
        # Skip very small or huge configs
        if M > 1024 or K > 4096 or N > 8192:
            continue

        print(f"[{i+1}/{len(kernels)}] {k['dir']} M={M} K={K} N={N} C={C}", end="", flush=True)

        rng = np.random.RandomState(42)
        inp = rng.randint(-127, 128, size=(M, K), dtype=np.int8)
        wt = rng.randint(-127, 128, size=(K, N), dtype=np.int8)

        try:
            xdna_out, timing = xrt_exec(k["xclbin"], k["insts"], M, K, N, inp, wt, ni=3)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # CPU reference for correctness check
        cpu_out = cpu_gemm(inp, wt, M, K, N)
        max_err = float(np.max(np.abs(cpu_out.astype(float) - xdna_out.astype(float))))
        bit_exact = bool(np.array_equal(cpu_out, xdna_out))

        flops = 2.0 * M * K * N
        gops = round(flops / (timing["total"]/1000) / 1e9, 2) if timing["total"] > 0 else 0

        r = {
            "M": M, "K": K, "N": N, "C": C,
            "total_ms": timing["total"],
            "kernel_ms": timing["kernel"],
            "upload_ms": timing["upload"],
            "download_ms": timing["download"],
            "gops": gops,
            "max_err": max_err,
            "bit_exact": bit_exact,
            "dir": k["dir"],
        }
        results.append(r)
        ex = "EXACT" if bit_exact else f"err={max_err:.1f}"
        print(f"  {timing['total']:>7.1f}ms  kernel={timing['kernel']:>7.1f}ms  GOPS={gops:>6.2f}  {ex}")

    # Save results
    jp = os.path.join(OUT, "phase18_all_kernels.json")
    with open(jp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {jp}")

    # Summary table sorted by total time
    if results:
        results.sort(key=lambda r: r["total_ms"])
        print(f"\n{'='*90}")
        print(f"ALL KERNELS: {len(results)} tested, {sum(1 for r in results if r['bit_exact'])} bit-exact")
        print(f"{'='*90}")
        print(f"{'M':>5} {'K':>5} {'N':>5} {'C':>2} {'total':>8} {'kernel':>8} {'GOPS':>7} {'err':>8} {'exact':>5}")
        for r in results:
            print(f"{r['M']:>5} {r['K']:>5} {r['N']:>5} {r['C']:>2} "
                  f"{r['total_ms']:>8.1f} {r['kernel_ms']:>8.1f} "
                  f"{r['gops']:>7.2f} {r['max_err']:>8.1f} {r['bit_exact']:>5}")

        # K=1024 vs K=32 comparison
        k1024 = [r for r in results if r["K"] == 1024 and r["C"] == 1]
        k32 = [r for r in results if r["K"] == 32 and r["C"] == 1]
        if k1024 and k32:
            print(f"\n{'='*90}")
            print(f"K=1024 (single submit) vs K=32 (32 submits) comparison")
            print(f"{'='*90}")
            # Match by M, N
            for r1 in k1024:
                for r3 in k32:
                    if r1["M"] == r3["M"] and r1["N"] == r3["N"]:
                        speedup = r3["total_ms"] / r1["total_ms"] if r1["total_ms"] > 0 else 0
                        print(f"M={r1['M']:>4} N={r1['N']:>5}: K=1024={r1['total_ms']:>7.1f}ms  K=32={r3['total_ms']:>7.1f}ms  speedup={speedup:.1f}x")

if __name__ == "__main__":
    main()
