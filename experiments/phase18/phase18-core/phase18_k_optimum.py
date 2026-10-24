#!/usr/bin/env python3
"""Phase 18-C: Test K optimum (1024, 2048, 4096) via force-compile."""
import json, os, sys, time, subprocess
import numpy as np

try:
    import pyxrt
except ImportError:
    sys.exit("ERROR: pyxrt not available")

KC = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"
OUT = "/home/ubuntu12738/phase18"
GEN = "/workspace/llama.cpp/experiments/npu-int8-gemm/gen_xdna_gemm.py"

def compile_kernel(M, K, N, nc, tm, tk, tn):
    wd = os.path.join(KC, f"M{M}_K{K}_N{N}_C{nc}_Tm{tm}_Tk{tk}_Tn{tn}")
    xp = os.path.join(wd, "design.xclbin")
    if os.path.exists(xp):
        return wd, xp, os.path.join(wd, "design.insts.bin"), 0
    cmd = [sys.executable, GEN, "--M", str(M), "--K", str(K), "--N", str(N),
           "--cols", str(nc), "--tile-m", str(tm), "--tile-k", str(tk),
           "--tile-n", str(tn), "--work-dir", wd]
    print(f"  Compiling M={M} K={K} N={N} C{nc} Tk={tk}...", end="", flush=True)
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        ct = time.time() - t0
        if res.returncode != 0:
            print(f" FAIL ({ct:.1f}s)")
            return None, None, None, 0
        ip = os.path.join(wd, "design.insts.bin")
        if not (os.path.exists(xp) and os.path.exists(ip)):
            print(f" NO OUTPUT ({ct:.1f}s)")
            return None, None, None, 0
        print(f" OK ({ct:.1f}s)")
        return wd, xp, ip, ct
    except subprocess.TimeoutExpired:
        print(f" TIMEOUT")
        return None, None, None, 0

def xrt_bench(xp, ip, M, K, N, ni=3):
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

    rng = np.random.RandomState(42)
    inp = rng.randint(-127, 128, size=(M, K), dtype=np.int8)
    wt = rng.randint(-127, 128, size=(K, N), dtype=np.int8)

    # Warmup
    for _ in range(1):
        in_bo.write(inp.tobytes(), 0); wt_bo.write(wt.tobytes(), 0)
        out_bo.write(b'\x00'*out_sz, 0)
        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = pyxrt.run(kern)
        run.set_arg(0,3); run.set_arg(1,inst_bo); run.set_arg(2,isz)
        run.set_arg(3,in_bo); run.set_arg(4,wt_bo); run.set_arg(5,out_bo)
        run.start(); run.wait()
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    times = []
    for _ in range(ni):
        t0=time.time()
        in_bo.write(inp.tobytes(),0); wt_bo.write(wt.tobytes(),0)
        out_bo.write(b'\x00'*out_sz,0)
        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t1=time.time()
        run=pyxrt.run(kern)
        run.set_arg(0,3); run.set_arg(1,inst_bo); run.set_arg(2,isz)
        run.set_arg(3,in_bo); run.set_arg(4,wt_bo); run.set_arg(5,out_bo)
        run.start(); run.wait()
        t2=time.time()
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw=out_bo.read(out_sz,0)
        t3=time.time()
        times.append(((t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000,(t3-t0)*1000))

    out=np.frombuffer(raw,dtype=np.int32).reshape(M,N).copy()
    avg_up=np.mean([t[0] for t in times])
    avg_kern=np.mean([t[1] for t in times])
    avg_dl=np.mean([t[2] for t in times])
    avg_tot=np.mean([t[3] for t in times])

    # Correctness
    cpu_out=inp.astype(np.int32) @ wt.astype(np.int32)
    max_err=float(np.max(np.abs(cpu_out.astype(float)-out.astype(float))))
    bit_exact=bool(np.array_equal(cpu_out,out))

    flops=2.0*M*K*N
    gops=round(flops/(avg_tot/1000)/1e9,2) if avg_tot>0 else 0

    return {"total":avg_tot,"kernel":avg_kern,"upload":avg_up,"download":avg_dl,
            "gops":gops,"max_err":max_err,"bit_exact":bit_exact,"flops":flops}

def main():
    os.makedirs(OUT, exist_ok=True)

    # Phase 18-C: K optimum for M=512, N=3072
    configs = []
    for K in [1024, 2048, 4096]:
        configs.append((512, K, 3072, 1, 64, 64, 64))  # C=1, Tk=64
    # Also test with different N for K=2048 and K=4096
    for K in [2048, 4096]:
        for N in [1024, 3072, 4096]:
            configs.append((512, K, N, 1, 64, 64, 64))

    results = []
    for M, K, N, nc, tm, tk, tn in configs:
        wd, xp, ip, ct = compile_kernel(M, K, N, nc, tm, tk, tn)
        if not xp:
            print(f"  SKIP M={M} K={K} N={N} C{nc}")
            continue
        try:
            perf = xrt_bench(xp, ip, M, K, N, ni=3)
        except Exception as e:
            print(f"  ERROR M={M} K={K} N={N}: {e}")
            continue
        perf["M"]=M; perf["K"]=K; perf["N"]=N; perf["C"]=nc
        perf["compile_s"]=ct; perf["tile_k"]=tk
        results.append(perf)
        ex = "EXACT" if perf["bit_exact"] else f"err={perf['max_err']:.1f}"
        print(f"  M={M} K={K} N={N} C{nc}: {perf['total']:.1f}ms kernel={perf['kernel']:.1f}ms "
              f"GOPS={perf['gops']:.2f} {ex}")

    jp = os.path.join(OUT, "phase18_k_optimum.json")
    with open(jp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON: {jp}")

    # Summary
    if results:
        print(f"\n{'='*70}")
        print(f"K OPTIMUM RESULTS")
        print(f"{'='*70}")
        print(f"{'M':>5} {'K':>5} {'N':>5} {'C':>2} {'total':>8} {'kernel':>8} {'GOPS':>7} {'err':>8} {'exact':>5}")
        for r in sorted(results, key=lambda x: (x["M"], x["N"], x["total"])):
            print(f"{r['M']:>5} {r['K']:>5} {r['N']:>5} {r['C']:>2} "
                  f"{r['total']:>8.1f} {r['kernel']:>8.1f} "
                  f"{r['gops']:>7.2f} {r['max_err']:>8.1f} {r['bit_exact']:>5}")

if __name__ == "__main__":
    main()
