#!/usr/bin/env python3
"""Phase 18 Kernel Performance Searcher - v3 (final fixed API)"""
import argparse, csv, json, os, sys, time, traceback, subprocess
import numpy as np

try:
    import pyxrt
except ImportError:
    sys.exit("ERROR: pyxrt not available")

KC = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

def check_shape(M, K, N, tm, tk, tn, nc):
    if M % 4 or K % 8 or N % 8: return ["AIE2 align"]
    if M % tm or K % tk or N % tn: return ["tile align"]
    if N % (tn * nc): return ["N%(tn*nc)"]
    return []

def get_wd(M, K, N, tm, tk, tn, nc):
    return os.path.join(KC, f"M{M}_K{K}_N{N}_C{nc}_Tm{tm}_Tk{tk}_Tn{tn}")

def xrt_exec(xp, ip, M, K, N, inp_i8, wt_i8, nw=1, ni=5):
    timing = {}
    t0 = time.time()
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xp)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    timing["init_ms"] = round((time.time()-t0)*1000, 3)
    in_sz = M * K; wt_sz = K * N; out_sz = M * N * 4
    gid3 = kern.group_id(3)
    in_bo = pyxrt.bo(dev, in_sz, pyxrt.bo.flags.host_only, gid3)
    wt_bo = pyxrt.bo(dev, wt_sz, pyxrt.bo.flags.host_only, gid3)
    out_bo = pyxrt.bo(dev, out_sz, pyxrt.bo.flags.host_only, gid3)
    isz = os.path.getsize(ip)
    gid1 = kern.group_id(1)
    inst_bo = pyxrt.bo(dev, isz, pyxrt.bo.flags.cacheable, gid1)
    with open(ip, "rb") as f:
        idata = f.read()
    # bo.write(data, offset) - 2nd arg is offset, not size
    inst_bo.write(idata, 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def run_one():
        t1 = time.time()
        in_bo.write(inp_i8.tobytes(), 0)
        wt_bo.write(wt_i8.tobytes(), 0)
        out_bo.write(b'\x00' * out_sz, 0)
        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        up = (time.time()-t1)*1000
        t2 = time.time()
        run = pyxrt.run(kern)
        run.set_arg(0, 3); run.set_arg(1, inst_bo); run.set_arg(2, isz)
        run.set_arg(3, in_bo); run.set_arg(4, wt_bo); run.set_arg(5, out_bo)
        run.start()
        sub = (time.time()-t2)*1000
        t3 = time.time(); run.wait()
        wt_ms = (time.time()-t3)*1000
        t4 = time.time()
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw = out_bo.read(out_sz, 0)
        dl = (time.time()-t4)*1000
        return np.frombuffer(raw, dtype=np.int32).reshape(M, N).copy(), up, sub, wt_ms, dl

    for _ in range(nw): run_one()
    s = {"u":0, "s":0, "w":0, "d":0}
    last = None
    for _ in range(ni):
        o, up, sub, wt, dl = run_one()
        s["u"] += up; s["s"] += sub; s["w"] += wt; s["d"] += dl; last = o
    n = ni
    timing["upload_ms"] = round(s["u"]/n, 3)
    timing["submit_ms"] = round(s["s"]/n, 3)
    timing["wait_ms"] = round(s["w"]/n, 3)
    timing["download_ms"] = round(s["d"]/n, 3)
    timing["total_ms"] = round(sum(s.values())/n, 3)
    timing["kernel_time_ms"] = round(s["w"]/n, 3)
    return last, timing

def cpu_gemm(a, b, M, K, N):
    return a.astype(np.int32) @ b.astype(np.int32)

def err_stats(ref, test):
    d = ref.astype(np.float64) - test.astype(np.float64)
    ad = np.abs(d); dn = np.abs(ref.astype(np.float64)) + 1.0
    return {"max_abs": float(np.max(ad)), "mean_abs": float(np.mean(ad)),
            "rmse": float(np.sqrt(np.mean(d**2))),
            "mismatch_pct": float(100*np.sum(ad>0)/ref.size),
            "max_rel": float(np.max(ad/dn)),
            "bit_exact": bool(np.array_equal(ref, test))}

def bench(M, K, N, tm, tk, tn, nc, iters=5, warmup=1, force=False):
    errs = check_shape(M, K, N, tm, tk, tn, nc)
    if errs:
        return {"error": ";".join(errs), "M":M, "K":K, "N":N,
                "tile_m":tm, "tile_k":tk, "tile_n":tn, "n_cols":nc}
    wd = get_wd(M, K, N, tm, tk, tn, nc)
    xp = os.path.join(wd, "design.xclbin")
    ip = os.path.join(wd, "design.insts.bin")
    r = {"M":M, "K":K, "N":N, "tile_m":tm, "tile_k":tk, "tile_n":tn, "n_cols":nc, "dtype":"int8"}
    if os.path.exists(xp) and os.path.exists(ip):
        r["compile_time_s"] = 0; r["compiled"] = "cached"
    elif force:
        print(f"  [COMPILE] M={M} K={K} N={N} C{nc}...")
        cmd = [sys.executable, "/workspace/llama.cpp/experiments/npu-int8-gemm/gen_xdna_gemm.py",
               "--M", str(M), "--K", str(K), "--N", str(N), "--cols", str(nc),
               "--tile-m", str(tm), "--tile-k", str(tk), "--tile-n", str(tn), "--work-dir", wd]
        t0 = time.time()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            ct = time.time() - t0
            if res.returncode != 0: r["error"] = "compile_fail"; return r
            if not (os.path.exists(xp) and os.path.exists(ip)): r["error"] = "compile_no_out"; return r
            r["compile_time_s"] = round(ct, 3); r["compiled"] = "fresh"
        except: r["error"] = "compile_timeout"; return r
    else:
        r["error"] = "not_compiled"; return r
    r["xclbin_size"] = os.path.getsize(xp)
    r["insts_size"] = os.path.getsize(ip)
    rng = np.random.RandomState(42)
    inp = rng.randint(-127, 128, size=(M, K), dtype=np.int8)
    wt = rng.randint(-127, 128, size=(K, N), dtype=np.int8)
    try:
        xo, tim = xrt_exec(xp, ip, M, K, N, inp, wt, nw=warmup, ni=iters)
        r.update(tim)
    except Exception as e:
        r["error"] = "xrt:" + str(e)[:200]
        traceback.print_exc(file=sys.stderr)
        return r
    t0 = time.time(); co = cpu_gemm(inp, wt, M, K, N)
    r["cpu_ref_ms"] = round((time.time()-t0)*1000, 3)
    acc = err_stats(co, xo)
    r["accuracy"] = acc; r["bit_exact"] = acc["bit_exact"]; r["max_abs_error"] = acc["max_abs"]
    flops = 2.0 * M * K * N
    r["gops"] = round(flops/(r["total_ms"]/1000)/1e9, 3) if r["total_ms"] > 0 else 0
    r["flops"] = flops; r["dma_bytes"] = M*K + K*N + M*N*4; r["status"] = "ok"
    return r

def focused(M, K, N):
    cfgs = []
    for nc in [1, 2, 4]:
        for tk in [32, 64, 128, 256]:
            if not check_shape(M, K, N, 64, tk, 64, nc):
                cfgs.append((M, K, N, 64, tk, 64, nc))
    return cfgs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int); p.add_argument("--K", type=int); p.add_argument("--N", type=int)
    p.add_argument("--iters", type=int, default=5); p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--force-compile", action="store_true")
    p.add_argument("--output-dir", default="/home/ubuntu12738/phase18")
    args = p.parse_args()
    if args.M and args.K and args.N:
        configs = focused(args.M, args.K, args.N)
    else:
        configs = [(512, 1024, 3072, 64, 32, 64, 1), (512, 1024, 3072, 64, 64, 64, 1)]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'='*70}\nPhase 18 Kernel Search  configs={len(configs)}  iters={args.iters}\n{'='*70}\n")
    results = []
    for i, (M, K, N, tm, tk, tn, nc) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] M={M} K={K} N={N} Tm={tm} Tk={tk} Tn={tn} C={nc}")
        r = bench(M, K, N, tm, tk, tn, nc, args.iters, args.warmup, args.force_compile)
        results.append(r)
        if r.get("status") == "ok":
            print(f"  -> {r['total_ms']:.1f}ms  GOPS={r.get('gops',0):.2f}  "
                  f"err={r.get('max_abs_error',-1):.1f}  exact={r.get('bit_exact',False)}")
        else:
            print(f"  -> {r.get('error', '?')}")
        sys.stdout.flush()
    jp = os.path.join(args.output_dir, "phase18_kernel_search.json")
    with open(jp, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\nJSON: {jp}")
    cp = os.path.join(args.output_dir, "phase18_kernel_search.csv")
    if results:
        ks = set()
        for r in results: ks.update(r.keys())
        with open(cp, "w", newline="") as f:
            w = csv.DictWriter(f, sorted(ks), extrasaction="ignore"); w.writeheader()
            for r in results:
                flat = {}
                for k, v in r.items():
                    if isinstance(v, dict):
                        for sk, sv in v.items(): flat[f"{k}_{sk}"] = sv
                    else: flat[k] = v
                w.writerow(flat)
        print(f"CSV: {cp}")
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        ok.sort(key=lambda r: r.get("total_ms", 99999))
        print(f"\n{'='*70}\nSUMMARY: {len(ok)}/{len(results)} succeeded\n{'='*70}")
        print(f"{'M':>5} {'K':>5} {'N':>5} {'C':>2} {'Tk':>4} {'total':>8} {'kernel':>8} {'GOPS':>7} {'err':>8} {'exact':>5}")
        for r in ok[:15]:
            print(f"{r['M']:>5} {r['K']:>5} {r['N']:>5} {r['n_cols']:>2} {r['tile_k']:>4} "
                  f"{r['total_ms']:>8.1f} {r.get('kernel_time_ms',0):>8.1f} "
                  f"{r.get('gops',0):>7.2f} {r.get('max_abs_error',-1):>8.1f} "
                  f"{r.get('bit_exact',False):>5}")
        ex = [r for r in ok if r.get("bit_exact")]
        if ex:
            print(f"\nBit-exact: {len(ex)}")
            for r in ex[:5]:
                print(f"  M={r['M']} K={r['K']} N={r['N']} C={r['n_cols']} Tk={r['tile_k']} "
                      f"time={r['total_ms']:.1f}ms G={r.get('gops',0):.2f}")

if __name__ == "__main__":
    main()
