#!/usr/bin/env python3
"""
Phase 20: Comprehensive XDNA Kernel Performance Analysis

Covers Steps I-II (baseline), IX (M scaling), XI (K sweep), XII (Roofline).
Uses CURRENT_RUNTIME_KERNEL: npu1_1col Tm64_Tk64_Tn64.
"""
import pyxrt
import numpy as np
import time
import os
import json

XCLBIN_DIR = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache"
N_WARMUP = 10
N_RUNS = 50

# Global device - reuse across all shapes
_dev = None

def get_dev():
    global _dev
    if _dev is None:
        _dev = pyxrt.device(0)
    return _dev

def load_engine(M, K, N, n_cols=1, tm=64, tk=64, tn=64):
    base = f"{XCLBIN_DIR}/M{M}_K{K}_N{N}_C{n_cols}_Tm{tm}_Tk{tk}_Tn{tn}"
    xp = f"{base}/design.xclbin"
    ip = f"{base}/design.insts.bin"
    if not os.path.exists(xp):
        return None
    dev = get_dev()
    xb = pyxrt.xclbin(xp)
    dev.register_xclbin(xb)
    inst_bytes = os.path.getsize(ip)
    with open(ip, "rb") as f:
        idata = f.read()
    hw = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(hw, "MLIR_AIE")
    g1 = kern.group_id(1)
    g3 = kern.group_id(3)
    ibo = pyxrt.bo(dev, inst_bytes, pyxrt.bo.flags.cacheable, g1)
    ibo.write(idata, 0)
    ibo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return {"dev": dev, "kern": kern, "ibo": ibo, "isz": inst_bytes, "g1": g1, "g3": g3}

def bench_one(engine, inp_i8, wgt_i8, M, K, N):
    dev = engine["dev"]
    kern = engine["kern"]
    g3 = engine["g3"]
    
    in_bo = pyxrt.bo(dev, max(M*K, 64), pyxrt.bo.flags.host_only, g3)
    wt_bo = pyxrt.bo(dev, max(K*N, 64), pyxrt.bo.flags.host_only, g3)
    out_bo = pyxrt.bo(dev, max(M*N*4, 64), pyxrt.bo.flags.host_only, g3)
    
    t0 = time.perf_counter()
    in_bo.write(inp_i8.tobytes(), 0)
    wt_bo.write(wgt_i8.tobytes(), 0)
    out_bo.write(b'\x00' * (M*N*4), 0)
    in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t1 = time.perf_counter()
    
    r = pyxrt.run(kern)
    r.set_arg(0, 3)
    r.set_arg(1, engine["ibo"])
    r.set_arg(2, engine["isz"])
    r.set_arg(3, in_bo)
    r.set_arg(4, wt_bo)
    r.set_arg(5, out_bo)
    r.start()
    r.wait()
    t2 = time.perf_counter()
    
    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    result = np.frombuffer(out_bo.read(M*N*4, 0), dtype=np.int32).copy()
    t3 = time.perf_counter()
    
    return {
        "kernel_ms": (t2 - t1) * 1000,
        "total_ms": (t3 - t0) * 1000,
        "upload_ms": (t1 - t0) * 1000,
        "download_ms": (t3 - t2) * 1000,
        "result": result,
    }

def cpu_ref(inp_i8, wgt_i8, M, K, N):
    return (inp_i8.reshape(M,K).astype(np.int32) @ wgt_i8.reshape(K,N).astype(np.int32)).astype(np.int32)

def measure_shape(M, K, N, n_cols=1):
    engine = load_engine(M, K, N, n_cols)
    if engine is None:
        return None
    
    np.random.seed(42)
    inp_i8 = np.random.randint(-127, 128, size=M*K, dtype=np.int8)
    wgt_i8 = np.random.randint(-127, 128, size=K*N, dtype=np.int8)
    
    ref = cpu_ref(inp_i8, wgt_i8, M, K, N)
    
    # Warmup
    for _ in range(N_WARMUP):
        bench_one(engine, inp_i8, wgt_i8, M, K, N)
    
    # Benchmark
    kernel_times = []
    total_times = []
    check_result = None
    for i in range(N_RUNS):
        t = bench_one(engine, inp_i8, wgt_i8, M, K, N)
        kernel_times.append(t["kernel_ms"])
        total_times.append(t["total_ms"])
        if i == 0:
            check_result = t["result"]
    
    max_err = int(np.max(np.abs(check_result - ref.flatten())))
    
    kt = np.array(kernel_times)
    tt = np.array(total_times)
    
    flops = 2.0 * M * K * N
    bytes_moved = M*K + K*N + M*N*4
    
    return {
        "M": M, "K": K, "N": N, "n_cols": n_cols,
        "correctness": max_err == 0,
        "max_err": max_err,
        "kernel_ms_mean": float(np.mean(kt)),
        "kernel_ms_p50": float(np.percentile(kt, 50)),
        "kernel_ms_p95": float(np.percentile(kt, 95)),
        "total_ms_mean": float(np.mean(tt)),
        "total_ms_p50": float(np.percentile(tt, 50)),
        "GOPS_kernel": flops / (np.mean(kt) * 1e-3) / 1e9,
        "GOPS_total": flops / (np.mean(tt) * 1e-3) / 1e9,
        "FLOPs": flops,
        "bytes_moved": bytes_moved,
        "AI": flops / bytes_moved,
        "DMA_GB_s": bytes_moved / (np.mean(kt) * 1e-3) / 1e9,
    }

def main():
    print("=" * 80)
    print("Phase 20: Comprehensive XDNA Kernel Performance Analysis")
    print("=" * 80)
    
    results = {"m_scaling": [], "k_sweep": [], "n_cols_test": []}
    
    # === STEP IX: M Scaling (K=1024, N=3072) ===
    print("\n" + "=" * 80)
    print("STEP IX: M Scaling (K=1024, N=3072)")
    print("=" * 80)
    print(f"{'M':>6} {'kernel_ms':>10} {'GOPS_k':>8} {'GOPS_t':>8} {'AI':>8} {'DMA GB/s':>10} {'correct':>7}")
    
    for M in [64, 128, 256, 512]:
        N = 3072
        K = 1024
        r = measure_shape(M, K, N)
        if r:
            results["m_scaling"].append(r)
            print(f"{M:6d} {r['kernel_ms_mean']:10.2f} {r['GOPS_kernel']:8.2f} {r['GOPS_total']:8.2f} "
                  f"{r['AI']:8.1f} {r['DMA_GB_s']:10.2f} {str(r['correctness']):>7}")
    
    # === STEP XI: K Sweep (M=512, N=3072) ===
    print("\n" + "=" * 80)
    print("STEP XI: K Sweep (M=512, N=3072)")
    print("=" * 80)
    print(f"{'K':>6} {'kernel_ms':>10} {'GOPS_k':>8} {'AI':>8} {'DMA GB/s':>10} {'correct':>7}")
    
    for K in [1024, 2048, 4096]:
        M = 512
        N = 3072
        r = measure_shape(M, K, N)
        if r:
            results["k_sweep"].append(r)
            print(f"{K:6d} {r['kernel_ms_mean']:10.2f} {r['GOPS_kernel']:8.2f} "
                  f"{r['AI']:8.1f} {r['DMA_GB_s']:10.2f} {str(r['correctness']):>7}")
    
    # === STEP X: N_COLS test (1 vs 4) ===
    print("\n" + "=" * 80)
    print("STEP X: N_COLS Test (M=512, K=1024)")
    print("=" * 80)
    print(f"{'N':>6} {'cols':>5} {'kernel_ms':>10} {'GOPS_k':>8} {'correct':>7}")
    
    for N in [3072]:
        for nc in [1, 4]:
            M, K = 512, 1024
            r = measure_shape(M, K, N, n_cols=nc)
            if r:
                results["n_cols_test"].append(r)
                print(f"{N:6d} {nc:5d} {r['kernel_ms_mean']:10.2f} {r['GOPS_kernel']:8.2f} "
                      f"{str(r['correctness']):>7}")
            else:
                print(f"{N:6d} {nc:5d} {'N/A':>10} {'N/A':>8} {'N/A':>7}")
    
    # === STEP XII: Roofline Analysis ===
    print("\n" + "=" * 80)
    print("STEP XII: Roofline Analysis")
    print("=" * 80)
    print("Phoenix XDNA1 specs (estimated):")
    print("  INT8 peak per column: ~2000 GOPS (conservative)")
    print("  DMA peak H2D: 14 GB/s (measured)")
    print("  DMA peak D2H: 116 GB/s (measured)")
    print()
    print(f"{'Shape':>20} {'AI':>8} {'GOPS':>8} {'Memory BW':>12} {'Bound':>12}")
    
    all_data = results["m_scaling"] + results["k_sweep"]
    for r in all_data:
        gops = r["GOPS_kernel"]
        ai = r["AI"]
        dma = r["DMA_GB_s"]
        peak_compute = 2000  # GOPS per column
        peak_mem = 14  # GB/s H2D (conservative)
        compute_bound_gops = peak_compute
        memory_bound_gops = peak_mem * ai
        
        if gops < memory_bound_gops * 0.8:
            bound = "MEMORY"
        elif gops < compute_bound_gops * 0.8:
            bound = "MIXED"
        else:
            bound = "COMPUTE"
        
        label = f"M={r['M']}_K={r['K']}_N={r['N']}"
        print(f"{label:>20} {ai:8.1f} {gops:8.2f} {dma:10.2f} GB/s {bound:>12}")
    
    # Save
    out_dir = "/Storage/Cache/llama.cpp/experiments/phase20"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/phase20_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # === FINAL SUMMARY ===
    print("\n" + "=" * 80)
    print("PHASE 20 KEY FINDINGS")
    print("=" * 80)
    print(f"1. M scaling: GOPS constant at ~{np.mean([r['GOPS_kernel'] for r in results['m_scaling']]):.1f} across M=64..512")
    print(f"2. K scaling: K=1024->{results['k_sweep'][0]['GOPS_kernel']:.1f}GOPS, "
          f"K=2048->{results['k_sweep'][1]['GOPS_kernel']:.1f}GOPS, "
          f"K=4096->{results['k_sweep'][2]['GOPS_kernel']:.1f}GOPS")
    if results['n_cols_test']:
        for r in results['n_cols_test']:
            print(f"3. N_COLS={r['n_cols']}: {r['GOPS_kernel']:.1f} GOPS")
    
    print(f"\nResults saved to {out_dir}/phase20_analysis.json")

if __name__ == "__main__":
    main()
