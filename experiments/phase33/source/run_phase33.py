#!/usr/bin/env python3
"""Phase 33 driver: XDNA capability boundary measurement.

Runs comprehensive sweeps to find XDNA's high-utilization workload regime:
  1. Null-kernel baseline (fixed overhead measurement)
  2. M/N/K scaling sweep with timing decomposition
  3. Batched GEMM (G=1,2,4,8,16,32,64,128)
  4. Fused GEMM chain (G=2,4,8,16)
  5. Tile shape sweep (m_tile, n_tile, k_chunk)
  6. Large-M sweep (512, 1024, 2048, 4096, 8192)
  7. Arithmetic intensity analysis

Results go to experiments/phase33/results/*.csv
"""
import os, sys, subprocess, csv, json, shutil, glob

REPO = "/workspace/llama.cpp"
PHASE33 = f"{REPO}/experiments/phase33"
CACHE = f"{REPO}/experiments/npu-int8-gemm/kernel_cache"
STUB = f"{REPO}/ggml/src/ggml-xdna/stub"

BENCH = f"{PHASE33}/source/bench_xdna_phase33"

# Shape taxonomy from Phase 33 spec
M_SWEEP = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
N_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
K_SWEEP = [128, 256, 512, 1024, 2048, 4096, 8192]

# Priority shapes: square, tall, wide, large-K, large-M, large-N
PRIORITY_SHAPES = [
    # Square M=N=K
    (64, 64, 64, "sq_64"),
    (128, 128, 128, "sq_128"),
    (256, 256, 256, "sq_256"),
    (512, 512, 512, "sq_512"),
    (1024, 1024, 1024, "sq_1024"),
    (2048, 2048, 2048, "sq_2048"),
    # Tall M >> N
    (1024, 256, 256, "tall_M1024"),
    (2048, 512, 512, "tall_M2048"),
    (4096, 1024, 1024, "tall_M4096"),
    # Wide N >> M
    (256, 1024, 1024, "wide_N1024"),
    (512, 2048, 2048, "wide_N2048"),
    # Large K
    (64, 8192, 64, "largeK_K8192"),
    (128, 4096, 128, "largeK_K4096"),
    (256, 2048, 256, "largeK_K2048"),
    # Real Qwen shapes
    (64, 1024, 3072, "qwen_ffn_M64"),
    (128, 1024, 3072, "qwen_ffn_M128"),
    (256, 1024, 3072, "qwen_ffn_M256"),
    (512, 1024, 3072, "qwen_ffn_M512"),
    (1024, 1024, 3072, "qwen_ffn_M1024"),
    (64, 1024, 1024, "qwen_qkvo_M64"),
    (128, 1024, 1024, "qwen_qkvo_M128"),
    (256, 1024, 1024, "qwen_qkvo_M256"),
    (512, 1024, 1024, "qwen_qkvo_M512"),
    # Large M variants
    (512, 1024, 1024, "largeM_512"),
    (1024, 1024, 1024, "largeM_1024"),
    (2048, 1024, 1024, "largeM_2048"),
]

# Tile shapes to sweep
TILE_SHAPES = [
    (32, 32, 128),
    (32, 64, 128),
    (32, 128, 128),
    (64, 32, 128),
    (64, 64, 128),
    (64, 128, 128),
    (64, 256, 128),
    (128, 64, 128),
    (128, 128, 128),
    (128, 256, 128),
]

# Batch sizes for batched GEMM
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

# Fused chain lengths
CHAIN_SIZES = [2, 4, 8, 16]

def kernel_path(M, K, N, cache=CACHE):
    base = f"M{M}_K{K}_N{N}"
    candidates = [
        f"{cache}/{base}_C1_Tm64_Tk64_Tn64",
        f"{cache}/{base}_C1_Tm64_Tk32_Tn64",
        f"{cache}/{base}_C1",
        f"{cache}/{base}_C1_Q8SCALE",
        f"{cache}/{base}_Q8SCALE",
        f"{cache}/{base}_C1_Tm64_Tk64_Tn64",
        f"{cache}/{base}",
    ]
    for d in candidates:
        xc = os.path.join(d, "design.xclbin")
        ins = os.path.join(d, "design.insts.bin")
        if os.path.exists(xc) and os.path.exists(ins):
            return xc, ins
    return None, None

def run_cmd(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"

def parse_csv_line(line):
    parts = line[4:].strip().split(",")
    return parts

def bench(mode, xclbin, insts, M, K, N, reps, tag="bench",
          m_tile=64, n_tile=64, k_chunk=256, G=1):
    if mode == "null":
        cmd = (f"cd {PHASE33}/source && "
               f"./bench_xdna_phase33 null {xclbin} {insts} {reps} {tag}")
    elif mode == "tile":
        cmd = (f"cd {PHASE33}/source && "
               f"./bench_xdna_phase33 tile {xclbin} {insts} {M} {K} {N} "
               f"{m_tile} {n_tile} {k_chunk} {reps} {tag}")
    elif mode in ("batch", "chain"):
        cmd = (f"cd {PHASE33}/source && "
               f"./bench_xdna_phase33 {mode} {xclbin} {insts} {M} {K} {N} {G} {reps} {tag}")
    else:
        cmd = (f"cd {PHASE33}/source && "
               f"./bench_xdna_phase33 {mode} {xclbin} {insts} {M} {K} {N} {reps} {tag}")
    rc, out, err = run_cmd(cmd, timeout=180)
    if rc != 0:
        return {"M": M, "K": K, "N": N, "G": G, "status": "fail", "reason": err.strip()[:200]}

    for line in out.splitlines():
        if line.startswith("CSV "):
            parts = parse_csv_line(line)
            return {
                "tag": parts[1], "mode": parts[0], "M": int(parts[2]), "K": int(parts[3]),
                "N": int(parts[4]), "m_tile": int(parts[5]), "n_tile": int(parts[6]),
                "k_chunk": int(parts[7]), "G": int(parts[8]), "reps": int(parts[9]),
                "med_ms": float(parts[10]), "min_ms": float(parts[11]), "max_ms": float(parts[12]),
                "gops": float(parts[13]), "eff_gops": float(parts[14]),
                "submit_ms": float(parts[15]), "dma_in_ms": float(parts[16]),
                "compute_ms": float(parts[17]), "dma_out_ms": float(parts[18]),
                "total_ms": float(parts[19]), "null_overhead": float(parts[20]),
                "arith_intensity": float(parts[21]), "status": "ok"
            }
    return {"M": M, "K": K, "N": N, "G": G, "status": "parse_fail", "raw": out[:500]}

def build_bench():
    print("[build] compiling phase33 benchmarks...")
    cmd = (f"cd {PHASE33}/source && "
           f"g++ -O2 -std=c++17 -I{STUB} bench_xdna_phase33.cpp "
           f"-o bench_xdna_phase33 -lxrt_coreutil -lpthread")
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        print(f"[build] FAILED: {cmd}\n{err}")
        return False
    print("[build] ok")
    return True

def phase1_null_kernel():
    """Measure fixed overhead of XRT submission without compute."""
    print("[phase1] null-kernel baseline...")
    # Use M128_K32_N1024 as smallest available kernel for null baseline
    xclbin, insts = kernel_path(128, 32, 1024)
    if not xclbin:
        xclbin, insts = kernel_path(128, 1024, 1024)

    if not xclbin:
        print("[phase1] SKIP: no kernel cache available")
        return []

    r = bench("null", xclbin, insts, 0, 0, 0, 5, "null_baseline")
    results = [r] if r.get("status") == "ok" else []
    return results

def phase2_shape_sweep():
    """Priority shape sweep with timing decomposition."""
    print("[phase2] priority shape sweep...")
    results = []
    for M, K, N, tag in PRIORITY_SHAPES:
        xclbin, insts = kernel_path(M, K, N)
        if not xclbin:
            print(f"  {tag}: SKIP (no kernel)")
            continue
        print(f"  {tag}: M={M} K={K} N={N}")
        r = bench("timing", xclbin, insts, M, K, N, 5, tag)
        if r and r.get("status") == "ok":
            results.append(r)
            print(f"    total={r['total_ms']:.2f}ms compute={r['compute_ms']:.2f}ms "
                  f"dma_in={r['dma_in_ms']:.2f}ms dma_out={r['dma_out_ms']:.2f}ms "
                  f"gops={r['gops']:.2f} eff={r['eff_gops']:.2f}")
        else:
            print(f"    FAIL: {r.get('reason', 'unknown')}")
    return results

def phase3_batched_gemm():
    """Batched GEMM: amortize submission over G independent GEMMs."""
    print("[phase3] batched GEMM...")
    results = []
    # Use a shape that has a cached kernel
    M, K, N = 512, 1024, 3072
    xclbin, insts = kernel_path(M, K, N)
    if not xclbin:
        print("[phase3] SKIP: no kernel for M=512 K=1024 N=3072")
        return results

    for G in BATCH_SIZES:
        tag = f"batch_G{G}"
        print(f"  G={G}: M={M} K={K} N={N}")
        r = bench("batch", xclbin, insts, M, K, N, 3, tag, G=G)
        if r and r.get("status") == "ok":
            results.append(r)
            print(f"    per_gemm={r['med_ms']:.4f}ms gops={r['gops']:.2f}")
        else:
            print(f"    FAIL: {r.get('reason', 'unknown')}")
    return results

def phase4_fused_chain():
    """Fused GEMM chain: G sequential GEMMs, all on device."""
    print("[phase4] fused chain...")
    results = []
    M, K, N = 512, 1024, 3072
    xclbin, insts = kernel_path(M, K, N)
    if not xclbin:
        print("[phase4] SKIP: no kernel for M=512 K=1024 N=3072")
        return results

    for G in CHAIN_SIZES:
        tag = f"chain_G{G}"
        print(f"  G={G}: M={M} K={K} N={N}")
        r = bench("chain", xclbin, insts, M, K, N, 3, tag, G=G)
        if r and r.get("status") == "ok":
            results.append(r)
            print(f"    per_gemm={r['med_ms']:.4f}ms gops={r['gops']:.2f}")
        else:
            print(f"    FAIL: {r.get('reason', 'unknown')}")
    return results

def phase5_tile_sweep():
    """Tile shape sweep for a fixed M/K/N."""
    print("[phase5] tile shape sweep...")
    results = []
    M, K, N = 512, 1024, 3072
    xclbin, insts = kernel_path(M, K, N)
    if not xclbin:
        print("[phase5] SKIP: no kernel for M=512 K=1024 N=3072")
        return results

    for m_tile, n_tile, k_chunk in TILE_SHAPES:
        tag = f"tile_m{m_tile}_n{n_tile}_k{k_chunk}"
        print(f"  {tag}: M={M} K={K} N={N}")
        r = bench("tile", xclbin, insts, M, K, N, 3, tag,
                  m_tile=m_tile, n_tile=n_tile, k_chunk=k_chunk)
        if r and r.get("status") == "ok":
            results.append(r)
            print(f"    med={r['med_ms']:.4f}ms gops={r['gops']:.2f}")
        else:
            print(f"    FAIL: {r.get('reason', 'unknown')}")
    return results

def phase6_large_M():
    """Large-M sweep: does larger M amortize fixed cost?"""
    print("[phase6] large-M sweep...")
    results = []
    K, N = 1024, 1024
    for M in [512, 1024, 2048, 4096, 8192]:
        xclbin, insts = kernel_path(M, K, N)
        if not xclbin:
            print(f"  M={M}: SKIP (no kernel)")
            continue
        tag = f"largeM_M{M}"
        print(f"  {tag}: M={M} K={K} N={N}")
        r = bench("timing", xclbin, insts, M, K, N, 5, tag)
        if r and r.get("status") == "ok":
            results.append(r)
            print(f"    total={r['total_ms']:.2f}ms compute={r['compute_ms']:.2f}ms gops={r['gops']:.2f}")
        else:
            print(f"    FAIL: {r.get('reason', 'unknown')}")
    return results

def write_csv(path, rows, fieldnames):
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

def main():
    os.makedirs(f"{PHASE33}/results", exist_ok=True)
    os.makedirs(f"{PHASE33}/raw", exist_ok=True)

    if not build_bench():
        sys.exit(1)

    all_results = {}

    # Phase 1: Null kernel
    null_results = phase1_null_kernel()
    all_results["null"] = null_results

    # Phase 2: Shape sweep
    shape_results = phase2_shape_sweep()
    all_results["shape"] = shape_results

    # Phase 3: Batched GEMM
    batch_results = phase3_batched_gemm()
    all_results["batch"] = batch_results

    # Phase 4: Fused chain
    chain_results = phase4_fused_chain()
    all_results["chain"] = chain_results

    # Phase 5: Tile sweep
    tile_results = phase5_tile_sweep()
    all_results["tile"] = tile_results

    # Phase 6: Large M
    largeM_results = phase6_large_M()
    all_results["largeM"] = largeM_results

    # Write CSVs
    fieldnames_shape = ["mode","tag","M","K","N","m_tile","n_tile","k_chunk","G","reps",
                        "med_ms","min_ms","max_ms","gops","eff_gops",
                        "submit_ms","dma_in_ms","compute_ms","dma_out_ms","total_ms",
                        "null_overhead","arith_intensity","status"]
    for key, rows in all_results.items():
        if rows:
            write_csv(f"{PHASE33}/results/{key}.csv", rows, fieldnames_shape)

    print(f"\n[done] null={len(null_results)} shape={len(shape_results)} "
          f"batch={len(batch_results)} chain={len(chain_results)} "
          f"tile={len(tile_results)} largeM={len(largeM_results)}")

    # Save all results as JSON for analysis
    with open(f"{PHASE33}/raw/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return 0

if __name__ == "__main__":
    sys.exit(main())
