#!/usr/bin/env python3
"""Phase 32 driver: systematic shape sweep + chain benchmark + analysis.

Sweeps M x K x N across Qwen-relevant and exploratory shapes, runs:
  - XDNA GEMM (existing benchmark_gemm.cpp)
  - Vulkan GEMM (existing vulkan_gemm_bench.cpp)
  - XDNA persistent chain (new bench_xdna_chain.cpp)

Results go to experiments/phase32/results/*.csv
"""
import os, sys, subprocess, json, csv, shutil, glob

REPO = "/workspace/llama.cpp"
PHASE32 = f"{REPO}/experiments/phase32"
CACHE = f"{REPO}/experiments/npu-int8-gemm/kernel_cache"
BUILD_VK = f"{REPO}/build-vk"
STUB = f"{REPO}/ggml/src/ggml-xdna/stub"

BENCH_XDNA = f"{PHASE32}/bench_xdna_sweep"
BENCH_VK   = f"{PHASE32}/bench_vk_sweep"
BENCH_CHAIN = f"{PHASE32}/bench_xdna_chain"

# Qwen3.5-0.8B real shapes (M, K, N, role)
QWEN_PREFILL_SHAPES = [
    (512,  1024, 3072, "ffn_gate_PP512"),
    (512,  1024, 3072, "ffn_up_PP512"),
    (512,  3072, 1024, "ffn_down_PP512"),
    (512,  1024, 1024, "q_proj_PP512"),
    (512,  1024, 512,  "k_proj_PP512"),
    (512,  1024, 512,  "v_proj_PP512"),
    (512,  1024, 1024, "o_proj_PP512"),
    (1024, 1024, 3072, "ffn_gate_PP1K"),
    (1024, 3072, 1024, "ffn_down_PP1K"),
    (2048, 1024, 3072, "ffn_gate_PP2K"),
    (2048, 3072, 1024, "ffn_down_PP2K"),
    (4096, 1024, 3072, "ffn_gate_PP4K"),
    (4096, 3072, 1024, "ffn_down_PP4K"),
    (8192, 1024, 3072, "ffn_gate_PP8K"),
    (8192, 3072, 1024, "ffn_down_PP8K"),
    (16384,1024, 3072, "ffn_gate_PP16K"),
    (16384,3072, 1024, "ffn_down_PP16K"),
    (32768,1024, 3072, "ffn_gate_PP32K"),
    (32768,3072, 1024, "ffn_down_PP32K"),
    (65536,1024, 3072, "ffn_gate_PP64K"),
    (65536,3072, 1024, "ffn_down_PP64K"),
]

QWEN_DECODE_SHAPES = [
    (1, 1024, 3072, "ffn_gate_decode"),
    (1, 1024, 3072, "ffn_up_decode"),
    (1, 3072, 1024, "ffn_down_decode"),
    (1, 1024, 1024, "q_proj_decode"),
    (1, 1024, 1024, "o_proj_decode"),
    (1, 1024, 512,  "k_proj_decode"),
    (1, 1024, 512,  "v_proj_decode"),
]

EXPLORATORY_SHAPES = [
    # small-M candidates
    (1,  1024, 1024, "M1_K1024_N1024"),
    (4,  1024, 1024, "M4_K1024_N1024"),
    (8,  1024, 1024, "M8_K1024_N1024"),
    (16, 1024, 1024, "M16_K1024_N1024"),
    (32, 1024, 1024, "M32_K1024_N1024"),
    (64, 1024, 1024, "M64_K1024_N1024"),
    (128,1024, 1024, "M128_K1024_N1024"),
    (256,1024, 1024, "M256_K1024_N1024"),
    (1,  1024, 3072, "M1_K1024_N3072"),
    (4,  1024, 3072, "M4_K1024_N3072"),
    (8,  1024, 3072, "M8_K1024_N3072"),
    (16, 1024, 3072, "M16_K1024_N3072"),
    (32, 1024, 3072, "M32_K1024_N3072"),
    (64, 1024, 3072, "M64_K1024_N3072"),
    (1,  1024, 512,  "M1_K1024_N512"),
    (4,  1024, 512,  "M4_K1024_N512"),
    (8,  1024, 512,  "M8_K1024_N512"),
    (16, 1024, 512,  "M16_K1024_N512"),
    (32, 1024, 512,  "M32_K1024_N512"),
    (64, 1024, 512,  "M64_K1024_N512"),
    # skinny N candidates
    (512, 1024, 256,  "M512_K1024_N256"),
    (512, 1024, 128,  "M512_K1024_N128"),
    (512, 1024, 64,   "M512_K1024_N64"),
    # K variants
    (512, 256, 1024,  "M512_K256_N1024"),
    (512, 512, 1024,  "M512_K512_N1024"),
    (512, 2048, 1024, "M512_K2048_N1024"),
    (512, 4096, 1024, "M512_K4096_N1024"),
    # 2-col candidates
    (512, 1024, 4096, "M512_K1024_N4096_c2"),
]

def kernel_path(M, K, N, cols=1, cache=CACHE):
    base = f"M{M}_K{K}_N{N}"
    candidates = [
        f"{cache}/{base}_C{cols}_Tm64_Tk64_Tn64",
        f"{cache}/{base}_C{cols}_Tm64_Tk32_Tn64",
        f"{cache}/{base}_C{cols}",
        f"{cache}/{base}_C{cols}_Q8SCALE",
        f"{cache}/{base}_C{cols}_Q8SCALEV",
        f"{cache}/{base}_Tm64_Tk64_Tn64",
        f"{cache}/{base}_Tm64_Tk32_Tn64",
        f"{cache}/{base}",
        f"{cache}/{base}_Q8SCALE",
        f"{cache}/{base}_Q8SCALEV",
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

def bench_xdna(M, K, N, reps=5, cache=CACHE):
    xc, ins = kernel_path(M, K, N, cache=cache)
    if not xc:
        return None
    cmd = (f"cd {REPO}/experiments/phase32 && "
           f"./bench_xdna_sweep {xc} {ins} {M} {K} {N} 1 4 1 16384 {reps} bench_xdna")
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        return {"M": M, "K": K, "N": N, "status": "fail", "reason": err.strip()[:200]}
    # Parse output line
    for line in out.splitlines():
        if line.startswith("CSV "):
            parts = line[4:].strip().split(",")
            # format: tag,M,K,N,in_b,out_b,cols,med,mn,gops,peak,util,maxrel,status
            if len(parts) >= 10:
                return {
                    "tag": parts[0], "M": M, "K": K, "N": N,
                    "cols": int(parts[6]),
                    "ms_med": float(parts[7]), "ms_min": float(parts[8]),
                    "gops": float(parts[9]), "status": "ok"
                }
    return {"M": M, "K": K, "N": N, "status": "parse_fail", "raw": out[:500]}

def bench_vulkan(M, K, N, iters=10):
    cmd = (f"cd {REPO}/experiments/phase32 && "
           f"LD_LIBRARY_PATH={BUILD_VK} ./bench_vk_sweep {M} {K} {N} {iters}")
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        return {"M": M, "K": K, "N": N, "status": "fail", "reason": err.strip()[:200]}
    for line in out.splitlines():
        if line.startswith("CSV "):
            parts = line[4:].strip().split(",")
            if len(parts) >= 6:
                return {
                    "tag": parts[0], "M": M, "K": K, "N": N,
                    "ms": float(parts[4]), "gops": float(parts[5]), "status": "ok"
                }
    return {"M": M, "K": K, "N": N, "status": "parse_fail", "raw": out[:500]}

def bench_chain(M, K, N, num_gemms=4, reps=3, cache=CACHE):
    xc, ins = kernel_path(M, K, N, cache=cache)
    if not xc:
        return None
    cmd = (f"cd {REPO}/experiments/phase32 && "
           f"./bench_xdna_chain {M} {K} {N} {num_gemms} {reps} {cache}")
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        return {"M": M, "K": K, "N": N, "G": num_gemms, "status": "fail", "reason": err.strip()[:200]}
    for line in out.splitlines():
        if line.startswith("CSV "):
            parts = line[4:].strip().split(",")
            if len(parts) >= 16:
                return {
                    "M": M, "K": K, "N": N, "G": num_gemms,
                    "cold_total": float(parts[5]), "cold_w": float(parts[6]),
                    "cold_in": float(parts[7]), "cold_exec": float(parts[8]),
                    "cold_out": float(parts[9]),
                    "persist_total": float(parts[10]), "persist_w": float(parts[11]),
                    "persist_in": float(parts[12]), "persist_exec": float(parts[13]),
                    "persist_out": float(parts[14]),
                    "speedup": float(parts[15]), "status": "ok"
                }
    return {"M": M, "K": K, "N": N, "G": num_gemms, "status": "parse_fail", "raw": out[:500]}

def build_benchmarks():
    """Build all benchmark binaries inside npu-dev."""
    print("[build] compiling phase32 benchmarks...")
    cmds = [
        # XDNA chain benchmark
        f"cd {REPO}/experiments/phase32 && "
        f"g++ -O2 -std=c++17 -I{STUB} bench_xdna_chain.cpp "
        f"-o bench_xdna_chain -lxrt_coreutil -lpthread",
    ]
    for cmd in cmds:
        rc, out, err = run_cmd(cmd)
        if rc != 0:
            print(f"[build] FAILED: {cmd}\n{err}")
            return False
        print(f"[build] ok")
    return True

def main():
    os.makedirs(f"{PHASE32}/results", exist_ok=True)
    os.makedirs(f"{PHASE32}/raw", exist_ok=True)

    # Check binaries
    for b in [BENCH_XDNA, BENCH_VK, BENCH_CHAIN]:
        if not os.path.exists(b):
            print(f"missing binary: {b}")
            print("build phase21 benchmarks first:")
            print(f"  g++ -O2 -std=c++17 -I{STUB} {REPO}/experiments/phase21/benchmark_gemm.cpp -o {BENCH_XDNA} -lxrt_coreutil")
            print(f"  g++ -O2 -std=c++17 -I{REPO}/ggml/include {REPO}/experiments/phase21/vulkan_gemm_bench.cpp -L{BUILD_VK} -lggml -lggml-base -lggml-vulkan -Wl,-rpath,{BUILD_VK} -o {BENCH_VK}")
            sys.exit(1)

    if not build_benchmarks():
        sys.exit(1)

    all_shapes = QWEN_PREFILL_SHAPES + QWEN_DECODE_SHAPES + EXPLORATORY_SHAPES

    # Phase 1: systematic GEMM sweep (XDNA + Vulkan)
    print("[phase1] systematic GEMM sweep...")
    xdna_results = []
    vk_results = []
    for M, K, N, role in all_shapes:
        print(f"  {role}: M={M} K={K} N={N}")
        xr = bench_xdna(M, K, N)
        if xr:
            xr["role"] = role
            xdna_results.append(xr)
            print(f"    XDNA: {xr.get('ms_med', 'N/A')} ms")
        vr = bench_vulkan(M, K, N)
        if vr:
            vr["role"] = role
            vk_results.append(vr)
            print(f"    VK:   {vr.get('ms', 'N/A')} ms")

    with open(f"{PHASE32}/results/xdna_sweep.csv", "w") as f:
        f.write("tag,role,M,K,N,cols,ms_med,ms_min,gops,status\n")
        for r in xdna_results:
            if r.get("status") == "ok":
                f.write(f"{r['tag']},{r['role']},{r['M']},{r['K']},{r['N']},{r.get('cols',1)},{r['ms_med']:.4f},{r['ms_min']:.4f},{r['gops']:.2f},{r['status']}\n")
            else:
                f.write(f"NA,{r['role']},{r['M']},{r['K']},{r['N']},0,0,0,0,{r['status']}:{r.get('reason','')[:50]}\n")

    with open(f"{PHASE32}/results/vk_sweep.csv", "w") as f:
        f.write("tag,role,M,K,N,ms,gops,status\n")
        for r in vk_results:
            if r.get("status") == "ok":
                f.write(f"{r['tag']},{r['role']},{r['M']},{r['K']},{r['N']},{r['ms']:.4f},{r['gops']:.2f},{r['status']}\n")
            else:
                f.write(f"NA,{r['role']},{r['M']},{r['K']},{r['N']},0,0,{r['status']}:{r.get('reason','')[:50]}\n")

    # Phase 2: persistent chain benchmark for selected shapes
    print("[phase2] persistent chain benchmark...")
    chain_shapes = [
        (512, 1024, 3072, 4),   # ffn_gate PP512
        (512, 3072, 1024, 4),   # ffn_down PP512
        (512, 1024, 1024, 4),   # q_proj PP512
        (1024, 1024, 3072, 4),  # ffn_gate PP1K
        (1024, 3072, 1024, 4),  # ffn_down PP1K
        (2048, 1024, 3072, 4),  # ffn_gate PP2K
        (64, 1024, 3072, 8),    # small-M FFN
        (64, 1024, 1024, 8),    # small-M Q proj
        (32, 1024, 3072, 8),    # small-M FFN
        (16, 1024, 3072, 8),    # small-M FFN
        (8, 1024, 3072, 8),     # small-M FFN
        (4, 1024, 3072, 8),     # small-M FFN
        (1, 1024, 3072, 8),     # decode FFN
    ]
    chain_results = []
    for M, K, N, G in chain_shapes:
        print(f"  chain M={M} K={K} N={N} G={G}")
        cr = bench_chain(M, K, N, G)
        if cr and cr.get("status") == "ok":
            chain_results.append(cr)
            print(f"    cold={cr['cold_total']:.2f}ms persist={cr['persist_total']:.2f}ms speedup={cr['speedup']:.2f}x")

    with open(f"{PHASE32}/results/chain.csv", "w") as f:
        f.write("M,K,N,G,cold_total_ms,cold_w_ms,cold_in_ms,cold_exec_ms,cold_out_ms,"
                "persist_total_ms,persist_w_ms,persist_in_ms,persist_exec_ms,persist_out_ms,speedup,status\n")
        for r in chain_results:
            f.write(f"{r['M']},{r['K']},{r['N']},{r['G']},"
                    f"{r['cold_total']:.4f},{r['cold_w']:.4f},{r['cold_in']:.4f},"
                    f"{r['cold_exec']:.4f},{r['cold_out']:.4f},"
                    f"{r['persist_total']:.4f},{r['persist_w']:.4f},{r['persist_in']:.4f},"
                    f"{r['persist_exec']:.4f},{r['persist_out']:.4f},"
                    f"{r['speedup']:.3f},ok\n")

    print(f"[done] xdna={len(xdna_results)} vk={len(vk_results)} chain={len(chain_results)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
