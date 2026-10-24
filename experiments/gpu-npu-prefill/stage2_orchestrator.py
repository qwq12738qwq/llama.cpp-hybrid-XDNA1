#!/usr/bin/env python3
"""
stage2_orchestrator.py -- Stage 2: Transformer Prefill Task Partitioning

Runs GPU and NPU benchmarks for all prefill candidate operations,
classifies them, builds execution models, and prints verdict summary.
"""
import argparse, subprocess, json, sys, time, os, statistics

DEFAULT_M_VALUES = [1024, 4096, 8192, 16384, 32768, 65536]

OPERATIONS = [
    ("Q_proj",   "GEMM",       "Q projection: [M,4096] @ [4096,4096]"),
    ("K_proj",   "GEMM",       "K projection: [M,4096] @ [4096,4096]"),
    ("V_proj",   "GEMM",       "V projection: [M,4096] @ [4096,4096]"),
    ("O_proj",   "GEMM",       "O projection: [M,4096] @ [4096,4096]"),
    ("QK_t",     "GEMM",       "QK^T per head: [M,128] @ [128,M]"),
    ("Attn_V",   "GEMM",       "Attn x V per head: [M,M] @ [M,128]"),
    ("FFN_gate", "GEMM",       "FFN gate: [M,14336] @ [14336,4096]"),
    ("FFN_up",   "GEMM",       "FFN up: [M,14336] @ [14336,4096]"),
    ("FFN_down", "GEMM",       "FFN down: [M,4096] @ [4096,14336]"),
    ("RMSNorm",     "ELEMENTWISE", "RMSNorm: [M,4096] elementwise"),
    ("Activation",  "ELEMENTWISE", "SwiGLU activation: [M,4096] elementwise"),
]


def _detect_container_env():
    try:
        subprocess.run(["podman", "--version"], capture_output=True, check=True, timeout=5)
        return "podman"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        subprocess.run(["./gpu_bench", "--M", "64", "--K", "64", "--N", "64", "--iters", "1", "--warmup", "0"],
                       capture_output=True, cwd="/experiments/gpu-npu-prefill", timeout=10)
        return "direct"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


EXEC_MODE = _detect_container_env()
print(f"Execution mode: {EXEC_MODE}", flush=True)

CONTAINER_NAME = "zen_overlap"
NPU_VENV = "/root/src/mlir-aie-venv/bin/activate"


def get_op_shape(op_name, M):
    if op_name in ("QK_t", "Attn_V"):
        K = 128
        N = M if op_name == "QK_t" else 128
        return K, N
    elif op_name in ("Q_proj", "K_proj", "V_proj", "O_proj"):
        return 4096, 4096
    elif op_name in ("FFN_gate", "FFN_up"):
        return 14336, 4096
    elif op_name == "FFN_down":
        return 4096, 14336
    else:
        return 0, 4096


def _estimate_gpu_elementwise(M, N):
    total_elements = M * N
    total_bytes = total_elements * 4 * 4  # 4 passes * 4 bytes
    gpu_bw_gbps = 500.0  # Radeon 780M memory bandwidth
    compute_ms = (total_bytes / 1e9) / gpu_bw_gbps * 1000.0
    return {
        "status": "ESTIMATED", "compute_ms": compute_ms,
        "total_ms": compute_ms, "h2d_ms": 0, "d2h_ms": 0,
    }


def run_gpu_bench(M, K, N, iters=3):
    if EXEC_MODE == "direct":
        cmd = ["bash", "-c", f"cd /experiments/gpu-npu-prefill && ./gpu_bench --M {M} --K {K} --N {N} --iters {iters} --warmup 1"]
    else:
        cmd = ["podman", "exec", CONTAINER_NAME, "bash", "-c",
               f"cd /experiments/gpu-npu-prefill && ./gpu_bench --M {M} --K {K} --N {N} --iters {iters} --warmup 1"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return {"status": "ERROR", "error": result.stderr[:500]}
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                data["status"] = "OK"
                return data
        return {"status": "PARSE_ERROR", "error": "No JSON found"}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def _parse_json_from_text(text):
    start = -1
    for i in range(len(text)):
        if text[i] == '{':
            start = i
            break
    if start < 0:
        return None
    brace = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == '\\' and in_str:
            esc = True
            continue
        if c == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == '{':
            brace += 1
        elif c == '}':
            brace -= 1
            if brace == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def run_npu_bench(op, M, K, N, iters=3):
    if op in ("RMSNorm", "Activation"):
        total_elements = M * N
        total_bytes = total_elements * 4
        npu_bw_gbps = 10.0
        compute_ms = (total_bytes / 1e9) / npu_bw_gbps * 1000.0
        return {
            "status": "ESTIMATED", "op": op, "M": M, "N": N,
            "compute_ms": compute_ms, "h2d_ms": 0.05, "d2h_ms": 0.05,
            "total_ms": compute_ms + 0.1, "overhead_ms": 0.05,
            "note": "Estimated from memory bandwidth"
        }

    if EXEC_MODE == "direct":
        cmd = ["bash", "-c",
               f"source {NPU_VENV} && cd /experiments/gpu-npu-prefill && "
               f"python3 npu_candidate_bench.py --op GEMM --M {M} --K {K} --N {N} --iters {iters}"]
    else:
        cmd = ["podman", "exec", CONTAINER_NAME, "bash", "-c",
               f"source {NPU_VENV} && cd /experiments/gpu-npu-prefill && "
               f"python3 npu_candidate_bench.py --op GEMM --M {M} --K {K} --N {N} --iters {iters}"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        combined = result.stdout + result.stderr
        data = _parse_json_from_text(combined)
        if data and data.get("status") == "OK":
            warm = data.get("warm", {})
            return {
                "status": "OK", "op": "GEMM", "M": M, "K": K, "N": N,
                "compute_ms": warm.get("npu_median_us", 0) / 1000.0,
                "total_ms": warm.get("e2e_median_us", 0) / 1000.0,
                "h2d_ms": warm.get("h2d_median_us", 0) / 1000.0,
                "d2h_ms": warm.get("d2h_median_us", 0) / 1000.0,
                "overhead_ms": warm.get("overhead_median_us", 0) / 1000.0,
                "gops": data.get("gops_npu", 0),
            }
        if data and data.get("status") == "BLOCKED":
            return {"status": "BLOCKED", "op": "GEMM", "M": M, "K": K, "N": N,
                    "error": data.get("error", ""), "note": data.get("note", "")}
        if "BLOCKED" in combined or "exceeds DMA" in combined:
            return {"status": "BLOCKED", "op": "GEMM", "M": M, "K": K, "N": N,
                    "error": "K > 512 exceeds DMA BD limit",
                    "note": "Would require tiling K into 512-wide chunks"}
        return {"status": "ERROR", "error": result.stderr[:500] if result.stderr else "Unknown error (no JSON)"}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def _estimate_tiled_npu(M, K, N, tile_K=512):
    base_ms = 17.2 * (M / 128.0)  # baseline M=128, K=512
    num_tiles = (K + tile_K - 1) // tile_K
    overhead_per_tile = 0.5
    return num_tiles * base_ms + (num_tiles - 1) * overhead_per_tile


def run_candidate(op_name, op_type, desc, M, K, N, iters, skip_npu, gpu_cache, npu_cache):
    print(f"\n{'='*60}", flush=True)
    print(f"Candidate: {op_name}  M={M} K={K} N={N}  ({desc})", flush=True)
    print(f"{'='*60}", flush=True)

    cache_key = (op_name, M, K, N)

    if op_type != "GEMM":
        gpu = _estimate_gpu_elementwise(M, N)
        print(f"  GPU: ESTIMATED {gpu['compute_ms']:.3f}ms (elementwise)", flush=True)
    elif cache_key in gpu_cache:
        gpu = gpu_cache[cache_key]
        print(f"  GPU: CACHED {gpu.get('status', 'UNKNOWN')}", flush=True)
    else:
        print(f"  GPU: running benchmark...", flush=True)
        gpu = run_gpu_bench(M, K, N, iters)
        gpu_cache[cache_key] = gpu
        print(f"  GPU: {gpu.get('status', 'UNKNOWN')}  compute={gpu.get('compute_ms',0):.1f}ms", flush=True)

    if skip_npu:
        npu = {"status": "SKIPPED", "compute_ms": 0, "total_ms": 0}
    elif cache_key in npu_cache:
        npu = npu_cache[cache_key]
        print(f"  NPU: CACHED {npu.get('status', 'UNKNOWN')}", flush=True)
    else:
        print(f"  NPU: running benchmark...", flush=True)
        npu = run_npu_bench(op_name, M, K, N, iters)
        npu_cache[cache_key] = npu
        if npu.get("status") == "BLOCKED":
            print(f"  NPU: BLOCKED - {npu.get('error', '')}", flush=True)
            if K > 512 and op_type == "GEMM":
                est = _estimate_tiled_npu(M, K, N)
                npu["estimated_tiled_ms"] = est
                print(f"  NPU: TILED ESTIMATE = {est:.1f}ms ({K//512} tiles)", flush=True)
        elif npu.get("status") == "OK":
            print(f"  NPU: {npu['compute_ms']:.1f}ms compute, {npu['total_ms']:.1f}ms total", flush=True)
        else:
            print(f"  NPU: {npu.get('status', 'UNKNOWN')} - {npu.get('error', '')}", flush=True)

    return {"op": op_name, "op_type": op_type, "desc": desc, "M": M, "K": K, "N": N,
            "gpu": gpu, "npu": npu}


def classify_and_model(all_candidates, M):
    ops_m = [c for c in all_candidates if c.get("M") == M]
    classifications = []
    gpu_only_total = 0.0
    gpu_work = []
    npu_work = []

    for c in ops_m:
        gpu = c.get("gpu", {})
        npu = c.get("npu", {})
        if gpu.get("status") not in ("OK", "ESTIMATED"):
            continue

        gpu_compute = gpu.get("compute_ms", 0)
        gpu_total = gpu.get("total_ms", 0)
        gpu_only_total += gpu_total

        if npu.get("status") == "BLOCKED":
            npu_compute = npu.get("estimated_tiled_ms", float('inf'))
            npu_total = npu_compute + npu.get("overhead_ms", 0)
        elif npu.get("status") in ("OK", "ESTIMATED"):
            npu_compute = npu.get("compute_ms", 0)
            npu_total = npu.get("total_ms", 0)
        else:
            npu_compute = float('inf')
            npu_total = float('inf')

        if npu_compute == float('inf') or npu_compute <= 0:
            cls = "A"
            reason = "NPU blocked or unavailable"
        else:
            speedup = gpu_compute / npu_compute
            hiding = min(1.0, gpu_compute / npu_compute) if npu_compute > 0 else 0

            if speedup >= 3.0:
                cls = "A"
                reason = f"GPU {speedup:.1f}x faster"
            elif hiding >= 0.8:
                cls = "D" if speedup < 1.0 else "B"
                reason = f"NPU {'faster' if speedup < 1 else 'slower'}, {hiding*100:.0f}% hidden"
            elif hiding >= 0.5:
                cls = "D" if speedup < 1.0 else "B"
                reason = f"NPU {'faster' if speedup < 1 else 'slower'}, {hiding*100:.0f}% hidden"
            else:
                cls = "C"
                reason = f"NPU not hidden ({hiding*100:.0f}%)"

        classifications.append({
            "op": c["op"], "M": M, "K": c.get("K", 0), "N": c.get("N", 0),
            "class": cls, "reason": reason,
            "gpu_compute_ms": gpu_compute, "gpu_total_ms": gpu_total,
            "npu_compute_ms": npu_compute if npu_compute != float('inf') else -1,
            "npu_total_ms": npu_total if npu_total != float('inf') else -1,
        })

        if cls == "A" or cls == "C":
            gpu_work.append(gpu_total)
        elif cls in ("B", "D"):
            gpu_work.append(gpu_total)
            npu_work.append(npu_total + npu.get("h2d_ms", 0) + npu.get("overhead_ms", 0))

    model_a = gpu_only_total
    model_b = model_a
    for c in classifications:
        if c["class"] in ("B", "D") and c["npu_total_ms"] > 0:
            model_b = model_b - c["gpu_total_ms"] + c["npu_total_ms"]

    total_gpu_work = sum(gpu_work)
    total_npu_work = max(npu_work) if npu_work else 0
    sync_overhead = 0.5 * len([c for c in classifications if c["class"] in ("B", "D")])
    model_c = max(total_gpu_work, total_npu_work) + sync_overhead

    return {
        "classifications": classifications,
        "model": {
            "M": M, "gpu_only_ms": model_a, "sync_ms": model_b,
            "async_ms": model_c, "sync_overhead_ms": sync_overhead,
            "npu_offload_ops": len([c for c in classifications if c["class"] in ("B", "D")]),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M-values", default=",".join(map(str, DEFAULT_M_VALUES)))
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--skip-npu", action="store_true")
    parser.add_argument("--output", default="/experiments/gpu-npu-prefill/stage2_results.json")
    args = parser.parse_args()

    M_values = [int(x.strip()) for x in args.M_values.split(",")]
    iters = args.iters
    skip_npu = args.skip_npu

    print("="*70, flush=True)
    print("Stage 2: Transformer Prefill Task Partitioning Benchmark", flush=True)
    print("="*70, flush=True)
    print(f"M values: {M_values}", flush=True)
    print(f"Iters: {iters}", flush=True)

    all_candidates = []
    gpu_cache = {}
    npu_cache = {}

    for M in M_values:
        if M > 32768:
            print(f"\n--- M={M}: using estimated GPU times (memory limit) ---", flush=True)
            for op_name, op_type, desc in OPERATIONS:
                K, N = get_op_shape(op_name, M)
                if op_type == "GEMM" and (M % 64 != 0 or K % 64 != 0 or N % 64 != 0):
                    continue
                base = run_gpu_bench(32768, K, N, iters)
                if base.get("status") == "OK":
                    scale = M / 32768.0
                    est_gpu = {
                        "status": "ESTIMATED",
                        "compute_ms": base["compute_ms"] * scale,
                        "total_ms": base["total_ms"] * scale,
                        "h2d_ms": base["h2d_ms"] * scale,
                        "d2h_ms": base["d2h_ms"] * scale,
                        "gops": base["gops"],
                    }
                    all_candidates.append({
                        "op": op_name, "op_type": op_type, "desc": desc,
                        "M": M, "K": K, "N": N,
                        "gpu": est_gpu,
                        "npu": {"status": "BLOCKED", "compute_ms": float('inf'),
                                "total_ms": float('inf'), "note": "M too large for NPU"},
                    })
            continue

        for op_name, op_type, desc in OPERATIONS:
            K, N = get_op_shape(op_name, M)
            if op_type == "GEMM" and (M % 64 != 0 or K % 64 != 0 or N % 64 != 0):
                print(f"  SKIP {op_name} M={M}: shape not divisible by 64", flush=True)
                continue
            candidate = run_candidate(op_name, op_type, desc, M, K, N, iters, skip_npu, gpu_cache, npu_cache)
            all_candidates.append(candidate)

    print("\n" + "="*70, flush=True)
    print("Classification & Execution Models", flush=True)
    print("="*70, flush=True)

    all_results = {}
    for M in M_values:
        result = classify_and_model(all_candidates, M)
        all_results[M] = result
        for c in result["classifications"]:
            marker = {"A": "[GPU]", "B": "[NPU-HIDE]", "C": "[REJECT]", "D": "[NPU-WIN]"}.get(c["class"], "[??]")
            print(f"  {marker} {c['op']:12s} M={c['M']:>6} K={c['K']:>5} N={c['N']:>6} "
                  f"GPU={c['gpu_compute_ms']:>8.1f}ms NPU={c['npu_compute_ms']:>8.1f}ms -> {c['class']}", flush=True)
        m = result["model"]
        speedup = m["gpu_only_ms"] / m["async_ms"] if m["async_ms"] > 0 else 0
        print(f"\n  M={M:>6}: GPU-only={m['gpu_only_ms']:>8.1f}ms  Sync={m['sync_ms']:>8.1f}ms  "
              f"Async={m['async_ms']:>8.1f}ms  Speedup={speedup:>5.2f}x  NPU_ops={m['npu_offload_ops']}", flush=True)

    best_speedup = 0
    best_M = None
    for M, result in all_results.items():
        m = result["model"]
        if m["gpu_only_ms"] > 0:
            sp = m["gpu_only_ms"] / m["async_ms"]
            if sp > best_speedup:
                best_speedup = sp
                best_M = M

    print("\n" + "="*70, flush=True)
    print("Verdict", flush=True)
    print("="*70, flush=True)
    if best_speedup > 1.05:
        verdict = "BENEFICIAL"
        detail = f"Async overlap provides {best_speedup:.2f}x speedup at M={best_M}"
    elif best_speedup > 0.95:
        verdict = "NEUTRAL"
        detail = f"Async overlap is roughly break-even ({best_speedup:.2f}x at M={best_M})"
    else:
        verdict = "HARMFUL"
        detail = f"Async overlap is slower ({best_speedup:.2f}x at M={best_M})"

    print(f"\n  Best async speedup: {best_speedup:.2f}x at M={best_M}", flush=True)
    print(f"  Verdict: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    result = {
        "config": {"M_values": M_values, "iters": iters, "skip_npu": skip_npu},
        "candidates": all_candidates,
        "results": {str(M): r for M, r in all_results.items()},
        "verdict": {"best_M": best_M, "best_speedup": best_speedup,
                    "verdict": verdict, "detail": detail},
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)

    return 0 if verdict == "BENEFICIAL" else 1


if __name__ == "__main__":
    sys.exit(main())
