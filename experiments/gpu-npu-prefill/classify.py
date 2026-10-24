#!/usr/bin/env python3
"""classify.py -- Operation classifier for GPU-NPU prefill partitioning.

Classifies each transformer operation into:
  Class A: GPU much faster -> keep on GPU
  Class B: NPU slower but can hide in GPU window -> candidate for async overlap
  Class C: NPU slower and causes GPU sync -> reject offload
  Class D: NPU and GPU can overlap, NPU result before dependent GPU op -> best candidate

Input: dict with gpu and npu results
Output: classification dict
"""
import json, sys, math


def classify_operation(op_name, M, K, N, gpu, npu):
    """Classify a single operation.
    
    Args:
        op_name: operation name (GEMM, ELEMENTWISE, etc.)
        M, K, N: matrix dimensions
        gpu: dict with compute_ms, total_ms, h2d_ms, d2h_ms
        npu: dict with compute_ms (npu_median_us/1000), overhead_ms, etc.
    
    Returns:
        classification dict
    """
    gpu_compute = gpu["compute_ms"]
    gpu_total = gpu["total_ms"]
    gpu_h2d = gpu.get("h2d_ms", 0)
    gpu_d2h = gpu.get("d2h_ms", 0)

    npu_compute = npu.get("compute_ms", 0)
    npu_total = npu.get("total_ms", 0)
    npu_h2d = npu.get("h2d_ms", 0)
    npu_d2h = npu.get("d2h_ms", 0)
    npu_overhead = npu.get("overhead_ms", 0)

    if gpu_compute <= 0 or npu_compute <= 0:
        return {
            "op": op_name, "M": M, "K": K, "N": N,
            "class": "UNKNOWN",
            "reason": "Missing timing data",
        }

    # Speedup ratio: how much faster is GPU vs NPU (compute only)
    speedup_ratio = gpu_compute / npu_compute

    # For async overlap, NPU compute must fit in GPU execution window
    # plus preparation time
    npu_available_window = gpu_compute  # GPU execution time is the window
    hiding_ratio = min(1.0, npu_compute / npu_available_window) if npu_available_window > 0 else 0

    # Total sequential time
    t_seq = gpu_total + npu_total

    # Ideal async overlap: max(gpu_total, npu_total + prep_time)
    # where prep_time is the time to prepare NPU data before/during GPU compute
    npu_prep = npu_h2d + npu_overhead
    t_async_ideal = max(gpu_total, npu_compute + npu_d2h + npu_prep)

    # Real async overlap includes synchronization overhead
    sync_overhead_ms = 0.1  # conservative estimate for event signaling
    t_async_real = max(gpu_total, npu_compute + npu_d2h + npu_prep) + sync_overhead_ms

    # GPU slowdown under async: does GPU have to wait for NPU?
    gpu_slowdown = max(0, t_async_real - gpu_total) / gpu_total * 100 if gpu_total > 0 else 0

    # NPU overhead fraction
    npu_overhead_frac = (npu_h2d + npu_d2h + npu_overhead) / npu_total * 100 if npu_total > 0 else 0

    # Classification logic
    if speedup_ratio >= 3.0:
        # GPU is 3x+ faster -> Class A
        cls = "A"
        reason = f"GPU {speedup_ratio:.1f}x faster than NPU"
    elif speedup_ratio >= 1.0 and hiding_ratio >= 0.8:
        # NPU slower but mostly hidden in GPU window -> Class B
        cls = "B"
        reason = f"NPU {1/speedup_ratio:.1f}x slower but {hiding_ratio*100:.0f}% hidden"
    elif speedup_ratio < 1.0 and hiding_ratio >= 0.8:
        # NPU faster and hidden -> Class D (best case)
        cls = "D"
        reason = f"NPU {speedup_ratio:.2f}x faster, {hiding_ratio*100:.0f}% hidden"
    elif speedup_ratio >= 1.0 and hiding_ratio >= 0.5:
        # NPU slower, partially hidden -> Class B
        cls = "B"
        reason = f"NPU {1/speedup_ratio:.1f}x slower, {hiding_ratio*100:.0f}% hidden"
    elif speedup_ratio < 1.0 and hiding_ratio >= 0.5:
        # NPU faster, partially hidden -> Class D
        cls = "D"
        reason = f"NPU {speedup_ratio:.2f}x faster, {hiding_ratio*100:.0f}% hidden"
    elif npu_overhead_frac > 50:
        # NPU has high overhead -> Class C
        cls = "C"
        reason = f"NPU overhead {npu_overhead_frac:.0f}% of total time, not worth offload"
    else:
        # NPU slower, not hidden -> Class C
        cls = "C"
        reason = f"NPU {1/speedup_ratio:.1f}x slower, not hidden ({hiding_ratio*100:.0f}%)"

    return {
        "op": op_name,
        "M": M, "K": K, "N": N,
        "class": cls,
        "reason": reason,
        "gpu_compute_ms": gpu_compute,
        "gpu_total_ms": gpu_total,
        "npu_compute_ms": npu_compute,
        "npu_total_ms": npu_total,
        "npu_overhead_ms": npu_overhead,
        "speedup_ratio": speedup_ratio,
        "hiding_ratio": hiding_ratio,
        "gpu_slowdown_pct": gpu_slowdown,
        "npu_overhead_pct": npu_overhead_frac,
        "t_seq_ms": t_seq,
        "t_async_ideal_ms": t_async_ideal,
        "t_async_real_ms": t_async_real,
        "speedup_vs_seq": t_seq / t_async_real if t_async_real > 0 else 0,
    }


def summarize_classifications(classifications):
    """Summarize classification results."""
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "UNKNOWN": 0}
    for c in classifications:
        cls = c.get("class", "UNKNOWN")
        counts[cls] = counts.get(cls, 0) + 1

    best_d = [c for c in classifications if c.get("class") == "D"]
    best_b = [c for c in classifications if c.get("class") == "B"]
    rejected = [c for c in classifications if c.get("class") in ("A", "C")]

    return {
        "counts": counts,
        "best_d_candidates": sorted(best_d, key=lambda x: x.get("speedup_vs_seq", 0), reverse=True),
        "best_b_candidates": sorted(best_b, key=lambda x: x.get("hiding_ratio", 0), reverse=True),
        "rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON file with benchmark results")
    parser.add_argument("--output", required=True, help="Output JSON file for classifications")
    args = parser.parse_args()

    import argparse
    with open(args.input) as f:
        data = json.load(f)

    classifications = []
    for entry in data.get("candidates", []):
        op = entry.get("op", "UNKNOWN")
        M = entry.get("M", 0)
        K = entry.get("K", 0)
        N = entry.get("N", 0)
        gpu = entry.get("gpu", {})
        npu = entry.get("npu", {})

        if gpu.get("status") != "OK" or npu.get("status") not in ("OK", "ESTIMATED"):
            continue

        cls = classify_operation(op, M, K, N, gpu, npu)
        classifications.append(cls)

    summary = summarize_classifications(classifications)
    result = {
        "classifications": classifications,
        "summary": summary,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Classified {len(classifications)} operations", flush=True)
    print(f"Counts: {summary['counts']}", flush=True)
    if summary["best_d_candidates"]:
        print(f"Best D candidates: {[c['op']+str(c['M']) for c in summary['best_d_candidates'][:5]]}", flush=True)


if __name__ == "__main__":
    import argparse
    main()
