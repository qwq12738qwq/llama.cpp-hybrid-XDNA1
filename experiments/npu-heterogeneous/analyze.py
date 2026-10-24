#!/usr/bin/env python3
"""analyze.py -- Phase 7A data analysis and reporting.

Loads results from experiments/npu-heterogeneous/results/,
computes summary statistics, and generates characterization data.

Usage:
    python3 analyze.py
    python3 analyze.py --results-dir /path/to/results
    python3 analyze.py --plot
"""
import argparse, json, os, sys, statistics
from pathlib import Path

def _stats(samples):
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0], "mean": sum(samples) / len(samples),
        "median": s[n // 2],
        "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)], "max": s[-1],
    }

def _safe_median(d, key):
    v = d.get(key, {})
    if isinstance(v, dict):
        return v.get("median", 0)
    return v if v is not None else 0

def load_results(results_dir):
    results = {}
    for f in Path(results_dir).glob("*.json"):
        if f.name.startswith("set_") or f.name == "sweep_summary.json":
            continue
        if "overlap" in f.name:
            continue
        if f.name.endswith("_gpu.json"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            label = data.get("label", f.stem)
            results[label] = data
        except Exception:
            pass
    return results

def load_overlap_results(results_dir):
    results = {}
    for f in Path(results_dir).glob("*_overlap.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            label = f.stem.replace("_overlap", "")
            results[label] = data
        except Exception:
            pass
    return results

def analyze_shape(data):
    M, K, N = data["M"], data["K"], data["N"]
    label = data.get("label", f"M{M}_K{K}_N{N}")

    cpu = data.get("cpu", {})
    gpu = data.get("gpu", {})
    npu = data.get("npu", {})
    xfer = data.get("transfer", {})

    cpu_compute = _safe_median(cpu, "compute_ms_median")
    cpu_total = _safe_median(cpu, "total_ms_median")

    gpu_compute = _safe_median(gpu, "compute_ms")
    gpu_total = _safe_median(gpu, "total_ms")
    gpu_h2d = _safe_median(gpu, "h2d_ms")
    gpu_d2h = _safe_median(gpu, "d2h_ms")

    npu_compute = _safe_median(npu, "npu_ms_median")
    npu_wall = _safe_median(npu, "wall_ms_median")
    npu_h2d = _safe_median(npu, "h2d_ms_median")
    npu_d2h = _safe_median(npu, "d2h_ms_median")
    npu_overhead = _safe_median(npu, "overhead_ms_median")
    npu_mm = npu.get("mismatches", -1)

    xfer_h2d = _safe_median(xfer, "h2d_A_us") / 1000.0 if xfer.get("status") == "OK" else 0
    xfer_d2h = _safe_median(xfer, "d2h_A_us") / 1000.0 if xfer.get("status") == "OK" else 0

    ops = 2.0 * M * K * N

    return {
        "label": label, "M": M, "K": K, "N": N,
        "cpu_compute_ms": cpu_compute,
        "cpu_total_ms": cpu_total,
        "gpu_compute_ms": gpu_compute,
        "gpu_total_ms": gpu_total,
        "gpu_h2d_ms": gpu_h2d,
        "gpu_d2h_ms": gpu_d2h,
        "npu_compute_ms": npu_compute,
        "npu_wall_ms": npu_wall,
        "npu_h2d_ms": npu_h2d,
        "npu_d2h_ms": npu_d2h,
        "npu_overhead_ms": npu_overhead,
        "npu_mismatches": npu_mm,
        "xfer_h2d_ms": xfer_h2d,
        "xfer_d2h_ms": xfer_d2h,
        "gops_cpu": ops / (cpu_compute * 1e-3) / 1e9 if cpu_compute > 0 else 0,
        "gops_gpu": ops / (gpu_compute * 1e-3) / 1e9 if gpu_compute > 0 else 0,
        "gops_npu": ops / (npu_compute * 1e-3) / 1e9 if npu_compute > 0 else 0,
        "status_cpu": cpu.get("status", "MISSING"),
        "status_gpu": gpu.get("status", "MISSING"),
        "status_npu": npu.get("status", "MISSING"),
    }

def classify_candidate(row):
    gpu_compute = row["gpu_compute_ms"]
    npu_compute = row["npu_compute_ms"]
    npu_wall = row["npu_wall_ms"]
    npu_h2d = row["npu_h2d_ms"]
    npu_overhead = row["npu_overhead_ms"]
    npu_total_e2e = npu_wall  # already includes h2d + compute + d2h + overhead

    if row["status_gpu"] != "OK" or row["status_npu"] != "OK":
        return "UNKNOWN", "Missing data"

    if npu_compute <= 0:
        return "UNKNOWN", "NPU compute is zero"

    speedup = gpu_compute / npu_compute
    hiding = min(1.0, gpu_compute / npu_wall) if npu_wall > 0 else 0

    if npu_wall >= gpu_compute * 1.5:
        return "A", f"NPU wall {npu_wall:.1f}ms > 1.5x GPU {gpu_compute:.1f}ms"
    elif hiding >= 0.8:
        if speedup >= 1.0:
            return "B", f"NPU hides {hiding*100:.0f}% of GPU, {speedup:.1f}x faster"
        else:
            return "D", f"NPU hides {hiding*100:.0f}% of GPU, {speedup:.1f}x slower"
    elif hiding >= 0.3:
        if speedup >= 1.0:
            return "B", f"NPU partial hide {hiding*100:.0f}%, {speedup:.1f}x faster"
        else:
            return "C", f"NPU partial hide {hiding*100:.0f}%, {speedup:.1f}x slower"
    else:
        return "C", f"NPU hides only {hiding*100:.0f}% of GPU"

def generate_report(all_rows, overlap_data):
    lines = []
    lines.append("# Phase 7A: MUL_MAT Operator Characterization Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    total = len(all_rows)
    ok_count = sum(1 for r in all_rows if r["status_cpu"] == "OK" and r["status_gpu"] == "OK")
    npu_ok = sum(1 for r in all_rows if r["status_npu"] == "OK")
    npu_blocked = sum(1 for r in all_rows if r["status_npu"] == "BLOCKED")

    lines.append(f"- Total shapes tested: {total}")
    lines.append(f"- CPU+GPU both OK: {ok_count}")
    lines.append(f"- NPU OK: {npu_ok}")
    lines.append(f"- NPU BLOCKED: {npu_blocked}")
    lines.append("")

    # Classification counts
    classes = {}
    for r in all_rows:
        cls, reason = classify_candidate(r)
        classes[cls] = classes.get(cls, 0) + 1
        r["class"] = cls
        r["class_reason"] = reason

    lines.append("### Classification Summary")
    lines.append("")
    lines.append("| Class | Meaning | Count |")
    lines.append("|-------|---------|-------|")
    class_desc = {
        "A": "GPU only (NPU too slow)",
        "B": "NPU offload candidate (hidden or faster)",
        "C": "NPU not beneficial",
        "D": "NPU wins (faster despite overhead)",
        "UNKNOWN": "Insufficient data",
    }
    for cls in ["B", "D", "A", "C", "UNKNOWN"]:
        if cls in classes:
            lines.append(f"| {cls} | {class_desc.get(cls, '')} | {classes[cls]} |")
    lines.append("")

    # Detailed table
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| Label | M | K | N | CPU(ms) | GPU(ms) | NPU(ms) | NPU wall(ms) | H2D(ms) | D2H(ms) | MM | Class |")
    lines.append("|-------|---|---|---|---------|---------|---------|--------------|---------|---------|----|-------|")
    for r in sorted(all_rows, key=lambda x: (x["M"], x["K"], x["N"])):
        cls, _ = classify_candidate(r)
        mm = r.get("npu_mismatches", -1)
        mm_str = "PASS" if mm == 0 else f"FAIL({mm})" if mm > 0 else "N/A"
        lines.append(f"| {r['label']} | {r['M']} | {r['K']} | {r['N']} "
                     f"| {r['cpu_compute_ms']:.2f} | {r['gpu_compute_ms']:.2f} "
                     f"| {r['npu_compute_ms']:.2f} | {r['npu_wall_ms']:.2f} "
                     f"| {r['npu_h2d_ms']:.2f} | {r['npu_d2h_ms']:.2f} "
                     f"| {mm_str} | {cls} |")
    lines.append("")

    # Overlap section
    if overlap_data:
        lines.append("## Overlap Results")
        lines.append("")
        for label, od in sorted(overlap_data.items()):
            results = od.get("results", [])
            if not results:
                continue
            walls = [r["overall_wall_ns"] / 1e6 for r in results]
            T_overlap = statistics.median(walls) if walls else 0

            gpu_execs = []
            npu_execs = []
            for r in results:
                gr = r.get("gpu_result")
                nr = r.get("npu_result")
                if gr and "gpu_exec_ns" in gr:
                    gpu_execs.append(gr["gpu_exec_ns"] / 1e6)
                if nr and nr.get("npu_us") is not None:
                    npu_execs.append(nr["npu_us"] / 1000.0)

            T_gpu = statistics.median(gpu_execs) if gpu_execs else 0
            T_npu = statistics.median(npu_execs) if npu_execs else 0
            T_seq = T_gpu + T_npu
            speedup = T_seq / T_overlap if T_overlap > 0 else 0
            efficiency = (T_seq - T_overlap) / min(T_gpu, T_npu) * 100 if T_seq > 0 else 0

            lines.append(f"### {label}")
            lines.append(f"- GPU exec: {T_gpu:.2f}ms")
            lines.append(f"- NPU exec: {T_npu:.2f}ms")
            lines.append(f"- Sequential: {T_seq:.2f}ms")
            lines.append(f"- Overlap: {T_overlap:.2f}ms")
            lines.append(f"- Speedup: {speedup:.2f}x")
            lines.append(f"- Efficiency: {efficiency:.1f}%")
            lines.append("")

    # Insights
    lines.append("## Key Insights")
    lines.append("")

    # Find shapes where NPU is best
    npu_best = [(r["label"], r["npu_wall_ms"], r["gpu_compute_ms"])
                for r in all_rows
                if r["status_npu"] == "OK" and r["npu_wall_ms"] < r["gpu_compute_ms"]]
    if npu_best:
        lines.append("### NPU faster than GPU (wall time)")
        for label, npu_w, gpu_c in sorted(npu_best, key=lambda x: x[1]):
            lines.append(f"- {label}: NPU {npu_w:.2f}ms vs GPU {gpu_c:.2f}ms")
        lines.append("")

    # Find shapes where NPU can hide GPU work
    hide_candidates = [(r["label"], r["npu_wall_ms"], r["gpu_compute_ms"])
                       for r in all_rows
                       if r["status_npu"] == "OK" and r["npu_wall_ms"] <= r["gpu_compute_ms"] * 1.2]
    if hide_candidates:
        lines.append("### NPU can potentially hide GPU work (wall <= 1.2x GPU)")
        for label, npu_w, gpu_c in sorted(hide_candidates, key=lambda x: x[1]):
            ratio = npu_w / gpu_c if gpu_c > 0 else 0
            lines.append(f"- {label}: NPU {npu_w:.2f}ms vs GPU {gpu_c:.2f}ms "
                        f"(ratio: {ratio:.2f})")
        lines.append("")

    # Transfer overhead analysis
    transfer_ok = [r for r in all_rows if r.get("xfer_h2d_ms", 0) > 0]
    if transfer_ok:
        avg_h2d = statistics.mean([r["xfer_h2d_ms"] for r in transfer_ok])
        avg_d2h = statistics.mean([r["xfer_d2h_ms"] for r in transfer_ok])
        lines.append(f"### Transfer Costs (average)")
        lines.append(f"- H2D: {avg_h2d:.3f}ms")
        lines.append(f"- D2H: {avg_d2h:.3f}ms")
        lines.append("")

    # Scaling analysis
    lines.append("### Scaling Observations")
    lines.append("")
    # Group by K
    by_k = {}
    for r in all_rows:
        by_k.setdefault(r["K"], []).append(r)
    for K in sorted(by_k.keys()):
        rows = sorted(by_k[K], key=lambda x: x["M"])
        lines.append(f"**K={K}:**")
        for r in rows:
            if r["status_gpu"] == "OK" and r["status_npu"] == "OK":
                ratio = r["npu_wall_ms"] / r["gpu_compute_ms"] if r["gpu_compute_ms"] > 0 else 0
                lines.append(f"- M={r['M']:>4}: GPU={r['gpu_compute_ms']:.2f}ms  "
                           f"NPU={r['npu_wall_ms']:.2f}ms  ratio={ratio:.2f}")
        lines.append("")

    return "\n".join(lines)

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "results", "phase7a_report.md"))
    parser.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    overlap_data = load_overlap_results(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    all_rows = [analyze_shape(data) for data in results.values()]
    report = generate_report(all_rows, overlap_data)

    with open(args.output, "w") as f:
        f.write(report)

    print(f"Report written to {args.output}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    classes = {}
    for r in all_rows:
        cls, _ = classify_candidate(r)
        classes[cls] = classes.get(cls, 0) + 1
    for cls in ["B", "D", "A", "C", "UNKNOWN"]:
        if cls in classes:
            print(f"  {cls}: {classes[cls]}")
    print("=" * 60)

    # Print top candidates
    candidates = [(r["label"], r["npu_wall_ms"], r["gpu_compute_ms"])
                  for r in all_rows
                  if r["status_npu"] == "OK"]
    if candidates:
        print("\nTOP NPU CANDIDATES (by NPU wall time):")
        for label, npu_w, gpu_c in sorted(candidates, key=lambda x: x[1])[:10]:
            ratio = npu_w / gpu_c if gpu_c > 0 else 0
            print(f"  {label}: NPU={npu_w:.2f}ms  GPU={gpu_c:.2f}ms  ratio={ratio:.2f}")

if __name__ == "__main__":
    main()
