#!/usr/bin/env python3
"""Phase 48: collect the PP scaling matrix logs into one machine-readable CSV.

Reads every <tag>_pp<N>.{bench,trace,verify}.log in results/ and emits one row
per (tag, pp) with every field of the phase 48 table. Run collect_pp_scaling.py
once per sweep; it appends nothing, it rewrites results/pp_scaling.csv.

usage: collect_pp_scaling.py [--results DIR] [--out FILE]
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys
from statistics import median

COLUMNS = [
    # identity
    "tag", "pp", "batch", "ubatch", "n_rounds",
    # throughput
    "vulkan_tps", "vulkan_ms", "hybrid_tps", "hybrid_ms", "hybrid_tps_sd",
    "hybrid_over_vulkan",
    # scheduler topology
    "n_splits", "n_transitions", "n_copies", "copy_bytes", "copy_mb",
    # node placement
    "n_nodes", "n_nodes_gpu", "n_nodes_xdna", "n_nodes_cpu",
    "n_mm_gpu", "n_mm_xdna", "n_mm_cpu", "cpu_fallback_count",
    # copies
    "v_to_x_n", "v_to_x_bytes", "v_to_x_ms",
    "x_to_v_n", "x_to_v_bytes", "x_to_v_ms",
    "c_to_x_n", "c_to_x_bytes", "c_to_x_ms",
    # timing breakdown (ms)
    "sched_wall_ms", "sched_submit_ms", "scheduler_sync_ms", "alloc_ms",
    "graph_build_ms", "set_inputs_ms", "n_ubatch",
    "gpu_busy_ms", "gpu_submit_ms", "gpu_sync_ms",
    "npu_total_ms", "npu_gemm_wall_ms", "npu_host_prep_ms",
    "npu_submit_ms", "npu_run_wait_ms", "npu_sync_from_ms", "npu_copyback_ms",
    "submit_overlap_ms",
    # NPU coverage
    "npu_gemm_count", "npu_gate_count", "npu_up_count", "npu_down_count",
    "npu_shape_M", "npu_shape_K", "npu_shape_N", "npu_shape_uniform",
    # memory
    "peak_rss_kb",
    # correctness
    "md5", "correctness",
]

KV = re.compile(r"([A-Za-z0-9_]+)=([^\s,]+)")


def newest(paths):
    return max(paths, key=os.path.getmtime) if paths else ""


def parse_kv_lines(path, prefix):
    out = []
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith(prefix):
                    continue
                out.append({k: v for k, v in KV.findall(line[len(prefix):])})
    except FileNotFoundError:
        pass
    return out


def parse_trace(path):
    """Aggregate the per-graph-compute instrumentation lines."""
    agg = {}
    sched = parse_kv_lines(path, "[SCHED-SUMMARY] ")
    # only the measured repetition, i.e. the last one
    sched_row = sched[-1] if sched else {}
    for row in sched:
        agg.setdefault("n_summaries", 0)
        agg["n_summaries"] += 1

    vk = parse_kv_lines(path, "[VK-SUMMARY] ")
    vk_row = vk[-1] if vk else {}
    xd = parse_kv_lines(path, "[XDNA-SUMMARY] ")
    ub = parse_kv_lines(path, "[LLAMA-UBATCH] ")

    def f(row, key, default=0.0):
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    def i(row, key, default=0):
        try:
            return int(float(row.get(key, default)))
        except (TypeError, ValueError):
            return default

    out = {}
    out["n_splits"] = i(sched_row, "n_splits")
    out["n_transitions"] = i(sched_row, "n_transitions")
    out["n_copies"] = i(sched_row, "n_copies")
    out["copy_bytes"] = i(sched_row, "copy_bytes")
    out["copy_mb"] = round(out["copy_bytes"] / (1024 * 1024), 3)
    out["n_nodes"] = i(sched_row, "n_nodes")
    out["n_nodes_gpu"] = i(sched_row, "n_nodes_gpu")
    out["n_nodes_xdna"] = i(sched_row, "n_nodes_xdna")
    out["n_nodes_cpu"] = i(sched_row, "n_nodes_cpu")
    out["n_mm_gpu"] = i(sched_row, "n_mm_gpu")
    out["n_mm_xdna"] = i(sched_row, "n_mm_xdna")
    out["n_mm_cpu"] = i(sched_row, "n_mm_cpu")
    out["cpu_fallback_count"] = out["n_mm_cpu"]
    out["v_to_x_n"] = i(sched_row, "v2x_n")
    out["v_to_x_bytes"] = i(sched_row, "v2x_bytes")
    out["v_to_x_ms"] = f(sched_row, "v2x_ms")
    out["x_to_v_n"] = i(sched_row, "x2v_n")
    out["x_to_v_bytes"] = i(sched_row, "x2v_bytes")
    out["x_to_v_ms"] = f(sched_row, "x2v_ms")
    out["c_to_x_n"] = i(sched_row, "c2x_n")
    out["c_to_x_bytes"] = i(sched_row, "c2x_bytes")
    out["c_to_x_ms"] = f(sched_row, "c2x_ms")
    # one [LLAMA-UBATCH] per ubatch of every decode pass; the trace run does a
    # warmup pass plus the measured pass, so len(ub) is the pass count
    n_passes = max(1, len(ub))
    ubatches_per_pass = max(1, len(ub) // 2)
    out["sched_wall_ms"] = f(sched_row, "wall_ms")
    out["sched_submit_ms"] = f(sched_row, "submit_ms")
    out["scheduler_sync_ms"] = f(sched_row, "sync_ms")
    out["alloc_ms"] = f(sched_row, "alloc_ms")
    out["submit_overlap_ms"] = f(sched_row, "overlap_ms")

    out["gpu_busy_ms"] = round(sum(f(r, "busy_ms") for r in vk) / n_passes, 3)
    out["gpu_submit_ms"] = round(sum(f(r, "submit_ms") for r in vk) / n_passes, 3)
    out["gpu_sync_ms"] = round(sum(f(r, "sync_ms") for r in vk) / n_passes, 3)

    # one [XDNA-SUMMARY] per NPU graph_compute; normalise to one ubatch
    if xd:
        def xsum(key):
            return round(sum(f(r, key) for r in xd) / n_passes, 3)

        def xisum(key):
            return round(sum(i(r, key) for r in xd) / n_passes)

        out["npu_gemm_count"] = xisum("n_gemm")
        out["npu_gate_count"] = xisum("gate")
        out["npu_up_count"] = xisum("up")
        out["npu_down_count"] = xisum("down")
        shapes = {(i(r, "M"), i(r, "K"), i(r, "N")) for r in xd if i(r, "n_gemm") > 0}
        out["npu_shape_M"] = max((s[0] for s in shapes), default=0)
        out["npu_shape_K"] = max((s[1] for s in shapes), default=0)
        out["npu_shape_N"] = max((s[2] for s in shapes), default=0)
        out["npu_shape_uniform"] = 1 if len(shapes) <= 1 else 0
        out["npu_total_ms"] = xsum("npu_total_ms")
        out["npu_gemm_wall_ms"] = xsum("gemm_wall_ms")
        out["npu_host_prep_ms"] = xsum("host_prep_ms")
        out["npu_submit_ms"] = xsum("submit_ms")
        out["npu_run_wait_ms"] = xsum("run_wait_ms")
        out["npu_sync_from_ms"] = xsum("sync_from_ms")
        out["npu_copyback_ms"] = xsum("copyback_ms")

    out["n_ubatch"] = ubatches_per_pass
    out["graph_build_ms"] = round(sum(f(r, "graph_build_ms") for r in ub) / n_passes, 3)
    out["set_inputs_ms"] = round(sum(f(r, "set_inputs_ms") for r in ub) / n_passes, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    results = args.results or os.path.join(here, "results")
    out_file = args.out or os.path.join(results, "pp_scaling.csv")

    rows = {}
    md5 = {}
    bench_rows = {}   # (base tag, pp) -> list of bench rows, for round merging

    for path in sorted(glob.glob(os.path.join(results, "*_pp*.bench.log"))):
        name = os.path.basename(path)
        m = re.match(r"(.+)_pp(\d+)\.bench\.log$", name)
        if not m:
            continue
        tag, pp = m.group(1), int(m.group(2))
        # strip the _r<N> interleaving suffix
        base = re.sub(r"_r\d+$", "", tag)
        bench = parse_kv_lines(path, "RESULT,")
        if not bench:
            continue
        b = bench[-1]
        bench_rows.setdefault((base, pp), []).append(b)

    for (base, pp), rounds in sorted(bench_rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = {
            "tag": base,
            "pp": pp,
            "batch": int(rounds[-1].get("batch", pp)),
            "ubatch": int(rounds[-1].get("ubatch", pp)),
            "peak_rss_kb": max(int(float(r.get("peak_rss_kb", 0))) for r in rounds),
        }
        if base == "vk":
            row["vulkan_tps"] = median([float(r.get("pp_tps", 0)) for r in rounds])
            row["vulkan_ms"] = median([float(r.get("pp_ms", 0)) for r in rounds])
        else:
            row["hybrid_tps"] = median([float(r.get("pp_tps", 0)) for r in rounds])
            row["hybrid_ms"] = median([float(r.get("pp_ms", 0)) for r in rounds])
            row["hybrid_tps_sd"] = median([float(r.get("pp_tps_sd", 0)) for r in rounds])
        row["n_rounds"] = len(rounds)
        key = f"{base}:{pp}"
        rows[key] = row

        # metrics come from the last round's trace; they are structural, not timing
        trace_path = newest(glob.glob(os.path.join(results, f"{base}_r*_pp{pp}.trace.err")))
        if not trace_path:
            trace_path = newest(glob.glob(os.path.join(results, f"{base}_pp{pp}.trace.err")))
        if not trace_path:
            trace_path = newest(glob.glob(os.path.join(results, f"{base}_pp{pp}.trace.log")))
        if trace_path:
            rows[key].update(parse_trace(trace_path))

        verify_path = newest(glob.glob(os.path.join(results, f"{base}_r*_pp{pp}.verify.log")))
        if not verify_path:
            verify_path = newest(glob.glob(os.path.join(results, f"{base}_pp{pp}.verify.log")))
        v = parse_kv_lines(verify_path, "VERIFY,")
        if v:
            md5[key] = v[-1].get("md5", "")

    # correctness: hybrid must match the Vulkan reference for the same pp
    for key, row in rows.items():
        tag, pp = key.split(":")
        row["md5"] = md5.get(key, "")
        ref = md5.get(f"vk:{pp}", "")
        if row["md5"] and ref:
            row["correctness"] = "ok" if row["md5"] == ref else "MISMATCH"
        elif row["md5"]:
            row["correctness"] = "no-reference"
        else:
            row["correctness"] = "FAILED"

    # ratio needs the Vulkan reference at the same pp
    for key, row in rows.items():
        _, pp = key.split(":")
        ref = rows.get(f"vk:{pp}")
        if ref and ref.get("vulkan_tps") and row.get("hybrid_tps"):
            row["hybrid_over_vulkan"] = round(row["hybrid_tps"] / ref["vulkan_tps"], 4)

    for c in COLUMNS:
        for row in rows.values():
            row.setdefault(c, "")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows, key=lambda k: (k.split(":")[0], int(k.split(":")[1]))):
            w.writerow(rows[key])

    print(f"wrote {out_file} ({len(rows)} rows)")
    for key in sorted(rows, key=lambda k: (k.split(":")[0], int(k.split(":")[1]))):
        r = rows[key]
        print(f"  {key:>16}  tps={r.get('hybrid_tps') or r.get('vulkan_tps')}"
              f"  splits={r.get('n_splits')} trans={r.get('n_transitions')}"
              f" copies={r.get('n_copies')} v2x_ms={r.get('v_to_x_ms')}"
              f" npu_gemm={r.get('npu_gemm_count')} cpu_mm={r.get('cpu_fallback_count')}"
              f" {r.get('correctness')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())