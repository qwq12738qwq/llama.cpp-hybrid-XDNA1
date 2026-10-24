#!/usr/bin/env python3
"""Phase 22: frozen Vulkan-only prompt-processing baseline.

Runs llama-bench (build-vk) on the real Qwen3.5-0.8B-Q8_0 model for a list of
prompt sizes, parses the per-rep samples from the JSON report and writes
median / p10 / p90 to results/phase22_baseline_vulkan.csv.

Usage:
  python bench_pp.py --bench build-vk/bin/llama-bench \
      --model /Storage/Cache/Models/Qwen3.5-0.8B-Q8_0.gguf \
      -p 512,2048,8192 --reps 5 --out results/phase22_baseline_vulkan.csv
"""
import argparse
import json
import os
import statistics
import subprocess


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="build-vk/bin/llama-bench")
    ap.add_argument("--model", required=True)
    ap.add_argument("-p", default="512,2048,8192")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--tag", default="vulkan_only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()

    pps = [int(x) for x in a.p.split(",")]
    rows = []
    for pp in pps:
        cmd = [a.bench, "-m", a.model, "-p", str(pp), "-n", "0",
               "-ngl", str(a.ngl), "-ub", str(a.ubatch),
               "-r", str(a.reps), "-o", "json"]
        if a.extra:
            cmd += a.extra.split()
        print("RUN", " ".join(cmd), flush=True)
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        js = json.loads(cp.stdout)
        rec = js[0]
        ts = rec["samples_ts"]
        med = statistics.median(ts)
        rows.append((a.tag, pp, a.reps, med, pct(ts, 0.10), pct(ts, 0.90),
                     rec["avg_ns"] / 1e6, 1000.0 / med if med else 0.0))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("tag,pp,reps,median_ts,p10_ts,p90_ts,avg_ms_per_run,ms_per_token\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f},"
                    f"{r[6]:.3f},{r[7]:.4f}\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
