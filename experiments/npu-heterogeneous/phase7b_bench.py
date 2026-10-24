#!/usr/bin/env python3
"""phase7b_bench.py -- Phase 7B benchmark harness.

Runs llama-bench at multiple context lengths and configurations.
Collects timing data for CPU, GPU (Vulkan), and XDNA backends.

Usage:
    python3 phase7b_bench.py
    python3 phase7b_bench.py --contexts 1024 4096 16384
"""
import argparse, json, os, subprocess, statistics, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LLAMA_BENCH = "/workspace/llama.cpp/build/bin/llama-bench"
MODEL = "/workspace/llama.cpp/models/Qwen3.5-0.8B-Q8_0.gguf"
RESULTS_DIR = SCRIPT_DIR / "results" / "phase7b"

def run_benchmark(ctx_size, ngl, device=None, r=3, warmup=False):
    cmd = [
        LLAMA_BENCH, "-m", MODEL,
        "-p", str(ctx_size), "-n", "1",
        "-r", str(r),
        "-ngl", str(ngl),
        "-b", "2048", "-ub", "512",
    ]
    if device:
        cmd += ["-dev", device]
    if not warmup:
        cmd += ["--no-warmup"]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/workspace/llama.cpp/build/bin"

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "error": "Benchmark timed out"}

    result = {
        "ctx_size": ctx_size,
        "ngl": ngl,
        "device": device or "auto",
        "status": "OK" if proc.returncode == 0 else "FAILED",
        "raw_output": output,
    }

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("|") and "qwen35" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 6:
                try:
                    backend = parts[3]
                    t_str = parts[-1].split("±")[0].strip()
                    test = parts[-2]
                    result["backend"] = backend
                    result["t_per_token"] = float(t_str)
                    result["test"] = test
                except (ValueError, IndexError):
                    pass

    if "backend" not in result:
        for line in output.split("\n"):
            if "backend=" in line or "t/s" in line:
                result["_debug_line"] = line

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", nargs="+", type=int,
                        default=[1024, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--warmup", action="store_true", default=True, help="Enable warmup runs")
    parser.add_argument("--no-warmup", action="store_false", dest="warmup", help="Disable warmup runs")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        {"ngl": 0, "device": None, "label": "cpu"},
        {"ngl": 999, "device": None, "label": "vulkan"},
        {"ngl": 999, "device": "XDNA0", "label": "xdna"},
    ]

    all_results = []
    for ctx in args.contexts:
        print(f"\n{'='*60}")
        print(f"Context size: {ctx}")
        print(f"{'='*60}")
        for cfg in configs:
            label = cfg["label"]
            print(f"\n  Running {label} ngl={cfg['ngl']} ctx={ctx}...")
            r = run_benchmark(ctx, cfg["ngl"], cfg["device"], args.r, args.warmup)
            r["label"] = label
            r["timestamp"] = time.time()
            all_results.append(r)

            if r.get("t_per_token"):
                test = r.get("test", "")
                print(f"  {label}: {r['t_per_token']:.2f} t/s ({test})")
            else:
                print(f"  {label}: FAILED - {r.get('status')}")

    out_path = RESULTS_DIR / f"bench_ctx_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
