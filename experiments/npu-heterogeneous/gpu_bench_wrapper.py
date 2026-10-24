#!/usr/bin/env python3
"""gpu_bench_wrapper.py -- Run gpu_bench.cpp and parse JSON output.

Usage:
    python3 gpu_bench_wrapper.py --M 128 --K 4096 --N 4096 --iters 10 --warmup 2
"""
import argparse, json, subprocess, sys, os, shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_BIN = os.path.join(SCRIPT_DIR, "gpu_bench")
GPU_CPP = os.path.join(SCRIPT_DIR, "gpu_bench.cpp")

def ensure_built():
    if os.path.exists(GPU_BIN) and os.access(GPU_BIN, os.X_OK):
        return True
    # Try to build
    if not os.path.exists(GPU_CPP):
        print(f"ERROR: gpu_bench.cpp not found at {GPU_CPP}", file=sys.stderr)
        return False
    print("Building gpu_bench...", file=sys.stderr)
    build_script = os.path.join(SCRIPT_DIR, "build_gpu_bench.sh")
    if os.path.exists(build_script):
        result = subprocess.run(["bash", build_script], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}", file=sys.stderr)
            return False
    else:
        # Fallback: direct g++ compile
        result = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
             "-I/usr/include/vulkan",
             GPU_CPP, "-o", GPU_BIN, "-lvulkan", "-lpthread"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}", file=sys.stderr)
            return False
    return os.path.exists(GPU_BIN)

def run_gpu_bench(M, K, N, iters=5, warmup=2, spv_path=None):
    if not ensure_built():
        return {"status": "BUILD_FAILED", "error": "Could not build gpu_bench"}

    cmd = [GPU_BIN, "--M", str(M), "--K", str(K), "--N", str(N),
           "--iters", str(iters), "--warmup", str(warmup)]
    if spv_path and os.path.exists(spv_path):
        cmd.extend(["--spv", spv_path])

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/vulkan:" + env.get("LD_LIBRARY_PATH", "")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                              cwd=SCRIPT_DIR, env=env)
        if proc.returncode != 0:
            return {"status": "ERROR", "error": proc.stderr[:500]}

        # Parse JSON from stdout (last JSON object)
        json_start = proc.stdout.rfind("{")
        if json_start < 0:
            return {"status": "PARSE_ERROR", "error": "No JSON in stdout",
                    "stdout": proc.stdout[:500], "stderr": proc.stderr[:500]}

        try:
            data = json.loads(proc.stdout[json_start:])
            data["status"] = "OK"
            return data
        except json.JSONDecodeError as e:
            return {"status": "PARSE_ERROR", "error": str(e),
                    "raw": proc.stdout[json_start:json_start+200]}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--spv", default=None)
    args = parser.parse_args()

    result = run_gpu_bench(args.M, args.K, args.N, args.iters, args.warmup, args.spv)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("status") == "OK" else 1)

if __name__ == "__main__":
    main()
