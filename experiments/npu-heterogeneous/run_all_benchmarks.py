#!/usr/bin/env python3
# run_all_benchmarks.py — Run comprehensive benchmarks for Phase 7B-R report

import subprocess
import json
import os

# Test configurations: (M, K, N, cols, description)
test_configs = [
    # 1-column baseline
    (64, 1024, 64, 1, "1col M64 N64"),
    (64, 1024, 128, 1, "1col M64 N128"),
    (64, 1024, 256, 1, "1col M64 N256"),
    (128, 1024, 64, 1, "1col M128 N64"),
    (128, 1024, 128, 1, "1col M128 N128"),
    (128, 1024, 256, 1, "1col M128 N256"),
    (128, 1024, 512, 1, "1col M128 N512"),
    (256, 1024, 64, 1, "1col M256 N64"),
    (256, 1024, 128, 1, "1col M256 N128"),
    (256, 1024, 256, 1, "1col M256 N256"),
    (512, 1024, 64, 1, "1col M512 N64"),
    
    # 2-column N-parallel
    (128, 1024, 128, 2, "2col M128 N128"),
    (128, 1024, 256, 2, "2col M128 N256"),
    (128, 1024, 512, 2, "2col M128 N512"),
    (256, 1024, 128, 2, "2col M256 N128"),
    (256, 1024, 256, 2, "2col M256 N256"),
    (256, 1024, 512, 2, "2col M256 N512"),
    
    # 4-column N-parallel
    (128, 1024, 256, 4, "4col M128 N256"),
    (128, 1024, 512, 4, "4col M128 N512"),
    (256, 1024, 256, 4, "4col M256 N256"),
    (256, 1024, 512, 4, "4col M256 N512"),
    (512, 1024, 256, 4, "4col M512 N256"),
]

results = []

for M, K, N, cols, desc in test_configs:
    # Skip invalid configs
    if N % (64 * cols) != 0:
        print(f"SKIP {desc}: N={N} not divisible by 64*{cols}")
        continue
    
    print(f"\nRunning {desc} (M={M} K={K} N={N} cols={cols})...")
    
    # Clear cache
    os.system("podman exec npu-dev rm -rf /home/ubuntu/.npu/cache 2>/dev/null")
    
    # Run benchmark
    cmd = f"podman exec npu-dev python3 /workspace/llama.cpp/experiments/npu-int8-gemm/int8_gemm_nparallel.py --M {M} --K {K} --N {N} --cols {cols} --iters 3 2>&1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    
    # Parse output
    output = result.stdout
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr}")
        results.append({
            "config": desc,
            "M": M, "K": K, "N": N, "cols": cols,
            "status": "FAILED",
            "error": result.stderr
        })
        continue
    
    # Extract metrics
    import re
    host_lat = re.search(r"Host submission latency.*?mean=([\d.]+)", output)
    npu_lat = re.search(r"NPU execution latency.*?mean=([\d.]+)", output)
    e2e_gops = re.search(r"Effective GOPS \(host e2e\): ([\d.]+)", output)
    npu_gops = re.search(r"Effective GOPS \(NPU exec\): ([\d.]+)", output)
    correctness = re.search(r"Correctness: (\d+) mismatches", output)
    
    if host_lat and npu_lat and e2e_gops and npu_gops and correctness:
        host_mean = float(host_lat.group(1))
        npu_mean = float(npu_lat.group(1))
        e2e = float(e2e_gops.group(1))
        npu = float(npu_gops.group(1))
        mismatches = int(correctness.group(1))
        
        result_data = {
            "config": desc,
            "M": M, "K": K, "N": N, "cols": cols,
            "host_latency_us": host_mean,
            "npu_latency_us": npu_mean,
            "e2e_gops": e2e,
            "npu_gops": npu,
            "mismatches": mismatches,
            "status": "PASS" if mismatches == 0 else "FAIL"
        }
        results.append(result_data)
        print(f"  PASS: host={host_mean:.1f}us npu={npu_mean:.1f}us e2e_gops={e2e:.2f} npu_gops={npu:.2f}")
    else:
        print(f"  FAILED to parse: {output[-500:]}")
        results.append({
            "config": desc,
            "M": M, "K": K, "N": N, "cols": cols,
            "status": "PARSE_FAILED",
            "output": output[-1000:]
        })

# Save results
with open("/home/ubuntu12738/Storage/Cache/llama.cpp/experiments/npu-heterogeneous/results/phase7br/multicolumn_raw_benchmarks.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nAll benchmarks complete. Results saved.")
for r in results:
    if r["status"] == "PASS":
        speedup = r["e2e_gops"] / 12.64  # baseline 1col M64 N64
        print(f"{r['config']:20s} host={r['host_latency_us']:7.1f}us npu={r['npu_latency_us']:7.1f}us e2e_gops={r['e2e_gops']:6.2f} npu_gops={r['npu_gops']:6.2f} speedup={speedup:.2f}x")
    else:
        print(f"{r['config']:20s} {r['status']}")