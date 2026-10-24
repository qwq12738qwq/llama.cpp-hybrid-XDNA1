#!/usr/bin/env python3
"""
Heterogeneous PP Test Framework

Tests:
- Test A: Vulkan-only candidate (baseline)
- Test B: XDNA-only candidate (not applicable for full PP)
- Test C: Sequential Vulkan + XDNA
- Test D: Concurrent Vulkan || XDNA

Measures end-to-end PP latency for different context lengths.
"""

import subprocess
import sys
import json
import os
import time
import numpy as np

def run_ggml_inference(model_path, prompt, n_predict, backend="auto", ctx_size=4096, n_threads=4):
    """Run llama-cli and measure PP time"""
    cmd = [
        "/workspace/llama.cpp/build/bin/llama-cli",
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "-c", str(ctx_size),
        "-t", str(n_threads),
        "--no-warmup"
    ]
    
    if backend != "auto":
        cmd.extend(["--backend", backend])
    
    env = os.environ.copy()
    env["GGML_XDNA_DEBUG"] = "1"
    
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    elapsed = time.perf_counter() - start
    
    return {
        "elapsed": elapsed,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

def measure_pp_latency(model_path, ctx_size, n_threads=4):
    """Measure PP latency for a given context size"""
    # Create a prompt of appropriate length
    # Use a simple repeating pattern
    base_prompt = "The quick brown fox jumps over the lazy dog. "
    num_repeats = max(1, (ctx_size * 4) // len(base_prompt))  # rough token estimate
    prompt = base_prompt * num_repeats
    
    # Run with n_predict=1 to measure PP only (first token)
    result = run_ggml_inference(model_path, prompt, 1, ctx_size=ctx_size, n_threads=n_threads)
    
    if result["returncode"] != 0:
        return {"error": result["stderr"]}
    
    # Parse timing from output
    # llama.cpp outputs timing info
    import re
    pp_time_match = re.search(r"pp time:\s+([\d.]+)\s*ms", result["stdout"] + result["stderr"])
    pp_time = float(pp_time_match.group(1)) / 1000.0 if pp_time_match else result["elapsed"]
    
    return {
        "ctx_size": ctx_size,
        "pp_time": pp_time,
        "elapsed": result["elapsed"]
    }

def main():
    model_path = "/Storage/Cache/Models/qwen3.5-0.8b-q8_0.gguf"
    
    if not os.path.exists(model_path):
        # Try to find model
        for root, dirs, files in os.walk("/Storage/Cache"):
            for f in files:
                if f.endswith(".gguf") and "qwen" in f.lower() and "0.8" in f:
                    model_path = os.path.join(root, f)
                    break
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please ensure Qwen3.5-0.8B Q8_0 model is available")
        return
    
    print(f"Using model: {model_path}")
    
    context_sizes = [4096, 8192, 16384, 32768, 65536]
    results = []
    
    for ctx in context_sizes:
        print(f"\nTesting PP with context size {ctx}...")
        
        # Test A: Vulkan-only
        print("  Test A: Vulkan-only...")
        os.environ["GGML_XDNA_DISABLE"] = "1"
        result_a = measure_pp_latency(model_path, ctx)
        print(f"    PP time: {result_a.get('pp_time', 'N/A'):.3f}s")
        
        # Test C: Sequential Vulkan + XDNA (XDNA for specific ops)
        print("  Test C: Sequential Vulkan + XDNA...")
        os.environ.pop("GGML_XDNA_DISABLE", None)
        result_c = measure_pp_latency(model_path, ctx)
        print(f"    PP time: {result_c.get('pp_time', 'N/A'):.3f}s")
        
        results.append({
            "ctx_size": ctx,
            "vulkan_only": result_a.get("pp_time"),
            "vulkan_xdna": result_c.get("pp_time"),
            "speedup": result_a.get("pp_time", 0) / result_c.get("pp_time", 1) if result_c.get("pp_time") else 0
        })
    
    # Summary
    print("\n" + "="*60)
    print("HETEROGENEOUS PP TEST RESULTS")
    print("="*60)
    print(f"{'Ctx':>8} | {'Vulkan (s)':>10} | {'Vulkan+XDNA (s)':>14} | {'Speedup':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['ctx_size']:>8} | {r['vulkan_only']:>10.3f} | {r['vulkan_xdna']:>14.3f} | {r['speedup']:>8.2f}x")
    
    # Save results
    with open("/workspace/llama.cpp/experiments/npu-int8-gemm/heterogeneous_pp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to heterogeneous_pp_results.json")

if __name__ == "__main__":
    main()