#!/usr/bin/env python3
"""
Validate Qwen GEMM shapes using iron runtime directly
"""

import subprocess
import sys
import json
import os
import re

def run_single_column(M, K, N, iters=10):
    """Run the int8_gemm_multicol_fixed.py generator for validation (single column)"""
    work_dir = f"/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/validation_M{M}_K{K}_N{N}"
    
    cmd = [
        "bash", "-c",
        f"source /Storage/Cache/src/mlir-aie-venv/bin/activate && "
        f"cd /workspace/llama.cpp && "
        f"python experiments/npu-int8-gemm/int8_gemm_multicol_fixed.py "
        f"--M {M} --K {K} --N {N} "
        f"--work-dir {work_dir} --iters {iters} 2>&1"
    ]
    
    print(f"  Testing single-column M={M}, K={K}, N={N}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    output = result.stdout + result.stderr
    if "PASS" in output and "FAIL" not in output:
        npu_gops_match = re.search(r"Effective GOPS \(NPU exec\): ([\d.]+)", output)
        npu_gops = float(npu_gops_match.group(1)) if npu_gops_match else 0
        return {"pass": True, "npu_gops": npu_gops, "output": output}
    else:
        return {"pass": False, "output": output}

def run_n_parallel(M, K, N, cols, iters=10):
    """Run the int8_gemm_nparallel.py generator for validation (N-parallel)"""
    # nparallel doesn't support --work-dir, it uses current directory
    work_dir = f"/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/validation_M{M}_K{K}_N{N}_C{cols}"
    
    cmd = [
        "bash", "-c",
        f"source /Storage/Cache/src/mlir-aie-venv/bin/activate && "
        f"cd {work_dir} && "
        f"python /workspace/llama.cpp/experiments/npu-int8-gemm/int8_gemm_nparallel.py "
        f"--M {M} --K {K} --N {N} --cols {cols} "
        f"--iters {iters} 2>&1"
    ]
    
    print(f"  Testing N-parallel M={M}, K={K}, N={N}, cols={cols}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    output = result.stdout + result.stderr
    if "PASS" in output and "FAIL" not in output:
        npu_gops_match = re.search(r"Effective GOPS \(NPU exec\): ([\d.]+)", output)
        npu_gops = float(npu_gops_match.group(1)) if npu_gops_match else 0
        return {"pass": True, "npu_gops": npu_gops, "output": output}
    else:
        return {"pass": False, "output": output}

def ensure_kernel(M, K, N, cols=1):
    """Ensure kernel is generated"""
    if cols == 1:
        work_dir = f"/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/validation_M{M}_K{K}_N{N}"
        gen_script = "gen_xdna_gemm.py"
        cols_arg = ""
    else:
        work_dir = f"/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/validation_M{M}_K{K}_N{N}_C{cols}"
        gen_script = "gen_xdna_gemm.py"
        cols_arg = f"--cols {cols}"
    
    if os.path.exists(f"{work_dir}/design.xclbin"):
        return True
    
    print(f"  Generating kernel...")
    cmd = [
        "bash", "-c",
        f"source /Storage/Cache/src/mlir-aie-venv/bin/activate && "
        f"cd {work_dir} && "
        f"python /workspace/llama.cpp/experiments/npu-int8-gemm/{gen_script} "
        f"--M {M} --K {K} --N {N} {cols_arg} 2>&1"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0

def main():
    # Key Qwen shapes from the shape matrix
    shapes = [
        # SSM (M=1, N=16) - Phase 9h microkernel (handled separately)
        {"M": 1, "K": 1024, "N": 16, "cols": 1, "name": "ssm_beta/alpha (decode)", "type": "phase9h"},
        
        # Decode shapes (M=1) - not supported by N-parallel generator (M>=4)
        # These would need Phase 9h microkernel with N-tiling
        
        # PP shapes (M=64 for M-tiling)
        {"M": 64, "K": 1024, "N": 64, "cols": 1, "name": "k_proj/v_proj (PP M=64)", "type": "single"},
        {"M": 64, "K": 1024, "N": 512, "cols": 1, "name": "q_proj/o_proj (PP M=64)", "type": "single"},
        {"M": 64, "K": 1024, "N": 3072, "cols": 4, "name": "up_proj/gate_proj (PP M=64)", "type": "nparallel"},
        {"M": 64, "K": 3072, "N": 1024, "cols": 2, "name": "down_proj (PP M=64)", "type": "nparallel"},
        
        # Larger PP shapes
        {"M": 128, "K": 1024, "N": 512, "cols": 1, "name": "q_proj/o_proj (PP M=128)", "type": "single"},
        {"M": 128, "K": 1024, "N": 3072, "cols": 4, "name": "up_proj/gate_proj (PP M=128)", "type": "nparallel"},
        {"M": 128, "K": 3072, "N": 1024, "cols": 2, "name": "down_proj (PP M=128)", "type": "nparallel"},
        {"M": 256, "K": 1024, "N": 512, "cols": 1, "name": "q_proj/o_proj (PP M=256)", "type": "single"},
        {"M": 256, "K": 1024, "N": 3072, "cols": 4, "name": "up_proj/gate_proj (PP M=256)", "type": "nparallel"},
    ]
    
    results = []
    for s in shapes:
        print(f"\n{'='*60}")
        print(f"Testing: {s['name']}")
        print(f"{'='*60}")
        
        if s['type'] == 'phase9h':
            print("  Skipping - Phase 9h microkernel tested separately")
            results.append({**s, "pass": True, "note": "Phase 9h microkernel validated separately"})
            continue
        
        if not ensure_kernel(s['M'], s['K'], s['N'], s['cols']):
            print(f"  Generation failed")
            results.append({**s, "pass": False, "error": "generation failed"})
            continue
        
        if s['type'] == 'single':
            result = run_single_column(s['M'], s['K'], s['N'], iters=10)
        else:
            result = run_n_parallel(s['M'], s['K'], s['N'], s['cols'], iters=10)
        
        result['name'] = s['name']
        results.append(result)
        
        if result['pass']:
            print(f"  PASS - NPU GOPS: {result['npu_gops']:.2f}")
        else:
            print(f"  FAIL")
            lines = result['output'].strip().split('\n')
            for line in lines[-10:]:
                print(f"  {line}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r.get('pass', False))
    total = len([r for r in results if r.get('type', '') != 'phase9h'])
    print(f"Passed: {passed}/{total}")
    
    for r in results:
        if r.get('type', '') == 'phase9h':
            print(f"  SKIP: {r['name']} (Phase 9h microkernel)")
        else:
            status = "PASS" if r.get('pass', False) else "FAIL"
            gops = f" ({r.get('npu_gops', 0):.2f} GOPS)" if r.get('pass', False) else ""
            print(f"  {status}: {r['name']}{gops}")

if __name__ == "__main__":
    main()