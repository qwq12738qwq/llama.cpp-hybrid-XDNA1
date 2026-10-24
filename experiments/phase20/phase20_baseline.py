#!/usr/bin/env python3
"""
Phase 20: XDNA Kernel Performance Research - Step I+II Baseline Freeze

CURRENT_RUNTIME_KERNEL: M512_K1024_N3072_C1_Tm64_Tk64_Tn64
- Single AIE core, npu1_1col
- Tile: 64x64x64
- INT8 GEMM with int32 accumulation

Measures:
- wall time, kernel time, DMA time, submit time, wait time
- GOPS, DMA GB/s, AIE utilization estimate
"""
import pyxrt
import numpy as np
import time
import os
import json

# ============================================================================
# Configuration
# ============================================================================
XCLBIN_DIR = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache"
SHAPES = [
    {"M": 512, "K": 1024, "N": 3072, "label": "baseline_M512_K1024_N3072"},
]
N_WARMUP = 10
N_RUNS = 100

# ============================================================================
# Kernel loading
# ============================================================================
def load_kernel(M, K, N, n_cols=1, tm=64, tk=64, tn=64):
    """Load pre-compiled kernel from cache."""
    base = f"{XCLBIN_DIR}/M{M}_K{K}_N{N}_C{n_cols}_Tm{tm}_Tk{tk}_Tn{tn}"
    xclbin_path = f"{base}/design.xclbin"
    inst_path = f"{base}/design.insts.bin"
    
    if not os.path.exists(xclbin_path) or not os.path.exists(inst_path):
        raise FileNotFoundError(f"Kernel not found: {xclbin_path}")
    
    device = pyxrt.device(0)
    xclbin_obj = pyxrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin_obj)
    uuid = xclbin_obj.get_uuid()
    
    inst_bytes = os.path.getsize(inst_path)
    with open(inst_path, "rb") as f:
        inst_data = f.read()
    
    hw_ctx = pyxrt.hw_context(device, uuid)
    kernel = pyxrt.kernel(hw_ctx, "MLIR_AIE")
    
    gid1 = kernel.group_id(1)
    gid3 = kernel.group_id(3)
    
    inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
    inst_bo.write(inst_data, 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    
    input_bytes = M * K
    weight_bytes = K * N
    output_bytes = M * N * 4
    
    input_bo = pyxrt.bo(device, max(input_bytes, 64), pyxrt.bo.flags.host_only, gid3)
    weight_bo = pyxrt.bo(device, max(weight_bytes, 64), pyxrt.bo.flags.host_only, gid3)
    output_bo = pyxrt.bo(device, max(output_bytes, 64), pyxrt.bo.flags.host_only, gid3)
    
    return {
        "device": device,
        "kernel": kernel,
        "inst_bo": inst_bo,
        "inst_bytes": inst_bytes,
        "input_bo": input_bo,
        "weight_bo": weight_bo,
        "output_bo": output_bo,
        "gid1": gid1,
        "gid3": gid3,
    }

# ============================================================================
# Single GEMM execution with timing breakdown
# ============================================================================
def run_gemm(engine, inp_i8, wgt_i8, M, K, N):
    """
    Execute one GEMM with detailed timing.
    
    Timing points:
      T_prepare      : host memcpy to BO maps
      T_bo_sync_in   : BO sync to device (H2D)
      T_submit       : run object creation + set_arg
      T_kernel       : run.start() to run.wait() (on-device DMA + compute)
      T_bo_sync_out  : BO sync from device (D2H)
      T_host_copy    : host read output
    """
    # Step 1: prepare
    t0 = time.perf_counter()
    
    inp_map = engine["input_bo"].map()
    wgt_map = engine["weight_bo"].map()
    out_map = engine["output_bo"].map()
    
    inp_map[:len(inp_i8)] = inp_i8.tobytes()
    wgt_map[:len(wgt_i8)] = wgt_i8.tobytes()
    zeros = np.zeros(M * N, dtype=np.int32).tobytes()
    out_map[:len(zeros)] = zeros
    
    t1 = time.perf_counter()
    
    # Step 2: sync to device
    engine["input_bo"].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    engine["weight_bo"].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    engine["output_bo"].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    
    t2 = time.perf_counter()
    
    # Step 3: run
    run = pyxrt.run(engine["kernel"])
    run.set_arg(0, 3)
    run.set_arg(1, engine["inst_bo"])
    run.set_arg(2, engine["inst_bytes"])
    run.set_arg(3, engine["input_bo"])
    run.set_arg(4, engine["weight_bo"])
    run.set_arg(5, engine["output_bo"])
    
    t3 = time.perf_counter()
    run.start()
    run.wait()
    t4 = time.perf_counter()
    
    # Step 4: sync from device
    engine["output_bo"].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    
    t5 = time.perf_counter()
    
    # Step 5: read output
    result = np.frombuffer(bytes(out_map[:M * N * 4]), dtype=np.int32).copy()
    
    t6 = time.perf_counter()
    
    return {
        "result": result,
        "T_prepare_ms": (t1 - t0) * 1000,
        "T_bo_sync_in_ms": (t2 - t1) * 1000,
        "T_submit_ms": (t3 - t2) * 1000,
        "T_kernel_ms": (t4 - t3) * 1000,
        "T_bo_sync_out_ms": (t5 - t4) * 1000,
        "T_host_copy_ms": (t6 - t5) * 1000,
        "T_total_ms": (t6 - t0) * 1000,
    }

# ============================================================================
# Correctness check
# ============================================================================
def cpu_gemm_ref(inp_i8, wgt_i8, M, K, N):
    """CPU reference INT8 GEMM."""
    A = inp_i8.reshape(M, K).astype(np.int32)
    B = wgt_i8.reshape(K, N).astype(np.int32)
    return (A @ B).astype(np.int32)

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 80)
    print("Phase 20 Step I+II: Baseline Freeze")
    print("CURRENT_RUNTIME_KERNEL: M512_K1024_N3072_C1_Tm64_Tk64_Tn64")
    print("=" * 80)
    
    all_results = {}
    
    for shape in SHAPES:
        M, K, N = shape["M"], shape["K"], shape["N"]
        label = shape["label"]
        
        print(f"\n{'='*80}")
        print(f"Shape: M={M}, K={K}, N={N}")
        print(f"{'='*80}")
        
        # Load kernel
        print("Loading kernel...")
        engine = load_kernel(M, K, N)
        
        # Generate data
        np.random.seed(42)
        inp_f32 = np.random.randn(M * K).astype(np.float32)
        wgt_f32 = np.random.randn(K * N).astype(np.float32)
        
        # Quantize to int8 (simple per-tensor for testing)
        inp_scale = float(np.max(np.abs(inp_f32))) / 127.0
        wgt_scale = float(np.max(np.abs(wgt_f32))) / 127.0
        inp_i8 = np.clip(np.round(inp_f32 / inp_scale), -127, 127).astype(np.int8)
        wgt_i8 = np.clip(np.round(wgt_f32 / wgt_scale), -127, 127).astype(np.int8)
        
        # Correctness check
        ref = cpu_gemm_ref(inp_i8, wgt_i8, M, K, N)
        warmup = run_gemm(engine, inp_i8, wgt_i8, M, K, N)
        max_err = int(np.max(np.abs(warmup["result"] - ref.flatten())))
        print(f"Correctness: max_err={max_err}")
        
        # Warmup
        for _ in range(N_WARMUP):
            run_gemm(engine, inp_i8, wgt_i8, M, K, N)
        
        # Timed runs
        timings = []
        for _ in range(N_RUNS):
            new_inp = np.random.randint(-127, 128, size=M*K, dtype=np.int8)
            t = run_gemm(engine, new_inp, wgt_i8, M, K, N)
            timings.append(t)
        
        # Statistics
        stats = {"M": M, "K": K, "N": N, "correctness_max_err": max_err}
        for field in ["T_prepare_ms", "T_bo_sync_in_ms", "T_submit_ms", 
                       "T_kernel_ms", "T_bo_sync_out_ms", "T_host_copy_ms", "T_total_ms"]:
            vals = [t[field] for t in timings]
            stats[f"{field}_mean"] = float(np.mean(vals))
            stats[f"{field}_p50"] = float(np.percentile(vals, 50))
            stats[f"{field}_p95"] = float(np.percentile(vals, 95))
            stats[f"{field}_min"] = float(np.min(vals))
            stats[f"{field}_max"] = float(np.max(vals))
        
        # GOPS
        flops = 2.0 * M * K * N
        kernel_ms = stats["T_kernel_ms_mean"]
        total_ms = stats["T_total_ms_mean"]
        stats["GOPS_kernel"] = flops / (kernel_ms * 1e-3) / 1e9 if kernel_ms > 0 else 0
        stats["GOPS_total"] = flops / (total_ms * 1e-3) / 1e9 if total_ms > 0 else 0
        
        # DMA bandwidth
        bytes_moved = M * K + K * N + M * N * 4
        stats["DMA_bytes"] = bytes_moved
        stats["DMA_GB_s_kernel"] = bytes_moved / (kernel_ms * 1e-3) / 1e9 if kernel_ms > 0 else 0
        
        # Roofline inputs
        stats["FLOPs"] = flops
        stats["arithmetic_intensity"] = flops / bytes_moved
        
        print(f"\n--- Timing Breakdown (mean) ---")
        print(f"  T_prepare:      {stats['T_prepare_ms_mean']:8.3f} ms")
        print(f"  T_bo_sync_in:   {stats['T_bo_sync_in_ms_mean']:8.3f} ms")
        print(f"  T_submit:       {stats['T_submit_ms_mean']:8.3f} ms")
        print(f"  T_kernel:       {stats['T_kernel_ms_mean']:8.3f} ms  (= DMA + AIE compute)")
        print(f"  T_bo_sync_out:  {stats['T_bo_sync_out_ms_mean']:8.3f} ms")
        print(f"  T_host_copy:    {stats['T_host_copy_ms_mean']:8.3f} ms")
        print(f"  T_total:        {stats['T_total_ms_mean']:8.3f} ms")
        print(f"\n--- Performance ---")
        print(f"  FLOPs:          {stats['FLOPs']:.2e}")
        print(f"  Arithmetic Int: {stats['arithmetic_intensity']:.2f} FLOP/byte")
        print(f"  GOPS (kernel):  {stats['GOPS_kernel']:.2f}")
        print(f"  GOPS (total):   {stats['GOPS_total']:.2f}")
        print(f"  DMA GB/s:       {stats['DMA_GB_s_kernel']:.2f}")
        print(f"  DMA bytes:      {stats['DMA_bytes']:.2e}")
        
        # AIE utilization estimate
        # Phoenix XDNA1: 4 columns, each with compute tiles
        # Theoretical peak: ~8000 GOPS (from Phase 18 report)
        # But we're using 1 column, so peak per column: ~2000 GOPS
        # Actually, the report says measured 50 GOPS at 0.65% utilization
        # So theoretical peak per column: 50 / 0.0065 = 7692 GOPS
        # This seems too high. Let me use the Phase 18 estimate.
        theoretical_peak = 8000  # GOPS total (4 columns)
        used_columns = 1
        aie_util = stats["GOPS_kernel"] / (theoretical_peak / 4 * used_columns) * 100
        stats["AIE_utilization_pct"] = aie_util
        print(f"  AIE util est:   {aie_util:.2f}% (vs {theoretical_peak/4:.0f} GOPS/column peak)")
        
        all_results[label] = stats
    
    # Save results
    out_dir = "/Storage/Cache/llama.cpp/experiments/phase20"
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "phase20_baseline.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {out_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
