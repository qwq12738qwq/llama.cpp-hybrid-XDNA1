#!/usr/bin/env python3
"""
Phase 20 Step V: Double Buffering Kernel

CURRENT_RUNTIME_KERNEL structure (single-buffer):
  runtime_sequence:
    DMA A (fill) -> start
    DMA B (fill) -> start  
    DMA C (drain, wait) -> start, await
    free all
    ... repeat for next M-tile

  AIE core:
    for each M-tile:
      for K/64 iterations:
        acquire A, B -> matmul -> release A, B
      release C

Optimized (double-buffered):
  Key insight: ObjectFifo depth=2 already exists.
  But runtime_sequence sends ALL data for one M-tile before starting compute.
  
  Better approach: Use 2 output buffers (C0, C1) and alternate.
  While AIE computes on C0, DMA drains C1 to host.
  
  Also: pre-fetch next M-tile's A/B while current M-tile computes.
"""
import pyxrt
import numpy as np
import time
import os
import json

XCLBIN_DIR = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

def load_kernel(M, K, N, n_cols=1, tm=64, tk=64, tn=64):
    base = f"{XCLBIN_DIR}/M{M}_K{K}_N{N}_C{n_cols}_Tm{tm}_Tk{tk}_Tn{tn}"
    xclbin_path = f"{base}/design.xclbin"
    inst_path = f"{base}/design.insts.bin"
    if not os.path.exists(xclbin_path):
        raise FileNotFoundError(f"Kernel not found: {xclbin_path}")
    device = pyxrt.device(0)
    xclbin_obj = pyxrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin_obj)
    inst_bytes = os.path.getsize(inst_path)
    with open(inst_path, "rb") as f:
        inst_data = f.read()
    hw_ctx = pyxrt.hw_context(device, xclbin_obj.get_uuid())
    kernel = pyxrt.kernel(hw_ctx, "MLIR_AIE")
    gid1 = kernel.group_id(1)
    gid3 = kernel.group_id(3)
    inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
    inst_bo.write(inst_data, 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return device, kernel, inst_bo, inst_bytes, gid3

def run_single_buffer(M, K, N, inp_i8, wgt_i8, engine_tuple):
    """CURRENT_RUNTIME_KERNEL: one XRT submit per M-tile group."""
    device, kernel, inst_bo, inst_bytes, gid3 = engine_tuple
    
    input_bytes = M * K
    weight_bytes = K * N
    output_bytes = M * N * 4
    
    in_bo = pyxrt.bo(device, max(input_bytes, 64), pyxrt.bo.flags.host_only, gid3)
    wt_bo = pyxrt.bo(device, max(weight_bytes, 64), pyxrt.bo.flags.host_only, gid3)
    out_bo = pyxrt.bo(device, max(output_bytes, 64), pyxrt.bo.flags.host_only, gid3)
    
    t0 = time.perf_counter()
    in_bo.write(inp_i8.tobytes(), 0)
    wt_bo.write(wgt_i8.tobytes(), 0)
    out_bo.write(np.zeros(M * N, dtype=np.int32).tobytes(), 0)
    in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t1 = time.perf_counter()
    
    run = pyxrt.run(kernel)
    run.set_arg(0, 3)
    run.set_arg(1, inst_bo)
    run.set_arg(2, inst_bytes)
    run.set_arg(3, in_bo)
    run.set_arg(4, wt_bo)
    run.set_arg(5, out_bo)
    run.start()
    run.wait()
    t2 = time.perf_counter()
    
    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    result = np.frombuffer(out_bo.read(output_bytes, 0), dtype=np.int32).copy()
    t3 = time.perf_counter()
    
    return {
        "result": result,
        "T_upload_ms": (t1 - t0) * 1000,
        "T_kernel_ms": (t2 - t1) * 1000,
        "T_download_ms": (t3 - t2) * 1000,
        "T_total_ms": (t3 - t0) * 1000,
    }

def run_double_buffer_tiled(M, K, N, inp_i8, wgt_i8, engine_tuple, tile_m=512):
    """
    Double-buffered: split M into tiles of tile_m.
    While tile[i] runs on NPU, prepare tile[i+1] data on host.
    Uses 2 sets of BOs alternating.
    """
    device, kernel, inst_bo, inst_bytes, gid3 = engine_tuple
    
    n_tiles = max(1, M // tile_m)
    actual_tile_m = M // n_tiles
    
    input_bytes = actual_tile_m * K
    weight_bytes = K * N
    output_bytes = actual_tile_m * N * 4
    
    # Create 2 sets of BOs for double buffering
    in_bo = [None, None]
    wt_bo = [None, None]
    out_bo = [None, None]
    for i in range(2):
        in_bo[i] = pyxrt.bo(device, max(input_bytes, 64), pyxrt.bo.flags.host_only, gid3)
        wt_bo[i] = pyxrt.bo(device, max(weight_bytes, 64), pyxrt.bo.flags.host_only, gid3)
        out_bo[i] = pyxrt.bo(device, max(output_bytes, 64), pyxrt.bo.flags.host_only, gid3)
    
    output = np.zeros(M * N, dtype=np.int32)
    
    t_total_start = time.perf_counter()
    
    for t in range(n_tiles):
        buf = t % 2
        m_offset = t * actual_tile_m
        tile_inp = inp_i8[m_offset * K : (m_offset + actual_tile_m) * K]
        
        # Upload current tile (overlaps with nothing on first iteration)
        t0 = time.perf_counter()
        in_bo[buf].write(tile_inp.tobytes(), 0)
        wt_bo[buf].write(wgt_i8.tobytes(), 0)
        out_bo[buf].write(np.zeros(actual_tile_m * N, dtype=np.int32).tobytes(), 0)
        in_bo[buf].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wt_bo[buf].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        out_bo[buf].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t1 = time.perf_counter()
        
        # Run kernel
        run = pyxrt.run(kernel)
        run.set_arg(0, 3)
        run.set_arg(1, inst_bo)
        run.set_arg(2, inst_bytes)
        run.set_arg(3, in_bo[buf])
        run.set_arg(4, wt_bo[buf])
        run.set_arg(5, out_bo[buf])
        run.start()
        run.wait()
        t2 = time.perf_counter()
        
        # Download result
        out_bo[buf].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        tile_out = np.frombuffer(out_bo[buf].read(output_bytes, 0), dtype=np.int32).copy()
        output[m_offset * N : (m_offset + actual_tile_m) * N] = tile_out
        t3 = time.perf_counter()
    
    t_total_end = time.perf_counter()
    
    return {
        "result": output,
        "T_total_ms": (t_total_end - t_total_start) * 1000,
        "n_tiles": n_tiles,
        "tile_m": actual_tile_m,
    }

def cpu_gemm_ref(inp_i8, wgt_i8, M, K, N):
    A = inp_i8.reshape(M, K).astype(np.int32)
    B = wgt_i8.reshape(K, N).astype(np.int32)
    return (A @ B).astype(np.int32)

def benchmark_shape(M, K, N, n_runs=50, n_warmup=10):
    print(f"\n{'='*80}")
    print(f"Shape: M={M}, K={K}, N={N}")
    print(f"{'='*80}")
    
    engine = load_kernel(M, K, N)
    
    np.random.seed(42)
    inp_f32 = np.random.randn(M * K).astype(np.float32)
    wgt_f32 = np.random.randn(K * N).astype(np.float32)
    inp_scale = float(np.max(np.abs(inp_f32))) / 127.0
    wgt_scale = float(np.max(np.abs(wgt_f32))) / 127.0
    inp_i8 = np.clip(np.round(inp_f32 / inp_scale), -127, 127).astype(np.int8)
    wgt_i8 = np.clip(np.round(wgt_f32 / wgt_scale), -127, 127).astype(np.int8)
    
    ref = cpu_gemm_ref(inp_i8, wgt_i8, M, K, N)
    
    # --- V1: Single buffer (CURRENT_RUNTIME_KERNEL) ---
    print("\n--- V1: Single Buffer (baseline) ---")
    for _ in range(n_warmup):
        run_single_buffer(M, K, N, inp_i8, wgt_i8, engine)
    
    v1_times = []
    for _ in range(n_runs):
        t = run_single_buffer(M, K, N, inp_i8, wgt_i8, engine)
        v1_times.append(t["T_kernel_ms"])
    
    v1_kernel = np.mean(v1_times)
    flops = 2.0 * M * K * N
    v1_gops = flops / (v1_kernel * 1e-3) / 1e9
    
    # Correctness with known input
    v1_check = run_single_buffer(M, K, N, inp_i8, wgt_i8, engine)
    v1_err = int(np.max(np.abs(v1_check["result"] - ref.flatten())))
    
    print(f"  Correctness: max_err={v1_err}")
    print(f"  Kernel: {v1_kernel:.2f} ms -> {v1_gops:.2f} GOPS")
    
    # --- V2: Double buffer tiled ---
    print("\n--- V2: Double Buffer Tiled ---")
    for tile_m in [512, 256, 128, 64]:
        if M % tile_m != 0:
            continue
        
        for _ in range(n_warmup):
            run_double_buffer_tiled(M, K, N, inp_i8, wgt_i8, engine, tile_m=tile_m)
        
        v2_times = []
        for _ in range(n_runs):
            t = run_double_buffer_tiled(M, K, N, inp_i8, wgt_i8, engine, tile_m=tile_m)
            v2_times.append(t["T_total_ms"])
        
        v2_check = run_double_buffer_tiled(M, K, N, inp_i8, wgt_i8, engine, tile_m=tile_m)
        v2_total = np.mean(v2_times)
        v2_err = int(np.max(np.abs(v2_check["result"] - ref.flatten())))
        v2_gops = flops / (v2_total * 1e-3) / 1e9
        n_tiles = M // tile_m
        
        print(f"  tile_m={tile_m:4d} ({n_tiles:2d} tiles): total={v2_total:.2f}ms → {v2_gops:.2f} GOPS err={v2_err}")
    
    return {
        "M": M, "K": K, "N": N,
        "V1_kernel_ms": v1_kernel,
        "V1_GOPS": v1_gops,
        "V1_err": v1_err,
    }

def main():
    print("=" * 80)
    print("Phase 20 Step V: Double Buffering Research")
    print("=" * 80)
    
    results = []
    
    shapes = [
        (512, 1024, 3072),
        (512, 2048, 3072),
        (512, 4096, 3072),
    ]
    
    for M, K, N in shapes:
        try:
            r = benchmark_shape(M, K, N)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Save
    out_dir = "/Storage/Cache/llama.cpp/experiments/phase20"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/phase20_double_buffer.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/phase20_double_buffer.json")

if __name__ == "__main__":
    main()
