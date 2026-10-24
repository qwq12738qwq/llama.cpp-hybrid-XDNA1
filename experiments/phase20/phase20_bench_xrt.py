#!/usr/bin/env python3
"""Benchmark pipelined kernel directly via XRT, bypassing GGML backend."""
import numpy as np
import time
import sys
import json

sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')

import pyxrt

M, K, N = 512, 1024, 3072
V2_XCLBIN = "/Storage/Cache/llama.cpp/experiments/phase20/v2_pipe.prj/design.xclbin"
V2_INSTS = "/Storage/Cache/llama.cpp/experiments/phase20/v2_pipe.prj/design.insts.bin"
V1_XCLBIN = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3072_C1_Tm64_Tk64_Tn64/design.xclbin"
V1_INSTS = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3072_C1_Tm64_Tk64_Tn64/design.insts.bin"

def benchmark_kernel(xclbin_path, insts_path, label, n_runs=20, n_warmup=5):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {label}")
    print(f"  xclbin: {xclbin_path} ({__import__('os').path.getsize(xclbin_path)} bytes)")
    print(f"  insts:  {insts_path} ({__import__('os').path.getsize(insts_path)} bytes)")

    # Open device
    device = pyxrt.device(0)
    xclbin_uuid = device.load_xclbin(xclbin_path)
    ctx = pyxrt.context(device, xclbin_uuid, "pipeline")

    # Read instruction binary
    with open(insts_path, 'rb') as f:
        insts_bin = f.read()

    # Create kernels and buffers
    # BO for instructions
    inst_bo = pyxrt.bo(device, len(insts_bin), pyxrt.bo.cacheable, 0)
    inst_bo.write(insts_bin, 0)
    inst_bo.sync(xcl.BO_SYNC_HOST_TO_DEVICE)

    # Create output BO to get issue_token
    # Actually, use the NPU kernel approach
    kernel = pyxrt.kernel(ctx, "pipeline", pyxrt.run.bo_type.npu)

    # Allocate A, B, C buffers
    a_size = M * K  # int8
    b_size = K * N  # int8
    c_size = M * N * 4  # int32
    total_size = a_size + b_size + c_size

    # For XRT NPU, we need to allocate BOs with proper flags
    # Use group_id from the kernel
    a_bo = pyxrt.bo(device, a_size, pyxrt.bo.normal, kernel.group_id(5))
    b_bo = pyxrt.bo(device, b_size, pyxrt.bo.normal, kernel.group_id(6))
    c_bo = pyxrt.bo(device, c_size, pyxrt.bo.normal, kernel.group_id(7))

    # Fill A, B with random data
    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)
    ref = A_np.astype(np.int32) @ B_np.astype(np.int32)

    a_bo.write(A_np.tobytes(), 0)
    a_bo.sync(xcl.BO_SYNC_HOST_TO_DEVICE)
    b_bo.write(B_np.tobytes(), 0)
    b_bo.sync(xcl.BO_SYNC_HOST_TO_DEVICE)

    # Warmup
    for _ in range(n_warmup):
        run = kernel(inst_bo, a_bo, b_bo, c_bo)
        run.wait()

    # Timed runs
    kernel_times = []
    for _ in range(n_runs):
        c_bo.write(np.zeros(c_size, dtype=np.uint8), 0)
        c_bo.sync(xcl.BO_SYNC_HOST_TO_DEVICE)
        t0 = time.perf_counter()
        run = kernel(inst_bo, a_bo, b_bo, c_bo)
        run.wait()
        t1 = time.perf_counter()
        kernel_times.append((t1 - t0) * 1000)

    # Read C
    c_bo.sync(xcl.BO_SYNC_TO_HOST)
    got = np.frombuffer(c_bo.read(), dtype=np.int32).reshape(M, N)

    max_err = int(np.max(np.abs(got - ref)))
    n_mismatch = int(np.sum(got != ref))

    kt = np.array(kernel_times)
    flops = 2.0 * M * K * N
    gops = flops / (np.mean(kt) * 1e-3) / 1e9

    print(f"  Correctness: max_err={max_err}, mismatches={n_mismatch}/{M*N}")
    print(f"  Kernel: {np.mean(kt):.2f} ms (p50={np.percentile(kt,50):.2f}, p95={np.percentile(kt,95):.2f})")
    print(f"  GOPS:   {gops:.2f}")

    del kernel
    del ctx
    del device
    return {"label": label, "kernel_ms": float(np.mean(kt)), "GOPS": gops,
            "max_err": max_err, "correct": max_err == 0}


if __name__ == "__main__":
    results = []
    try:
        r = benchmark_kernel(V1_XCLBIN, V1_INSTS, "V1 baseline")
        results.append(r)
    except Exception as e:
        print(f"V1 failed: {e}")

    try:
        r = benchmark_kernel(V2_XCLBIN, V2_INSTS, "V2 pipelined")
        results.append(r)
    except Exception as e:
        print(f"V2 failed: {e}")

    if len(results) == 2:
        print(f"\n{'='*60}")
        print(f"V1 vs V2 Pipelined")
        print(f"  V1: {results[0]['GOPS']:.2f} GOPS, {results[0]['kernel_ms']:.2f} ms")
        print(f"  V2: {results[1]['GOPS']:.2f} GOPS, {results[1]['kernel_ms']:.2f} ms")
        print(f"  Speedup: {results[1]['GOPS']/results[0]['GOPS']:.2f}x")

    with open("/Storage/Cache/llama.cpp/experiments/phase20/phase20_v2_pipe.json", "w") as f:
        json.dump(results, f, indent=2)
