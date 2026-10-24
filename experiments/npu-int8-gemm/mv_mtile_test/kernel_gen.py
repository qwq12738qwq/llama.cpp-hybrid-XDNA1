#!/usr/bin/env python3
"""
Generate MLIR-AIE design for large-M Matrix-Vector Multiply using mm kernel
M_tile=64, K=32, N=1, INT8 weights (dequantized on host)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, '/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')
import aie.iron as iron
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, Out

def build_mv_kernel_design(M_tile, K, N, work_dir):
    """Build M_tile x K x N matrix-vector multiply kernel using mm kernel."""
    
    print(f"\n=== Building MV Kernel: M_tile={M_tile}, K={K}, N={N} ===")
    
    # Use the built-in mm kernel for INT8 x INT8 -> INT32
    matmul_kernel = mm(
        dim_m=M_tile,
        dim_k=K,
        dim_n=N,
        input_dtype=np.int8,
        output_dtype=np.int32,
        vectorized=False,
    )
    
    # Types
    A_ty = np.ndarray[(M_tile, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M_tile, N), np.dtype[np.int32]]
    
    # Stream types (flattened for FIFOs)
    a_ty = np.ndarray[(M_tile * K,), np.dtype[np.int8]]
    b_ty = np.ndarray[(K * N,), np.dtype[np.int8]]
    c_ty = np.ndarray[(M_tile * N,), np.dtype[np.int32]]
    
    # FIFOs
    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=[(M_tile * K, 1)], name="A_L2L1"
    )
    
    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=[(K * N, 1)], name="B_L2L1"
    )
    
    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=[(M_tile, N), (N, 1)] if M_tile > 1 else [(1, N), (N, 1)], 
        name="C_L2L3"
    )
    
    # Access patterns
    tap_A = TensorAccessPattern((M_tile, K), 0, [M_tile, K], [K, 1])
    tap_B = TensorAccessPattern((K, N), 0, [K, N], [N, 1])
    tap_C = TensorAccessPattern((M_tile, N), 0, [M_tile, N], [N, 1])
    
    def core_fn(of_a, of_b, of_c, matmul):
        for _ in iron.controlflow.range_(1):
            elem_out = of_c.acquire(1)
            for i in iron.controlflow.range_(M_tile * N):
                elem_out[i] = 0
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matmul(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)
    
    worker = iron.Worker(
        core_fn,
        [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), matmul_kernel],
    )
    
    def sequence(A, B, C, a_prod, b_prod, c_cons):
        tg = iron.TaskGroup()
        a_prod.fill(A, tap=tap_A, group=tg)
        b_prod.fill(B, tap=tap_B, group=tg)
        c_cons.drain(C, tap=tap_C, group=tg, wait=True)
        tg.finish()
    
    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
    )
    
    program = iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()
    
    # Compile
    xclbin_path = os.path.join(work_dir, "design.xclbin")
    insts_path = os.path.join(work_dir, "design.insts.bin")
    pdi_path = os.path.join(work_dir, "main.pdi")
    
    t0 = time.time()
    try:
        @iron.jit
        def design(A: In, B: In, C: Out, *, M_tile: CompileTime[int], K: CompileTime[int], N: CompileTime[int]):
            return build_mv_kernel_design(M_tile=M_tile, K=K, N=N, work_dir=work_dir)
        
        compiled = design.specialize(M_tile=M_tile, K=K, N=N).compile(
            xclbin_path=xclbin_path,
            inst_path=insts_path,
            pdi_path=pdi_path,
        )
    except Exception as exc:
        print(f"  COMPILE FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return False, None
    compile_time = time.time() - t0
    print(f"  Compile time: {compile_time:.1f}s")
    
    return True, (xclbin_path, insts_path, pdi_path)


def test_kernel(M_tile, K, N, artifacts):
    """Test the compiled kernel with INT8 inputs."""
    xclbin_path, insts_path, pdi_path = artifacts
    
    print(f"\n=== Testing Kernel: M_tile={M_tile}, K={K}, N={N} ===")
    
    import xrt
    
    device = xrt.device(0)
    xclbin = xrt.xclbin(xclbin_path)
    uuid = device.register_xclbin(xclbin)
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(device, uuid, "MLIR_AIE")
    
    rng = np.random.default_rng(42)
    all_pass = True
    
    for pattern_name, pattern_fn in [
        ("all_ones", lambda: (
            np.ones((M_tile, K), dtype=np.int8),
            np.ones((K, N), dtype=np.int8)
        )),
        ("random", lambda: (
            rng.integers(-128, 127, size=(M_tile, K), dtype=np.int8),
            rng.integers(-128, 127, size=(K, N), dtype=np.int8)
        )),
    ]:
        A_np, B_np = pattern_fn()
        C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
        
        A = iron.tensor(A_np, dtype=np.int8, device="npu")
        B = iron.tensor(B_np, dtype=np.int8, device="npu")
        C = iron.tensor((M_tile, N), dtype=np.int32, device="npu")
        
        design(A, B, C, M_tile=M_tile, K=K, N=N)
        C_out = C.numpy()
        
        mismatches = int(np.sum(C_out != C_ref))
        max_abs_error = int(np.max(np.abs(C_out.astype(np.int64) - C_ref.astype(np.int64))))
        
        if mismatches == 0 and max_abs_error == 0:
            print(f"  {pattern_name}: PASS")
        else:
            print(f"  {pattern_name}: FAIL - {mismatches} mismatches, max_abs_error={max_abs_error}")
            all_pass = False
    
    return all_pass


def main():
    base_dir = "/home/ubuntu12738/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/mv_mtile_test"
    os.makedirs(base_dir, exist_ok=True)
    
    # Test different M_tile sizes
    test_configs = [
        (16, 32, 1),
        (32, 32, 1),
        (64, 32, 1),
    ]
    
    results = {}
    
    for M_tile, K, N in test_configs:
        work_dir = os.path.join(base_dir, f"M{M_tile}_K{K}_N{N}")
        os.makedirs(work_dir, exist_ok=True)
        
        success, artifacts = build_mv_kernel_design(M_tile, K, N, work_dir)
        results[(M_tile, K, N)] = {"compile_success": success, "artifacts": artifacts}
        
        if success and artifacts:
            test_pass = test_kernel(M_tile, K, N, artifacts)
            results[(M_tile, K, N)]["test_pass"] = test_pass
            
            if not test_pass:
                print(f"\n*** FAILURE at M_tile={M_tile} ***")
        else:
            print(f"\n*** COMPILE FAILURE at M_tile={M_tile} ***")
    
    print("\n=== MV Kernel Summary ===")
    for config, result in results.items():
        status = "PASS" if result.get("test_pass", False) else ("COMPILE_FAIL" if not result["compile_success"] else "TEST_FAIL")
        print(f"  M_tile={config[0]:2d} K={config[1]:2d} N={config[2]}: {status}")
    
    import json
    with open(os.path.join(base_dir, "mv_mtile_results.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()