#!/usr/bin/env python3
"""
Stage 8: M Dimension Coverage - Test M=1,2,4,8,16 with K=32 N=16
"""
import os, sys, time
import numpy as np
sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')
import aie.iron as iron
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, Out

def build_design(A, B, C, *, M, K, N):
    matmul_kernel = mm(
        dim_m=M,
        dim_k=K,
        dim_n=N,
        input_dtype=np.int8,
        output_dtype=np.int32,
        vectorized=False,
    )
    
    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]
    
    a_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    b_ty = np.ndarray[(K * N,), np.dtype[np.int8]]
    c_ty = np.ndarray[(M * N,), np.dtype[np.int32]]
    
    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=[(M * K, 1)], name="A_L2L1"
    )
    
    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=[(K * N, 1)], name="B_L2L1"
    )
    
    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=[(M, N), (N, 1)] if M > 1 else [(1, N), (N, 1)], name="C_L2L3"
    )
    
    tap_A = TensorAccessPattern((M, K), 0, [M, K], [K, 1])
    tap_B = TensorAccessPattern((K, N), 0, [K, N], [N, 1])
    tap_C = TensorAccessPattern((M, N), 0, [M, N], [N, 1])
    
    def core_fn(of_a, of_b, of_c, matmul):
        for _ in iron.controlflow.range_(1):
            elem_out = of_c.acquire(1)
            for i in iron.controlflow.range_(M * N):
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
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def build_and_test(M, K, N, work_dir):
    print(f"\n=== Testing M={M} K={K} N={N} ===")
    
    @iron.jit
    def design(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int]):
        return build_design(A, B, C, M=M, K=K, N=N)
    
    os.makedirs(work_dir, exist_ok=True)
    xclbin_path = os.path.join(work_dir, "design.xclbin")
    insts_path = os.path.join(work_dir, "design.insts.bin")
    pdi_path = os.path.join(work_dir, "main.pdi")
    
    t0 = time.time()
    try:
        compiled = design.specialize(M=M, K=K, N=N).compile(
            xclbin_path=xclbin_path,
            inst_path=insts_path,
            pdi_path=pdi_path,
        )
    except Exception as exc:
        print(f"  COMPILE FAILED: {exc}")
        return False, None
    compile_time = time.time() - t0
    print(f"  Compile time: {compile_time:.1f}s")
    
    rng = np.random.default_rng(42)
    all_pass = True
    
    for pattern_name, pattern_fn in [
        ("all_ones", lambda: (np.ones((M, K), dtype=np.int8), np.ones((K, N), dtype=np.int8))),
        ("arange_A", lambda: (np.tile(np.arange(K, dtype=np.int8), (M, 1)), np.ones((K, N), dtype=np.int8))),
        ("arange_B", lambda: (np.ones((M, K), dtype=np.int8), np.arange(K*N, dtype=np.int8).reshape(K, N))),
        ("random", lambda: (rng.integers(-128, 127, size=(M, K), dtype=np.int8), 
                            rng.integers(-128, 127, size=(K, N), dtype=np.int8))),
        ("pos_only", lambda: (rng.integers(0, 127, size=(M, K), dtype=np.int8), 
                              rng.integers(0, 127, size=(K, N), dtype=np.int8))),
        ("neg_only", lambda: (rng.integers(-128, -1, size=(M, K), dtype=np.int8), 
                              rng.integers(-128, -1, size=(K, N), dtype=np.int8))),
        ("small_vals", lambda: (rng.integers(-2, 2, size=(M, K), dtype=np.int8), 
                                rng.integers(-2, 2, size=(K, N), dtype=np.int8))),
    ]:
        A_np, B_np = pattern_fn()
        C_ref = (A_np.astype(np.int32) @ B_np.astype(np.int32)).astype(np.int32)
        
        A = iron.tensor(A_np, dtype=np.int8, device="npu")
        B = iron.tensor(B_np, dtype=np.int8, device="npu")
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        
        design(A, B, C, M=M, K=K, N=N)
        C_out = C.numpy()
        
        mismatches = int(np.sum(C_out != C_ref))
        max_abs_error = int(np.max(np.abs(C_out.astype(np.int64) - C_ref.astype(np.int64))))
        
        if mismatches == 0 and max_abs_error == 0:
            print(f"  {pattern_name}: PASS")
        else:
            print(f"  {pattern_name}: FAIL - {mismatches} mismatches, max_abs_error={max_abs_error}")
            all_pass = False
    
    return all_pass, (xclbin_path, insts_path, pdi_path)


def main():
    base_dir = "/home/ubuntu12738/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/m_dimension_coverage"
    M_values = [1, 2, 4, 8, 16]
    K, N = 32, 16
    
    results = {}
    
    for M in M_values:
        work_dir = os.path.join(base_dir, f"M{M}_K{K}_N{N}")
        success, artifacts = build_and_test(M, K, N, work_dir)
        results[M] = {"success": success, "artifacts": artifacts}
        if not success:
            print(f"\n*** FAILURE at M={M} ***")
    
    print("\n=== M Dimension Coverage Summary ===")
    for M, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        print(f"  M={M:2d}: {status}")
    
    import json
    with open(os.path.join(base_dir, "m_dimension_results.json"), "w") as f:
        json.dump({str(k): v["success"] for k, v in results.items()}, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()