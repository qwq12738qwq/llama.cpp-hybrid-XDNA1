#!/usr/bin/env python3
# trace_with_existing.py — Use existing int8_gemm.py with trace inputs

import sys
import os
os.environ['MLIR_AIE_INSTALL_DIR'] = '/root/src/mlir-aie'

sys.path.insert(0, "/home/ubuntu12738/APPS/Projects/llama.cpp/experiments/npu-int8-gemm")

import time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern


_TILE_M = 64
_TILE_K = 64
_TILE_N = 64


def _build_design(M, K, N):
    """Copy from int8_gemm.py"""
    m, k, n = _TILE_M, _TILE_K, _TILE_N

    matmul_kernel = mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=np.int8,
        output_dtype=np.int32,
        vectorized=True,
    )
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

    a_ty = np.ndarray[(m * k,), np.dtype[np.int8]]
    b_ty = np.ndarray[(k * n,), np.dtype[np.int8]]
    c_ty = np.ndarray[(m * n,), np.dtype[np.int32]]

    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
    )

    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
    )

    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(m, n),
        offset=0,
        sizes=[m // r, r, n // t, t],
        strides=[r * n, t, r * t, 1],
    )
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3"
    )

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in iron.controlflow.range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            for i in iron.controlflow.range_(m * n):
                elem_out[i] = 0
            for _ in iron.controlflow.range_(K // k):
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

    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n)
    )
    if K > 512:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, N // n), tile_group_col_major=False
        )[0]
        b_taps = [b_tap]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
        )[0]
        b_taps = [b_tap]

    c_taps = []
    for tile_row in range(M // m):
        for tile_col in range(N // n):
            offset = tile_row * m * N + tile_col * n
            c_tap = TensorAccessPattern(
                tensor_dims=(M, N),
                offset=offset,
                sizes=[m, n],
                strides=[N, 1],
            )
            c_taps.append(c_tap)

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // m):
            for tile_col in range(N // n):
                tg = iron.TaskGroup()
                a_prod.fill(A, tap=a_taps[tile_row], group=tg)
                b_prod.fill(B, tap=b_tap, group=tg)
                c_cons.drain(C, tap=c_taps[tile_row * (N // n) + tile_col], group=tg, wait=True)
                tg.finish()

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
    )
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def create_trace_inputs(M, K, N):
    """Create input matrices with unique tile IDs for tracing."""
    m, k, n = _TILE_M, _TILE_K, _TILE_N
    
    # A matrix: all 1s (so output = sum of B tiles along K)
    A = np.ones((M, K), dtype=np.int8)
    
    # B matrix: each tile has unique value
    B = np.zeros((K, N), dtype=np.int8)
    K_tiles = K // k
    N_tiles = N // n
    
    for kt in range(K_tiles):
        for nt in range(N_tiles):
            tile_id = kt * N_tiles + nt
            # Unique value per tile
            val = (tile_id * 7 + 1) % 127
            if val == 0:
                val = 1
            row_start = kt * k
            col_start = nt * n
            B[row_start:row_start+k, col_start:col_start+n] = val
    
    return A, B, K_tiles, N_tiles


def cpu_gemm(A, B):
    """CPU reference: int8 multiply, int32 accumulation."""
    A_i32 = A.astype(np.int32)
    B_i32 = B.astype(np.int32)
    return A_i32 @ B_i32


def analyze_output(C_npu, C_cpu, M, N, K_tiles, N_tiles):
    """Compare NPU output with CPU reference."""
    m, n = _TILE_M, _TILE_N
    
    print(f"\nOutput Analysis (M={M}, N={N}, K_tiles={K_tiles}, N_tiles={N_tiles}):")
    
    for tile_row in range(M // m):
        for tile_col in range(N // n):
            row_start = tile_row * m
            col_start = tile_col * n
            
            npu_tile = C_npu[row_start:row_start+m, col_start:col_start+n]
            cpu_tile = C_cpu[row_start:row_start+m, col_start:col_start+n]
            
            npu_sum = np.sum(npu_tile)
            cpu_sum = np.sum(cpu_tile)
            
            match = np.array_equal(npu_tile, cpu_tile)
            
            print(f"  C[{tile_row},{tile_col}]: NPU_sum={npu_sum}, CPU_sum={cpu_sum}, Match={match}")
            
            if not match:
                # Show first few elements
                print(f"    NPU[0:4,0:4] = {npu_tile[0:4, 0:4].flatten()}")
                print(f"    CPU[0:4,0:4] = {cpu_tile[0:4, 0:4].flatten()}")


def main():
    test_cases = [
        (64, 1024, 64),   # PASS baseline
        (64, 1024, 128),  # FAIL
    ]

    for M, K, N in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: M={M}, K={K}, N={N}")
        
        A, B, K_tiles, N_tiles = create_trace_inputs(M, K, N)
        print(f"Created trace inputs: K_tiles={K_tiles}, N_tiles={N_tiles}")
        
        # Build and compile
        print("Building design...")
        design_module = _build_design(M, K, N)
        
        # Compile with @iron.jit wrapper
        @iron.jit
        def int8_gemm_design(
            A: In,
            B: In,
            C: Out,
            *,
            M: CompileTime[int] = M,
            K: CompileTime[int] = K,
            N: CompileTime[int] = N,
        ):
            return _build_design(M, K, N)
        
        work_dir = f"/experiments/npu-int8-gemm/tile_order_forensics/run_M{M}_K{K}_N{N}"
        os.makedirs(work_dir, exist_ok=True)
        compile_kwargs = {
            "xclbin_path": os.path.join(work_dir, "design.xclbin"),
            "inst_path": os.path.join(work_dir, "design.insts.bin"),
        }
        
        print("Compiling...")
        try:
            compiled = int8_gemm_design.specialize(M=M, K=K, N=N).compile(**compile_kwargs)
            print(f"  xclbin: {compiled[0]}")
            print(f"  insts:  {compiled[1]}")
        except Exception as e:
            print(f"Compilation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Run
        A_t = iron.tensor(A, dtype=np.int8, device="npu")
        B_t = iron.tensor(B, dtype=np.int8, device="npu")
        C_t = iron.tensor((M, N), dtype=np.int32, device="npu")
        
        print("Running...")
        try:
            ret = int8_gemm_design(A_t, B_t, C_t, M=M, K=K, N=N)
            C_npu = C_t.numpy()
        except Exception as e:
            print(f"Execution failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # CPU reference
        C_cpu = cpu_gemm(A, B)
        
        # Analyze
        analyze_output(C_npu, C_cpu, M, N, K_tiles, N_tiles)


if __name__ == "__main__":
    main()