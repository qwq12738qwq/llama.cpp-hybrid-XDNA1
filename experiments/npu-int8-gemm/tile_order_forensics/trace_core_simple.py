#!/usr/bin/env python3
# trace_core_simple.py — Use @iron.jit pattern to trace core acquire order
# Uses unique tile IDs in B matrix to infer acquire sequence

import sys
import os
os.environ['MLIR_AIE_INSTALL_DIR'] = '/root/src/mlir-aie'

sys.path.insert(0, "/home/ubuntu12738/APPS/Projects/llama.cpp/experiments/npu-int8-gemm")

import argparse
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
    # For K > 512 the single column-major L3L2 B BD exceeds aiecc's
    # iteration-wrap limit.  Switch B to row-major tiling so the BD
    # stays within the limit.  Keep A unchanged.
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
    
    # A matrix: all 1s (so output = sum of B tiles)
    A = np.ones((M, K), dtype=np.int8)
    
    # B matrix: each tile has unique value
    B = np.zeros((K, N), dtype=np.int8)
    K_tiles = K // k
    N_tiles = N // n
    
    for kt in range(K_tiles):
        for nt in range(N_tiles):
            tile_id = kt * N_tiles + nt
            # Unique value per tile: use (tile_id * 7 + 1) % 127
            val = (tile_id * 7 + 1) % 127
            if val == 0:
                val = 1
            row_start = kt * k
            col_start = nt * n
            B[row_start:row_start+k, col_start:col_start+n] = val
    
    return A, B, K_tiles, N_tiles


def analyze_output(C, M, N, K_tiles, N_tiles):
    """Analyze output to infer acquire order."""
    m, n = _TILE_M, _TILE_N
    
    print(f"\nOutput Analysis (M={M}, N={N}, K_tiles={K_tiles}, N_tiles={N_tiles}):")
    print(f"Expected: each output tile sums K_tiles B-tiles")
    
    for tile_row in range(M // m):
        for tile_col in range(N // n):
            row_start = tile_row * m
            col_start = tile_col * n
            tile_data = C[row_start:row_start+m, col_start:col_start+n]
            total = np.sum(tile_data)
            
            # Expected sum if core acquires in order: B[0,tile_col] + B[1,tile_col] + ... + B[K_tiles-1,tile_col]
            expected_sum = 0
            for kt in range(K_tiles):
                tile_id = kt * N_tiles + tile_col
                val = (tile_id * 7 + 1) % 127
                if val == 0:
                    val = 1
                expected_sum += val * (m * n)  # each element in tile
            
            # The output tile is m x n, each element gets the sum
            # Actually, matmul: C[i,j] = sum_k A[i,k] * B[k,j]
            # A = 1, so C[i,j] = sum_k B[k,j]
            # For a given output tile (tile_row, tile_col), it accumulates over all K tiles
            # B tile (kt, tile_col) contributes to this output tile
            # Each element in output tile gets sum of column j of all B tiles
            
            # Let's check a specific element
            elem_0_0 = tile_data[0, 0]
            elem_mid = tile_data[m//2, n//2]
            
            print(f"  C[{tile_row},{tile_col}]: C[0,0]={elem_0_0}, C[mid]={elem_mid}, sum={total}")
            print(f"    Expected per-element sum: {expected_sum // (m*n) if m*n > 0 else 'N/A'}")
            
            # The per-element value should equal sum of B column values
            # B[k_tile, tile_col][k_local, j_local] where j_local varies
            # For j_local=0: B[kt, tile_col][:, 0] = val (all same in our construction)
            # So each element gets K_tiles * val
            # But val differs per K tile
            
            # Let's compute expected per-element for column 0 of this output tile
            expected_elem = 0
            for kt in range(K_tiles):
                tile_id = kt * N_tiles + tile_col
                val = (tile_id * 7 + 1) % 127
                if val == 0:
                    val = 1
                expected_elem += val
            print(f"    Expected C[i,0] = {expected_elem}")


def main():
    test_cases = [
        (64, 1024, 64),   # PASS baseline
        (64, 1024, 128),  # FAIL
        (64, 1024, 256),  # FAIL
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
        compiled = int8_gemm_design.specialize(M=M, K=K, N=N).compile(**compile_kwargs)
        print(f"  xclbin: {compiled[0]}")
        print(f"  insts:  {compiled[1]}")
        
        # Run
        A_t = iron.tensor(A, dtype=np.int8, device="npu")
        B_t = iron.tensor(B, dtype=np.int8, device="npu")
        C_t = iron.tensor((M, N), dtype=np.int32, device="npu")
        
        print("Running...")
        ret = int8_gemm_design(A_t, B_t, C_t, M=M, K=K, N=N)
        
        C_out = C_t.numpy()
        analyze_output(C_out, M, N, K_tiles, N_tiles)


if __name__ == "__main__":
    main()