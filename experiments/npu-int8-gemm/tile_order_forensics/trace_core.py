#!/usr/bin/env python3
# trace_core.py — Instrument core to trace actual acquire order
# Goal: Observe what tile IDs the core receives at each acquire

import sys
import os
os.environ['MLIR_AIE_INSTALL_DIR'] = '/root/src/mlir-aie'

sys.path.insert(0, "/home/ubuntu12738/APPS/Projects/llama.cpp/experiments/npu-int8-gemm")

import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
from aie.utils import config


_TILE_M = 64
_TILE_K = 64
_TILE_N = 64


def build_debug_design(M, K, N, work_dir):
    """Build design with debug kernel that logs acquire order."""
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
    tap_A_L2L1 = TensorTiler2D.group_tiler((M, K), (m, k), (M // m, K // k))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
    )

    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    if K > 512:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, N // n), tile_group_col_major=False
        )[0]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
        )[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=b_tap.transformation_dims, name="B_L2L1"
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

    # Debug kernel source - encodes tile IDs in output
    debug_kernel_source = f"""
// Debug kernel that traces acquire order
// Each B tile gets a unique ID: (k_tile * N_tiles + n_tile) * 1000
// The kernel accumulates these IDs to trace order

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"
#include <cstdint>

extern "C" void debug_matmul_i8_i32(
    const int8_t* __restrict a,
    const int8_t* __restrict b,
    int32_t* __restrict c
) {{
    // Tile dimensions: {m}x{k} * {k}x{n} -> {m}x{n}
    constexpr int M = {m};
    constexpr int K = {k};
    constexpr int N = {n};
    
    // We can't easily track tile index from inside kernel without passing it
    // Instead, we'll encode the tile ID in the B matrix data itself
    // This kernel just does the matmul, the tracing is done via input data
    for (int i = 0; i < M; ++i) {{
        for (int j = 0; j < N; ++j) {{
            int32_t sum = 0;
            for (int kk = 0; kk < K; ++kk) {{
                sum += static_cast<int32_t>(a[i * K + kk]) * static_cast<int32_t>(b[kk * N + j]);
            }}
            c[i * N + j] += sum;
        }}
    }}
}}
"""

    # Write debug kernel
    os.makedirs(work_dir, exist_ok=True)
    kernel_file = os.path.join(work_dir, "debug_matmul.cc")
    with open(kernel_file, 'w') as f:
        f.write(debug_kernel_source)

    # Use ExternalFunction for debug kernel
    from aie.iron.kernel import ExternalFunction
    debug_matmul = ExternalFunction(
        "debug_matmul_i8_i32",
        source_file=kernel_file,
        arg_types=[
            np.ndarray[(m * k,), np.dtype[np.int8]],
            np.ndarray[(k * n,), np.dtype[np.int8]],
            np.ndarray[(m * n,), np.dtype[np.int32]],
        ],
        include_dirs=[str(config.cxx_header_path())],
        compile_flags=[f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"],
        object_file_name=f"debug_matmul_{m}_{k}_{n}.o",
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
        [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), debug_matmul],
    )

    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n)
    )
    if K > 512:
        b_taps = [b_tap]
    else:
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
    prog = iron.Program(iron.get_current_device(), rt, workers=[worker])

    os.makedirs(work_dir, exist_ok=True)
    mlir_path = os.path.join(work_dir, "design.mlir")
    module = prog.resolve_program()
    with open(mlir_path, 'w') as f:
        f.write(str(module))
    print(f"Emitted MLIR to {mlir_path}")

    return prog, b_tap


def create_trace_inputs(M, K, N):
    """Create input matrices with unique tile IDs for tracing."""
    m, k, n = _TILE_M, _TILE_K, _TILE_N
    
    # A matrix: all 1s
    A = np.ones((M, K), dtype=np.int8)
    
    # B matrix: each tile has unique ID
    B = np.zeros((K, N), dtype=np.int8)
    K_tiles = K // k
    N_tiles = N // n
    
    for kt in range(K_tiles):
        for nt in range(N_tiles):
            tile_id = kt * N_tiles + nt
            # Fill tile with tile_id (mod 128 for int8)
            val = (tile_id * 7 + 1) % 127  # Unique non-zero pattern per tile
            if val == 0:
                val = 1
            row_start = kt * k
            col_start = nt * n
            B[row_start:row_start+k, col_start:col_start+n] = val
    
    return A, B


def decode_output(C, M, N):
    """Decode output to infer acquire order."""
    m, n = _TILE_M, _TILE_N
    K_tiles = 1024 // _TILE_K  # Hardcoded for K=1024
    N_tiles = N // n
    
    print(f"Output shape: {C.shape}")
    print(f"Output tiles: {M//m} x {N//n}")
    
    # Each output tile accumulates over K tiles
    # The value in each output tile = sum over K of (A_tile * B_tile)
    # Since A=1, output = sum of B tile values for that output tile
    
    for tile_row in range(M // m):
        for tile_col in range(N // n):
            row_start = tile_row * m
            col_start = tile_col * n
            tile_data = C[row_start:row_start+m, col_start:col_start+n]
            total = np.sum(tile_data)
            print(f"  C[{tile_row},{tile_col}]: sum={total}, sample={tile_data.flat[:4]}")


def main():
    print("Core Acquire Order Trace")
    print("=" * 60)

    # Test cases
    test_cases = [
        (64, 1024, 64),   # PASS baseline
        (64, 1024, 128),  # FAIL
    ]

    for M, K, N in test_cases:
        print(f"\n{'='*60}")
        print(f"Shape: M={M}, K={K}, N={N}")
        
        work_dir = f"/experiments/npu-int8-gemm/tile_order_forensics/core_M{M}_K{K}_N{N}"
        
        prog, b_tap = build_debug_design(M, K, N, work_dir)
        
        # Create trace inputs
        A, B = create_trace_inputs(M, K, N)
        print(f"Created trace inputs")
        print(f"  B tile values (first few):")
        K_tiles = K // _TILE_K
        N_tiles = N // _TILE_N
        for kt in range(min(K_tiles, 4)):
            for nt in range(min(N_tiles, 4)):
                val = B[kt*_TILE_K, nt*_TILE_N]
                print(f"    B[{kt},{nt}] = {val}")

        # Compile
        try:
            xclbin_path = os.path.join(work_dir, "design.xclbin")
            inst_path = os.path.join(work_dir, "design.insts.bin")
            compiled = prog.resolve_program(xclbin_path=xclbin_path, inst_path=inst_path)
            print(f"Compiled: xclbin={compiled[0]}")
            
            # Run
            A_t = iron.tensor(A, dtype=np.int8, device="npu")
            B_t = iron.tensor(B, dtype=np.int8, device="npu")
            C_t = iron.tensor((M, N), dtype=np.int32, device="npu")
            
            # Need to call the design - but it's not a @iron.jit function
            # The design is in prog, we need to extract the runtime sequence
            # For now, just show the input preparation
            print("Ready to run (but need @iron.jit wrapper)")
            
        except Exception as e:
            print(f"Compilation/execution failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()