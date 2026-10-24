#!/usr/bin/env python3
# trace_bd.py — Generate and inspect lowered MLIR DMA BD chain
# Goal: Reconstruct tile sequence from BD descriptors

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


def build_design_with_bd(M, K, N, work_dir):
    """Build design and emit MLIR for BD inspection."""
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
    # B tiling: col-major for K<=512, row-major for K>512
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

    # Emit MLIR
    os.makedirs(work_dir, exist_ok=True)
    mlir_path = os.path.join(work_dir, "design.mlir")
    module = prog.resolve_program()
    with open(mlir_path, 'w') as f:
        f.write(str(module))
    print(f"Emitted MLIR to {mlir_path}")

    # Also try to compile and get the lowered MLIR
    try:
        xclbin_path = os.path.join(work_dir, "design.xclbin")
        inst_path = os.path.join(work_dir, "design.insts.bin")
        compiled = prog.resolve_program(xclbin_path=xclbin_path, inst_path=inst_path)
        print(f"Compiled successfully")
        print(f"  xclbin: {compiled[0]}")
        print(f"  insts:  {compiled[1]}")
    except Exception as e:
        print(f"Compilation failed: {e}")

    return prog, b_tap


def parse_bd_from_mlir(mlir_path):
    """Parse aie.dma_bd operations from MLIR."""
    with open(mlir_path, 'r') as f:
        content = f.read()

    # Find aie.dma_bd ops
    import re
    bd_pattern = r'aie\.dma_bd\s+[^:]+:\s*\(([^)]+)\)\s*->\s*\([^)]+\)\s*\{[^}]*\}'
    matches = re.findall(r'aie\.dma_bd\s+[^:]+:\s*\(([^)]+)\)\s*->\s*\([^)]+\)\s*\{([^}]*)\}', content)

    print(f"\nFound {len(matches)} DMA BD operations")
    for i, (args, attrs) in enumerate(matches):
        print(f"  BD[{i}]: args=({args}) attrs={{{attrs}}}")

    # Also look for aie.objectfifo
    of_pattern = r'aie\.objectfifo\s+@([^:]+):\s*\(([^)]+)\)\s*->\s*\([^)]+\)\s*\{([^}]*)\}'
    of_matches = re.findall(of_pattern, content)
    print(f"\nFound {len(of_matches)} ObjectFifos")
    for name, args, attrs in of_matches:
        print(f"  ObjectFifo @{name}: args=({args}) attrs={{{attrs}}}")

    return matches


def main():
    print("DMA BD Chain Trace")
    print("=" * 60)

    test_cases = [
        (64, 64, 64),    # PASS baseline
        (64, 128, 64),   # K=128, N=64
        (64, 128, 128),  # K=128, N=128
        (64, 1024, 64),  # PASS baseline
        (64, 1024, 128), # FAIL
        (64, 1024, 256), # FAIL
    ]

    for M, K, N in test_cases:
        work_dir = f"/experiments/npu-int8-gemm/tile_order_forensics/bd_M{M}_K{K}_N{N}"
        print(f"\n{'='*60}")
        print(f"Shape: M={M}, K={K}, N={N}")
        prog, b_tap = build_design_with_bd(M, K, N, work_dir)

        # Parse MLIR for BD info
        mlir_path = os.path.join(work_dir, "design.mlir")
        if os.path.exists(mlir_path):
            parse_bd_from_mlir(mlir_path)

        # Print B TAP transformation_dims
        print(f"\n  B TAP transformation_dims: {b_tap.transformation_dims}")
        print(f"  B TAP sizes: {b_tap.sizes}")
        print(f"  B TAP strides: {b_tap.strides}")


if __name__ == "__main__":
    main()