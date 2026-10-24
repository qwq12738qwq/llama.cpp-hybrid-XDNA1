#!/usr/bin/env python3
"""Phase 35: Phoenix XDNA1 GEMM scaling dataflow optimization experiments.

Builds on Phase 34's proven gen_phase21.py whole-array GEMM. Adds:
  - baseline  : Phase 34 default (fifo_depth=2)
  - depth     : same topology, deeper ObjectFifos (depth=4/6/8)
  - abcast    : A broadcast via single shim fifo + split to all rows
  - persistent: workers loop internally, sequence runs once

Fixed shape: M=512, K=2048, N=2048, INT8
Tile size:   m=64, k=64, n=64

Usage:
  python gen_phase35.py --variant baseline --tiles 4 --work-dir /tmp/p35/baseline_t4
  python gen_phase35.py --variant depth --tiles 16 --fifo-depth 4 --work-dir /tmp/p35/depth16_t4
  python gen_phase35.py --variant persistent --tiles 16 --work-dir /tmp/p35/persist_t16
"""
import argparse
import os
import sys
import time

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Runtime,
    TaskGroup,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

_TM = 64
_TK = 64
_TN = 64

_RELU_SRC = (
    "#include <stdint.h>\n"
    "extern \"C\" void relu_i32(int32_t * c) {\n"
    "    for (int32_t i = 0; i < {mn}; i++) if (c[i] < 0) c[i] = 0;\n"
    "}\n"
)
_NOOP_SRC = (
    "#include <stdint.h>\n"
    "extern \"C\" void relu_i32(int32_t * c) { (void)c; }\n"
)


def _mm_kernel(dtype_in, dtype_out):
    return kernels.mm(
        dim_m=_TM, dim_k=_TK, dim_n=_TN,
        input_dtype=dtype_in, output_dtype=dtype_out,
    )


def build_baseline(M, K, N, n_rows, n_cols, fifo_depth=2,
                   dtype_in=np.int8, dtype_out=np.int32):
    """Phase 34 baseline: same dataflow as gen_phase21.py n-split."""
    m, k, n = _TM, _TK, _TN
    n_cores = n_rows * n_cols
    matmul_kernel = _mm_kernel(dtype_in, dtype_out)
    r, s, t = matmul_kernel.mac_dims
    zero_kernel = matmul_kernel.zero

    assert M % (m * n_rows) == 0
    assert K % k == 0
    assert N % (n * n_cols) == 0
    assert m % r == 0 and k % s == 0 and n % t == 0

    nb_m_per_col = M // m // n_rows
    nb_n_per_col = N // n // n_cols
    n_tiles_per_core = nb_m_per_col * nb_n_per_col

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]

    b_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    c_l2_ty = np.ndarray[(m * n * n_rows,), np.dtype[dtype_out]]
    a_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos = []
    A_l2l1_fifos = []
    B_l3l2_fifos = []
    B_l2l1_fifos = []
    C_l1l2_fifos = [[] for _ in range(n_rows)]
    C_l2l3_fifos = []

    n_shim_A = n_rows if n_cols > n_rows else n_cols
    n_A_per_shim = n_rows // n_cols if n_cols < n_rows else 1

    for i in range(n_shim_A):
        a_l3l2 = ObjectFifo(
            np.ndarray[(m * k * n_A_per_shim,), np.dtype[dtype_in]],
            name=f"A_L3L2_{i}", depth=fifo_depth,
        )
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_per_shim
        stop_row = start_row + n_A_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        a_dims = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]] * (stop_row - start_row)
        a_tmp = a_l3l2.cons().split(
            of_offsets,
            obj_types=[a_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp)

    for col in range(n_cols):
        b_l3l2 = ObjectFifo(b_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(obj_type=b_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims)
        )

        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(c_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_dims)
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_rows)]
        c_tmp = c_l2l3.prod().join(
            of_offsets,
            obj_types=[c_l1_ty] * n_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_rows)],
            depths=[fifo_depth] * n_rows,
        )
        for j in range(n_rows):
            C_l1l2_fifos[j].append(c_tmp[j])

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        n_rows, n_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
        ),
    )
    flat_workers = [w for row in workers for w in row]

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_A_per_shim, k), (1, K // k),
        pattern_repeat=N // n // n_cols, prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N), (k, n),
        tile_group_repeats=(K // k, N // n // n_cols),
        tile_group_steps=(1, n_cols),
        tile_group_col_major=True, prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N), (m * n_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_cols),
        tile_group_steps=(1, n_cols), prune_step=False,
    )

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]
    nA, nB = len(A_prods), len(B_prods)

    def sequence(A, B, C, *fifos):
        A_hs = fifos[:nA]
        B_hs = fifos[nA:nA + nB]
        C_hs = fifos[nA + nB:]
        c_index = 0
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_rows - row_base]
                )
                for col in range(n_cols):
                    C_hs[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        tile_offset = ((row_base + tile_row) * n_shim_A + col) % len(A_tiles)
                        if col < n_rows:
                            A_hs[col].fill(A, tap=A_tiles[tile_offset], group=tg)
                        B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    print(f"[baseline] M={M} K={K} N={N} tiles={n_cores} ({n_rows}x{n_cols}) "
          f"fifo_depth={fifo_depth} n_shim_A={n_shim_A} n_A_per_shim={n_A_per_shim} "
          f"A_tiles={len(A_tiles)} B_tiles={len(B_tiles)} C_tiles={len(C_tiles)} "
          f"n_tiles_per_core={n_tiles_per_core} "
          f"tb_blocks={iron.ceildiv(M // m // n_rows, tb_max_n_rows)}",
          flush=True)
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + A_prods + B_prods + C_conses)
    return iron.Program(iron.get_current_device(), rt, workers=flat_workers).resolve_program()


def build_2dsplit(M, K, N, n_rows, n_cols, fifo_depth=2,
                  dtype_in=np.int8, dtype_out=np.int32):
    """2D-split: A is M-sliced across rows, B is N-sliced across columns."""
    m, k, n = _TM, _TK, _TN
    n_cores = n_rows * n_cols
    matmul_kernel = _mm_kernel(dtype_in, dtype_out)
    r, s, t = matmul_kernel.mac_dims
    zero_kernel = matmul_kernel.zero

    assert M % (m * n_rows) == 0
    assert K % k == 0
    assert N % (n * n_cols) == 0

    nb_m_per_row = M // m // n_rows
    nb_n_per_col = N // n // n_cols
    n_tiles_per_core = nb_m_per_row * nb_n_per_col

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    a_l2_ty = np.ndarray[(m * k,), np.dtype[dtype_in]]
    b_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    c_l2_ty = np.ndarray[(m * n,), np.dtype[dtype_out]]
    a_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos = []
    A_l2l1_fifos = []
    B_l3l2_fifos = []
    B_l2l1_fifos = []
    C_l1l2_fifos = [[] for _ in range(n_rows)]
    C_l2l3_fifos = []

    for row in range(n_rows):
        a_l3l2 = ObjectFifo(a_l2_ty, name=f"A_L3L2_{row}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
        A_l2l1_fifos.append(
            a_l3l2.cons().forward(obj_type=a_l1_ty, name=f"A_L2L1_{row}", dims_to_stream=a_dims)
        )

    for col in range(n_cols):
        b_l3l2 = ObjectFifo(b_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(obj_type=b_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims)
        )

        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(c_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_dims)
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_rows)]
        c_tmp = c_l2l3.prod().join(
            of_offsets,
            obj_types=[c_l1_ty] * n_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_rows)],
            depths=[fifo_depth] * n_rows,
        )
        for j in range(n_rows):
            C_l1l2_fifos[j].append(c_tmp[j])

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        n_rows, n_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
        ),
    )
    flat_workers = [w for row in workers for w in row]

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k),
        pattern_repeat=N // n // n_cols, prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N), (k, n),
        tile_group_repeats=(K // k, N // n // n_cols),
        tile_group_steps=(1, n_cols),
        tile_group_col_major=True, prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N), (m * n_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_cols),
        tile_group_steps=(1, n_cols), prune_step=False,
    )

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]
    nA, nB = len(A_prods), len(B_prods)

    def sequence(A, B, C, *fifos):
        A_hs = fifos[:nA]
        B_hs = fifos[nA:nA + nB]
        C_hs = fifos[nA + nB:]
        c_index = 0
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_rows - row_base]
                )
                for col in range(n_cols):
                    C_hs[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        tile_offset = ((row_base + tile_row) + col) % len(A_tiles)
                        if col < n_rows:
                            A_hs[col].fill(A, tap=A_tiles[tile_offset], group=tg)
                        B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    print(f"[2dsplit] M={M} K={K} N={N} tiles={n_cores} ({n_rows}x{n_cols}) "
          f"fifo_depth={fifo_depth} A_tiles={len(A_tiles)} B_tiles={len(B_tiles)} "
          f"C_tiles={len(C_tiles)} n_tiles_per_core={n_tiles_per_core} "
          f"tb_blocks={iron.ceildiv(M // m // n_rows, tb_max_n_rows)}",
          flush=True)
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + A_prods + B_prods + C_conses)
    return iron.Program(iron.get_current_device(), rt, workers=flat_workers).resolve_program()


def build_persistent(M, K, N, n_rows, n_cols, fifo_depth=2, n_iters=4,
                     dtype_in=np.int8, dtype_out=np.int32):
    """Persistent workers: workers loop n_iters internally, sequence runs once."""
    m, k, n = _TM, _TK, _TN
    n_cores = n_rows * n_cols
    matmul_kernel = _mm_kernel(dtype_in, dtype_out)
    r, s, t = matmul_kernel.mac_dims
    zero_kernel = matmul_kernel.zero

    assert M % (m * n_rows) == 0
    assert K % k == 0
    assert N % (n * n_cols) == 0

    nb_m_per_col = M // m // n_rows
    nb_n_per_col = N // n // n_cols
    n_tiles_per_core = nb_m_per_col * nb_n_per_col

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    b_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    c_l2_ty = np.ndarray[(m * n * n_rows,), np.dtype[dtype_out]]
    a_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos = []
    A_l2l1_fifos = []
    B_l3l2_fifos = []
    B_l2l1_fifos = []
    C_l1l2_fifos = [[] for _ in range(n_rows)]
    C_l2l3_fifos = []

    n_shim_A = n_rows if n_cols > n_rows else n_cols
    n_A_per_shim = n_rows // n_cols if n_cols < n_rows else 1

    for i in range(n_shim_A):
        a_l3l2 = ObjectFifo(
            np.ndarray[(m * k * n_A_per_shim,), np.dtype[dtype_in]],
            name=f"A_L3L2_{i}", depth=fifo_depth,
        )
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_per_shim
        stop_row = start_row + n_A_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        a_dims = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]] * (stop_row - start_row)
        a_tmp = a_l3l2.cons().split(
            of_offsets,
            obj_types=[a_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp)

    for col in range(n_cols):
        b_l3l2 = ObjectFifo(b_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(obj_type=b_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims)
        )

        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(c_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_dims)
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_rows)]
        c_tmp = c_l2l3.prod().join(
            of_offsets,
            obj_types=[c_l1_ty] * n_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_rows)],
            depths=[fifo_depth] * n_rows,
        )
        for j in range(n_rows):
            C_l1l2_fifos[j].append(c_tmp[j])

    def core_fn(in_a, in_b, out_c, zero, matmul):
        outer = range_(n_iters)
        for _ in outer:
            loop = range_(n_tiles_per_core)
            for _ in loop:
                elem_out = out_c.acquire(1)
                zero(elem_out)
                for _ in range_(K // k):
                    elem_in_a = in_a.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out)
                    in_a.release(1)
                    in_b.release(1)
                out_c.release(1)

    workers = Worker.grid(
        n_rows, n_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
            while_true=False,
        ),
    )
    flat_workers = [w for row in workers for w in row]

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_A_per_shim, k), (1, K // k),
        pattern_repeat=N // n // n_cols, prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N), (k, n),
        tile_group_repeats=(K // k, N // n // n_cols),
        tile_group_steps=(1, n_cols),
        tile_group_col_major=True, prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N), (m * n_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_cols),
        tile_group_steps=(1, n_cols), prune_step=False,
    )

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]
    nA, nB = len(A_prods), len(B_prods)

    def sequence(A, B, C, *fifos):
        A_hs = fifos[:nA]
        B_hs = fifos[nA:nA + nB]
        C_hs = fifos[nA + nB:]
        c_index = 0
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_rows - row_base]
                )
                for col in range(n_cols):
                    C_hs[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        tile_offset = ((row_base + tile_row) * n_shim_A + col) % len(A_tiles)
                        if col < n_rows:
                            A_hs[col].fill(A, tap=A_tiles[tile_offset], group=tg)
                        B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    print(f"[persistent] M={M} K={K} N={N} tiles={n_cores} ({n_rows}x{n_cols}) "
          f"fifo_depth={fifo_depth} n_iters={n_iters} n_tiles_per_core={n_tiles_per_core} "
          f"A_tiles={len(A_tiles)} B_tiles={len(B_tiles)} C_tiles={len(C_tiles)}",
          flush=True)
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + A_prods + B_prods + C_conses)
    return iron.Program(iron.get_current_device(), rt, workers=flat_workers).resolve_program()


def build(M, K, N, n_cores, variant="baseline", fifo_depth=2, n_persist_iters=4,
          dtype_in=np.int8, dtype_out=np.int32):
    if variant == "baseline":
        if n_cores <= 4:
            n_rows, n_cols = 1, n_cores
        elif n_cores == 8:
            n_rows, n_cols = 2, 4
        elif n_cores == 16:
            n_rows, n_cols = 4, 4
        else:
            raise ValueError(f"Unsupported tile count for baseline: {n_cores}")
        return build_baseline(M, K, N, n_rows, n_cols, fifo_depth=fifo_depth,
                              dtype_in=dtype_in, dtype_out=dtype_out)

    elif variant == "2d":
        if n_cores == 1:
            n_rows, n_cols = 1, 1
        elif n_cores == 4:
            n_rows, n_cols = 2, 2
        elif n_cores == 8:
            n_rows, n_cols = 2, 4
        elif n_cores == 16:
            n_rows, n_cols = 4, 4
        else:
            raise ValueError(f"Unsupported tile count for 2d: {n_cores}")
        return build_2dsplit(M, K, N, n_rows, n_cols, fifo_depth=fifo_depth,
                               dtype_in=dtype_in, dtype_out=dtype_out)

    elif variant == "persistent":
        if n_cores <= 4:
            n_rows, n_cols = 1, n_cores
        elif n_cores == 8:
            n_rows, n_cols = 2, 4
        elif n_cores == 16:
            n_rows, n_cols = 4, 4
        else:
            raise ValueError(f"Unsupported tile count for persistent: {n_cores}")
        return build_persistent(M, K, N, n_rows, n_cols, fifo_depth=fifo_depth,
                                n_iters=n_persist_iters, dtype_in=dtype_in, dtype_out=dtype_out)

    else:
        raise ValueError(f"Unknown variant: {variant}")


def cpu_int8_gemm(A, B):
    A_i32 = A.astype(np.int32)
    B_i32 = B.astype(np.int32)
    return A_i32 @ B_i32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--tiles", type=int, required=True, choices=[1, 2, 4, 8, 16])
    ap.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "depth", "2d", "persistent"])
    ap.add_argument("--fifo-depth", type=int, default=2)
    ap.add_argument("--n-persist-iters", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--dtype-in", type=str, choices=["i8", "i16", "bf16"], default="i8")
    ap.add_argument("--dtype-out", type=str, choices=["i8", "i16", "i32", "bf16", "f32"], default="i32")
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cores = args.tiles
    variant = args.variant
    if variant == "depth":
        variant = "baseline"

    dtype_in_map = {"i8": np.int8, "i16": np.int16, "bf16": np.uint16}
    dtype_out_map = {"i8": np.int8, "i16": np.int16, "i32": np.int32, "bf16": np.uint16, "f32": np.float32}
    dtype_in = dtype_in_map[args.dtype_in]
    dtype_out = dtype_out_map[args.dtype_out]

    iron.set_current_device(from_name("npu", n_cols=4))

    @iron.jit
    def design(A: In, B: In, C: Out, *,
               M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
               mt: CompileTime[int] = _TM, kt: CompileTime[int] = _TK, nt: CompileTime[int] = _TN,
               rows: CompileTime[int] = 4, cols: CompileTime[int] = 4,
               din: CompileTime[str] = args.dtype_in, dout: CompileTime[str] = args.dtype_out,
               act: CompileTime[str] = "none"):
        return build(M, K, N, n_cores, variant, fifo_depth=args.fifo_depth,
                      n_persist_iters=args.n_persist_iters,
                      dtype_in=dtype_in, dtype_out=dtype_out)

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    print(f"[phase35] M={M} K={K} N={N} tiles={n_cores} variant={variant} "
          f"fifo_depth={args.fifo_depth} dtype={args.dtype_in}->{args.dtype_out}")
    t0 = time.time()
    compiled = design.specialize(
        M=M, K=K, N=N, mt=_TM, kt=_TK, nt=_TN,
        rows=4, cols=4, din=args.dtype_in, dout=args.dtype_out, act="none",
    ).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"[phase35] compiled in {time.time() - t0:.1f}s -> {compiled[0]}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
