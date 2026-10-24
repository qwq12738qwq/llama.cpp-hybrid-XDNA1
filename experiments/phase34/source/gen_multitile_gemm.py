#!/usr/bin/env python3
"""Phase 34: Phoenix XDNA whole-array multi-tile INT8 GEMM.

Builds a true distributed GEMM across N AIE compute tiles using IRON
Worker.grid + ObjectFifo broadcast/distribute.  Only int8->int32,
no scale dequant, no FP32/BF16.

Dataflows
---------
N-split  : B and C are N-sliced across columns, A is broadcast.
M-split  : A and C are M-sliced across rows,      B is broadcast.
2D-split : A is M-sliced across rows, B is N-sliced across columns,
           each (row, col) pair computes its own C tile.

Usage
-----
  source /Storage/Cache/ironenv-v143/bin/activate
  python gen_multitile_gemm.py --M 512 --K 2048 --N 2048 --tiles 4 --split n --work-dir /tmp/p34
  python gen_multitile_gemm.py --M 512 --K 2048 --N 2048 --tiles 16 --split 2d --work-dir /tmp/p34
"""

import argparse
import math
import os
import sys
import time

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Runtime,
    TaskGroup,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

_TM = 64
_TK = 64
_TN = 64


def _build_nsplit(M, K, N, n_rows, n_cols):
    """N-split: distribute B and C across n_cols columns, broadcast A."""
    m, k, n = _TM, _TK, _TN
    n_cores = n_rows * n_cols

    assert M % (m * n_rows) == 0, f"M={M} not divisible by {m * n_rows}"
    assert K % k == 0, f"K={K} not divisible by {k}"
    assert N % (n * n_cols) == 0, f"N={N} not divisible by {n * n_cols}"
    assert m % 4 == 0 and k % 8 == 0 and n % 8 == 0

    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k, dim_n=n,
        input_dtype=np.int8, output_dtype=np.int32,
    )
    r, s, t = matmul_kernel.mac_dims
    zero_kernel = matmul_kernel.zero

    A_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    B_ty = np.ndarray[(K * N,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.int32]]
    b_l2_ty = np.ndarray[(k * n,), np.dtype[np.int8]]
    c_l2_ty = np.ndarray[(m * n * n_rows,), np.dtype[np.int32]]
    a_l1_ty = np.ndarray[(m, k), np.dtype[np.int8]]
    b_l1_ty = np.ndarray[(k, n), np.dtype[np.int8]]
    c_l1_ty = np.ndarray[(m, n), np.dtype[np.int32]]

    fifo_depth = 2
    n_per_col = N // n_cols
    nb_n_per_col = n_per_col // n
    nb_m_per_col = M // m // n_rows
    n_tiles_per_core = nb_m_per_col * nb_n_per_col

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
            np.ndarray[(m * k * n_A_per_shim,), np.dtype[np.int8]],
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
            b_l3l2.cons().forward(
                obj_type=b_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims,
            )
        )

        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(
            c_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_dims,
        )
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

    A_tiles = []
    for mt in range(nb_m_per_col):
        offset = mt * m * n_rows * K
        a_tap = TensorAccessPattern(
            tensor_dims=(M, K),
            offset=offset,
            sizes=[m * n_rows, K],
            strides=[K, 1],
        )
        A_tiles.append(a_tap)

    B_tiles = []
    for col in range(n_cols):
        col_n_offset = col * n_per_col
        b_tap = TensorAccessPattern(
            tensor_dims=(K, N),
            offset=col_n_offset,
            sizes=[K, n_per_col],
            strides=[N, 1],
        )
        B_tiles.append(b_tap)

    C_tiles = []
    for col in range(n_cols):
        col_n_offset = col * n_per_col
        for mt in range(nb_m_per_col):
            for nt in range(nb_n_per_col):
                n_tile_global = col * nb_n_per_col + nt
                offset = (mt * m * n_rows) * N + col_n_offset + nt * n
                c_tap = TensorAccessPattern(
                    tensor_dims=(M, N),
                    offset=offset,
                    sizes=[m * n_rows, n],
                    strides=[N, 1],
                )
                C_tiles.append(c_tap)

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]
    nA, nB = len(A_prods), len(B_prods)

    def sequence(A, B, C, *fifos):
        A_hs = fifos[:nA]
        B_hs = fifos[nA:nA + nB]
        C_hs = fifos[nA + nB:]
        c_index = 0
        for mt in range(nb_m_per_col):
            for col in range(n_cols):
                tg = TaskGroup()
                for row in range(n_rows):
                    tile_offset = (mt * n_shim_A + row) % len(A_tiles)
                    A_hs[row].fill(A, tap=A_tiles[tile_offset], group=tg)
                B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                C_hs[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                c_index += 1
                tg.finish()

    print(f"[gen_nsplit] M={M} K={K} N={N} tiles={n_cores} ({n_rows}x{n_cols}) "
          f"n_shim_A={n_shim_A} n_A_per_shim={n_A_per_shim} "
          f"A_tiles={len(A_tiles)} B_tiles={len(B_tiles)} C_tiles={len(C_tiles)} "
          f"n_tiles_per_core={n_tiles_per_core}",
          flush=True)
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + A_prods + B_prods + C_conses)
    return iron.Program(iron.get_current_device(), rt, workers=flat_workers).resolve_program()


def _build_msplit(M, K, N, n_rows, n_cols):
    """M-split: distribute A and C across n_rows rows, broadcast B."""
    m, k, n = _TM, _TK, _TN
    n_cores = n_rows * n_cols

    assert M % (m * n_rows) == 0
    assert K % k == 0
    assert N % n == 0
    assert n_cols == 1, "M-split uses a single column"

    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k, dim_n=n,
        input_dtype=np.int8, output_dtype=np.int32,
    )
    r, s, t = matmul_kernel.mac_dims
    zero_kernel = matmul_kernel.zero

    A_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    B_ty = np.ndarray[(K * N,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.int32]]
    a_l2_ty = np.ndarray[(m * k,), np.dtype[np.int8]]
    c_l2_ty = np.ndarray[(m * n,), np.dtype[np.int32]]
    a_l1_ty = np.ndarray[(m, k), np.dtype[np.int8]]
    b_l1_ty = np.ndarray[(k, n), np.dtype[np.int8]]
    c_l1_ty = np.ndarray[(m, n), np.dtype[np.int32]]

    fifo_depth = 2
    nb_m_per_row = M // m // n_rows
    nb_n = N // n
    n_tiles_per_core = nb_m_per_row * nb_n

    A_l3l2_fifos = []
    A_l2l1_fifos = []
    B_l3l2_fifos = []
    B_l2l1_fifos = []
    C_l1l2_fifos = []
    C_l2l3_fifos = []

    for row in range(n_rows):
        a_l3l2 = ObjectFifo(a_l2_ty, name=f"A_L3L2_{row}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
        A_l2l1_fifos.append(
            a_l3l2.cons().forward(
                obj_type=a_l1_ty, name=f"A_L2L1_{row}", dims_to_stream=a_dims,
            )
        )

        c_l2l3 = ObjectFifo(
            c_l2_ty, name=f"C_L2L3_{row}", depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        C_l2l3_fifos.append(c_l2l3)
        C_l1l2_fifos.append(
            c_l2l3.prod().forward(obj_type=c_l1_ty, name=f"C_L1L2_{row}")
        )

    b_l3l2 = ObjectFifo(
        np.ndarray[(K * N,), np.dtype[np.int8]], name="B_L3L2", depth=fifo_depth,
    )
    B_l3l2_fifos.append(b_l3l2)
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    B_l2l1_fifos.append(
        b_l3l2.cons().forward(
            obj_type=b_l1_ty, name="B_L2L1", dims_to_stream=b_dims,
        )
    )

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
                B_l2l1_fifos[0].cons(),
                C_l1l2_fifos[row].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
        ),
    )
    flat_workers = [w for row in workers for w in row]

    A_tiles = []
    for row in range(n_rows):
        offset = row * m * K
        a_tap = TensorAccessPattern(
            tensor_dims=(M, K),
            offset=offset,
            sizes=[m, K],
            strides=[K, 1],
        )
        A_tiles.append(a_tap)

    B_tiles = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), prune_step=False,
    )[0]

    C_tiles = []
    for row in range(n_rows):
        for nt in range(nb_n):
            offset = (row * m * N + nt * n) * 4
            c_tap = TensorAccessPattern(
                tensor_dims=(M, N),
                offset=offset,
                sizes=[m, n],
                strides=[N, 1],
            )
            C_tiles.append(c_tap)

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]
    nA, nB = len(A_prods), len(B_prods)

    def sequence(A, B, C, *fifos):
        A_hs = fifos[:nA]
        B_hs = fifos[nA:nA + nB]
        C_hs = fifos[nA + nB:]
        for row in range(n_rows):
            for nt in range(nb_n):
                tg = TaskGroup()
                A_hs[row].fill(A, tap=A_tiles[row], group=tg)
                B_hs[0].fill(B, tap=B_tiles, group=tg)
                C_hs[row].drain(C, tap=C_tiles[row * nb_n + nt], wait=True, group=tg)
                tg.finish()

    print(f"[gen_msplit] M={M} K={K} N={N} tiles={n_cores} ({n_rows}x{n_cols}) "
          f"n_tiles_per_core={n_tiles_per_core}", flush=True)
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + A_prods + B_prods + C_conses)
    return iron.Program(iron.get_current_device(), rt, workers=flat_workers).resolve_program()


def _build_2dsplit(M, K, N, n_rows, n_cols):
    """2D-split: A is M-sliced across rows, B is N-sliced across columns."""
    m, k, n = _TM, _TK, _TN
    n_cores = n_rows * n_cols

    assert M % (m * n_rows) == 0
    assert K % k == 0
    assert N % (n * n_cols) == 0

    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k, dim_n=n,
        input_dtype=np.int8, output_dtype=np.int32,
    )
    r, s, t = matmul_kernel.mac_dims
    zero_kernel = matmul_kernel.zero

    A_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    B_ty = np.ndarray[(K * N,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.int32]]
    a_l2_ty = np.ndarray[(m * k,), np.dtype[np.int8]]
    b_l2_ty = np.ndarray[(k * n,), np.dtype[np.int8]]
    c_l2_ty = np.ndarray[(m * n,), np.dtype[np.int32]]
    a_l1_ty = np.ndarray[(m, k), np.dtype[np.int8]]
    b_l1_ty = np.ndarray[(k, n), np.dtype[np.int8]]
    c_l1_ty = np.ndarray[(m, n), np.dtype[np.int32]]

    fifo_depth = 2
    nb_m_per_row = M // m // n_rows
    nb_n_per_col = N // n // n_cols
    n_tiles_per_core = nb_m_per_row * nb_n_per_col

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
            a_l3l2.cons().forward(
                obj_type=a_l1_ty, name=f"A_L2L1_{row}", dims_to_stream=a_dims,
            )
        )

        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(
            np.ndarray[(m * n * n_cols,), np.dtype[np.int32]],
            name=f"C_L2L3_{row}", depth=fifo_depth, dims_to_stream=c_dims,
        )
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_cols)]
        c_tmp = c_l2l3.prod().join(
            of_offsets,
            obj_types=[c_l1_ty] * n_cols,
            names=[f"C_L1L2_{row}_{col}" for col in range(n_cols)],
            depths=[fifo_depth] * n_cols,
        )
        for col in range(n_cols):
            C_l1l2_fifos[row].append(c_tmp[col])

    for col in range(n_cols):
        b_l3l2 = ObjectFifo(b_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(
                obj_type=b_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims,
            )
        )

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

    A_tiles = []
    for row in range(n_rows):
        offset = row * m * K
        a_tap = TensorAccessPattern(
            tensor_dims=(M, K),
            offset=offset,
            sizes=[m, K],
            strides=[K, 1],
        )
        A_tiles.append(a_tap)

    B_tiles = []
    for col in range(n_cols):
        offset = col * n
        b_tap = TensorAccessPattern(
            tensor_dims=(K, N),
            offset=offset,
            sizes=[K, n],
            strides=[N, 1],
        )
        B_tiles.append(b_tap)

    C_tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            offset = row * m * N + col * n
            c_tap = TensorAccessPattern(
                tensor_dims=(M, N),
                offset=offset,
                sizes=[m, n],
                strides=[N, 1],
            )
            C_tiles.append(c_tap)

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]
    nA, nB = len(A_prods), len(B_prods)

    def sequence(A, B, C, *fifos):
        A_hs = fifos[:nA]
        B_hs = fifos[nA:nA + nB]
        C_hs = fifos[nA + nB:]
        for row in range(n_rows):
            for col in range(n_cols):
                tg = TaskGroup()
                A_hs[row].fill(A, tap=A_tiles[row], group=tg)
                B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                idx = row * n_cols + col
                C_hs[row].drain(C, tap=C_tiles[idx], wait=True, group=tg)
                tg.finish()

    print(f"[gen_2dsplit] M={M} K={K} N={N} tiles={n_cores} ({n_rows}x{n_cols}) "
          f"n_tiles_per_core={n_tiles_per_core}", flush=True)
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + A_prods + B_prods + C_conses)
    return iron.Program(iron.get_current_device(), rt, workers=flat_workers).resolve_program()


def build(M, K, N, n_cores, split="n"):
    if split == "n":
        if n_cores <= 4:
            n_rows, n_cols = 1, n_cores
        elif n_cores == 8:
            n_rows, n_cols = 2, 4
        elif n_cores == 16:
            n_rows, n_cols = 4, 4
        else:
            raise ValueError(f"Unsupported tile count for N-split: {n_cores}")
        return _build_nsplit(M, K, N, n_rows, n_cols)
    elif split == "m":
        if n_cores <= 4:
            n_rows, n_cols = n_cores, 1
        elif n_cores == 8:
            n_rows, n_cols = 8, 1  # needs 8 core rows, only 4 available -> FAIL
        elif n_cores == 16:
            n_rows, n_cols = 16, 1  # needs 16 core rows -> FAIL
        else:
            raise ValueError(f"Unsupported tile count for M-split: {n_cores}")
        return _build_msplit(M, K, N, n_rows, n_cols)
    elif split == "2d":
        if n_cores == 1:
            n_rows, n_cols = 1, 1
        elif n_cores == 4:
            n_rows, n_cols = 2, 2
        elif n_cores == 16:
            n_rows, n_cols = 4, 4
        else:
            raise ValueError(f"Unsupported tile count for 2D-split: {n_cores}")
        return _build_2dsplit(M, K, N, n_rows, n_cols)
    else:
        raise ValueError(f"Unknown split mode: {split}")


def cpu_int8_gemm(A, B):
    A_i32 = A.astype(np.int32)
    B_i32 = B.astype(np.int32)
    return A_i32 @ B_i32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--tiles", type=int, required=True, choices=[1, 2, 4, 8, 16])
    ap.add_argument("--split", type=str, choices=["n", "m", "2d"], default="n")
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cores = args.tiles

    iron.set_current_device(from_name("npu", n_cols=4))

    @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
    def design(A: In, B: In, C: Out, *,
               M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N):
        return build(M, K, N, n_cores, args.split)

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    print(f"[phase34] M={M} K={K} N={N} tiles={n_cores} split={args.split}")
    t0 = time.time()
    compiled = design.specialize(M=M, K=K, N=N).compile(
        xclbin_path=xp, inst_path=ip, pdi_path=pp
    )
    print(f"[phase34] compiled in {time.time() - t0:.1f}s -> {compiled[0]}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
