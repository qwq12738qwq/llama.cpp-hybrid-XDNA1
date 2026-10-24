#!/usr/bin/env python3
"""Phase 21: whole-array native XDNA1 GEMM generator (IRON, mmul-based).

Adapted from the official mlir-aie whole_array example
(programming_examples/basic/matrix_multiplication/whole_array), but built for
the installed mlir-aie 1.4.3 IRON Runtime API.

The official source example passes nested lists as Runtime() fn_args, which
1.4.3 no longer marshals structurally (each element maps back to the whole
list). We flatten the prod/cons endpoints and collect them with *fifos, the
pattern already used by experiments/phase20i/gen_chain.py.

C = A @ B, both plain row-major on the host. Each compute tile owns one
(m, n) output sub-tile and streams (m, k) x (k, n) inputs through three
layers of ObjectFifos (L3->L2 shim, L2->L1 MemTile, L1 compute).

Optional fused activation off the accumulator: --activation relu runs a relu
on the (m,n) tile inside the compute core before it is released to MemTile,
so the elementwise op rides the GEMM datapath instead of a separate submit.

Usage:
  python gen_gemm.py --M 512 --K 512 --N 512 -m 64 -k 64 -n 32 \
      --n-aie-cols 4 --dtype-in i8 --dtype-out i32 --activation relu --work-dir /tmp/g
"""
import argparse
import os
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
    StreamDims,
    TaskGroup,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.helpers.taplib import TensorTiler2D

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


def build_design(M, K, N, m, k, n, n_aie_cols, dtype_in, dtype_out, activation="none"):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k, dim_n=n,
        input_dtype=dtype_in, output_dtype=dtype_out,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    assert M % (m * n_aie_rows) == 0, "A must tile into (m*n_aie_rows, k) blocks"
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0
    assert m % r == 0 and k % s == 0 and n % t == 0

    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos = []
    A_l2l1_fifos = []
    B_l3l2_fifos = []
    B_l2l1_fifos = []
    C_l1l2_fifos = [[] for _ in range(n_aie_rows)]
    C_l2l3_fifos = []

    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        a_dims = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]] * (stop_row - start_row)
        a_tmp_fifos = a_l3l2.cons().split(
            of_offsets,
            obj_types=[A_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp_fifos)

    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims)
        )

        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_dims)
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_aie_rows)]
        c_tmp_fifos = c_l2l3.prod().join(
            of_offsets,
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j].append(c_tmp_fifos[j])

    act_kernel = None
    if activation in ("relu", "noop"):
        src = (_NOOP_SRC if activation == "noop" else _RELU_SRC).replace("{mn}", str(m * n))
        act_kernel = ExternalFunction(
            "relu_i32",
            source_string=src,
            arg_types=[np.ndarray[(m * n,), np.dtype[dtype_out]]],
        )

    def core_fn(in_a, in_b, out_c, zero, matmul, act):
        loop = range(1)
        if n_tiles_per_core > 1:
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
            if act is not None:
                act(elem_out)
            out_c.release(1)

    workers = Worker.grid(
        n_aie_rows, n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
                act_kernel,
            ],
            stack_size=0x2000 if act_kernel is not None else 0xD00,
        ),
    )

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_A_tiles_per_shim, k), (1, K // k),
        pattern_repeat=N // n // n_aie_cols, prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N), (k, n),
        tile_group_repeats=(K // k, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        tile_group_col_major=True, prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N), (m * n_aie_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols), prune_step=False,
    )
    flat_workers = [w for row in workers for w in row]

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
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )
                for col in range(n_aie_cols):
                    C_hs[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        tile_offset = ((row_base + tile_row) * n_shim_mem_A + col) % len(A_tiles)
                        if col < n_aie_rows:
                            A_hs[col].fill(A, tap=A_tiles[tile_offset], group=tg)
                        B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    print(f"[gen] cols={n_aie_cols} act={activation} n_shim_mem_A={n_shim_mem_A} "
          f"n_A_tiles_per_shim={n_A_tiles_per_shim} A_tiles={len(A_tiles)} "
          f"B_tiles={len(B_tiles)} C_tiles={len(C_tiles)} nA={nA} nB={nB} nC={len(C_conses)} "
          f"n_tiles_per_core={n_tiles_per_core} "
          f"tb_blocks={iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)}", flush=True)
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + A_prods + B_prods + C_conses)
    return iron.Program(iron.get_current_device(), rt, workers=flat_workers).resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("-m", dest="mt", type=int, default=64)
    ap.add_argument("-k", dest="kt", type=int, default=64)
    ap.add_argument("-n", dest="nt", type=int, default=32)
    ap.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], required=True)
    ap.add_argument("--dtype-in", type=str, choices=["i8", "i16", "bf16"], default="i8")
    ap.add_argument("--dtype-out", type=str, choices=["i8", "i16", "i32", "bf16", "f32"], default="i32")
    ap.add_argument("--activation", type=str, choices=["none", "relu", "noop"], default="none")
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    iron.set_current_device(from_name("npu", n_cols=args.n_aie_cols))

    @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
    def design(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int],
               N: CompileTime[int], mt: CompileTime[int], kt: CompileTime[int],
               nt: CompileTime[int], cols: CompileTime[int],
               din: CompileTime[str], dout: CompileTime[str], act: CompileTime[str]):
        return build_design(M, K, N, mt, kt, nt, cols, str_to_dtype(din), str_to_dtype(dout), act)

    print(f"[gen_gemm] M={args.M} K={args.K} N={args.N} "
          f"tile=({args.mt},{args.kt},{args.nt}) cols={args.n_aie_cols} "
          f"{args.dtype_in}->{args.dtype_out} act={args.activation}")
    t0 = time.time()
    compiled = design.specialize(
        M=args.M, K=args.K, N=args.N, mt=args.mt, kt=args.kt, nt=args.nt,
        cols=args.n_aie_cols, din=args.dtype_in, dout=args.dtype_out, act=args.activation,
    ).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"[gen_gemm] compiled in {time.time() - t0:.1f}s -> {compiled[0]}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
