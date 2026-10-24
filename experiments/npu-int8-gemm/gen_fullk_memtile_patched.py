"""Full-K Q8_0 GEMM on XDNA (Phoenix npu1) with MemTile staging.

Three-layer ObjectFifo hierarchy (mirrors the official whole_array.py example),
so the number of shim DMA streams is 3 * n_cols instead of 2 * n_tiles + 1:

    A: L3 -> L2  split   -> per-row L2 -> L1
    B: L3 -> L2  forward -> per-col L2 -> L1 (shared by the column's rows)
    C: L1 -> L2  join    -> per-col L2 -> L3

M is partitioned col-major: tile (row, col) handles M-block `col * n_rows + row`,
so each column's rows cover a contiguous M range and C drains stay contiguous.
"""

import argparse
import os
import time

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
from aie.iron.controlflow import range_ as iron_range
from aie.iron.runtime.taskgroup import TaskGroup
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config

KERNEL_SRC_TMPL = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

// a_packed: [m, kc] int8  then  [m, kc/32] float32
// b_packed: [kc, n] int8  then  [kc/32, n] float32
// c:        [m, n] float32, accumulated across K-chunks
extern "C" void mm_chunk_vec(
    const int8_t * __restrict a_packed,
    const int8_t * __restrict b_packed,
    float * __restrict c,
    int32_t m, int32_t kc, int32_t n)
{
    const int32_t nblk = kc / 32;
    const int8_t * a_data = a_packed;
    const float  * a_scales = (const float *)(a_packed + m * kc);
    const int8_t * b_data = b_packed;
    const float  * b_scales = (const float *)(b_packed + kc * n);

    constexpr int32_t VEC = {vec};
    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = 0; j < n; j += VEC) {
            aie::vector<float, VEC> cv = aie::load_v<VEC>(c + i * n + j);
            for (int32_t kb = 0; kb < nblk; kb++) {
                aie::accum<acc32, VEC> pacc = aie::zeros<acc32, VEC>();
                const int8_t * ar = a_data + i * kc + kb * 32;
                const int8_t * bc = b_data + (kb * 32) * n + j;
                for (int32_t l = 0; l < 32; l++) {
                    aie::vector<int8_t, VEC> av = aie::broadcast<int8_t, VEC>(ar[l]);
                    aie::vector<int8_t, VEC> bv = aie::load_v<VEC>(bc + l * n);
                    pacc = aie::mac(pacc, av, bv);
                }
                aie::vector<int32_t, VEC> pv = pacc.to_vector<int32_t>();
                float as_i = a_scales[i * nblk + kb];
                aie::vector<float, VEC> bsv = aie::load_v<VEC>(b_scales + kb * n + j);
                aie::vector<float, VEC> t1 = aie::mul(as_i, bsv).to_vector<float>();
                aie::vector<float, VEC> t2 = aie::mul(t1, aie::to_float(pv)).to_vector<float>();
                cv = aie::add(cv, t2);
            }
            aie::store_v(c + i * n + j, cv);
        }
    }
}
"""


def build(M, K, N, m_tile, n_tile, k_chunk, n_rows, n_cols):
    n_tiles = n_rows * n_cols
    assert K % k_chunk == 0 and k_chunk % 32 == 0
    assert M % (m_tile * n_tiles) == 0 and N % n_tile == 0
    n_mt = M // m_tile
    n_nt = N // n_tile
    n_kc = K // k_chunk
    assert n_mt == n_tiles, f"M/m_tile={n_mt} must equal n_rows*n_cols={n_tiles}"
    nblk = k_chunk // 32
    vec_len = 64 if n_tile % 64 == 0 else 32
    assert n_tile % vec_len == 0, "n_tile must be a multiple of 32"

    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4
    c_elems = m_tile * n_tile
    c_bytes = c_elems * 4
    BD_MAX = 16383 * 4
    for nm, v in (("a_bytes", a_bytes), ("b_bytes", b_bytes)):
        if v > BD_MAX:
            raise ValueError(f"{nm}={v} exceeds BD limit {BD_MAX}")

    depth_l1, depth_c1 = 1, 1
    depth_l2 = 1
    stack_bytes = 0x2000
    l1_per_tile = stack_bytes + depth_l1 * a_bytes + depth_l1 * b_bytes + depth_c1 * c_bytes
    L1_MAX = 64 * 1024
    print(f"  per-tile L1: stack={stack_bytes} A={depth_l1 * a_bytes} "
          f"B={depth_l1 * b_bytes} C={depth_c1 * c_bytes} total={l1_per_tile}/{L1_MAX}")
    if l1_per_tile > L1_MAX:
        raise ValueError(f"per-tile L1 {l1_per_tile} exceeds {L1_MAX}")

    # A per column: n_rows A-chunks + 1 B + n_rows C-chunks, one L3L2 fifo each.
    l2_per_col = depth_l2 * (n_rows * a_bytes + b_bytes + n_rows * c_bytes)
    L2_MAX = 128 * 1024
    print(f"  per-col L2: A={depth_l2 * n_rows * a_bytes} B={depth_l2 * b_bytes} "
          f"C={depth_l2 * n_rows * c_bytes} total={l2_per_col}/{L2_MAX}")
    if l2_per_col > L2_MAX:
        raise ValueError(f"per-col L2 {l2_per_col} exceeds {L2_MAX}")

    # One L3L2 fifo per column per direction -> 3 * n_cols shim DMA streams.
    n_shim = 3 * n_cols
    print(f"  shim streams: A={n_cols} B={n_cols} C={n_cols} total={n_shim} "
          f"(vs per-tile-objectfifo={2 * n_tiles + 1})")
    if n_shim > 16:
        raise ValueError(f"{n_shim} shim streams exceeds budget")

    fused = ExternalFunction(
        "mm_chunk_vec",
        source_string=KERNEL_SRC_TMPL.replace("{vec}", str(vec_len)),
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.float32]],
            np.int32, np.int32, np.int32,
        ],
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.float32]]
    A_l2_ty = np.ndarray[(n_rows * a_bytes,), np.dtype[np.int8]]
    B_l2_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    C_l2_ty = np.ndarray[(n_rows * c_elems,), np.dtype[np.float32]]

    a_l3l2, b_l3l2, c_l2l3 = [], [], []
    A_l2l1, B_l2l1 = [], []
    C_l1l2 = [[None] * n_cols for _ in range(n_rows)]

    for col in range(n_cols):
        a2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=depth_l2)
        a_l3l2.append(a2)
        A_l2l1.append(a2.cons().split(
            [row * a_bytes for row in range(n_rows)],
            obj_types=[a_ty] * n_rows,
            names=[f"A_L2L1_{col}_{row}" for row in range(n_rows)],
            depths=[depth_l1] * n_rows,
        ))

        b2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=depth_l2)
        b_l3l2.append(b2)
        B_l2l1.append(b2.cons().forward(
            obj_type=b_ty, name=f"B_L2L1_{col}", depth=depth_l1))

        c2 = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=depth_l2)
        c_l2l3.append(c2)
        c_subs = c2.prod().join(
            [row * c_elems for row in range(n_rows)],
            obj_types=[c_ty] * n_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_rows)],
            depths=[depth_c1] * n_rows,
        )
        for row in range(n_rows):
            C_l1l2[row][col] = c_subs[row]

    def core_fn(of_a, of_b, of_c, kern, m, kc, n, nnt, nkc):
        for _nt in iron_range(nnt):
            elem_c = of_c.acquire(1)
            for x in iron_range(m * n):
                elem_c[x] = 0.0
            for _kc in iron_range(nkc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c, m, kc, n)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    workers = Worker.grid(
        n_rows,
        n_cols,
        lambda row, col: Worker(
            core_fn,
            [A_l2l1[col][row].cons(), B_l2l1[col].cons(), C_l1l2[row][col].prod(),
             fused, m_tile, k_chunk, n_tile, n_nt, n_kc],
            stack_size=stack_bytes,
        ),
    )

    A_ty = np.ndarray[(n_tiles * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_nt * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    # A is laid out column-major by (col, kc, row) so each column's chunk is
    # contiguous: a single contiguous BD per fill, no strided layout needed.
    a_taps = [[TensorAccessPattern(
        tensor_dims=(n_tiles * n_kc * a_bytes,),
        offset=(col * n_kc + kc) * n_rows * a_bytes,
        sizes=[n_rows * a_bytes], strides=[1])
        for kc in range(n_kc)] for col in range(n_cols)]

    # B differs per n-tile (a different [k_chunk, n_tile] slab), and core_fn
    # acquires B inside the nt loop, so the host walks (nt, kc).
    b_taps = [[TensorAccessPattern(
        tensor_dims=(n_nt * n_kc * b_bytes,),
        offset=(nt * n_kc + kc) * b_bytes, sizes=[b_bytes], strides=[1])
        for kc in range(n_kc)] for nt in range(n_nt)]

    c_taps = [[TensorAccessPattern(
        tensor_dims=(M, N),
        offset=col * n_rows * m_tile * N + nt * n_tile,
        sizes=[n_rows * m_tile, n_tile], strides=[N, 1])
        for nt in range(n_nt)] for col in range(n_cols)]

    def sequence(A, B, C, a_p, b_p, c_c):
        # core_fn re-acquires A and B on every nt iteration, so the host must
        # supply n_nt * n_kc chunks per column, not n_kc. One TaskGroup per
        # (col, nt) keeps live BDs at 2*n_kc + 1 regardless of n_nt.
        tg_fill = TaskGroup()
        for col in range(n_cols):
            for nt in range(n_nt):
                for kc in range(n_kc):
                    a_p[col].fill(A, tap=a_taps[col][kc], group=tg_fill)
                    b_p[col].fill(B, tap=b_taps[nt][kc], group=tg_fill)
        tg_fill.finish()
        tg_drain = TaskGroup()
        for col in range(n_cols):
            for nt in range(n_nt):
                c_c[col].drain(C, tap=c_taps[col][nt], wait=True, group=tg_drain)
        tg_drain.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty,
                            [f.prod() for f in a_l3l2], [f.prod() for f in b_l3l2],
                            [f.cons() for f in c_l2l3]])
    flat_workers = [w for row in workers for w in row]
    return iron.Program(iron.get_current_device(), rt,
                        workers=flat_workers).resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, required=True)
    ap.add_argument("--n-tile", type=int, required=True)
    ap.add_argument("--k-chunk", type=int, required=True)
    ap.add_argument("--n-rows", type=int, default=4)
    ap.add_argument("--n-cols", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N, args.m_tile, args.n_tile,
                     args.k_chunk, args.n_rows, args.n_cols)

    print(f"E4 memtile: M={args.M} K={args.K} N={args.N} m_tile={args.m_tile} "
          f"n_tile={args.n_tile} k_chunk={args.k_chunk} "
          f"grid={args.n_rows}x{args.n_cols}")
    t0 = time.time()
    design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time()-t0:.1f}s")

    import glob
    cores = sorted(glob.glob(os.path.join(args.work_dir, "**", "els_main_core_*"),
                             recursive=True))
    if not cores:
        cores = sorted(glob.glob(os.path.join(args.work_dir, "**", "elfs_main_core_*"),
                                 recursive=True))
    print(f"compute tile artifacts ({len(cores)}): "
          f"{[os.path.basename(c) for c in cores]}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
