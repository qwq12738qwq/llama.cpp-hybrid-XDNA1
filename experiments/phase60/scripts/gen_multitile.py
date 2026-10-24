#!/usr/bin/env python3
"""Phase 60: raw MLIR-AIE multi-tile GEMM generator (r rows x c cols).

Uses physical tile placement with logical-ish explicit fifos per core.
Each core has its own A/B/C fifos; host DMA feeds each. No shared L2L1
broadcast at MLIR level (avoids iron's overlapping-segment issue).
"""
import argparse, os, sys

MT = 32
NT = 64
KC = 256
NBLK = KC // 32
MA = MT // 4
NA = NT // 8
KA = KC // 8
KB = KC // 8

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void int8_pure_mmul_v2(const int8_t * __restrict at,
                                   const int8_t * __restrict bt,
                                   int32_t      * __restrict c) {
    constexpr int MT=32, NT=64, KCC=256, NBLK=KCC/32;
    constexpr int MA=MT/4, NA=NT/8, KA=KCC/8, KB=KCC/8;
    const int8_t * a_tiles = at; const int8_t * b_tiles = bt;
    using MMUL = aie::mmul<4,8,8,int8_t,int8_t,accauto>;
    for (int bb=0; bb<NBLK; bb++) for (int i=0; i<MA; i++) for (int j=0; j<NA; j++) {
        MMUL acc;
        for (int kk=0; kk<4; kk++) {
            aie::vector<int8_t,32> av = aie::load_v<32>(a_tiles + (i*KA + bb*4 + kk)*32);
            aie::vector<int8_t,64> bv = aie::load_v<64>(b_tiles + (j*KB + bb*4 + kk)*64);
            acc.mac(av,bv);
        }
        alignas(32) int32_t tmp[32];
        aie::store_v(tmp, acc.to_vector<int32_t>());
        for (int rr=0; rr<4; rr++) for (int ll=0; ll<8; ll++)
            c[(i*4+rr)*NT + j*8 + ll] += tmp[rr*8+ll];
    }
}
"""


def build_mlir(M, K, N, rows, cols):
    assert M % (MT*rows) == 0 or (M % MT == 0)
    assert N % (NT*cols) == 0 or (N % NT == 0)
    assert K % KC == 0
    nb_m = M // MT
    nb_n = N // NT
    nb_k = K // KC
    tileA = MT*KC
    tileB = KC*NT
    tileC32 = MT*NT

    # partition m-tiles across rows, n-tiles across cols
    m_per_row = nb_m // rows
    n_per_col = nb_n // cols
    assert m_per_row >= 1 and n_per_col >= 1

    lines = []
    lines.append(f"// Phase60 multi-tile: M={M} K={K} N={N} rows={rows} cols={cols}")
    lines.append("module {")
    # device: use npu1 (4 cols) if cols<=4 else npu1
    lines.append("  aie.device(npu1) {")

    # Compute tiles: (col, 2+ri)
    cores = []
    for ri in range(rows):
        for ci in range(cols):
            r = 2 + ri
            c = ci
            lines.append(f"    %tile_core_r{ri}_c{ci} = aie.tile({c}, {r})")
            cores.append((ri, ci))
    # Distributed shims: A on col0, B and C spread across all columns to
    # avoid exhausting a single ShimNOCTile's DMA channel budget.
    lines.append("    %shim_a = aie.tile(0, 0)")
    lines.append("    %mem_a = aie.tile(0, 1)")
    for ci in range(cols):
        lines.append(f"    %shim_col{ci} = aie.tile({ci}, 0)")
        lines.append(f"    %mem_col{ci} = aie.tile({ci}, 1)")

    # ObjectFifos: A one per row
    for ri in range(rows):
        lines.append(f"    aie.objectfifo @A_r{ri}(%shim_a, {{%mem_a}}, 2 : i32) : !aie.objectfifo<memref<{tileA}xi8>>")
    # B one per col (on its own column shim)
    for ci in range(cols):
        lines.append(f"    aie.objectfifo @B_c{ci}(%shim_col{ci}, {{%mem_col{ci}}}, 2 : i32) : !aie.objectfifo<memref<{tileB}xi8>>")
    # A_L2L1: mem_a -> cores in that row
    for ri in range(rows):
        core_list = ", ".join([f"%tile_core_r{ri}_c{ci}" for ci in range(cols)])
        lines.append(f"    aie.objectfifo @A2_r{ri}(%mem_a, {{{core_list}}}, 2 : i32) : !aie.objectfifo<memref<{tileA}xi8>>")
        lines.append(f"    aie.objectfifo.link [@A_r{ri}] -> [@A2_r{ri}]([] [0])")
    # B_L2L1: mem_col{ci} -> cores in that col
    for ci in range(cols):
        core_list = ", ".join([f"%tile_core_r{ri}_c{ci}" for ri in range(rows)])
        lines.append(f"    aie.objectfifo @B2_c{ci}(%mem_col{ci}, {{{core_list}}}, 2 : i32) : !aie.objectfifo<memref<{tileB}xi8>>")
        lines.append(f"    aie.objectfifo.link [@B_c{ci}] -> [@B2_c{ci}]([] [0])")
    # C: per core fifo L1->L2->shim. Use two shim groups to spread: ci even -> c0, odd -> c1.
    for (ri, ci) in cores:
        lines.append(f"    aie.objectfifo @C_r{ri}_c{ci}(%tile_core_r{ri}_c{ci}, {{%mem_col{ci}}}, 1 : i32) : !aie.objectfifo<memref<{tileC32}xi32>>")
    for (ri, ci) in cores:
        lines.append(f"    aie.objectfifo @C2_r{ri}_c{ci}(%mem_col{ci}, {{%shim_col{ci}}}, 1 : i32) : !aie.objectfifo<memref<{tileC32}xi32>>")
        lines.append(f"    aie.objectfifo.link [@C_r{ri}_c{ci}] -> [@C2_r{ri}_c{ci}]([] [0])")

    # kernel decl
    lines.append('    func.func private @"int8_pure_mmul_v2"(memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) attributes {link_with = "int8_pure_mmul_v2.o"}')
    # Wait tile sizes: MT*KC = 32*256=8192, KC*NT=256*64=16384, MT*NT=2048. OK for MT=32,KC=256,NT=64.

    # core functions
    for (ri, ci) in cores:
        mt_start = ri * m_per_row
        mt_end = (ri+1) * m_per_row
        nt_start = ci * n_per_col
        nt_end = (ci+1) * n_per_col
        n_tiles = (mt_end-mt_start)*(nt_end-nt_start)
        lines.append(f"    %core_r{ri}_c{ci} = aie.core(%tile_core_r{ri}_c{ci}) {{")
        lines.append(f"      %c0 = arith.constant 0 : index")
        lines.append(f"      %c1 = arith.constant 1 : index")
        lines.append(f"      %cnk = arith.constant {nb_k} : index")
        lines.append(f"      %ctiles = arith.constant {n_tiles} : index")
        lines.append(f"      scf.for %t = %c0 to %ctiles step %c1 {{")
        lines.append(f"        %cout = aie.objectfifo.acquire @C_r{ri}_c{ci}(Produce, 1) : memref<{tileC32}xi32>")
        lines.append(f"        %zc = arith.constant 0 : i32")
        lines.append(f"        %csz = arith.constant {tileC32} : index")
        lines.append(f"        scf.for %i = %c0 to %csz step %c1 {{ memref.store %zc, %cout[%i] : memref<{tileC32}xi32> }}")
        lines.append(f"        %ck = arith.constant 0 : index")
        lines.append(f"        scf.for %k = %ck to %cnk step %c1 {{")
        lines.append(f"          %ain = aie.objectfifo.acquire @A2_r{ri}(Consume, 1) : memref<{tileA}xi8>")
        lines.append(f"          %bin = aie.objectfifo.acquire @B2_c{ci}(Consume, 1) : memref<{tileB}xi8>")
        lines.append(f"          func.call @int8_pure_mmul_v2(%ain, %bin, %cout) : (memref<{tileA}xi8>, memref<{tileB}xi8>, memref<{tileC32}xi32>) -> ()")
        lines.append(f"          aie.objectfifo.release @A2_r{ri}(Consume, 1)")
        lines.append(f"          aie.objectfifo.release @B2_c{ci}(Consume, 1)")
        lines.append(f"        }}")
        lines.append(f"        aie.objectfifo.release @C_r{ri}_c{ci}(Produce, 1)")
        lines.append(f"      }}")
        lines.append(f"      aie.end")
        lines.append(f"    }}")

    # runtime sequence
    lines.append(f"    aie.runtime_sequence(%A: memref<{M}x{K}xi8>, %B: memref<{K}x{N}xi8>, %Cout: memref<{M}x{N}xi32>) {{")
    # For each core's tiles, issue A/B/C DMA. We'll generate per (ri,ci) a nested loop over its m/n tiles.
    for (ri, ci) in cores:
        mt_start = ri * m_per_row
        mt_end = (ri+1) * m_per_row
        nt_start = ci * n_per_col
        nt_end = (ci+1) * n_per_col
        for mt in range(mt_start, mt_end):
            for nt in range(nt_start, nt_end):
                prefix = f"r{ri}c{ci}_m{mt}_n{nt}"
                a_off = mt * MT * K
                b_off = nt * NT  # B is [K,N], slice col nt -> offset nt*NT (element), but need strides; runtime uses offset in elements
                # C offset in elements (int32): (mt*MT*N + nt*NT)
                c_off = (mt * MT * N + nt * NT)
                lines.append(f"      // core r{ri}c{ci} m{mt} n{nt}")
                lines.append(f"      %a_{prefix} = aiex.dma_configure_task_for @A_r{ri} {{")
                lines.append(f"        aie.dma_bd(%A : memref<{M}x{K}xi8> offset = {a_off} len = {tileA} sizes = [1, {MT}, {KC}] strides = [0, {K}, 1])")
                lines.append(f"        aie.end")
                lines.append(f"      }}")
                lines.append(f"      aiex.dma_start_task(%a_{prefix})")
                lines.append(f"      %b_{prefix} = aiex.dma_configure_task_for @B_c{ci} {{")
                lines.append(f"        aie.dma_bd(%B : memref<{K}x{N}xi8> offset = {nt*NT} len = {tileB} sizes = [1, {KC}, {NT}] strides = [0, {N}, 1])")
                lines.append(f"        aie.end")
                lines.append(f"      }}")
                lines.append(f"      aiex.dma_start_task(%b_{prefix})")
                lines.append(f"      %c_{prefix} = aiex.dma_configure_task_for @C2_r{ri}_c{ci} {{")
                lines.append(f"        aie.dma_bd(%Cout : memref<{M}x{N}xi32> offset = {c_off} len = {tileC32} sizes = [1, {MT}, {NT}] strides = [0, {N}, 1])")
                lines.append(f"        aie.end")
                lines.append(f"      }} {{issue_token = true}}")
                lines.append(f"      aiex.dma_start_task(%c_{prefix})")
                lines.append(f"      aiex.dma_await_task(%c_{prefix})")
                lines.append(f"      aiex.dma_free_task(%a_{prefix})")
                lines.append(f"      aiex.dma_free_task(%b_{prefix})")
                lines.append(f"      aiex.dma_free_task(%c_{prefix})")
    lines.append("    }")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()
    M, K, N = args.M, args.K, args.N
    os.makedirs(args.work_dir, exist_ok=True)
    mlir = build_mlir(M, K, N, args.rows, args.cols)
    path = os.path.join(args.work_dir, "aie.mlir")
    with open(path, "w") as f:
        f.write(mlir)
    print(f"Generated {path} ({len(mlir)} bytes)")

if __name__ == "__main__":
    main()
