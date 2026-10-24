#!/usr/bin/env python3
"""Phase 40M: Generate MLIR for Q8SCALEV-C4 mmul v2 kernel.

Generates a complete AIE MLIR program for M=512 K=1024 N=3584 C=4
using the pre-compiled q8_mmul_scale kernel object.

The kernel uses tile-major layout with per-block scales:
  A: 64x256 int8 tiles + 64x8x4 float32 scales = 18432 bytes
  B: 256x64 int8 tiles + 8x64x4 float32 scales = 18432 bytes
  C: 64x64 float32 accumulator = 16384 bytes

Usage:
  python3 gen_mlir_q8_mmul_v2.py --out /tmp/q8mmul_v2/design.prj/aie.mlir
"""
import argparse
import os

M = 512
K = 1024
N = 3584
N_COLS = 4

M_TILE = 64
N_TILE = 64
KC = 256
NBLK = KC // 32

N_MT = M // M_TILE
N_NT_C = (N // N_COLS) // N_TILE
N_KC = K // KC

A_CHUNK = M_TILE * KC + M_TILE * NBLK * 4
B_CHUNK = KC * N_TILE + NBLK * N_TILE * 4
C_ELEMS = M_TILE * N_TILE


def gen_mlir():
    lines = []
    a = lines.append

    a('// Multi-column Q8 mmul GEMM: M=512 K=1024 N=3584 n_cols=4')
    a('// Tile: Tm=64 Tk=256 Tn=64')
    a(f'// n_per_col={N//N_COLS} nb_m={NBLK} nb_k={N_KC} nb_n={N_NT_C}')
    a('module {')
    a('  aie.device(npu1) {')

    # Logical tiles: 4 columns x 4 rows = 16 cores
    core_idx = 0
    mem_idx = 0
    shim_idx = 0
    cores = []
    mems = []
    shims = []
    for col in range(N_COLS):
        for row in range(4):
            cores.append(f'%logical_core_{core_idx}')
            mems.append(f'%logical_mem_{mem_idx}')
            shims.append(f'%logical_shim_noc_{shim_idx}')
            core_idx += 1
            mem_idx += 1
            shim_idx += 1

    # Declare logical tiles
    for i, (c, m, s) in enumerate(zip(cores, mems, shims)):
        a(f'    {c} = aie.logical_tile<CoreTile>(?, ?)')
        a(f'    {m} = aie.logical_tile<MemTile>(?, ?)')
        a(f'    {s} = aie.logical_tile<ShimNOCTile>(?, ?)')

    # ObjectFifos for A (broadcast from shim to 4 cores per column)
    for col in range(N_COLS):
        row_cores = [cores[col*4 + r] for r in range(4)]
        mem = mems[col*4]
        shim = shims[col*4]
        a(f'    aie.objectfifo @A_L2L1_{col}({mem} dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {{{", ".join(row_cores)}}}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>>')
        a(f'    aie.objectfifo @A_L3L2_{col}({shim}, {{{mem}}}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>')
        a(f'    aie.objectfifo.link [@A_L3L2_{col}] -> [@A_L2L1_{col}]([] [0])')

    # ObjectFifos for B (each column has its own B stream)
    for col in range(N_COLS):
        row_cores = [cores[col*4 + r] for r in range(4)]
        mem = mems[4 + col]
        shim = shims[4 + col]
        a(f'    aie.objectfifo @B_L2L1_{col}({mem} dimensionsToStream [<size = 8, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {{{", ".join(row_cores)}}}, 2 : i32) : !aie.objectfifo<memref<64x32xi8>>')
        a(f'    aie.objectfifo @B_L3L2_{col}({shim}, {{{mem}}}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>')
        a(f'    aie.objectfifo.link [@B_L3L2_{col}] -> [@B_L2L1_{col}]([] [0])')

    # ObjectFifos for C (4 columns, 4 rows each)
    for col in range(N_COLS):
        for row in range(4):
            core = cores[col*4 + row]
            mem = mems[8 + col]
            shim = shims[8 + col]
            a(f'    aie.objectfifo @C_L1L2_{col}_{row}({core}, {{{mem}}}, 2 : i32) : !aie.objectfifo<memref<64x64xf32>>')
        a(f'    aie.objectfifo @C_L2L3_{col}({mem} dimensionsToStream [<size = 16, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {{{shims[8+col]}}}, 2 : i32) : !aie.objectfifo<memref<8192xf32>>')
        links = ', '.join([f'@C_L1L2_{col}_{row}' for row in range(4)])
        offsets = ', '.join([str(row * 2048) for row in range(4)])
        a(f'    aie.objectfifo.link [{links}] -> [@C_L2L3_{col}]([{offsets}] [])')

    # Kernel declaration
    a(f'    func.func private @q8_mmul_scale(memref<{A_CHUNK}xi8>, memref<{B_CHUNK}xi8>, memref<{C_ELEMS}xf32>) attributes {{link_with = "q8_mmul_scale.o"}}')

    # Core definitions
    for col in range(N_COLS):
        for row in range(4):
            core = cores[col*4 + row]
            a(f'    %{col*4+row} = aie.core({core}) {{')
            a('      %c0 = arith.constant 0 : index')
            a('      %c1 = arith.constant 1 : index')
            a('      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {')
            a(f'        %c0_14 = arith.constant 0 : index')
            a(f'        %c{N_NT_C} = arith.constant {N_NT_C} : index')
            a('        %c1_15 = arith.constant 1 : index')
            a(f'        scf.for %arg1 = %c0_14 to %c{N_NT_C} step %c1_15 {{')
            a(f'          %4 = aie.objectfifo.acquire @C_L1L2_{col}_{row}(Produce, 1) : memref<64x64xf32>')
            a('          %c0_16 = arith.constant 0 : index')
            a(f'          %c{C_ELEMS} = arith.constant {C_ELEMS} : index')
            a('          %c1_17 = arith.constant 1 : index')
            a(f'          scf.for %arg2 = %c0_16 to %c{C_ELEMS} step %c1_17 {{')
            a('            %cst = arith.constant 0.000000e+00 : f32')
            a(f'            memref.store %cst, %4[%arg2] : memref<64x64xf32>')
            a('          }')
            a('          %c0_18 = arith.constant 0 : index')
            a('          %c4 = arith.constant 4 : index')
            a('          %c1_19 = arith.constant 1 : index')
            a('          scf.for %arg2 = %c0_18 to %c4 step %c1_19 {')
            a(f'            %5 = aie.objectfifo.acquire @A_L2L1_{col}(Consume, 1) : memref<64x64xi8>')
            a(f'            %6 = aie.objectfifo.acquire @B_L2L1_{col}(Consume, 1) : memref<64x32xi8>')
            a('            %collapse_a = memref.collapse_shape %5 [[0, 1]] : memref<64x64xi8> into memref<4096xi8>')
            a('            %collapse_b = memref.collapse_shape %6 [[0, 1]] : memref<64x32xi8> into memref<2048xi8>')
            a(f'            %collapse_c = memref.collapse_shape %4 [[0, 1]] : memref<64x64xf32> into memref<4096xf32>')
            a(f'            func.call @q8_mmul_scale(%collapse_a, %collapse_b, %collapse_c) : (memref<{A_CHUNK}xi8>, memref<{B_CHUNK}xi8>, memref<{C_ELEMS}xf32>) -> ()')
            a(f'            aie.objectfifo.release @A_L2L1_{col}(Consume, 1)')
            a(f'            aie.objectfifo.release @B_L2L1_{col}(Consume, 1)')
            a('          }')
            a(f'          aie.objectfifo.release @C_L1L2_{col}_{row}(Produce, 1)')
            a('        }')
            a('      }')
            a('      aie.end')
            a('    } {stack_size = 4096 : i32}')

    # Runtime sequence
    a('    aie.runtime_sequence(%arg0: memref<524288xi8>, %arg1: memref<3670016xi8>, %arg2: memref<1835008xf32>) {')

    task_id = 0
    for col in range(N_COLS):
        for nt_c in range(N_NT_C):
            nt = col * N_NT_C + nt_c
            for mt in range(N_MT):
                for kc in range(N_KC):
                    # A DMA
                    a_offset = (mt * N_KC + kc) * A_CHUNK
                    b_offset = ((col * N_NT_C + nt_c) * N_KC + kc) * B_CHUNK
                    c_offset = (mt * N + (col * N_NT_C + nt_c) * N_TILE) * 4

                    # A DMA
                    a(f'      %a{task_id} = aiex.dma_configure_task_for @A_L3L2_{col} {{')
                    a(f'        aie.dma_bd(%arg0 : memref<524288xi8> offset = {a_offset} len = {A_CHUNK} sizes = [1, 1, 1, {A_CHUNK}] strides = [0, 0, 0, 1])')
                    a('        aie.end')
                    a(f'      }}')
                    a(f'      aiex.dma_start_task(%a{task_id})')

                    # B DMA
                    a(f'      %b{task_id} = aiex.dma_configure_task_for @B_L3L2_{col} {{')
                    a(f'        aie.dma_bd(%arg1 : memref<3670016xi8> offset = {b_offset} len = {B_CHUNK} sizes = [1, 1, 1, {B_CHUNK}] strides = [0, 0, 0, 1])')
                    a('        aie.end')
                    a(f'      }}')
                    a(f'      aiex.dma_start_task(%b{task_id})')

                    task_id += 1

            # C DMA for this N-tile
            c_offset = (N_NT_C * col + nt_c) * N_TILE * 4
            for mt in range(N_MT):
                c_off = (mt * N + (col * N_NT_C + nt_c) * N_TILE) * 4
                a(f'      %c{task_id} = aiex.dma_configure_task_for @C_L2L3_{col} {{')
                a(f'        aie.dma_bd(%arg2 : memref<1835008xf32> offset = {c_off} len = {C_ELEMS*4} sizes = [1, 1, {N_TILE}, {N_TILE}] strides = [0, 0, {N}, 1])')
                a('        aie.end')
                a(f'      }} {{issue_token = true}}')
                a(f'      aiex.dma_start_task(%c{task_id})')
                a(f'      aiex.dma_await_task(%c{task_id})')
                a(f'      aiex.dma_free_task(%a{task_id - N_MT*N_KC + mt*N_KC + (N_KC-1)})')
                a(f'      aiex.dma_free_task(%b{task_id - N_MT*N_KC + mt*N_KC + (N_KC-1)})')
                a(f'      aiex.dma_free_task(%c{task_id})')
                task_id += 1

    a('    }')
    a('  }')
    a('}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(gen_mlir())
    print(f'Written MLIR to {args.out}')


if __name__ == '__main__':
    main()
