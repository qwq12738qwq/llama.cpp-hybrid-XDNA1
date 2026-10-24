#!/usr/bin/env python3
"""MLIR Generator for multi-column AIE2 INT8 GEMM.

For mlir-aie 1.4.3 (no iron module, direct MLIR text generation).
Requires Vitis aietools for xclbin generation (aiecc pipeline).

Key API changes from pre-1.4.3:
- !aie.objectfifosubview<memref<...>> is REMOVED
- aie.objectfifo.acquire returns memref<...> directly
- aie.objectfifo.subview.access is REMOVED

Target: Phoenix XDNA1 NPU1 (4 columns)
"""

def gen_core_fn(c, tile_A, tile_B, tile_C32, nb_mn, nb_k, kernel_name="matmul_i8_i32"):
    """Generate core function for column c."""
    lines = []
    lines.append(f"    // Core {c}")
    lines.append(f"    %core{c} = aie.core(%tile_core{c}) {{")
    lines.append(f"      %c0 = arith.constant 0 : index")
    lines.append(f"      %c1 = arith.constant 1 : index")
    lines.append(f"      %c_mn = arith.constant {nb_mn} : index")
    lines.append(f"      %c_C32 = arith.constant {tile_C32} : index")
    lines.append(f"      %c_k = arith.constant {nb_k} : index")
    lines.append(f"      scf.for %iter = %c0 to %c_mn step %c1 {{")
    lines.append(f"        %a = aie.objectfifo.acquire @A_L2L1(Consume, 1) : memref<{tile_A}xi8>")
    lines.append(f"        %b = aie.objectfifo.acquire @B_L2L1(Consume, 1) : memref<{tile_B}xi8>")
    lines.append(f"        %c_out = aie.objectfifo.acquire @C_L1L2_{c}(Produce, 1) : memref<{tile_C32}xi32>")
    lines.append(f"        %c0_i32 = arith.constant 0 : i32")
    lines.append(f"        scf.for %i = %c0 to %c_C32 step %c1 {{")
    lines.append(f"          memref.store %c0_i32, %c_out[%i] : memref<{tile_C32}xi32>")
    lines.append(f"        }}")
    lines.append(f"        scf.for %k = %c0 to %c_k step %c1 {{")
    lines.append(f"          func.call @matmul_i8_i32(%a, %b, %c_out) : (memref<{tile_A}xi8>, memref<{tile_B}xi8>, memref<{tile_C32}xi32>) -> ()")
    lines.append(f"        }}")
    lines.append(f"        aie.objectfifo.release @A_L2L1(Consume, 1)")
    lines.append(f"        aie.objectfifo.release @B_L2L1(Consume, 1)")
    lines.append(f"        aie.objectfifo.release @C_L1L2_{c}(Produce, 1)")
    lines.append(f"      }}")
    lines.append(f"      aie.end")
    lines.append(f"    }} {{stack_size = 2048 : i32}}")
    return "\n".join(lines)


def gen_runtime_seq(M, K, N, Tm, Tk, Tn, n_cols, tile_A, tile_B, tile_C):
    """Generate runtime sequence with N-split across columns."""
    nb_m = M // Tm
    nb_k = K // Tk
    nb_n = N // Tn
    n_per_col = N // n_cols
    nb_n_per_col = n_per_col // Tn

    lines = []
    lines.append(f"    aie.runtime_sequence(%A: memref<{M}x{K}xi8>, %B: memref<{K}x{N}xi8>, %C_out: memref<{M}x{N}xi32>) {{")

    for col in range(n_cols):
        col_n_offset = col * nb_n_per_col
        for m_t in range(nb_m):
            for n_t in range(nb_n_per_col):
                n_tile_global = col * nb_n_per_col + n_t
                a_off = m_t * Tm * K
                b_off = n_tile_global * Tn
                c_off_linear = (m_t * Tm * N + n_tile_global * Tn) * 4

                prefix = f"col{col}_m{m_t}_n{n_t}"
                lines.append(f"      // Col {col} M-tile {m_t} N-tile {n_t}")
                lines.append(f"      %a_{prefix} = aiex.dma_configure_task_for @A_L3L2 {{")
                lines.append(f"        aie.dma_bd(%A : memref<{M}x{K}xi8> offset = {a_off} len = {tile_A} sizes = [1, {Tm}, {Tk}] strides = [0, {K}, 1])")
                lines.append(f"        aie.end")
                lines.append(f"      }}")
                lines.append(f"      aiex.dma_start_task(%a_{prefix})")
                lines.append(f"      %b_{prefix} = aiex.dma_configure_task_for @B_L3L2 {{")
                lines.append(f"        aie.dma_bd(%B : memref<{K}x{N}xi8> offset = {b_off} len = {tile_B} sizes = [1, {Tk}, {Tn}] strides = [0, {N}, 1])")
                lines.append(f"        aie.end")
                lines.append(f"      }}")
                lines.append(f"      aiex.dma_start_task(%b_{prefix})")
                lines.append(f"      %c_{prefix} = aiex.dma_configure_task_for @C_L2L3_{col} {{")
                lines.append(f"        aie.dma_bd(%C_out : memref<{M}x{N}xi32> offset = {c_off_linear} len = {tile_C} sizes = [1, {Tm}, {Tn}] strides = [0, {N}, 1])")
                lines.append(f"        aie.end")
                lines.append(f"      }} {{issue_token = true}}")
                lines.append(f"      aiex.dma_start_task(%c_{prefix})")
                lines.append(f"      aiex.dma_await_task(%c_{prefix})")
                lines.append(f"      aiex.dma_free_task(%a_{prefix})")
                lines.append(f"      aiex.dma_free_task(%b_{prefix})")
                lines.append(f"      aiex.dma_free_task(%c_{prefix})")

    lines.append("    }")
    return "\n".join(lines)


def generate_mlir(M, K, N, Tm=64, Tk=64, Tn=64, n_cols=1):
    """Generate complete MLIR for multi-column INT8 GEMM on AIE2."""
    tile_A = Tm * Tk
    tile_B = Tk * Tn
    tile_C = Tm * Tn * 4
    tile_C32 = Tm * Tn
    nb_m = M // Tm
    nb_k = K // Tk
    nb_n = N // Tn
    nb_mn = nb_m * nb_n
    n_per_col = N // n_cols
 
    assert N % n_cols == 0, f"N={N} must be divisible by n_cols={n_cols}"
    assert n_per_col % Tn == 0, f"N/n_cols={n_per_col} must be divisible by Tn={Tn}"
    assert K % Tk == 0, f"K={K} must be divisible by Tk={Tk}"

    # Column allocation: 0..n_cols-1 for cores, n_cols for A, n_cols+1 for B
    lines = []
    lines.append(f"// Multi-column INT8 GEMM: M={M} K={K} N={N} n_cols={n_cols}")
    lines.append(f"// Tile: Tm={Tm} Tk={Tk} Tn={Tn}")
    lines.append(f"// n_per_col={n_per_col} nb_m={nb_m} nb_k={nb_k} nb_n={nb_n}")
    lines.append("module {")
    if n_cols == 1:
        lines.append("  aie.device(npu1_1col) {")
    elif n_cols == 2:
        lines.append("  aie.device(npu1_2col) {")
    elif n_cols == 3:
        lines.append("  aie.device(npu1_3col) {")
    else:
        lines.append("  aie.device(npu1) {")

    # Tiles
    for c in range(n_cols):
        lines.append(f"    %tile_shim_c{c} = aie.tile({c}, 0)")
        lines.append(f"    %tile_mem_c{c} = aie.tile({c}, 1)")
        lines.append(f"    %tile_core{c} = aie.tile({c}, 2)")

    shim_a_col = n_cols
    shim_b_col = n_cols + 1
    lines.append(f"    %tile_shim_a = aie.tile({shim_a_col}, 0)")
    lines.append(f"    %tile_mem_a = aie.tile({shim_a_col}, 1)")
    lines.append(f"    %tile_shim_b = aie.tile({shim_b_col}, 0)")
    lines.append(f"    %tile_mem_b = aie.tile({shim_b_col}, 1)")
    lines.append("")

    # ObjectFifos
    core_tiles = ", ".join([f"%tile_core{c}" for c in range(n_cols)])
    lines.append(f"    aie.objectfifo @A_L3L2(%tile_shim_a, {{%tile_mem_a}}, 1 : i32) : !aie.objectfifo<memref<{tile_A}xi8>>")
    lines.append(f"    aie.objectfifo @A_L2L1(%tile_mem_a, {{{core_tiles}}}, 1 : i32) : !aie.objectfifo<memref<{tile_A}xi8>>")
    lines.append(f"    aie.objectfifo.link [@A_L3L2] -> [@A_L2L1]([] [0])")
    lines.append("")
    lines.append(f"    aie.objectfifo @B_L3L2(%tile_shim_b, {{%tile_mem_b}}, 1 : i32) : !aie.objectfifo<memref<{tile_B}xi8>>")
    lines.append(f"    aie.objectfifo @B_L2L1(%tile_mem_b, {{{core_tiles}}}, 1 : i32) : !aie.objectfifo<memref<{tile_B}xi8>>")
    lines.append(f"    aie.objectfifo.link [@B_L3L2] -> [@B_L2L1]([] [0])")
    lines.append("")

    for c in range(n_cols):
        lines.append(f"    aie.objectfifo @C_L1L2_{c}(%tile_core{c}, {{%tile_mem_c{c}}}, 1 : i32) : !aie.objectfifo<memref<{tile_C32}xi32>>")
        lines.append(f"    aie.objectfifo @C_L2L3_{c}(%tile_mem_c{c}, {{%tile_shim_c{c}}}, 1 : i32) : !aie.objectfifo<memref<{tile_C32}xi32>>")
        lines.append(f"    aie.objectfifo.link [@C_L1L2_{c}] -> [@C_L2L3_{c}]([] [0])")
    lines.append("")

    # Kernel declaration
    lines.append(f'    func.func private @"matmul_i8_i32"(memref<{tile_A}xi8>, memref<{tile_B}xi8>, memref<{tile_C32}xi32>) attributes {{link_with = "matmul_i8_i32.o"}}')
    lines.append("")

    # Core functions
    for c in range(n_cols):
        lines.append(gen_core_fn(c, tile_A, tile_B, tile_C32, nb_mn, nb_k))
        lines.append("")

    # Runtime sequence
    lines.append(gen_runtime_seq(M, K, N, Tm, Tk, Tn, n_cols, tile_A, tile_B, tile_C))
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys, os
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 3072
    n_cols = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    outdir = sys.argv[5] if len(sys.argv) > 5 else "."
 
    # Use tile sizes matching production kernels:
    # For Q8_0 (K=32 block): Tm=64, Tk=32, Tn=64
    # For I8 (full K): Tm=64, Tk=64, Tn=64
    if K == 32:
        Tm, Tk, Tn = 64, 32, 64
    else:
        Tm, Tk, Tn = 64, 64, 64
 
    mlir = generate_mlir(M, K, N, Tm, Tk, Tn, n_cols)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "aie.mlir")
    with open(path, "w") as f:
        f.write(mlir)
    print(f"Generated: {path} ({len(mlir)} bytes)")
    print(f"Shape: M={M} K={K} N={N} n_cols={n_cols} Tm={Tm} Tk={Tk} Tn={Tn}")
