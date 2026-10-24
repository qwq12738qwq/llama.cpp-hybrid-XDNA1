#!/usr/bin/env python3
"""MLIR Generator for multi-column AIE2 INT8 GEMM using logical tiles (IRON-style).

Uses aie.logical_tile for auto-placement instead of physical aie.tile.
This avoids aiecc BD validation issues with full-device column count.

Target: Phoenix XDNA1 NPU1, mlir-aie 1.4.3
"""

def gen_core_fn(c, tile_A, tile_B, tile_C32, nb_mn, nb_k, kernel_name="matmul_i8_i32"):
    """Generate core function for column c using its logical core tile."""
    lines = []
    lines.append(f"    // Core {c}")
    lines.append(f"    %core{c} = aie.core(%logical_core_{c}) {{")
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
    """Generate runtime sequence with N-split across columns using logical tiles."""
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
    """Generate complete MLIR for multi-column INT8 GEMM on AIE2 using logical tiles."""
    tile_A = Tm * Tk
    tile_B = Tk * Tn
    tile_C = Tm * Tn
    tile_C32 = Tm * Tn
    nb_m = M // Tm
    nb_k = K // Tk
    nb_n = N // Tn
    nb_mn = nb_m * nb_n
    n_per_col = N // n_cols

    assert N % n_cols == 0, f"N={N} must be divisible by n_cols={n_cols}"
    assert n_per_col % Tn == 0, f"N/n_cols={n_per_col} must be divisible by Tn={Tn}"
    assert K % Tk == 0, f"K={K} must be divisible by Tk={Tk}"

    lines = []
    lines.append(f"// Multi-column INT8 GEMM: M={M} K={K} N={N} n_cols={n_cols}")
    lines.append(f"// Tile: Tm={Tm} Tk={Tk} Tn={Tn}")
    lines.append(f"// n_per_col={n_per_col} nb_m={nb_m} nb_k={nb_k} nb_n={nb_n}")
    lines.append("module {")

    # Use logical device for auto-placement (IRON-style)
    if n_cols == 1:
        lines.append("  aie.device(npu1_1col) {")
    elif n_cols == 2:
        lines.append("  aie.device(npu1_2col) {")
    elif n_cols == 3:
        lines.append("  aie.device(npu1_3col) {")
    else:
        lines.append("  aie.device(npu1) {")

    # Per-column input and output logical tiles (each column is independent like Phase 18 C4)
    for c in range(n_cols):
        lines.append(f"    %logical_core_{c} = aie.logical_tile<CoreTile>(?, ?)")
        lines.append(f"    %logical_shim_a_{c} = aie.logical_tile<ShimNOCTile>(?, ?)")
        lines.append(f"    %logical_shim_b_{c} = aie.logical_tile<ShimNOCTile>(?, ?)")
        lines.append(f"    %logical_shim_c_{c} = aie.logical_tile<ShimNOCTile>(?, ?)")

    lines.append("")

    # Per-column ObjectFifos - single FIFO per path like IRON (producer -> consumer)
    for c in range(n_cols):
        # A path: shim (producer) -> core (consumer)
        lines.append(f"    aie.objectfifo @A_{c}(%logical_shim_a_{c}, {{%logical_core_{c}}}, 1 : i32) : !aie.objectfifo<memref<{tile_A}xi8>>")
        # B path: shim (producer) -> core (consumer)
        lines.append(f"    aie.objectfifo @B_{c}(%logical_shim_b_{c}, {{%logical_core_{c}}}, 1 : i32) : !aie.objectfifo<memref<{tile_B}xi8>>")
        # C path: core (producer) -> shim (consumer)
        lines.append(f"    aie.objectfifo @C_{c}(%logical_core_{c}, {{%logical_shim_c_{c}}}, 1 : i32) : !aie.objectfifo<memref<{tile_C32}xi32>>")
    lines.append("")

    # Kernel declaration
    lines.append(f'    func.func private @"matmul_i8_i32"(memref<{tile_A}xi8>, memref<{tile_B}xi8>, memref<{tile_C32}xi32>) attributes {{link_with = "matmul_i8_i32.o"}}')
    lines.append("")

    # Core functions - each uses its own logical core tile and per-column FIFOs
    for c in range(n_cols):
        lines.append(gen_core_fn_col(c, tile_A, tile_B, tile_C32, nb_mn, nb_k))
        lines.append("")

    # Runtime sequence
    lines.append(gen_runtime_seq_col(M, K, N, Tm, Tk, Tn, n_cols, tile_A, tile_B, tile_C))
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines)


def gen_core_fn_col(c, tile_A, tile_B, tile_C32, nb_mn, nb_k, kernel_name="matmul_i8_i32"):
    """Generate core function for column c using its logical core tile and per-column FIFOs."""
    lines = []
    lines.append(f"    // Core {c}")
    lines.append(f"    %core{c} = aie.core(%logical_core_{c}) {{")
    lines.append(f"      %c0 = arith.constant 0 : index")
    lines.append(f"      %c1 = arith.constant 1 : index")
    lines.append(f"      %c_mn = arith.constant {nb_mn} : index")
    lines.append(f"      %c_C32 = arith.constant {tile_C32} : index")
    lines.append(f"      %c_k = arith.constant {nb_k} : index")
    lines.append(f"      scf.for %iter = %c0 to %c_mn step %c1 {{")
    lines.append(f"        %a = aie.objectfifo.acquire @A_{c}(Consume, 1) : memref<{tile_A}xi8>")
    lines.append(f"        %b = aie.objectfifo.acquire @B_{c}(Consume, 1) : memref<{tile_B}xi8>")
    lines.append(f"        %c_out = aie.objectfifo.acquire @C_{c}(Produce, 1) : memref<{tile_C32}xi32>")
    lines.append(f"        %c0_i32 = arith.constant 0 : i32")
    lines.append(f"        scf.for %i = %c0 to %c_C32 step %c1 {{")
    lines.append(f"          memref.store %c0_i32, %c_out[%i] : memref<{tile_C32}xi32>")
    lines.append(f"        }}")
    lines.append(f"        scf.for %k = %c0 to %c_k step %c1 {{")
    lines.append(f"          func.call @matmul_i8_i32(%a, %b, %c_out) : (memref<{tile_A}xi8>, memref<{tile_B}xi8>, memref<{tile_C32}xi32>) -> ()")
    lines.append(f"        }}")
    lines.append(f"        aie.objectfifo.release @A_{c}(Consume, 1)")
    lines.append(f"        aie.objectfifo.release @B_{c}(Consume, 1)")
    lines.append(f"        aie.objectfifo.release @C_{c}(Produce, 1)")
    lines.append(f"      }}")
    lines.append(f"      aie.end")
    lines.append(f"    }} {{stack_size = 2048 : i32}}")
    return "\n".join(lines)


def gen_runtime_seq_col(M, K, N, Tm, Tk, Tn, n_cols, tile_A, tile_B, tile_C):
    """Generate runtime sequence with N-split across columns using logical tiles.
    
    Uses 4D memref format with [1, 1, ...] leading dimensions to match
    working Phase 18 kernels and avoid BD validation errors.
    """
    nb_m = M // Tm
    nb_k = K // Tk
    nb_n = N // Tn
    n_per_col = N // n_cols
    nb_n_per_col = n_per_col // Tn

    lines = []
    # 4D memref format: [1, 1, Tm, Tk] etc. to match Phase 18 working kernels
    lines.append(f"    aie.runtime_sequence(%A: memref<1x1x{Tm}x{Tk}xi8>, %B: memref<1x1x{Tk}x{Tn}xi8>, %C_out: memref<1x1x{Tm}x{Tn}xi32>) {{")

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
                # A: 4D memref [1, 1, Tm, Tk] with strides [0, 0, Tk, 1]
                # len in ELEMENTS (not bytes): Tm * Tk for xi8
                lines.append(f"      %a_{prefix} = aiex.dma_configure_task_for @A_{col} {{")
                lines.append(f"        aie.dma_bd(%A : memref<1x1x{Tm}x{Tk}xi8> offset = {a_off} len = {Tm * Tk} sizes = [1, 1, {Tm}, {Tk}] strides = [0, 0, {Tk}, 1])")
                lines.append(f"        aie.end")
                lines.append(f"      }}")
                lines.append(f"      aiex.dma_start_task(%a_{prefix})")
                # B: 4D memref [1, 1, Tk, Tn] with strides [0, 0, Tn, 1]
                # len in ELEMENTS (not bytes): Tk * Tn for xi8
                lines.append(f"      %b_{prefix} = aiex.dma_configure_task_for @B_{col} {{")
                lines.append(f"        aie.dma_bd(%B : memref<1x1x{Tk}x{Tn}xi8> offset = {b_off} len = {Tk * Tn} sizes = [1, 1, {Tk}, {Tn}] strides = [0, 0, {Tn}, 1])")
                lines.append(f"        aie.end")
                lines.append(f"      }}")
                lines.append(f"      aiex.dma_start_task(%b_{prefix})")
                # C: 4D memref [1, 1, Tm, Tn] with strides [0, 0, Tn, 1]
                # len in ELEMENTS (not bytes): Tm * Tn for xi32
                lines.append(f"      %c_{prefix} = aiex.dma_configure_task_for @C_{col} {{")
                lines.append(f"        aie.dma_bd(%C_out : memref<1x1x{Tm}x{Tn}xi32> offset = {c_off_linear} len = {Tm * Tn} sizes = [1, 1, {Tm}, {Tn}] strides = [0, 0, {Tn}, 1])")
                lines.append(f"        aie.end")
                lines.append(f"      }} {{issue_token = true}}")
                lines.append(f"      aiex.dma_start_task(%c_{prefix})")
                lines.append(f"      aiex.dma_await_task(%c_{prefix})")
                lines.append(f"      aiex.dma_free_task(%a_{prefix})")
                lines.append(f"      aiex.dma_free_task(%b_{prefix})")
                lines.append(f"      aiex.dma_free_task(%c_{prefix})")

    lines.append("    }")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys, os
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 3072
    n_cols = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    outdir = sys.argv[5] if len(sys.argv) > 5 else "."

    # Use tile sizes matching production kernels:
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