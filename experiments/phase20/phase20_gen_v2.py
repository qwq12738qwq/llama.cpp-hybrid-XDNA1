#!/usr/bin/env python3
"""
Phase 20 Step V: Generate V2 kernel - vectorized INT8 GEMM with double-buffered DMA.

CURRENT_RUNTIME_KERNEL uses scalar matmul_i8_i32.
V2 uses IRON's vectorized mm() kernel + proper ObjectFifo depth=2 pipelining.

Target: M=512, K=1024, N=3072, npu1_1col, tile 64x64x64
"""
import sys, os
sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')

import numpy as np
import aie.iron as iron
from aie.iron import In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern

# Target shape
M, K, N = 512, 1024, 3072
tile_m, tile_k, tile_n = 64, 64, 64

print(f"Generating V2 vectorized kernel: M={M}, K={K}, N={N}")
print(f"Tile: {tile_m}x{tile_k}x{tile_n}")

# Create vectorized matmul kernel
matmul_kernel = mm(
    dim_m=tile_m,
    dim_k=tile_k,
    dim_n=tile_n,
    input_dtype=np.int8,
    output_dtype=np.int32,
    vectorized=True,
)
r, s, t = matmul_kernel.mac_dims
print(f"Vectorized MAC dims: r={r}, s={s}, t={t}")

# Type definitions
A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

a_ty = np.ndarray[(tile_m * tile_k,), np.dtype[np.int8]]
b_ty = np.ndarray[(tile_k * tile_n,), np.dtype[np.int8]]
c_ty = np.ndarray[(tile_m * tile_n,), np.dtype[np.int32]]

# ObjectFifos with depth=2 for double buffering
fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2", depth=2)
tap_A_L2L1 = TensorTiler2D.group_tiler((tile_m, tile_k), (r, s), (tile_m // r, tile_k // s))[0]
fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
    dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
)

fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2", depth=2)
tap_B_L2L1 = TensorTiler2D.group_tiler((tile_k, tile_n), (s, t), (tile_k // s, tile_n // t))[0]
fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
    dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
)

fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2", depth=2)
tap_C_L1L2 = TensorAccessPattern(
    tensor_dims=(tile_m, tile_n),
    offset=0,
    sizes=[tile_m // r, r, tile_n // t, t],
    strides=[r * tile_n, t, r * t, 1],
)
fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
    dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3"
)

# AIE core function with double-buffered DMA/compute overlap
def core_fn(of_a, of_b, of_c, matmul):
    n_m_tiles = M // tile_m
    n_n_tiles = N // tile_n
    n_k_tiles = K // tile_k
    
    for _ in iron.controlflow.range_(n_m_tiles * n_n_tiles):
        # Acquire output buffer
        elem_out = of_c.acquire(1)
        # Zero output
        for i in iron.controlflow.range_(tile_m * tile_n):
            elem_out[i] = 0
        # K-dimension reduction
        for _ in iron.controlflow.range_(n_k_tiles):
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

# DMA tiling patterns
a_taps = TensorTiler2D.group_tiler(
    (M, K), (tile_m, tile_k), (1, K // tile_k), pattern_repeat=(N // tile_n)
)
b_tap = TensorTiler2D.group_tiler(
    (K, N), (tile_k, tile_n), (K // tile_k, N // tile_n), tile_group_col_major=True
)[0]
c_taps = TensorTiler2D.group_tiler((M, N), (tile_m, tile_n), (1, N // tile_n))

# Runtime sequence
def sequence(A, B, C, a_prod, b_prod, c_cons):
    for tile_row in range(M // tile_m):
        tg = iron.TaskGroup()
        a_prod.fill(A, tap=a_taps[tile_row], group=tg)
        b_prod.fill(B, tap=b_tap, group=tg)
        c_cons.drain(C, tap=c_taps[tile_row], group=tg, wait=True)
        tg.finish()

rt = iron.Runtime(
    sequence,
    [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
)
prog = iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

# Write MLIR
out_dir = "/Storage/Cache/llama.cpp/experiments/phase20"
os.makedirs(out_dir, exist_ok=True)
mlir_path = f"{out_dir}/v2_vectorized_M{M}_K{K}_N{N}.mlir"
with open(mlir_path, 'w') as f:
    f.write(str(prog))
print(f"MLIR written to {mlir_path}")

# Compile
print("\nCompiling with aiecc...")
import subprocess
prj_dir = f"{out_dir}/v2_vectorized_M{M}_K{K}_N{N}.mlir.prj"
cmd = [
    "/Storage/Cache/src/mlir-aie-venv/bin/aiecc",
    "--aie-generate-xclbin",
    "--aie-generate-npu-insts",
    "--no-xrt-sync",
    "--xclbin-name=design.xclbin",
    "--npu-insts-name=insts.bin",
    "--device=npu1",
    mlir_path,
]
print(f"  CMD: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if result.returncode != 0:
    print(f"  aiecc FAILED (rc={result.returncode})")
    print(f"  STDERR: {result.stderr[:2000]}")
    sys.exit(1)
else:
    print(f"  aiecc SUCCESS")
    # Check output
    for f_name in ["design.xclbin", "insts.bin"]:
        if os.path.exists(f_name):
            print(f"  {f_name}: {os.path.getsize(f_name)} bytes")
        else:
            # Check in prj directory
            prj_f = os.path.join(prj_dir, f_name)
            if os.path.exists(prj_f):
                print(f"  {prj_f}: {os.path.getsize(prj_f)} bytes")
            else:
                print(f"  {f_name}: NOT FOUND")
