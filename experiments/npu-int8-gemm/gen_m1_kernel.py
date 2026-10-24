#!/usr/bin/env python3
"""
Phase 9-F: Generate M=1, K=1024, N=16 XCLBIN for Qwen3.5 SSM decode.

Uses aie.iron (the declarative API that produced the proven M=16 kernel).
M=1 is handled by the scalar path: tile_m=1, tile_k=32, tile_n=16.
K=1024 is chunked into 32 iterations of 32.
B tile = 32*16 = 512 bytes (within aiecc DMA BD limit of 1023).
A tile = 32 bytes.
C tile = 16 int32 = 64 bytes.

No vectorized mmul (requires M % 4 == 0); scalar matmul is correct.

Output: exactly 1 x 16 int32 values. Never writes an M=16 footprint.
"""
import os
import sys
import json
import time

import numpy as np
import aie.iron as iron
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(WORK_DIR, "k1024_m1_n16")

M, K, N = 1, 1024, 16
TILE_M, TILE_K, TILE_N = 1, 32, 16

# Scalar matmul kernel (no vectorization needed for M=1)
matmul_kernel = mm(
    dim_m=TILE_M,
    dim_k=TILE_K,
    dim_n=TILE_N,
    input_dtype=np.int8,
    output_dtype=np.int32,
    vectorized=False,
)

A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

a_ty = np.ndarray[(TILE_M * TILE_K,), np.dtype[np.int8]]
b_ty = np.ndarray[(TILE_K * TILE_N,), np.dtype[np.int8]]
c_ty = np.ndarray[(TILE_M * TILE_N,), np.dtype[np.int32]]

# Single-tile object fifos
# dims_to_stream is a list of (size, stride) pairs
fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
    dims_to_stream=[(TILE_M * TILE_K, 1)], name="A_L2L1"
)

fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
    dims_to_stream=[(TILE_K * TILE_N, 1)], name="B_L2L1"
)

# C output: use 4D dims_to_stream with positive strides to avoid stride=0 issue
fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
    dims_to_stream=[(1, 16), (16, 1)], name="C_L2L3"
)

def core_fn(of_a, of_b, of_c, matmul):
    elem_out = of_c.acquire(1)
    for _ in iron.controlflow.range_(K // TILE_K):
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

# DMA access patterns
tap_A = TensorTiler2D.group_tiler(
    (M, K), (TILE_M, TILE_K), (M // TILE_M, K // TILE_K), pattern_repeat=(N // TILE_N)
)[0]
tap_B = TensorTiler2D.group_tiler(
    (K, N), (TILE_K, TILE_N), (K // TILE_K, N // TILE_N), tile_group_col_major=True
)[0]
tap_C = TensorTiler2D.group_tiler(
    (M, N), (TILE_M, TILE_N), (M // TILE_M, N // TILE_N)
)[0]

def sequence(A, B, C, a_prod, b_prod, c_cons):
    tg = iron.TaskGroup()
    a_prod.fill(A, tap=tap_A, group=tg)
    b_prod.fill(B, tap=tap_B, group=tg)
    c_cons.drain(C, tap=tap_C, group=tg, wait=True)
    tg.finish()

rt = iron.Runtime(
    sequence,
    [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
)
prog = iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

mlir_path = os.path.join(OUT_DIR, "design.prj", "aie.mlir")
os.makedirs(os.path.dirname(mlir_path), exist_ok=True)
with open(mlir_path, "w") as f:
    f.write(str(prog))
print(f"[M1-GEN] Wrote MLIR to {mlir_path}")

# Compile with aiecc
aiecc = "/Storage/Cache/src/mlir-aie-venv/bin/aiecc"
PYTHONPATH = "/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python"
env = os.environ.copy()
env["PYTHONPATH"] = PYTHONPATH
env["PATH"] = "/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/llvm-aie/bin:" + env["PATH"]

t0 = time.time()
cmd = (
    f"PYTHONPATH={PYTHONPATH} PATH={env['PATH']} {aiecc} "
    f"--output-dir={OUT_DIR} "
    f"--tmpdir={OUT_DIR}/design.prj "
    f"--xclbin-name={OUT_DIR}/design.xclbin "
    f"--pdi-name=main.pdi "
    f"--elf-name=matmul_i8_i32_m1_k1024_n16.o "
    f"{mlir_path}"
)
print(f"[M1-GEN] Running: {cmd}")
result = os.system(cmd)
compile_time = time.time() - t0

xclbin_path = os.path.join(OUT_DIR, "design.xclbin")
insts_path = os.path.join(OUT_DIR, "design.insts.bin")

compile_info = {
    "phase": "9f",
    "kernel": "m1_k1024_n16_matmul_i8_i32",
    "M": M, "K": K, "N": N,
    "tile_m": TILE_M, "tile_k": TILE_K, "tile_n": TILE_N,
    "vectorized": False,
    "scalar_path": True,
    "k_chunked": True,
    "k_chunks": K // TILE_K,
    "a_tile_bytes": TILE_M * TILE_K,
    "b_tile_bytes": TILE_K * TILE_N,
    "c_tile_bytes": TILE_M * TILE_N * 4,
    "compile_time_s": round(compile_time, 2),
    "return_code": result,
    "xclbin_path": xclbin_path,
    "insts_path": insts_path,
    "xclbin_exists": os.path.exists(xclbin_path),
    "insts_exists": os.path.exists(insts_path),
    "xclbin_size_bytes": os.path.getsize(xclbin_path) if os.path.exists(xclbin_path) else 0,
    "insts_size_bytes": os.path.getsize(insts_path) if os.path.exists(insts_path) else 0,
}

out_json = os.path.join(WORK_DIR, "phase9f_compile.json")
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w") as f:
    json.dump(compile_info, f, indent=2)
print(f"[M1-GEN] Compile info written to {out_json}")
print(f"[M1-GEN] xclbin_exists={compile_info['xclbin_exists']} "
      f"xclbin_size={compile_info['xclbin_size_bytes']} "
      f"insts_size={compile_info['insts_size_bytes']}")
