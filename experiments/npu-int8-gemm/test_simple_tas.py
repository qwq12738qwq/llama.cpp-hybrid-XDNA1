import sys
sys.path.insert(0, '/experiments/npu-int8-gemm')

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
import numpy as np
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern

M, K, N = 128, 1024, 128
tile_m, tile_k, tile_n = 64, 64, 64

matmul_kernel = mm(
    dim_m=tile_m,
    dim_k=tile_k,
    dim_n=tile_n,
    input_dtype=np.int8,
    output_dtype=np.int32,
    vectorized=True,
)
r, s, t = matmul_kernel.mac_dims

A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

a_ty = np.ndarray[(tile_m * tile_k,), np.dtype[np.int8]]
b_ty = np.ndarray[(tile_k * tile_n,), np.dtype[np.int8]]
c_ty = np.ndarray[(tile_m * tile_n,), np.dtype[np.int32]]

fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
tap_A_L2L1 = TensorTiler2D.group_tiler((tile_m, tile_k), (r, s), (tile_m // r, tile_k // s))[0]
fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
    dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
)

fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
tap_B_L2L1 = TensorTiler2D.group_tiler((tile_k, tile_n), (s, t), (tile_k // s, tile_n // t))[0]
fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
    dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
)

fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
tap_C_L1L2 = TensorAccessPattern(
    tensor_dims=(tile_m, tile_n),
    offset=0,
    sizes=[tile_m // r, r, tile_n // t, t],
    strides=[r * tile_n, t, r * t, 1],
)
fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
    dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3"
)

def core_fn(of_a, of_b, of_c, matmul):
    for _ in iron.controlflow.range_(M // tile_m * N // tile_n):
        elem_out = of_c.acquire(1)
        for i in iron.controlflow.range_(tile_m * tile_n):
            elem_out[i] = 0
        for _ in iron.controlflow.range_(K // tile_k):
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

a_taps = TensorTiler2D.group_tiler(
    (M, K), (tile_m, tile_k), (1, K // tile_k), pattern_repeat=(N // tile_n)
)

# Use simple 2D TAS for B (row-major)
b_taps = []
offset = 0
chunk_size = 512
remaining = K
while remaining > 0:
    cur_chunk = min(chunk_size, remaining)
    # Simple 2D TAS: [rows, cols] with row-major strides
    b_tap = TensorAccessPattern(
        tensor_dims=(K, N),
        offset=offset,
        sizes=[cur_chunk, N],
        strides=[N, 1],
    )
    b_taps.append(b_tap)
    offset += cur_chunk * N
    remaining -= cur_chunk

c_taps = TensorTiler2D.group_tiler((M, N), (tile_m, tile_n), (1, N // tile_n))

def sequence(A, B, C, a_prod, b_prod, c_cons):
    for tile_row in range(M // tile_m):
        tg = iron.TaskGroup()
        a_prod.fill(A, tap=a_taps[tile_row], group=tg)
        for b_tap in b_taps:
            b_prod.fill(B, tap=b_tap, group=tg)
        c_cons.drain(C, tap=c_taps[tile_row], group=tg, wait=True)
        tg.finish()

rt = iron.Runtime(
    sequence,
    [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
)
prog = iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

with open('/experiments/npu-int8-gemm/k1024_simple_tas.mlir', 'w') as f:
    f.write(str(prog))

print("Written to k1024_simple_tas.mlir")
