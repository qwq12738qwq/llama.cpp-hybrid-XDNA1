import sys
sys.path.insert(0, '/experiments/npu-int8-gemm')

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
import numpy as np
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
import time

M, K, N = 128, 1024, 64
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

# Chunk B into 2 chunks of 512 rows each, even though N=64 doesn't need it
b_taps = []
offset = 0
chunk_size = 512
remaining = K
while remaining > 0:
    cur_chunk = min(chunk_size, remaining)
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

@iron.jit
def int8_gemm_design(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = M,
    K: CompileTime[int] = K,
    N: CompileTime[int] = N,
):
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

# Compile
compiled = int8_gemm_design.specialize(M=M, K=K, N=N).compile()
print(f"Compiled: {compiled}")

# Random inputs
rng = np.random.default_rng(42)
A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

A = iron.tensor(A_np, dtype=np.int8, device="npu")
B = iron.tensor(B_np, dtype=np.int8, device="npu")
C = iron.tensor((M, N), dtype=np.int32, device="npu")

# Warmup
int8_gemm_design(A, B, C, M=M, K=K, N=N)

# CPU reference
ref = A_np.astype(np.int32) @ B_np.astype(np.int32)

# Run
C = iron.tensor((M, N), dtype=np.int32, device="npu")
start = time.perf_counter()
ret = int8_gemm_design(A, B, C, M=M, K=K, N=N)
e2e_us = (time.perf_counter() - start) * 1e6

got = C.numpy()
mismatches = int(np.sum(got != ref))
print(f"K={K}, N={N}: {mismatches} mismatches out of {M*N}")
if mismatches == 0:
    print("PASS")
else:
    print("FAIL")
    print(f"got[:8] = {got[:8].flatten()}")
    print(f"ref[:8] = {ref[:8].flatten()}")
