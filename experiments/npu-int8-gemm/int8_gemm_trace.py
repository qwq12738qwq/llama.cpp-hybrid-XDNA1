#!/usr/bin/env python3
# int8_gemm_trace.py — int8_gemm.py with trace input support

import argparse
import sys
import time
import statistics
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.benchmark import run_iters
from aie.utils.verify import assert_pass

_TILE_M = 64
_TILE_K = 64
_TILE_N = 64


def _build_design(M, K, N):
    m, k, n = _TILE_M, _TILE_K, _TILE_N

    matmul_kernel = mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=np.int8,
        output_dtype=np.int32,
        vectorized=True,
    )
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

    a_ty = np.ndarray[(m * k,), np.dtype[np.int8]]
    b_ty = np.ndarray[(k * n,), np.dtype[np.int8]]
    c_ty = np.ndarray[(m * n,), np.dtype[np.int32]]

    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
    )

    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
    )

    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(m, n),
        offset=0,
        sizes=[m // r, r, n // t, t],
        strides=[r * n, t, r * t, 1],
    )
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3"
    )

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in iron.controlflow.range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            for i in iron.controlflow.range_(m * n):
                elem_out[i] = 0
            for _ in iron.controlflow.range_(K // k):
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
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n)
    )
    if K > 512:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, N // n), tile_group_col_major=False
        )[0]
        b_taps = [b_tap]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
        )[0]
        b_taps = [b_tap]

    c_taps = []
    for tile_row in range(M // m):
        for tile_col in range(N // n):
            offset = tile_row * m * N + tile_col * n
            c_tap = TensorAccessPattern(
                tensor_dims=(M, N),
                offset=offset,
                sizes=[m, n],
                strides=[N, 1],
            )
            c_taps.append(c_tap)

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // m):
            for tile_col in range(N // n):
                tg = iron.TaskGroup()
                a_prod.fill(A, tap=a_taps[tile_row], group=tg)
                b_prod.fill(B, tap=b_tap, group=tg)
                c_cons.drain(C, tap=c_taps[tile_row * (N // n) + tile_col], group=tg, wait=True)
                tg.finish()

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
    )
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def cpu_int8_gemm(A, B):
    A_i32 = A.astype(np.int32)
    B_i32 = B.astype(np.int32)
    return A_i32 @ B_i32


def create_trace_inputs(M, K, N):
    """Create input matrices with unique tile IDs for tracing."""
    m, k, n = _TILE_M, _TILE_K, _TILE_N
    
    A = np.ones((M, K), dtype=np.int8)
    
    B = np.zeros((K, N), dtype=np.int8)
    K_tiles = K // k
    N_tiles = N // n
    
    for kt in range(K_tiles):
        for nt in range(N_tiles):
            tile_id = kt * N_tiles + nt
            val = (tile_id * 7 + 1) % 127
            if val == 0:
                val = 1
            row_start = kt * k
            col_start = nt * n
            B[row_start:row_start+k, col_start:col_start+n] = val
    
    return A, B, K_tiles, N_tiles


def analyze_output(C_npu, C_cpu, M, N, K_tiles, N_tiles):
    m, n = _TILE_M, _TILE_N
    
    print(f"\nOutput Analysis (M={M}, N={N}, K_tiles={K_tiles}, N_tiles={N_tiles}):")
    
    for tile_row in range(M // m):
        for tile_col in range(N // n):
            row_start = tile_row * m
            col_start = tile_col * n
            
            npu_tile = C_npu[row_start:row_start+m, col_start:col_start+n]
            cpu_tile = C_cpu[row_start:row_start+m, col_start:col_start+n]
            
            npu_sum = np.sum(npu_tile)
            cpu_sum = np.sum(cpu_tile)
            
            match = np.array_equal(npu_tile, cpu_tile)
            
            print(f"  C[{tile_row},{tile_col}]: NPU_sum={npu_sum}, CPU_sum={cpu_sum}, Match={match}")
            
            if not match:
                print(f"    NPU[0,0]={npu_tile[0,0]}, CPU[0,0]={cpu_tile[0,0]}")
                print(f"    NPU[mid]={npu_tile[m//2,n//2]}, CPU[mid]={cpu_tile[m//2,n//2]}")
                
                # Expected per-element value for this output tile
                # C[i,j] = sum_k A[i,k] * B[k,j]
                # A=1, so C[i,j] = sum_k B[k,j]
                # For output tile (0, tile_col), it accumulates B[kt, tile_col] for all kt
                # Each element in output tile column j gets sum of B[:, j] across K tiles
                # Since B tiles have constant value per tile, column j in tile (kt, tile_col) = val
                expected_elem = 0
                for kt in range(K_tiles):
                    tile_id = kt * N_tiles + tile_col
                    val = (tile_id * 7 + 1) % 127
                    if val == 0:
                        val = 1
                    expected_elem += val
                print(f"    Expected per-element: {expected_elem}")


def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0],
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--trace", action="store_true", help="Use trace inputs")
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N

    r, s, t = 4, 8, 8
    if M % r != 0 or K % s != 0 or N % t != 0:
        sys.exit(f"Shape {M}x{K}x{N} not compatible with aie2 int8 MAC dims ({r},{s},{t}).")
    if M % _TILE_M != 0 or K % _TILE_K != 0 or N % _TILE_N != 0:
        sys.exit(f"Shape {M}x{K}x{N} not divisible by tile size ({_TILE_M},{_TILE_K},{_TILE_N}).")

    dev = iron.get_current_device()
    cols = dev.cols
    print(f"device: {type(dev).__name__}  columns: {cols}")

    print(f"Building INT8 GEMM: A={M}x{K}, B={K}x{N}, C={M}x{N}  tile=({_TILE_M},{_TILE_K},{_TILE_N})")

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
        return _build_design(M, K, N)

    # Create inputs
    if args.trace:
        print("Using trace inputs...")
        A_np, B_np, K_tiles, N_tiles = create_trace_inputs(M, K, N)
        print(f"  K_tiles={K_tiles}, N_tiles={N_tiles}")
        print(f"  B[0,0]={B_np[0,0]}, B[0,1]={B_np[0,64] if N>=128 else 'N/A'}")
        print(f"  B[64,0]={B_np[64,0] if K>=128 else 'N/A'}")
    else:
        rng = np.random.default_rng(42)
        A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)
        K_tiles, N_tiles = K // _TILE_K, N // _TILE_N

    compiled = int8_gemm_design.specialize(M=M, K=K, N=N).compile()
    print(f"  xclbin: {compiled[0]}")
    print(f"  insts:  {compiled[1]}")

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    # Warmup
    int8_gemm_design(A, B, C, M=M, K=K, N=N)

    # CPU reference
    ref = cpu_int8_gemm(A_np, B_np)

    # Benchmark
    host_times = []
    for i in range(args.iters):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        start = time.perf_counter()
        ret = int8_gemm_design(A, B, C, M=M, K=K, N=N)
        e2e_us = (time.perf_counter() - start) * 1e6
        host_times.append(e2e_us)

    got = C.numpy()
    
    if args.trace:
        analyze_output(got, ref, M, N, K_tiles, N_tiles)
    else:
        mismatches = int(np.sum(got != ref))
        print(f"\nCorrectness check: {mismatches} mismatches out of {M*N}")
        if mismatches == 0:
            print("PASS")
        else:
            print("FAIL")
            print("got[:8] =", got[:8].flatten())
            print("ref[:8] =", ref[:8].flatten())


if __name__ == "__main__":
    main()