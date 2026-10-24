#!/usr/bin/env python3
"""CPU-side simulation of the Q8 mmul kernel to verify packing correctness.

Simulates the EXACT memory accesses the C++ kernel does, using the packed
buffers. Compares against ref_q8_0_gemm.
"""
import sys
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase40")
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase39")

import numpy as np
from ref_q8_gemm import ref_q8_0_gemm
from bench_q8_c4 import pack_A_mmul, pack_B_mmul

M_TILE = 64
N_TILE = 64
KC = 256
NBLK = KC // 32
KA = KC // 8
KB = KC // 8


def pack_B_mmul_c4(B, Bs, K, N, n_tile, k_chunk, n_cols):
    """Pack B for C4: reorder tiles by (col, nt) for multi-worker access."""
    n_nt, n_kc = N // n_tile, K // k_chunk
    nb = k_chunk // 32
    n_nt_c = n_nt // n_cols

    full_B = pack_B_mmul(B, Bs, K, N, n_tile, k_chunk)

    b_bytes = k_chunk * n_tile + nb * n_tile * 4
    full_reshaped = full_B.reshape(n_nt, n_kc, b_bytes)

    reordered = np.empty((n_cols, n_nt_c, n_kc, b_bytes), dtype=np.int8)
    for col in range(n_cols):
        for nt_c in range(n_nt_c):
            src_nt = col * n_nt_c + nt_c
            reordered[col, nt_c] = full_reshaped[src_nt]

    return reordered.reshape(-1)


def simulate_kernel_c1(A_packed, B_packed, M, K, N):
    """Simulate the C1 mmul kernel on CPU using exact memory accesses."""
    n_mt = M // M_TILE
    n_nt = N // N_TILE
    n_kc = K // KC

    a_bytes = M_TILE * KC + M_TILE * NBLK * 4
    b_bytes = KC * N_TILE + NBLK * N_TILE * 4

    C = np.zeros((M, N), dtype=np.float64)

    A_packed = A_packed.reshape(n_mt, n_kc, -1)
    B_packed = B_packed.reshape(n_nt, n_kc, -1)

    for mt in range(n_mt):
        for nt in range(n_nt):
            a_tile = A_packed[mt, 0]
            b_tile = B_packed[nt, 0]

            as_scales = np.frombuffer(a_tile[M_TILE * KC:].tobytes(), dtype=np.float32).reshape(M_TILE, NBLK)
            bs_scales = np.frombuffer(b_tile[KC * N_TILE:].tobytes(), dtype=np.float32).reshape(NBLK, N_TILE)

            for bb in range(NBLK):
                for i in range(16):
                    for j in range(8):
                        tmp = np.zeros((4, 8), dtype=np.int32)
                        for kk in range(4):
                            a_offset = (i * KA + bb * 4 + kk) * 32
                            b_offset = (j * KB + bb * 4 + kk) * 64
                            av = a_tile[a_offset:a_offset + 32].reshape(4, 8)
                            bv = b_tile[b_offset:b_offset + 64].reshape(8, 8)
                            tmp += av.astype(np.int32) @ bv.astype(np.int32)

                        bsv = bs_scales[bb, j*8:(j+1)*8]
                        for rr in range(4):
                            row = mt * M_TILE + i * 4 + rr
                            col = nt * N_TILE + j * 8
                            asi = as_scales[i * 4 + rr, bb]
                            pvf = tmp[rr, :].astype(np.float64)
                            f = asi * bsv * pvf
                            C[row, col:col+8] += f

    return C.astype(np.float32)


def simulate_kernel_c4(A_packed, B_packed, M, K, N, n_cols=4):
    """Simulate the C4 mmul kernel on CPU using exact memory accesses."""
    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    n_kc = K // KC

    a_bytes = M_TILE * KC + M_TILE * NBLK * 4
    b_bytes = KC * N_TILE + NBLK * N_TILE * 4

    C = np.zeros((M, N), dtype=np.float64)

    A_packed = A_packed.reshape(n_mt, n_kc, -1)
    B_packed = B_packed.reshape(n_cols, n_nt_c, n_kc, -1)

    for col in range(n_cols):
        for nt_c in range(n_nt_c):
            for mt in range(n_mt):
                a_tile = A_packed[mt, 0]
                b_tile = B_packed[col, nt_c, 0]

                as_scales = np.frombuffer(a_tile[M_TILE * KC:].tobytes(), dtype=np.float32).reshape(M_TILE, NBLK)
                bs_scales = np.frombuffer(b_tile[KC * N_TILE:].tobytes(), dtype=np.float32).reshape(NBLK, N_TILE)

                nt = col * n_nt_c + nt_c

                for bb in range(NBLK):
                    for i in range(16):
                        for j in range(8):
                            tmp = np.zeros((4, 8), dtype=np.int32)
                            for kk in range(4):
                                a_offset = (i * KA + bb * 4 + kk) * 32
                                b_offset = (j * KB + bb * 4 + kk) * 64
                                av = a_tile[a_offset:a_offset + 32].reshape(4, 8)
                                bv = b_tile[b_offset:b_offset + 64].reshape(8, 8)
                                tmp += av.astype(np.int32) @ bv.astype(np.int32)

                            bsv = bs_scales[bb, j*8:(j+1)*8]
                            for rr in range(4):
                                row = mt * M_TILE + i * 4 + rr
                                col_start = nt * N_TILE + j * 8
                                asi = as_scales[i * 4 + rr, bb]
                                pvf = tmp[rr, :].astype(np.float64)
                                f = asi * bsv * pvf
                                C[row, col_start:col_start+8] += f

    return C.astype(np.float32)


def main():
    shapes = [
        (64, 256, 64, 1),
        (64, 256, 256, 4),
    ]

    for M, K, N, n_cols in shapes:
        print("")
        print("=" * 60)
        print(f"Testing M={M} K={K} N={N} n_cols={n_cols}")
        print("=" * 60)

        for test_name, uniform in [
            ("uniform", True),
            ("two-scale", False),
            ("non-uniform", False),
        ]:
            print("")
            print(f"--- {test_name} ---")

            rng = np.random.default_rng(42)
            A = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
            B = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
            nb = K // 32

            if uniform:
                As = np.ones((M, nb), np.float32)
                Bs = np.ones((nb, N), np.float32)
            else:
                As = rng.uniform(0.001, 0.02, size=(M, nb)).astype(np.float32)
                Bs = rng.uniform(0.001, 0.02, size=(nb, N)).astype(np.float32)

            if test_name == "two-scale":
                As = np.ones((M, nb), np.float32)
                As[:, nb//2:] = 2.0
                Bs = np.ones((nb, N), np.float32)
                Bs[nb//2:, :] = 2.0

            if test_name == "non-uniform":
                rng = np.random.default_rng(42)
                As = np.power(2.0, rng.integers(0, 8, size=(M, nb))).astype(np.float32)
                Bs = np.power(2.0, rng.integers(0, 8, size=(nb, N))).astype(np.float32)

            C_ref = ref_q8_0_gemm(M, K, N, A, As, B, Bs)

            # Test C1 packing + simulation
            Ap = pack_A_mmul(A, As, M, K, M_TILE, KC)
            Bp = pack_B_mmul(B, Bs, K, N, N_TILE, KC)
            C_sim_c1 = simulate_kernel_c1(Ap, Bp, M, K, N)

            diff_c1 = np.abs(C_sim_c1.astype(np.float64) - C_ref.astype(np.float64))
            max_err_c1 = float(np.max(diff_c1))
            mask = np.abs(C_ref) > 1e-6
            rel_err_c1 = float(np.max(diff_c1[mask] / np.abs(C_ref[mask]))) if np.any(mask) else 0.0
            bad_c1 = int(np.sum(diff_c1 > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))

            print(f"  C1 simulation: max_err={max_err_c1:.3e} rel_err={rel_err_c1:.3e} bad={bad_c1}")

            # Test C4 packing + simulation
            if n_cols > 1:
                Bp_c4 = pack_B_mmul_c4(B, Bs, K, N, N_TILE, KC, n_cols)
                C_sim_c4 = simulate_kernel_c4(Ap, Bp_c4, M, K, N, n_cols)

                diff_c4 = np.abs(C_sim_c4.astype(np.float64) - C_ref.astype(np.float64))
                max_err_c4 = float(np.max(diff_c4))
                rel_err_c4 = float(np.max(diff_c4[mask] / np.abs(C_ref[mask]))) if np.any(mask) else 0.0
                bad_c4 = int(np.sum(diff_c4 > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))

                print(f"  C4 simulation: max_err={max_err_c4:.3e} rel_err={rel_err_c4:.3e} bad={bad_c4}")


if __name__ == "__main__":
    main()
