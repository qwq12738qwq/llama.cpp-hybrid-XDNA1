#!/usr/bin/env python3
# trace_tas.py — Trace TensorTiler2D group_tiler output order
# Goal: Reconstruct logical tile index sequence from TAS

import sys
sys.path.insert(0, "/home/ubuntu12738/APPS/Projects/llama.cpp/experiments/npu-int8-gemm")

import numpy as np
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern

# Tile sizes matching the GEMM kernel
_TILE_M = 64
_TILE_K = 64
_TILE_N = 64
MAC_R, MAC_S, MAC_T = 4, 8, 8


def trace_b_tas(K, N, tile_group_col_major):
    """Trace B tensor TAS tile order for given K, N."""
    k, n = _TILE_K, _TILE_N
    print(f"\n=== B TAS: K={K}, N={N}, tile_group_col_major={tile_group_col_major} ===")
    print(f"  K_div_k = {K//k}, N_div_n = {N//n}")

    taps = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), tile_group_col_major=tile_group_col_major
    )
    print(f"  Number of TAPs: {len(taps)}")

    for i, tap in enumerate(taps):
        # Extract tile coordinates from transformation dims
        td = tap.transformation_dims
        print(f"  TAP[{i}]: transformation_dims={td}, offset={tap.offset}")
        print(f"    sizes={tap.sizes}, strides={tap.strides}")

    return taps


def trace_a_tas(M, K):
    """Trace A tensor TAS tile order."""
    m, k = _TILE_M, _TILE_K
    print(f"\n=== A TAS: M={M}, K={K} ===")
    print(f"  M_div_m = {M//m}, K_div_k = {K//k}")

    taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (M // m, K // k)
    )
    print(f"  Number of TAPs: {len(taps)}")

    for i, tap in enumerate(taps):
        td = tap.transformation_dims
        print(f"  TAP[{i}]: transformation_dims={td}, offset={tap.offset}")
        print(f"    sizes={tap.sizes}, strides={tap.strides}")

    return taps


def trace_c_tas(M, N):
    """Trace C tensor TAS tile order (drain order)."""
    m, n = _TILE_M, _TILE_N
    print(f"\n=== C TAS (drain): M={M}, N={N} ===")
    print(f"  M_div_m = {M//m}, N_div_n = {N//n}")

    taps = []
    for tile_row in range(M // m):
        for tile_col in range(N // n):
            offset = tile_row * m * N + tile_col * n
            tap = TensorAccessPattern(
                tensor_dims=(M, N),
                offset=offset,
                sizes=[m, n],
                strides=[N, 1],
            )
            taps.append(tap)
            print(f"  TAP[{len(taps)-1}]: row={tile_row}, col={tile_col}, offset={offset}")

    return taps


def reconstruct_logical_order(taps, K, N, name):
    """Reconstruct logical tile coordinates from TAPs."""
    k, n = _TILE_K, _TILE_N
    logical = []
    for i, tap in enumerate(taps):
        # For B tensor: tap covers k x n region
        # transformation_dims tells us the order of tiles within the group
        td = tap.transformation_dims
        # The first dim is typically the slowest varying
        if len(td) >= 2:
            # Approximate: assume td[0] corresponds to K dimension tiles, td[1] to N dimension tiles
            pass
        logical.append({
            'tap_index': i,
            'offset': tap.offset,
            'transformation_dims': td,
        })
    return logical


def main():
    print("TensorTiler2D Tile Order Forensics")
    print("=" * 60)

    # Test matrix: M=16, K=16, N=64/128/256 (small for debugging)
    # But the TAS logic uses _TILE_M, _TILE_K, _TILE_N = 64, 64, 64
    # So we need M,K,N that are multiples of 64

    test_cases = [
        # (M, K, N)
        (64, 64, 64),   # N_div_n = 1
        (64, 64, 128),  # N_div_n = 2
        (64, 64, 256),  # N_div_n = 4
        (64, 128, 64),  # K_div_k = 2
        (64, 128, 128), # K_div_k = 2, N_div_n = 2
        (64, 1024, 64), # K=1024, N_div_n = 1
        (64, 1024, 128),# K=1024, N_div_n = 2
        (64, 1024, 256),# K=1024, N_div_n = 4
    ]

    for M, K, N in test_cases:
        print(f"\n{'='*60}")
        print(f"Shape: M={M}, K={K}, N={N}")
        print(f"  Tiles: M/{_TILE_M}={M//_TILE_M}, K/{_TILE_K}={K//_TILE_K}, N/{_TILE_N}={N//_TILE_N}")

        # A TAS (always col-major in current code)
        a_taps = trace_a_tas(M, K)

        # B TAS: col-major for K<=512, row-major for K>512
        b_col_major = K <= 512
        b_taps = trace_b_tas(K, N, b_col_major)

        # C TAS (drain order)
        c_taps = trace_c_tas(M, N)

        # For B, also trace the opposite to compare
        if K > 512:
            print(f"\n  --- B TAS (col-major for comparison) ---")
            trace_b_tas(K, N, True)


if __name__ == "__main__":
    main()