#!/usr/bin/env python3
# trace_tas.py — Trace TensorTiler2D group_tiler output order

import sys
sys.path.insert(0, "/home/ubuntu12738/Storage/Cache/llama.cpp/experiments/npu-int8-gemm")

import numpy as np
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern

_TILE_M = 64
_TILE_K = 64
_TILE_N = 64


def trace_a_tas(M, K, N):
    """Trace A tensor TAS tile order for given M, K, N."""
    m, k = _TILE_M, _TILE_K
    print(f"\n=== A TAS: M={M}, K={K}, N={N} ===")
    print(f"  M_div_m = {M//m}, K_div_k = {K//k}, N_div_n = {N//_TILE_N}")

    # Buggy version: pattern_repeat=(N//n)
    taps_buggy = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // _TILE_N)
    )
    print(f"  BUGGY pattern_repeat={N//_TILE_N}: num_taps={len(taps_buggy)}")
    for i, tap in enumerate(taps_buggy):
        td = tap.transformation_dims
        print(f"    TAP[{i}]: td={td}, offset={tap.offset}, sizes={tap.sizes}, strides={tap.strides}")

    # Fixed version: pattern_repeat=1
    taps_fixed = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=1
    )
    print(f"  FIXED pattern_repeat=1: num_taps={len(taps_fixed)}")
    for i, tap in enumerate(taps_fixed):
        td = tap.transformation_dims
        print(f"    TAP[{i}]: td={td}, offset={tap.offset}, sizes={tap.sizes}, strides={tap.strides}")


def main():
    test_cases = [
        (64, 1024, 64),
        (64, 1024, 128),
        (128, 1024, 64),
        (128, 1024, 128),
        (128, 1024, 256),
        (256, 1024, 128),
    ]

    for M, K, N in test_cases:
        trace_a_tas(M, K, N)


if __name__ == "__main__":
    main()
