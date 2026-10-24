#!/usr/bin/env python3
"""Phase 43: Bounded topology search for 2x4 and 3x4 mmul-v2 topologies.

Tests CPU simulation correctness and estimates resource budgets before
any hardware compilation attempt.

The kernel computation is independent of topology, so CPU simulation
uses the v1 generator's reference functions.

Usage:
  python3 experiments/phase43/test_topology_sweep.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase40'))
from gen_q8_c4_mmul_v2 import (
    pack_A, pack_B, simulate_kernel_cpu, ref_q8_gemm,
    M_TILE, N_TILE, KC
)

import numpy as np


def count_resources(n_rows, n_cols):
    """Estimate ObjectFifo count for v4 topology."""
    n_shim_mem_A = n_rows if n_cols > n_rows else n_cols
    n_A_tiles_per_shim = n_rows // n_cols if n_cols < n_rows else 1

    a_l3l2 = n_shim_mem_A
    a_l2l1 = n_rows  # from split
    b_l3l2 = n_rows * n_cols
    b_l2l1 = n_rows * n_cols  # from forward
    c_l2l3 = n_cols
    c_l1l2 = n_rows * n_cols  # from join

    total = a_l3l2 + a_l2l1 + b_l3l2 + b_l2l1 + c_l2l3 + c_l1l2
    return {
        'A_L3L2': a_l3l2,
        'A_L2L1': a_l2l1,
        'B_L3L2': b_l3l2,
        'B_L2L1': b_l2l1,
        'C_L2L3': c_l2l3,
        'C_L1L2': c_l1l2,
        'total': total,
    }


def test_topology_cpu(n_rows, n_cols, M=512, K=1024, N=3584):
    """Run CPU simulation for a given topology. Topology doesn't affect
    the kernel computation, so result should match reference."""
    rng = np.random.default_rng(42)
    nb = K // 32
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    As = np.ones((M, nb), dtype=np.float32)
    Bs = np.ones((nb, N), dtype=np.float32)

    C_ref = ref_q8_gemm(A_i8, As, B_i8, Bs)
    Ap = pack_A(A_i8, As, M, K, M_TILE, KC)
    Bp = pack_B(B_i8, Bs, K, N, N_TILE, KC)
    C_dev = simulate_kernel_cpu(Ap, Bp, M, K, N, M_TILE, N_TILE, KC)

    diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
    max_err = float(np.max(diff))
    bad = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))

    return {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'workers': n_rows * n_cols,
        'max_err': max_err,
        'bad': bad,
        'total': M * N,
        'correct': bad == 0,
        'resources': count_resources(n_rows, n_cols),
    }


def test_down_projection(M=512, K=3584, N=1024):
    """Test K=3584 down projection (ffn_down shape)."""
    rng = np.random.default_rng(42)
    nb = K // 32
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    As = np.ones((M, nb), dtype=np.float32)
    Bs = np.ones((nb, N), dtype=np.float32)

    C_ref = ref_q8_gemm(A_i8, As, B_i8, Bs)
    Ap = pack_A(A_i8, As, M, K, M_TILE, KC)
    Bp = pack_B(B_i8, Bs, K, N, N_TILE, KC)
    C_dev = simulate_kernel_cpu(Ap, Bp, M, K, N, M_TILE, N_TILE, KC)

    diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
    max_err = float(np.max(diff))
    bad = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))

    return {
        'shape': f'{M}x{K}x{N}',
        'max_err': max_err,
        'bad': bad,
        'total': M * N,
        'correct': bad == 0,
    }


def main():
    print("Phase 43: Topology Sweep (CPU Simulation)")
    print("=" * 60)
    print()

    # Baseline 1x4
    print("1. Baseline 1x4 (v1)")
    r1 = test_topology_cpu(1, 4)
    print(f"   workers={r1['workers']}  correct={r1['correct']}  "
          f"max_err={r1['max_err']:.4e}  bad={r1['bad']}/{r1['total']}")
    print(f"   resources: {r1['resources']}")
    print()

    # 2x4
    print("2. Topology 2x4")
    r2 = test_topology_cpu(2, 4)
    print(f"   workers={r2['workers']}  correct={r2['correct']}  "
          f"max_err={r2['max_err']:.4e}  bad={r2['bad']}/{r2['total']}")
    print(f"   resources: {r2['resources']}")
    print()

    # 3x4
    print("3. Topology 3x4")
    r3 = test_topology_cpu(3, 4)
    print(f"   workers={r3['workers']}  correct={r3['correct']}  "
          f"max_err={r3['max_err']:.4e}  bad={r3['bad']}/{r3['total']}")
    print(f"   resources: {r3['resources']}")
    print()

    # Resource budget analysis
    print("4. Resource Budget Analysis")
    print("   v2 (4x4 shared-B): 36 ObjectFifos -> hardware race")
    print("   v3 (16x1 per-B): 48 L3/L2 FIFOs -> Shim/NOCTile budget exceeded")
    print("   v4 (4x4 per-worker-B): 36 MemTile channels -> MemTile budget exceeded")
    print()
    for name, r in [('1x4', r1), ('2x4', r2), ('3x4', r3)]:
        total = r['resources']['total']
        status = 'PASS' if total <= 36 else 'AT RISK' if total <= 40 else 'LIKELY FAIL'
        print(f"   {name}: {total} ObjectFifos -> {status}")
    print()

    # Down projection
    print("5. Down Projection (K=3584, N=1024)")
    r_down = test_down_projection()
    print(f"   shape={r_down['shape']}  correct={r_down['correct']}  "
          f"max_err={r_down['max_err']:.4e}  bad={r_down['bad']}/{r_down['total']}")
    print()

    # Selection
    print("6. Selection (based on CPU simulation + resource budget)")
    valid = []
    for name, r in [('1x4', r1), ('2x4', r2), ('3x4', r3)]:
        if r['correct'] and r['resources']['total'] <= 36:
            valid.append(name)
    print(f"   Valid topologies (CPU sim + budget <= 36): {valid}")
    if valid:
        print(f"   Selected: {valid[0]} (first valid)")
    print()
    print("NOTE: Hardware compilation required to confirm resource allocation.")
    print("      CPU simulation verifies kernel correctness only.")


if __name__ == "__main__":
    main()
