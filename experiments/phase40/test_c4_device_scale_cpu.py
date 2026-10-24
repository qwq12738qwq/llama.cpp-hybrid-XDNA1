#!/usr/bin/env python3
"""Phase 40E/F: CPU simulation of device-side scale kernel.

Simulates the C4_WA + device-side scaling kernel on CPU to verify
mathematical correctness before running on actual hardware.

The kernel does:
  1. int8 x int8 -> int32 mmul (preserved)
  2. per-block scale application on device
  3. float accumulation
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ref_q8_gemm import gen_q8_0_data, q8_0_gemm_ref


def simulate_c4_device_scale_kernel(A_i8, As, B_i8, Bs, m_tile=64, n_tile=64, k_chunk=256):
    """Simulate the C4 device-scale kernel on CPU."""
    M, K = A_i8.shape
    N = B_i8.shape[1]
    nb = K // 32
    n_kc = K // k_chunk
    nblk = k_chunk // 32
    
    C = np.zeros((M, N), dtype=np.float64)
    
    for mt in range(M // m_tile):
        for nt in range(N // n_tile):
            m_start = mt * m_tile
            m_end = m_start + m_tile
            n_start = nt * n_tile
            n_end = n_start + n_tile
            
            C_tile = np.zeros((m_tile, n_tile), dtype=np.float64)
            
            for kc in range(n_kc):
                k_start = kc * k_chunk
                
                for blk in range(nblk):
                    kb = kc * nblk + blk
                    k_blk_start = k_start + blk * 32
                    k_blk_end = k_blk_start + 32
                    
                    # Get int8 block
                    A_blk = A_i8[m_start:m_end, k_blk_start:k_blk_end]
                    B_blk = B_i8[k_blk_start:k_blk_end, n_start:n_end]
                    
                    # mmul<4,8,8>: int8 x int8 -> int32
                    C_partial = A_blk.astype(np.int64) @ B_blk.astype(np.int64)
                    
                    # Apply per-block scales on device
                    scale_a = As[m_start:m_end, kb]
                    scale_b = Bs[kb, n_start:n_end]
                    scale_matrix = scale_a[:, np.newaxis] * scale_b[np.newaxis, :]
                    C_tile += C_partial * scale_matrix
            
            C[m_start:m_end, n_start:n_end] = C_tile
    
    return C.astype(np.float32)


def test_pattern(pattern, M=512, K=1024, N=3584):
    """Test a specific scale pattern."""
    print(f"Pattern: {pattern}")
    A_i8, As, B_i8, Bs = gen_q8_0_data(M, K, N, seed=42, scale_pattern=pattern)
    
    # Reference
    C_ref = q8_0_gemm_ref(A_i8, As, B_i8, Bs)
    
    # Simulated device kernel
    C_dev = simulate_c4_device_scale_kernel(A_i8, As, B_i8, Bs)
    
    # Compare (account for float16 precision loss in scale packing)
    diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
    # Use relative tolerance of 1% for float16 scale precision
    rel_tol = 1e-2
    max_err = float(np.max(diff))
    rel_err = float(np.max(diff[np.abs(C_ref) > 1e-6] / np.abs(C_ref[np.abs(C_ref) > 1e-6])))
    bad = int(np.sum(diff > rel_tol * np.maximum(np.abs(C_ref), 1.0)))
    
    print(f"  max_err={max_err:.4e} rel_err={rel_err:.4e} bad={bad}/{M*N}")
    if bad == 0:
        print(f"  PASS: Device-side scale kernel mathematically correct")
    else:
        print(f"  FAIL: Device-side scale kernel has errors")
    print()
    
    return bad == 0


def main():
    print("Phase 40E/F: CPU simulation of C4 device-scale kernel")
    print("=" * 60)
    print()
    
    patterns = ['uniform', 'two_scale', 'geometric', 'realistic']
    results = {}
    
    for pattern in patterns:
        results[pattern] = test_pattern(pattern)
    
    print("Summary:")
    all_pass = True
    for pattern, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {pattern:12s}: {status}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print()
        print("Phase 40E/F PASS: Device-side scale kernel is mathematically correct")
        print("for all test patterns including realistic random scales.")
    else:
        print()
        print("Phase 40E/F FAIL: Some patterns failed")


if __name__ == "__main__":
    main()
