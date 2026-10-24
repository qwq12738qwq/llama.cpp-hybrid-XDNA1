#!/usr/bin/env python3
"""Phase 40F: 32 K-block correctness test.

Tests Q8_0 GEMM with M=512, K=1024, N=3584 (32 K-blocks) using:
  - Test A: uniform scales (all 1.0)
  - Test B: two scales (first half=1, second half=2)
  - Test C: geometric scales [1, 2, 4, 8, ...]
  - Test D: realistic random scales

Compares host-folded approach (Phase 39C - WRONG) vs correct per-block scaling.
"""
import numpy as np
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ref_q8_gemm import gen_q8_0_data, q8_0_gemm_ref


def pack_q8_0(A_i8, As, B_i8, Bs):
    """Pack Q8_0 tensors in the format expected by the kernel."""
    M, K = A_i8.shape
    N = B_i8.shape[1]
    nb = K // 32
    
    # A: [M, K] -> [M, nb*34]
    A_packed = np.zeros((M, nb * 34), dtype=np.uint8)
    for m in range(M):
        for b in range(nb):
            scale_bytes = struct.pack('<e', float(As[m, b]))
            block = A_i8[m, b*32:(b+1)*32].astype(np.int8)
            A_packed[m, b*34:b*34+2] = np.frombuffer(scale_bytes, dtype=np.uint8)
            A_packed[m, b*34+2:b*34+34] = np.frombuffer(bytes(block), dtype=np.uint8)
    
    # B: [K, N] -> [N*nb, 34]
    B_packed = np.zeros((N, nb * 34), dtype=np.uint8)
    for n in range(N):
        for b in range(nb):
            scale_bytes = struct.pack('<e', float(Bs[b, n]))
            block = B_i8[b*32:(b+1)*32, n].astype(np.int8)
            B_packed[n, b*34:b*34+2] = np.frombuffer(scale_bytes, dtype=np.uint8)
            B_packed[n, b*34+2:b*34+34] = np.frombuffer(bytes(block), dtype=np.uint8)
    
    return A_packed, B_packed


def host_fold_gemm(A_packed, B_packed, M, K, N):
    """WRONG host-folded approach from Phase 39C."""
    nb = K // 32
    A_i8 = np.zeros((M, K), dtype=np.int8)
    As = np.zeros((M, nb), dtype=np.float32)
    for m in range(M):
        for b in range(nb):
            scale = np.frombuffer(A_packed[m, b*34:b*34+2].tobytes(), dtype=np.float16)[0]
            As[m, b] = scale
            A_i8[m, b*32:(b+1)*32] = A_packed[m, b*34+2:b*34+34].astype(np.int8)
    
    B_i8 = np.zeros((K, N), dtype=np.int8)
    Bs = np.zeros((nb, N), dtype=np.float32)
    for n in range(N):
        for b in range(nb):
            scale = np.frombuffer(B_packed[n, b*34:b*34+2].tobytes(), dtype=np.float16)[0]
            Bs[b, n] = scale
            B_i8[b*32:(b+1)*32, n] = B_packed[n, b*34+2:b*34+34].astype(np.int8)
    
    # Host-side global scale folding
    C_fold = np.zeros((M, N), dtype=np.float64)
    for b in range(nb):
        k_start = b * 32
        k_end = k_start + 32
        A_blk = A_i8[:, k_start:k_end]
        B_blk = B_i8[k_start:k_end, :]
        
        amax = 1e-9
        for m in range(M):
            for kk in range(32):
                val = As[m, b] * A_blk[m, kk]
                if abs(val) > amax:
                    amax = abs(val)
        gs_a = amax / 127.0
        
        amax = 1e-9
        for n in range(N):
            for kk in range(32):
                val = Bs[b, n] * B_blk[kk, n]
                if abs(val) > amax:
                    amax = abs(val)
        gs_b = amax / 127.0
        gs = gs_a * gs_b
        
        A_q = np.round(A_blk.astype(np.float64) * As[:, b:b+1].astype(np.float64) / gs_a).astype(np.int64)
        B_q = np.round(B_blk.astype(np.float64) * Bs[b:b+1, :].astype(np.float64) / gs_b).astype(np.int64)
        p = A_q @ B_q
        C_fold += p * gs
    
    return C_fold.astype(np.float32)


def main():
    M, K, N = 512, 1024, 3584
    nb = K // 32
    patterns = ['uniform', 'two_scale', 'geometric', 'realistic']
    
    print(f"Phase 40F: 32 K-block correctness test")
    print(f"  M={M} K={K} N={N} nb={nb}")
    print()
    
    results = []
    for pattern in patterns:
        print(f"Test: {pattern}")
        A_i8, As, B_i8, Bs = gen_q8_0_data(M, K, N, seed=42, scale_pattern=pattern)
        
        # Reference
        C_ref = q8_0_gemm_ref(A_i8, As, B_i8, Bs)
        
        # Host-folded
        A_packed, B_packed = pack_q8_0(A_i8, As, B_i8, Bs)
        C_fold = host_fold_gemm(A_packed, B_packed, M, K, N)
        
        # Compare
        diff = np.abs(C_fold.astype(np.float64) - C_ref.astype(np.float64))
        max_err = float(np.max(diff))
        rel_err = float(np.max(diff[np.abs(C_ref) > 1e-6] / np.abs(C_ref[np.abs(C_ref) > 1e-6])))
        bad = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))
        
        print(f"  Host-folded: max_err={max_err:.4e} rel_err={rel_err:.4e} bad={bad}/{M*N}")
        
        if max_err > 0:
            print(f"  FAIL: Host folding produces catastrophic error")
        else:
            print(f"  PASS: Host folding correct for this pattern")
        
        results.append({
            'pattern': pattern,
            'max_err': max_err,
            'rel_err': rel_err,
            'bad': bad,
            'passed': max_err == 0.0
        })
        print()
    
    print("Summary:")
    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  {r['pattern']:12s}: {status} (max_err={r['max_err']:.4e})")
    
    # Expected: uniform=pass, two_scale=fail, geometric=fail, realistic=fail
    expected_failures = ['two_scale', 'geometric', 'realistic']
    all_correct = all(not r['passed'] for r in results if r['pattern'] in expected_failures)
    if all_correct:
        print()
        print("Phase 40F PASS: Host folding fails as expected for non-uniform scales")
    else:
        print()
        print("Phase 40F INCONCLUSIVE: Check results above")


if __name__ == "__main__":
    main()
