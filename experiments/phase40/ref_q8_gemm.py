#!/usr/bin/env python3
"""Phase 40B: CPU reference model for Q8_0 GEMM with per-K-block scales.

This reference preserves every Q8_0 block scale independently and is used
to verify AIE kernel correctness.

Q8_0 layout:
  - Block size: 32 elements along K
  - Each block: [fp16 scale][32 x int8 quants]
  - Dequant: float = int8 * fp16_to_fp32(scale)

GEMM semantics:
  C[m,n] = sum_b (sum_i A_int8[m,b*32+i] * B_int8[b*32+i,n]) * As[m,b] * Bs[b,n]
"""
import argparse
import numpy as np


def gen_q8_0_data(M, K, N, seed=0, scale_pattern='uniform'):
    rng = np.random.default_rng(seed)
    nb = K // 32
    
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    
    if scale_pattern == 'uniform':
        As = np.ones((M, nb), dtype=np.float32)
        Bs = np.ones((nb, N), dtype=np.float32)
    elif scale_pattern == 'two_scale':
        As = np.ones((M, nb), dtype=np.float32)
        Bs = np.ones((nb, N), dtype=np.float32)
        As[:, nb//2:] = 2.0
        Bs[nb//2:, :] = 2.0
    elif scale_pattern == 'geometric':
        # Use powers of 2 up to a reasonable max for float16
        factors = np.array([min(2.0 ** i, 65504.0) for i in range(nb)], dtype=np.float32)
        As = np.ones((M, nb), dtype=np.float32) * factors[np.newaxis, :]
        Bs = np.ones((nb, N), dtype=np.float32) * factors[:, np.newaxis]
    elif scale_pattern == 'realistic':
        As = rng.uniform(0.001, 0.02, size=(M, nb)).astype(np.float32)
        Bs = rng.uniform(0.001, 0.02, size=(nb, N)).astype(np.float32)
    else:
        raise ValueError(f"Unknown scale pattern: {scale_pattern}")
    
    return A_i8, As, B_i8, Bs


def q8_0_gemm_ref(A_i8, As, B_i8, Bs):
    """Reference Q8_0 GEMM preserving per-block scales."""
    M, K = A_i8.shape
    N = B_i8.shape[1]
    nb = K // 32
    
    C = np.zeros((M, N), dtype=np.float64)
    for b in range(nb):
        k_start = b * 32
        k_end = k_start + 32
        A_blk = A_i8[:, k_start:k_end].astype(np.int64)
        B_blk = B_i8[k_start:k_end, :].astype(np.int64)
        p = A_blk @ B_blk
        scale = As[:, b:b+1].astype(np.float64) * Bs[b:b+1, :].astype(np.float64)
        C += p * scale
    
    return C.astype(np.float32)


def pack_q8_0_block(i8_block, scale):
    """Pack int8 block (32 elements) + fp16 scale into Q8_0 bytes."""
    import struct
    scale_f16 = struct.pack('<e', scale)
    return scale_f16 + bytes(i8_block & 0xFF)


def verify_against_host_folding(A_i8, As, B_i8, Bs, M, K, N):
    """Show that host-side global scale folding is numerically wrong."""
    nb = K // 32
    C_ref = q8_0_gemm_ref(A_i8, As, B_i8, Bs)
    
    # Host-folded approach (Phase 39C - WRONG)
    C_fold = np.zeros((M, N), dtype=np.float64)
    for b in range(nb):
        k_start = b * 32
        k_end = k_start + 32
        A_blk = A_i8[:, k_start:k_end]
        B_blk = B_i8[k_start:k_end, :]
        
        # Compute global scale for this K-block
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
        
        # Re-quantize and compute
        A_q = np.round(A_blk.astype(np.float64) * As[:, b:b+1].astype(np.float64) / gs_a).astype(np.int64)
        B_q = np.round(B_blk.astype(np.float64) * Bs[b:b+1, :].astype(np.float64) / gs_b).astype(np.int64)
        p = A_q @ B_q
        C_fold += p * gs
    
    err = np.abs(C_fold - C_ref)
    max_err = float(np.max(err))
    rel_err = float(np.max(err[np.abs(C_ref) > 1e-6] / np.abs(C_ref[np.abs(C_ref) > 1e-6])))
    return max_err, rel_err, C_ref, C_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--pattern", type=str, default='geometric',
                    choices=['uniform', 'two_scale', 'geometric', 'realistic'])
    ap.add_argument("--verify-folding", action='store_true')
    args = ap.parse_args()
    
    M, K, N = args.M, args.K, args.N
    A_i8, As, B_i8, Bs = gen_q8_0_data(M, K, N, seed=42, scale_pattern=args.pattern)
    
    print(f"Q8_0 GEMM reference: M={M} K={K} N={N} pattern={args.pattern}")
    print(f"  nb={K//32} As shape={As.shape} Bs shape={Bs.shape}")
    
    C_ref = q8_0_gemm_ref(A_i8, As, B_i8, Bs)
    print(f"  C_ref[0,0]={C_ref[0,0]:.6f} C_ref[M-1,N-1]={C_ref[M-1,N-1]:.6f}")
    
    if args.verify_folding:
        max_err, rel_err, _, C_fold = verify_against_host_folding(A_i8, As, B_i8, Bs, M, K, N)
        print(f"  Host-folded: max_err={max_err:.4e} rel_err={rel_err:.4e}")
        print(f"  C_fold[0,0]={C_fold[0,0]:.6f} C_fold[M-1,N-1]={C_fold[M-1,N-1]:.6f}")


if __name__ == "__main__":
    main()
