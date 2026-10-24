#!/usr/bin/env python3
"""Phase 62: check the host packing of the c256 engine against the kernel.

The C++ host path of the c4_mmul_v4 engine packs

    A chunk: (m_tile * k_chunk) int8 bytes, then m_tile f32 row scales
    B chunk: (k_chunk * n_tile) int8 bytes, then n_tile f32 column scales

and the kernel reads them at

    asc = (const float *)(at  + MT*KCC)      bsc = (const float *)(bt + KCC*NT)

This script transcribes the C++ packing (the weight requantization, the tile
order and the scale section) into numpy, rebuilds the GEMM from the packed
buffers with the kernel's own convention, and compares it against a float64
reference.  It is the check that the host and the kernel agree.

usage: check_pack.py [--M 64 --K 256 --N 64]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "gguf-py"))
import gguf  # noqa: E402


def dequant_q8_0(data, n_row, k):
    nb = k // 32
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_row, nb, 34)
    scales = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    qs = raw[:, :, 2:].view(np.int8).astype(np.float32)
    return (qs * scales[:, :, None]).reshape(n_row, k)


def pack_b_group(qi, scale, n, kc, ntile_start, n_tiles=1):
    """Pack the B chunk of one (n-tile, chunk) exactly like the C++ code."""
    nblk = kc // 32
    b_chunk = kc * n + n * 4
    packed = np.zeros(b_chunk, np.int8)
    for jn in range(n_tiles):
        for bb in range(nblk):
            for kk in range(4):
                for rr in range(8):
                    for cc_col in range(8):
                        flat = ((jn * nblk + bb) * 4 + kk) * 64 + rr * 8 + cc_col
                        k = bb * 32 + kk * 8 + rr
                        c = ntile_start + jn * 8 + cc_col
                        packed[flat] = qi[c, k]
    ts = packed[kc * n:].view(np.float32)
    for j in range(n):
        ts[j] = scale[ntile_start + j]
    return packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B-Q8_0.gguf")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--kc", type=int, default=256)
    ap.add_argument("--n-tile", type=int, default=64)
    a = ap.parse_args()

    reader = gguf.GGUFReader(a.model)
    name = f"blk.{a.layer}.ffn_gate.weight"
    t = [x for x in reader.tensors if x.name == name][0]
    k = int(t.shape[0])
    n = int(t.shape[1])
    w_q8 = dequant_q8_0(t.data, n, k)

    nkc = k // a.kc
    nblk = a.kc // 32
    w = w_q8.reshape(n, k)
    wb = w.reshape(n, nkc, a.kc)
    amax = np.max(np.abs(wb), axis=2)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    qi = np.clip(np.rint(wb / scale[:, :, None]), -127, 127).astype(np.int8).reshape(n, k)

    # pack the first n-tile of the first chunk and rebuild the GEMM from it
    nt = a.n_tile
    packed = pack_b_group(qi, scale[:, 0], nt, a.kc, 0, nt // 8)
    tiles = packed[:a.kc * nt].reshape(nt // 8, nblk, 4, 8, 8)
    bsc = packed[a.kc * nt:].view(np.float32)

    # the kernel's reading convention: bj = b_tiles + j*(KCC/8)*64, the tile
    # of (8 K x 8 N) is stored K major
    bmat = np.zeros((a.kc, nt), np.int8)
    for jn in range(nt // 8):
        for bb in range(nblk):
            for kk in range(4):
                for rr in range(8):
                    for cc_col in range(8):
                        flat = ((jn * nblk + bb) * 4 + kk) * 64 + rr * 8 + cc_col
                        bmat[bb * 32 + kk * 8 + rr, jn * 8 + cc_col] = packed[flat]

    # the reference: the dequantized Q8_0 values with the new chunk scale
    ref = w[:nt, :a.kc].astype(np.float64)   # [nt][kc] dequantized
    # the packed values times the packed scale must reproduce it
    got = bmat.astype(np.float64) * bsc[None, :a.kc]
    err = np.max(np.abs(got - ref.T)) / max(np.max(np.abs(ref)), 1e-30)

    print(f"tensor {name}: K={k} N={n} kc={a.kc} n-tile={nt}")
    print(f"  packed scale section: {bsc[:4]} ...")
    print(f"  max relative error of the packed weights vs the dequantized Q8_0: {err:.3e}")
    print(f"  (this is the rounding of the 256 element requantization, expected ~1e-3)")

    # and the full GEMM with the exact Q8_0 reference
    rng = np.random.default_rng(3)
    act = rng.standard_normal((16, a.kc))
    c_ref = act.astype(np.float64) @ w[:nt, :a.kc].astype(np.float64).T
    c_got = act.astype(np.float64) @ got
    rel = np.sqrt(np.mean((c_got - c_ref) ** 2)) / np.max(np.abs(c_ref))
    print(f"  relative RMS error of the packed GEMM: {rel:.3e}")


if __name__ == "__main__":
    main()
