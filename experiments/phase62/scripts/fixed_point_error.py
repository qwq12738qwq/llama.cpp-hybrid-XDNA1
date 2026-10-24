#!/usr/bin/env python3
"""Phase 62: the bit budget of a fixed-point (integer) Q8_0 scale.

The phase 62 kernel changed the quantization granularity to get the scale
application out of the block loop.  The alternative is to keep the exact Q8_0
weights and apply the scale in the integer domain:

    V     = w_fix * (P >> S)          (int32, accumulated over the chunk)
    C    += to_float(V) * sA[m,c] * 2^-e[c]

with `w_fix` the fixed point mantissa of the weight block scale (one common
exponent per K chunk) and `sA[m,c]` the activation scale of the chunk.

This script measures the two quantities that decide whether that is viable on
the real weights:

  * the dynamic range of the weight scales inside one K chunk: the mantissa
    has to cover the whole range, so the range costs bits,
  * the magnitude of the int32 partial sums: they decide the down-shift S,

and it computes the error of the scheme for a range of (mantissa bits, S)
combinations against the float64 reference of the exact Q8_0 GEMM.

usage: fixed_point_error.py [--layers 0 12 23] [--M 256]
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
    return qs, scales, nb


def quantize_rows(x, block):
    """int8 with one scale per `block` elements, per row."""
    m, k = x.shape
    nb = k // block
    xb = x.reshape(m, nb, block)
    amax = np.max(np.abs(xb), axis=2)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(xb / scale[:, :, None]), -127, 127).astype(np.int8)
    return q, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B-Q8_0.gguf")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 12, 23])
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--block", type=int, default=256)
    a = ap.parse_args()

    reader = gguf.GGUFReader(a.model)
    tensors = []
    for t in reader.tensors:
        if t.tensor_type != 8:
            continue
        for kind in ("ffn_gate", "ffn_down"):
            if kind in t.name and ".weight" in t.name:
                for L in a.layers:
                    if t.name.startswith(f"blk.{L}."):
                        tensors.append(t)
    rng = np.random.default_rng(11)

    print("the dynamic range of the Q8_0 weight scales inside one K chunk")
    print("(the fixed point mantissa has to cover the range, so the range costs bits)")
    print(f"{'tensor':28s} {'median':>8s} {'p99':>8s} {'max':>8s} "
          f"{'|P| med':>9s} {'|P| p999':>9s} {'|P| max':>9s}")
    stats = []
    for t in tensors:
        n = int(t.shape[1])
        k = int(t.shape[0])
        q, sc, nb = dequant_q8_0(t.data, n, k)
        nkc = k // a.block
        nblk = a.block // 32
        rngs = []
        for c in range(nkc):
            s = sc[:, c * nblk:(c + 1) * nblk]        # [n, nblk]
            smax = s.max(axis=1)
            smin = s.min(axis=1)
            smin = np.where(smin > 0, smin, smax)     # an all zero block
            rngs.extend((smax / np.maximum(smin, 1e-30)).tolist())
        # the magnitude of the int32 partial sums of one 32 element block
        act = rng.integers(-127, 128, size=(a.M, 32)).astype(np.int32)
        p = np.abs(act @ q[:, 0, :].astype(np.int32).T)
        stats.append((t, q, sc, n, k, nkc, nblk))
        print(f"{t.name:28s} {np.median(rngs):8.2f} {np.percentile(rngs, 99):8.2f} "
              f"{max(rngs):8.2f} {np.median(p):9.0f} {np.percentile(p, 99.9):9.0f} "
              f"{p.max():9.0f}")

    print()
    print("the error of the fixed point scheme (relative RMS of the GEMM output,")
    print("against the float64 reference of the exact Q8_0 weights)")
    print(f"{'tensor':28s} {'mant':>5s} {'S':>3s} {'error':>10s} {'int32 bits':>11s}")
    for t, q, sc, n, k, nkc, nblk in stats:
        act = rng.standard_normal((a.M, k)).astype(np.float32)
        act *= rng.uniform(0.5, 2.0, size=(a.M, 1)).astype(np.float32)
        # q is [n][block][32] and sc is [n][block]: the K index is 32*block+k,
        # which is the order of the reshape (no transpose)
        w_ref = (q.astype(np.float32) * sc[:, :, None]).reshape(n, k)
        ref = act.astype(np.float64) @ w_ref.T.astype(np.float64)
        # the activation quantized per 32 (the Q8_0 reference path)
        aq32, s32 = quantize_rows(act, 32)
        got32 = (aq32.astype(np.float64) * s32[:, :, None]).reshape(a.M, k).astype(np.float64) \
            @ w_ref.T.astype(np.float64)
        e32 = np.sqrt(np.mean((got32 - ref) ** 2)) / np.max(np.abs(ref))
        print(f"{t.name:28s} {'-':>5s} {'-':>3s} {e32:10.3e} {'(Q8_0 ref)':>11s}")
        for mant_bits, S in ((12, 0), (12, 4), (8, 4), (6, 4)):
            aq, sa = quantize_rows(act, a.block)      # the activation per chunk
            got = np.zeros_like(ref)
            for c in range(nkc):
                sl = slice(c * a.block, (c + 1) * a.block)
                s_blk = sc[:, c * nblk:(c + 1) * nblk].T        # [nblk, n]
                e = np.log2(np.maximum(s_blk.max(axis=0), 1e-30))
                # the mantissa is the scale relative to the chunk exponent;
                # the down-shift by S is compensated by S extra mantissa bits
                mant = np.rint(s_blk * np.exp2(mant_bits + S - e[None, :])).astype(np.int64)
                # the int32 partial sums of the chunk: p[m, b, n]
                aq_blk = aq[:, c, :].reshape(a.M, nblk, 32).astype(np.int64)
                # the weight blocks are stored [n][block][32], the einsum
                # wants [b][k][n]
                q_blk = q[:, c * nblk:(c + 1) * nblk, :].astype(np.int64).transpose(1, 2, 0)
                p = np.einsum("mbk,bkn->mbn", aq_blk, q_blk)
                # round to nearest, a truncating shift biases the result
                v = ((p + (1 << (S - 1))) >> S) if S > 0 else p
                v = v * mant[None, :, :]
                # the mantissa carries mant_bits fractional bits
                got += (v.sum(axis=1).astype(np.float64) * np.exp2(-mant_bits)
                        * sa[:, c][:, None] * np.exp2(e)[None, :])
            err = np.sqrt(np.mean((got - ref) ** 2)) / np.max(np.abs(ref))
            # the down-shift does not buy bits: the mantissa needs the same
            # bits back, so the accumulator needs mant_bits + 20 + log2(nblk)
            bits = mant_bits + 20 + int(np.log2(nblk))
            print(f"{t.name:28s} {mant_bits:5d} {S:3d} {err:10.3e} {bits:11d}")


if __name__ == "__main__":
    main()
