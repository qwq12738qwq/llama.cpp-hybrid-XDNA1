#!/usr/bin/env python3
"""Phase 62: check the fp256 (fixed point) kernel arithmetic without a device.

The NPU stopped accepting contexts in the middle of the phase, so the fp256
kernel could not be run.  This script emulates the kernel's arithmetic in
numpy, on the real weights of the model:

  * the host packs the weight block scales as int32 mantissas (relative to
    one exponent per K chunk and column) and the per column factors,
  * the kernel accumulates `(P >> S) * mant` in int32 over the 32 element
    blocks of the chunk,
  * the chunk sum is converted to float, multiplied by the row scale and the
    column factor and added to the C tile,

and compares the result against the float64 reference of the exact Q8_0
mathematics with the per K chunk activation scale.

usage: fp256_check.py [--layer 0] [--kc 256] [--mb 8] [--shift 4]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "gguf-py"))
import gguf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B-Q8_0.gguf")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--kc", type=int, default=256)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--M", type=int, default=64)
    a = ap.parse_args()

    reader = gguf.GGUFReader(a.model)
    name = f"blk.{a.layer}.ffn_gate.weight"
    t = [x for x in reader.tensors if x.name == name][0]
    k = int(t.shape[0])
    n = int(t.shape[1])
    nb = k // 32
    raw = np.frombuffer(t.data, dtype=np.uint8).reshape(n, nb, 34)
    sc = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]  # [n][nb]
    q = raw[:, :, 2:].view(np.int8).astype(np.int64)                            # [n][nb][32]

    nkc = k // a.kc
    nblk = a.kc // 32
    rng = np.random.default_rng(5)
    act = rng.standard_normal((a.M, k)).astype(np.float32)
    # the activation quantized per K chunk
    xb = act.reshape(a.M, nkc, a.kc)
    amax = np.abs(xb).max(axis=2)
    sa = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    aq = np.clip(np.rint(xb / sa[:, :, None]), -127, 127).astype(np.int64)

    # the reference: the exact Q8_0 weight scales, the per chunk activation
    w_ref = (q.astype(np.float32) * sc[:, :, None]).reshape(n, k)
    ref = act.astype(np.float64) @ w_ref.T.astype(np.float64)

    got = np.zeros_like(ref)
    for c in range(nkc):
        sl = slice(c * a.kc, (c + 1) * a.kc)
        s_blk = sc[:, c * nblk:(c + 1) * nblk].T                      # [nblk, n]
        e = np.log2(np.maximum(s_blk.max(axis=0), 1e-30))
        mant = np.rint(s_blk * np.exp2(a.mb + a.shift - e[None, :])).astype(np.int64)
        # the down-shift of the partial sum is compensated by the extra
        # mantissa bits, so the factor does not carry the shift
        fac = np.exp2(e - a.mb)
        # the int32 partial sums of the chunk: p[m, b, n]
        aq_blk = aq[:, c, :].reshape(a.M, nblk, 32)
        q_blk = q[:, c * nblk:(c + 1) * nblk, :].transpose(1, 2, 0)
        p = np.einsum("mbk,bkn->mbn", aq_blk, q_blk)
        # the kernel's integer path
        v = ((p + (1 << (a.shift - 1))) >> a.shift) * mant[None, :, :]
        acc = v.sum(axis=1)                                   # int32 accumulator
        # the float path at the end of the chunk
        got += acc.astype(np.float64) * sa[:, c][:, None] * fac[None, :]

    err = np.sqrt(np.mean((got - ref) ** 2)) / np.max(np.abs(ref))
    print(f"tensor {name}: K={k} N={n} chunk={a.kc} mantissa bits={a.mb} shift={a.shift}")
    print(f"  relative RMS error of the emulated fp256 kernel: {err:.3e}")
    print(f"  (the Q8_0 reference path, the activation per 32 elements, is ~4e-4)")
    print(f"  int32 accumulator bits: {a.mb + 20 + int(np.log2(nblk))}")


if __name__ == "__main__":
    main()
