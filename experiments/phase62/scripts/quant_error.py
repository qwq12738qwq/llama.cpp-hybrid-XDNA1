#!/usr/bin/env python3
"""Phase 62: the numerical cost of the per-256-block (c256) quantization.

The phase 62 kernel moves the Q8_0 scale application out of the 32-element
block loop, which means the weight block scale has to cover 256 elements and
the activation block scale has to cover the K chunk (256 elements) as well.

This script measures what that costs, on the real weights of the model:

  * W32 : the Q8_0 weights as they are stored (32-element blocks) - the
          reference the model currently computes with,
  * W256: the same weights requantized to 256-element blocks (dequantize,
          then quantize again with one scale per 256 elements),
  * A32 / A256: the activation quantized with 32- and 256-element blocks.

and reports the relative error of the GEMM output for each combination,
against the dequantized Q8_0 reference.

usage: quant_error.py [--layers 0 12] [--M 512]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "gguf-py"))
import gguf  # noqa: E402


def dequant_q8_0(data, n_row, k):
    """Q8_0 rows of k elements, 32-element blocks: [fp16 scale][32 int8]."""
    nb = k // 32
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_row, nb, 34)
    scales = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    qs = raw[:, :, 2:].view(np.int8).astype(np.float32)
    out = (qs * scales[:, :, None]).reshape(n_row, k)
    return out, scales, qs.astype(np.int8)


def requantize(w, block):
    """Requantize a float matrix to int8 with one scale per `block` elements
    along the K axis.  Returns the dequantized values and the scales."""
    n, k = w.shape
    nb = k // block
    wb = w.reshape(n, nb, block)
    amax = np.max(np.abs(wb), axis=2)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(wb / scale[:, :, None]), -127, 127).astype(np.int8)
    return (q.astype(np.float32) * scale[:, :, None]).reshape(n, k), scale


def rel_err(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    rms = np.sqrt(np.mean(d ** 2))
    return rms / max(np.max(np.abs(b)), 1e-30), float(d.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B-Q8_0.gguf")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 12, 23])
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--block", type=int, default=256)
    a = ap.parse_args()

    reader = gguf.GGUFReader(a.model)
    wanted = []
    for t in reader.tensors:
        if t.tensor_type != 8:      # Q8_0
            continue
        for kind in ("ffn_gate", "ffn_up", "ffn_down"):
            if kind in t.name and ".weight" in t.name:
                for L in a.layers:
                    if t.name.startswith(f"blk.{L}."):
                        wanted.append((t, kind, L))

    rng = np.random.default_rng(7)
    print(f"model {a.model}  M={a.M}  block={a.block}")
    print(f"{'tensor':28s} {'K':>6s} {'N':>6s} "
          f"{'W32 vs ref':>12s} {'W256 vs ref':>12s} "
          f"{'A32 GEMM':>11s} {'A256 GEMM':>11s} {'W256+A256':>11s}")
    for t, kind, L in wanted:
        n_row = int(t.shape[1])     # K
        k = int(t.shape[0])
        w_ref, _, _ = dequant_q8_0(t.data, n_row, k)
        w256, _ = requantize(w_ref, a.block)

        e_w32, _ = rel_err(w_ref, w_ref)      # 0 by construction
        e_w256, _ = rel_err(w256, w_ref)

        # a realistic activation: a zero-mean gaussian with a per-row scale
        act = rng.standard_normal((a.M, k)).astype(np.float32)
        act *= (rng.uniform(0.5, 2.0, size=(a.M, 1))).astype(np.float32)

        a32, _ = requantize(act, 32)
        a256, _ = requantize(act, a.block)

        c_ref = act.astype(np.float64) @ w_ref.T.astype(np.float64)
        c_a32 = a32.astype(np.float64) @ w_ref.T.astype(np.float64)
        c_a256 = a256.astype(np.float64) @ w_ref.T.astype(np.float64)
        c_w256 = act.astype(np.float64) @ w256.T.astype(np.float64)
        c_both = a256.astype(np.float64) @ w256.T.astype(np.float64)

        e_a32, _ = rel_err(c_a32, c_ref)
        e_a256, _ = rel_err(c_a256, c_ref)
        e_both, _ = rel_err(c_both, c_ref)

        print(f"{t.name:28s} {k:6d} {n_row:6d} "
              f"{e_w32:12.3e} {e_w256:12.3e} "
              f"{e_a32:11.3e} {e_a256:11.3e} {e_both:11.3e}")


if __name__ == "__main__":
    main()
