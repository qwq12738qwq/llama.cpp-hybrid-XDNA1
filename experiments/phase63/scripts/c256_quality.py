#!/usr/bin/env python3
"""Phase 63: the model level quality cost of the c256 weight requantization.

The phase 62 c256 engine (c4_mmul_v4) replaces the per-32-element Q8_0
weight scale with one scale per K chunk (256 elements).  The NPU is not
available, so the quality cost of that change is measured here at the model
level instead:

  * the FFN weights of the model are requantized to 256-element blocks,
  * the model is written out as F16 (the dequantized values are exactly the
    values the int8 GEMM of the c256 engine sees, so the GEMM of the two is
    the same up to the float rounding of the scale multiply),
  * the perplexity of the converted model is measured on the CPU and
    compared with the perplexity of the original Q8_0 model.

The result is a lower bound on the cost of the c256 engine: the converted
model does not quantize the activation, while the c256 engine quantizes it
per 256 elements as well.

usage: c256_quality.py [--model MODEL] [--out OUT] [--chunks 6]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gguf-py"))
import gguf  # noqa: E402


def dequant_q8_0(data, n_row, k):
    nb = k // 32
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_row, nb, 34)
    scales = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    qs = raw[:, :, 2:].view(np.int8).astype(np.float32)
    return qs, scales


def requantize_256(qs, scales, n_row, k, block=256):
    """The c256 requantization: one scale per block of 256 elements."""
    nb = k // 32
    nbc = k // block
    w = (qs * scales[:, :, None]).reshape(n_row, k)
    wb = w.reshape(n_row, nbc, block)
    amax = np.max(np.abs(wb), axis=2)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(wb / scale[:, :, None]), -127, 127).astype(np.int8)
    return (q.astype(np.float32) * scale[:, :, None]).reshape(n_row, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models",
                                                   "Qwen3.5-0.8B-Q8_0.gguf"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models",
                                                  "Qwen3.5-0.8B-c256-F16.gguf"))
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--skip-convert", action="store_true")
    a = ap.parse_args()

    if not a.skip_convert:
        t0 = time.time()
        reader = gguf.GGUFReader(a.model)
        arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
        writer = gguf.GGUFWriter(a.out, arch=arch)
        # the metadata, copied field by field
        for field in reader.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or \
                    field.name.startswith("GGUF."):
                continue
            val_type = field.types[0]
            sub_type = (field.types[-1]
                        if val_type == gguf.GGUFValueType.ARRAY else None)
            writer.add_key_value(field.name, field.contents(), val_type,
                                 sub_type=sub_type)
        n_conv = 0
        for t in reader.tensors:
            data = t.data
            name = t.name
            if t.tensor_type == 8 and ("ffn_gate" in name or "ffn_up" in name
                                       or "ffn_down" in name):
                n_row = int(t.shape[1])
                k = int(t.shape[0])
                qs, sc = dequant_q8_0(data, n_row, k)
                w = requantize_256(qs, sc, n_row, k)
                # a ggml tensor with ne = (K, N) is stored as a (N, K)
                # array with ne0 contiguous, and the gguf writer reverses
                # the shape, so the array is the (N, K) one
                writer.add_tensor(name,
                                  np.ascontiguousarray(w.astype(np.float16)),
                                  raw_dtype=gguf.GGMLQuantizationType.F16)
                n_conv += 1
            else:
                writer.add_tensor(name, data, raw_dtype=t.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(f"converted {n_conv} FFN weight tensors to the 256 element "
              f"quantization and wrote {a.out} in {time.time() - t0:.1f}s")

    # the perplexity of the converted model
    corpus = os.path.join(ROOT, "experiments", "phase61", "ppl", "wiki.test.raw")
    binp = os.path.join(ROOT, "build", "bin", "llama-perplexity")
    for tag, model in (("c256-F16", a.out), ("Q8_0", a.model)):
        cmd = [binp, "-m", model, "-f", corpus, "-c", "512",
               "--chunks", str(a.chunks), "-ngl", "0"]
        env = dict(os.environ)
        env["GGML_XDNA_DISABLE"] = "1"
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        out = p.stdout + p.stderr
        line = [l for l in out.splitlines() if "Final estimate" in l]
        print(f"{tag:10s} {line[0].strip() if line else out[-300:]} "
              f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
