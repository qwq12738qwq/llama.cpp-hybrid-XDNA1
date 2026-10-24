#!/usr/bin/env python3
"""Patch the FFN weights of a Q8_0 GGUF to bf16 (or f16) rounded values,
stored as F32, to measure the model-level quality of an NPU float matrix path.

The NPU bf16 path computes bf16(dequant(Q8_0_w)) x bf16(act).  Storing the
rounded weights as F32 makes any backend (CPU here) compute with exactly the
weight-side bf16 error and no extra quantization noise.

The metadata (kv) section is copied verbatim; only the tensor-info section
and the data blob are rewritten.

usage: patch_gguf_bf16.py <in.gguf> <out.gguf> <bf16|f16>
"""
import struct
import sys

import numpy as np
from ml_dtypes import bfloat16

# GGUF value types
GU, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCF = {GU: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
       F32: "<f", BOOL: "<B", U64: "<Q", I64: "<q", F64: "<d"}

# ggml tensor types
F32_T, F16_T, Q8_0_T = 0, 1, 8


def rd_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)


def skip_value(f, vt):
    if vt == STR:
        rd_string(f)
    elif vt == ARR:
        et = struct.unpack("<I", f.read(4))[0]
        cnt = struct.unpack("<Q", f.read(8))[0]
        for _ in range(cnt):
            skip_value(f, et)
    else:
        f.read(struct.calcsize(SCF[vt]))


def main():
    inp, outp, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    f = open(inp, "rb")
    assert f.read(4) == b"GGUF"
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    kv_start = f.tell()
    for _ in range(n_kv):
        rd_string(f)
        vt = struct.unpack("<I", f.read(4))[0]
        skip_value(f, vt)
    kv_end = f.tell()
    f.seek(kv_start)
    kv_bytes = f.read(kv_end - kv_start)

    infos = []
    for _ in range(n_tensors):
        name = rd_string(f).decode()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos.append((name, dims, ttype, off))
    align = 32
    data_start = (f.tell() + align - 1) // align * align
    f.seek(data_start)
    blob = f.read()
    f.close()

    round_fn = (lambda x: x.astype(bfloat16).astype(np.float32)) if mode in ("bf16", "f16store") else (lambda x: x.astype(np.float16).astype(np.float32) if mode == "f16" else x.copy())
    store_type = F16_T if mode == "f16store" else F32_T

    def tsize(ttype, dims):
        ne = int(np.prod(dims))
        if ttype == F32_T:
            return ne * 4
        if ttype == F16_T:
            return ne * 2
        if ttype == Q8_0_T:
            return ne // 32 * 34
        raise SystemExit(f"unhandled ggml type {ttype} in {name}")

    patched = {}
    report = []
    for name, dims, ttype, off in infos:
        if ttype == Q8_0_T and any(s in name for s in (".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight")) and (not __import__("os").environ.get("P61_ONLY","") or name.split(".")[1] in __import__("os").environ["P61_ONLY"].split(",")):
            ne = int(np.prod(dims))
            nb = ne // 32
            raw = np.frombuffer(blob, np.uint8, nb * 34, off).reshape(nb, 34)
            sc = raw[:, :2].view(np.float16).astype(np.float32)
            q = raw[:, 2:].view(np.int8).astype(np.float32)
            deq = (q * sc).ravel()
            rounded = round_fn(deq)
            err = rounded - deq
            rel = float(np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(deq**2)))
            report.append((name, ne, rel))
            patched[name] = (rounded.astype(np.float16) if mode == "f16store" else rounded.astype(np.float32)).tobytes()

    body = []
    run_off = 0
    new_infos = []
    for name, dims, ttype, off in infos:
        if name in patched:
            data = patched[name]
            new_infos.append((name, dims, store_type, run_off))
            body.append(data)
            run_off += len(data)
        else:
            new_infos.append((name, dims, ttype, run_off))
            body.append(blob[off:off + tsize(ttype, dims)])
            run_off += tsize(ttype, dims)
        run_off = (run_off + align - 1) // align * align

    with open(outp, "wb") as g:
        g.write(b"GGUF" + struct.pack("<I", ver) + struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv))
        g.write(kv_bytes)
        for name, dims, ttype, off in new_infos:
            nb = name.encode()
            g.write(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)))
            for d in dims:
                g.write(struct.pack("<Q", d))
            g.write(struct.pack("<I", ttype) + struct.pack("<Q", off))
        pad = (align - (g.tell() % align)) % align
        g.write(b"\x00" * pad)
        for i, b in enumerate(body):
            g.write(b)
            if i != len(body) - 1:
                pad = (align - (g.tell() % align)) % align
                g.write(b"\x00" * pad)
    print(f"wrote {outp}")
    for name, ne, rel in report[:3] + report[-2:]:
        print(f"  {name:32s} {ne:>10} rel_rms {rel:.3e}")
    print(f"  {len(report)} tensors patched; total rel-rms "
          f"{np.sqrt(np.mean([r*8*1 for _,_,r in report])):.3e} (per-tensor mean of rms, not summed)")


if __name__ == "__main__":
    main()
