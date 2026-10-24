#!/usr/bin/env python3
"""Phase 21: extract real Qwen3.5-0.8B GEMM shapes from the gguf tensor inventory.

For each 2D weight tensor (ggml mul_mat operand, ne0=K, ne1=N) prints
name, K, N, type and byte size, then a de-duplicated shape table with counts.
Usage:
  PYTHONPATH=gguf-py python3 extract_qwen_shapes.py <model.gguf> [out.csv]
"""
import sys
import numpy as np
from collections import Counter
from gguf import GGUFReader


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.5-0.8B-Q8_0.gguf"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    r = GGUFReader(path)

    print(f"[meta] tensors={len(r.tensors)} kv={len(r.fields)}")
    for k in ("general.architecture", "qwen3.5.block_count", "qwen3.5.embedding_length",
              "qwen3.5.feed_forward_length", "qwen3.5.attention.head_count",
              "qwen3.5.attention.head_count_kv"):
        if k in r.fields:
            f = r.fields[k]
            try:
                print(f"  {k} = {f.contents()}")
            except Exception:
                print(f"  {k} = {f.parts[f.data[0]][0]}")

    rows = []
    for t in r.tensors:
        shp = tuple(int(x) for x in t.shape)
        if len(shp) < 2:
            continue
        tname = str(t.tensor_type)
        # gguf stores ne0 first (contiguous dim) -> K = shp[0], N = shp[1]
        K, N = shp[0], shp[1]
        nbytes = int(t.data.nbytes) if hasattr(t, "data") else 0
        rows.append((t.name, K, N, tname, nbytes))

    print(f"[shapes] {len(rows)} 2D tensors")
    cnt = Counter()
    for name, K, N, ty, nb in rows:
        if "token_embd" in name or "output" in name:
            continue
        cnt[(K, N, ty)] += 1

    lines = ["K,N,type,count"]
    for (K, N, ty), c in sorted(cnt.items(), key=lambda x: -x[1]):
        lines.append(f"{K},{N},{ty},{c}")
        print(f"  K={K:5d} N={N:5d} {ty:10s} x{c}")

    # per-role sample (first occurrence of each distinct weight name suffix)
    print("[sample roles]")
    seen = set()
    for name, K, N, ty, nb in rows:
        role = name.split(".", 2)[-1] if name.count(".") >= 2 else name
        if role in seen:
            continue
        seen.add(role)
        print(f"  {name:55s} K={K:5d} N={N:5d} {ty:8s} {nb} B")

    if out:
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
