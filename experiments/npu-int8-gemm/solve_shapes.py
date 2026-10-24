#!/usr/bin/env python3
"""Enumerate single-submit Full-K MemTile-staged GEMM shapes and rank them.

Constraints mirror gen_fullk_memtile.build(). A candidate is only reported if
every constraint passes; ranking follows the Phase 20F priority order.
"""
import argparse
import itertools
import json

L1_MAX = 64 * 1024
L2_MAX = 128 * 1024
BD_MAX = 16383 * 4          # max length of one buffer descriptor
MEMTILE_BD_MAX = 16         # simultaneously active BDs per tile (aiecc)
SHIM_STREAM_MAX = 16        # 3 * n_cols must fit
STACK_BYTES = 0x2000


def evaluate(M, K, N, m_tile, n_tile, k_chunk, n_rows, n_cols):
    n_tiles = n_rows * n_cols
    if K % k_chunk or k_chunk % 32:
        return None
    if M % (m_tile * n_tiles) or N % n_tile:
        return None
    if M // m_tile != n_tiles:
        return None
    n_nt = N // n_tile
    n_kc = K // k_chunk
    nblk = k_chunk // 32
    vec_len = 64 if n_tile % 64 == 0 else 32
    if n_tile % vec_len:
        return None

    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4
    c_bytes = m_tile * n_tile * 4
    if a_bytes > BD_MAX or b_bytes > BD_MAX:
        return None

    l1 = STACK_BYTES + a_bytes + b_bytes + c_bytes
    if l1 > L1_MAX:
        return None
    l2 = n_rows * (a_bytes + c_bytes) + b_bytes
    if l2 > L2_MAX:
        return None
    n_shim = 3 * n_cols
    if n_shim > SHIM_STREAM_MAX:
        return None

    # MemTile BD demand measured in Phase 20E: n_nt * (2*n_kc + 1) per tile.
    memtile_bd = n_nt * (2 * n_kc + 1)
    if memtile_bd > MEMTILE_BD_MAX:
        return None
    # Compute-tile BD: a+b per kc plus one C per nt, same accounting.
    compute_bd = n_nt + n_nt * n_kc * 2
    if compute_bd > MEMTILE_BD_MAX:
        return None

    return {
        "M": M, "K": K, "N": N, "m_tile": m_tile, "n_tile": n_tile,
        "k_chunk": k_chunk, "n_rows": n_rows, "n_cols": n_cols,
        "tiles": n_tiles, "n_nt": n_nt, "n_kc": n_kc,
        "a_bytes": a_bytes, "b_bytes": b_bytes, "c_bytes": c_bytes,
        "l1": l1, "l2": l2, "memtile_bd": memtile_bd, "compute_bd": compute_bd,
        "vec_len": vec_len,
    }


def divisors_multiples(lo, hi, step):
    out = []
    for v in range(lo, hi + 1, step):
        out.append(v)
    return out


def rank_key(c):
    """Phase 20F priority: 16 tiles, N descending, K descending, single submit."""
    p1 = c["tiles"] == 16 and c["N"] >= 2048 and c["K"] >= 1024
    p2 = c["tiles"] == 16 and c["N"] >= 2048 and c["K"] < 1024
    tier = 0 if p1 else (1 if p2 else 2)
    return (tier, -c["N"], -c["K"], -c["tiles"], c["l1"], c["memtile_bd"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", default="32,64,128,256,512,1024,2048,3072,4096")
    ap.add_argument("--N", default="1024,1280,1536,1792,2048,2304,2560,3072,3584,4096,6144")
    ap.add_argument("--m-tiles", default="16,32,64,128")
    ap.add_argument("--n-tiles", default="32,64,128,192,256,384,416,448,512")
    ap.add_argument("--rows", default="4")
    ap.add_argument("--cols", default="4")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    Ks = [int(x) for x in a.K.split(",")]
    Ns = [int(x) for x in a.N.split(",")]
    mts = [int(x) for x in a.m_tiles.split(",")]
    nts = [int(x) for x in a.n_tiles.split(",")]
    rows = [int(x) for x in a.rows.split(",")]
    cols = [int(x) for x in a.cols.split(",")]

    found = []
    for n_rows, n_cols in itertools.product(rows, cols):
        n_tiles = n_rows * n_cols
        for m_tile in mts:
            M = m_tile * n_tiles
            for K in Ks:
                kc_candidates = [k for k in divisors_multiples(32, K, 32) if K % k == 0]
                for N in Ns:
                    for n_tile in nts:
                        if N % n_tile:
                            continue
                        for k_chunk in kc_candidates:
                            r = evaluate(M, K, N, m_tile, n_tile, k_chunk, n_rows, n_cols)
                            if r:
                                found.append(r)

    # keep one best record per (M,K,N,n_tile) - prefer k_chunk == K (n_kc == 1)
    best = {}
    for r in found:
        key = (r["M"], r["K"], r["N"], r["n_tile"], r["tiles"])
        cur = best.get(key)
        if cur is None or r["n_kc"] < cur["n_kc"] or (
            r["n_kc"] == cur["n_kc"] and r["memtile_bd"] < cur["memtile_bd"]
        ):
            best[key] = r

    ranked = sorted(best.values(), key=rank_key)
    print(f"total feasible records: {len(found)}, unique shapes: {len(ranked)}\n")
    print("=== N >= 2048 single-submit ===")
    big = [r for r in ranked if r["N"] >= 2048]
    for r in big[: a.top]:
        print(
            f'M={r["M"]:5d} K={r["K"]:5d} N={r["N"]:5d} m_tile={r["m_tile"]:3d} '
            f'n_tile={r["n_tile"]:4d} k_chunk={r["k_chunk"]:5d} n_nt={r["n_nt"]:2d} '
            f'n_kc={r["n_kc"]} L1={r["l1"]:6d} L2={r["l2"]:6d} BD={r["memtile_bd"]:2d}'
        )
    if not big:
        print("  (none)")

    print("\n=== N >= 1024, 16 tiles, ranked ===")
    mid = [r for r in ranked if r["N"] >= 1024 and r["tiles"] == 16]
    for r in mid[: a.top]:
        print(
            f'M={r["M"]:5d} K={r["K"]:5d} N={r["N"]:5d} m_tile={r["m_tile"]:3d} '
            f'n_tile={r["n_tile"]:4d} k_chunk={r["k_chunk"]:5d} n_nt={r["n_nt"]:2d} '
            f'n_kc={r["n_kc"]} L1={r["l1"]:6d} L2={r["l2"]:6d} BD={r["memtile_bd"]:2d}'
        )

    # per-K reachable N ceiling
    print("\n=== max N per (K, tiles=16) ===")
    for K in sorted({r["K"] for r in ranked if r["tiles"] == 16}):
        sub = [r for r in ranked if r["K"] == K and r["tiles"] == 16]
        if sub:
            top = max(sub, key=lambda r: r["N"])
            print(f'  K={K:5d}: N_max={top["N"]:5d} (n_tile={top["n_tile"]}, n_nt={top["n_nt"]}, k_chunk={top["k_chunk"]})')

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(ranked, f, indent=1)
        print(f"\nwrote {len(ranked)} records to {a.json_out}")


if __name__ == "__main__":
    main()
