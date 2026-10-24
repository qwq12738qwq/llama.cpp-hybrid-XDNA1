#!/usr/bin/env python3
"""Phase 63: verify the data flow of the MemTile relay design offline.

The phase 63 design (experiments/phase63/gen_mtile.py) splits the output of a
column between its workers, relays the operands through the MemTile and
gathers the C pieces back through the MemTile.  The NPU is not available in
this session, so the data flow is verified here instead:

  * the host packing is transcribed (the activation group, the weight chunk
    and the C pieces),
  * the kernel's reading convention is applied to the packed buffers (the
    8x8 int8 tiles, the row scales and the column scales),
  * the result of every worker is gathered the way the MemTile gathers it,
  * and the result is compared with a float64 reference of the exact Q8_0
    mathematics.

usage: check_relay_flow.py [--M 64 --K 256 --N 128 --cols 2 --rows 4]
"""
import argparse
import numpy as np


def quantize_rows(x, block):
    """int8 with one scale per `block` elements, per row (the host side)."""
    m, k = x.shape
    nb = k // block
    xb = x.reshape(m, nb, block)
    amax = np.max(np.abs(xb), axis=2)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(xb / scale[:, :, None]), -127, 127).astype(np.int8)
    return q, scale


def pack_a_group(A_i8, As, k_chunk, rows, wm):
    """The activation group of one (m-tile, k-chunk): the concatenation of the
    per worker chunks, each of them the int8 tiles of its rows followed by
    its row scales."""
    ma = wm // 8
    nblk = k_chunk // 32
    group = np.zeros((rows, wm * k_chunk + wm * 4), np.int8)
    for r in range(rows):
        tiles = group[r, :wm * k_chunk].reshape(ma, nblk, 4, 64)
        for im in range(ma):
            for bb in range(nblk):
                for l in range(4):
                    r0 = r * wm + im * 8
                    c0 = bb * 32 + l * 8
                    tiles[im, bb, l] = A_i8[r0:r0 + 8, c0:c0 + 8].reshape(-1)
        sc = group[r, wm * k_chunk:].view(np.float32)
        sc[:] = As[r * wm:(r + 1) * wm]
    return group


def pack_b_chunk(B_i8, Bs, n_tile, k_chunk):
    """The weight chunk of one (n-tile, k-chunk): the int8 tiles followed by
    the column scales."""
    na = n_tile // 8
    nblk = k_chunk // 32
    chunk = np.zeros(k_chunk * n_tile + n_tile * 4, np.int8)
    tiles = chunk[:k_chunk * n_tile].reshape(na, nblk, 4, 64)
    for jn in range(na):
        for bb in range(nblk):
            for l in range(4):
                r0 = bb * 32 + l * 8
                c0 = jn * 8
                tiles[jn, bb, l] = B_i8[r0:r0 + 8, c0:c0 + 8].reshape(-1)
    sc = chunk[k_chunk * n_tile:].view(np.float32)
    # the per chunk column scale
    sc[:] = Bs
    return chunk


def kernel_worker(a_chunk, b_chunk, wm, wn, k_chunk, n_tile):
    """The kernel's reading convention: the int8 tiles of the 8x8 matrix unit
    and the per chunk scales, applied to the packed buffers."""
    ma, na = wm // 8, wn // 8
    nblk = k_chunk // 32
    at = a_chunk[:wm * k_chunk].reshape(ma, nblk, 4, 8, 8)
    asc = a_chunk[wm * k_chunk:].view(np.float32)
    bt = b_chunk[:k_chunk * n_tile].reshape(na, nblk, 4, 8, 8)
    bsc = b_chunk[k_chunk * n_tile:].view(np.float32)

    out = np.zeros((wm, wn), np.float64)
    for im in range(ma):
        for jn in range(na):
            p = np.zeros((8, 8), np.int64)
            for bb in range(nblk):
                for l in range(4):
                    p += at[im, bb, l].astype(np.int64) @ bt[jn, bb, l].astype(np.int64)
            for r in range(8):
                for c in range(8):
                    out[im * 8 + r, jn * 8 + c] += (float(p[r, c])
                                                     * float(asc[im * 8 + r])
                                                     * float(bsc[jn * 8 + c]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--m-tile", type=int, default=64)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    a = ap.parse_args()

    M, K, N = a.M, a.K, a.N
    rows = a.rows
    wm, wn = a.m_tile // rows, a.n_tile
    nkc = K // a.k_chunk
    rng = np.random.default_rng(17)

    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    # the activation is quantized per K chunk, the weights per 32 element block
    aq, asc = quantize_rows(rng.standard_normal((M, K)).astype(np.float32), a.k_chunk)
    # the c256 kernel applies one scale per K chunk and column: the weights
    # are requantized to that granularity (the phase 62 c256 engine does the
    # same on the host)
    Bs = (rng.uniform(0.002, 0.05, size=(K // a.k_chunk, N)) *
          np.power(2.0, rng.integers(-3, 4, size=(K // a.k_chunk, N)))).astype(np.float32)

    # the reference: the exact Q8_0 mathematics with the per chunk activation
    # the weight scale of the chunk, expanded to the elements
    w_ref = B_i8.astype(np.float32) * np.repeat(Bs, a.k_chunk, axis=0)
    act = (aq.astype(np.float32) * asc[:, :, None]).reshape(M, K)
    ref = act.astype(np.float64) @ w_ref.astype(np.float64)

    # the packed buffers and the per worker result
    got = np.zeros((M, N), np.float64)
    n_mt = M // a.m_tile
    n_nt = N // a.n_tile
    n_nt_col = n_nt // a.cols
    for c in range(a.cols):
        for mt in range(n_mt):
            for nt in range(n_nt_col):
                nt_global = c * n_nt_col + nt
                for kc in range(nkc):
                    sl = slice(kc * a.k_chunk, (kc + 1) * a.k_chunk)
                    agroup = pack_a_group(
                        aq[mt * a.m_tile:(mt + 1) * a.m_tile, kc, :],
                        asc[mt * a.m_tile:(mt + 1) * a.m_tile, kc],
                        a.k_chunk, rows, wm)
                    bchunk = pack_b_chunk(
                        B_i8[sl, nt_global * a.n_tile:(nt_global + 1) * a.n_tile],
                        Bs[kc, nt_global * a.n_tile:(nt_global + 1) * a.n_tile],
                        a.n_tile, a.k_chunk)
                    for r in range(rows):
                        cpart = kernel_worker(agroup[r], bchunk, wm, wn, a.k_chunk, a.n_tile)
                        # the MemTile gathers the pieces at their row offsets
                        got[mt * a.m_tile + r * wm:mt * a.m_tile + (r + 1) * wm,
                            nt_global * a.n_tile:(nt_global + 1) * a.n_tile] += cpart

    err = np.abs(got - ref)
    rms = float(np.sqrt(np.mean(err ** 2)))
    refmax = float(np.max(np.abs(ref)))
    print(f"the relay data flow: M={M} K={K} N={N} cols={a.cols} rows={rows} "
          f"(a worker computes {wm}x{wn} outputs)")
    print(f"  max abs error {err.max():.3e}  relative RMS {rms / refmax:.3e} "
          f"(the float32 rounding level is ~1e-7)")
    ok = rms / refmax < 1e-6
    print("  VERDICT:", "the packed layout and the kernel convention agree"
          if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
