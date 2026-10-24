#!/usr/bin/env python3
"""Phase 21: weight-residency analysis from the whole-array roofline.

Fit NPU GEMM time vs M at fixed K,N:  ms = a*M + b.
  a = per-row (activation) cost, b = fixed cost (weight B DMA + dispatch).
If B (K*N bytes) were re-streamed every submit, b and the small-M cost would
be dominated by the weight transfer; a small b means B is amortised (resident
across M-tiles / depth-2 L2 reuse).

Writes results/residency.csv and prints the fit.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOF = os.path.join(HERE, "results", "roofline.csv")
OUT = os.path.join(HERE, "results", "residency.csv")


def main():
    pts = []
    with open(ROOF) as f:
        for r in csv.DictReader(f):
            if r["in_b"] == "1" and r["cols"] == "4" and r["K"] == "4096" and r["N"] == "4096":
                pts.append((int(r["M"]), float(r["ms_med"])))
    pts.sort()
    n = len(pts)
    sx = sum(m for m, _ in pts)
    sy = sum(t for _, t in pts)
    sxx = sum(m * m for m, _ in pts)
    sxy = sum(m * t for m, t in pts)
    a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - a * sx) / n

    K = N = 4096
    print(f"[fit] ms = {a:.6f} * M + {b:.3f}   (K=N=4096, i8, 4 cols)")
    print(f"[fit] weight B = K*N = {K*N/1e6:.1f} MB; per-row A = K = {K/1e3:.1f} KB")
    print(f"[fit] fixed cost b = {b:.3f} ms  (dispatch + residual weight/stream)")
    rows = [("M", "ms_med", "pred_ms", "resid_ms")]
    for m, t in pts:
        pred = a * m + b
        rows.append((m, f"{t:.4f}", f"{pred:.4f}", f"{t - pred:+.4f}"))
        print(f"  M={m:5d} ms={t:8.4f} pred={pred:8.4f} resid={t - pred:+.4f}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a_ms_per_row", "b_fixed_ms", "K", "N", "cols", "dtype"])
        w.writerow([f"{a:.6f}", f"{b:.4f}", K, N, 4, "i8"])
        w.writerows(rows)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
