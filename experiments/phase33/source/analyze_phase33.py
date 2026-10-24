#!/usr/bin/env python3
"""Phase 33 analysis: identify XDNA sweet spot, answer Q1-Q3.

Reads experiments/phase33/results/*.csv and produces:
  - phase33_verdict.txt: Q1, Q2, Q3 answers
  - phase33_summary.csv: all results combined
  - phase33_analysis.md: detailed analysis
"""
import csv, os, sys, json

PHASE33 = "/workspace/llama.cpp/experiments/phase33"
RESULTS = f"{PHASE33}/results"

def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def main():
    all_rows = []
    for fname in os.listdir(RESULTS):
        if fname.endswith(".csv"):
            rows = read_csv(os.path.join(RESULTS, fname))
            all_rows.extend(rows)

    # Q1: Find shapes where compute efficiency is high
    print("=== Q1: XDNA High-Utilization Regime ===\n")

    # Filter valid rows
    valid = [r for r in all_rows if r.get("status") == "ok" and float(r.get("med_ms", 0)) > 0]

    # Compute efficiency = eff_gops / theoretical_peak
    # Phoenix XDNA1 single tile mmul<4,8,8> at ~1GHz = ~256 GOPS per tile
    # C1 kernels use 1 tile; C4 use 4 tiles
    THEORETICAL_PEAK_GOPS_PER_TILE = 256.0

    for r in sorted(valid, key=lambda x: float(x.get("eff_gops", 0)), reverse=True)[:10]:
        eff = float(r.get("eff_gops", 0))
        util = eff / THEORETICAL_PEAK_GOPS_PER_TILE * 100
        print(f"  {r['mode']:8s} {r['tag']:20s} M={r['M']:>5} K={r['K']:>5} N={r['N']:>5} "
              f"med={float(r['med_ms']):7.2f}ms compute={float(r['compute_ms']):6.2f}ms "
              f"eff={eff:7.1f} GOPS util={util:4.1f}%")

    # Q2: Decompose overhead
    print("\n=== Q2: Overhead Decomposition ===\n")

    if valid:
        avg_submit = sum(float(r.get("submit_ms", 0)) for r in valid) / len(valid)
        avg_dma_in = sum(float(r.get("dma_in_ms", 0)) for r in valid) / len(valid)
        avg_compute = sum(float(r.get("compute_ms", 0)) for r in valid) / len(valid)
        avg_dma_out = sum(float(r.get("dma_out_ms", 0)) for r in valid) / len(valid)
        avg_total = sum(float(r.get("total_ms", 0)) for r in valid) / len(valid)

        print(f"  Average over {len(valid)} measurements:")
        print(f"    submit_setup:  {avg_submit:.3f} ms ({avg_submit/avg_total*100:.1f}%)")
        print(f"    dma_in:        {avg_dma_in:.3f} ms ({avg_dma_in/avg_total*100:.1f}%)")
        print(f"    compute:       {avg_compute:.3f} ms ({avg_compute/avg_total*100:.1f}%)")
        print(f"    dma_out:       {avg_dma_out:.3f} ms ({avg_dma_out/avg_total*100:.1f}%)")
        print(f"    total:         {avg_total:.3f} ms")

    # Q3: Verdict
    print("\n=== Q3: Verdict ===\n")

    max_eff = max((float(r.get("eff_gops", 0)) for r in valid), default=0)
    max_util = max_eff / THEORETICAL_PEAK_GOPS_PER_TILE * 100

    if max_util > 15.0:
        verdict = "XDNA_SWEET_SPOT_FOUND"
        rationale = (f"Found shapes achieving {max_util:.1f}% compute utilization. "
                     "XDNA has a valid high-throughput regime for large resident workloads.")
    elif max_util > 5.0:
        verdict = "XDNA_WORKLOAD_MISMATCH"
        rationale = (f"Max compute utilization {max_util:.1f}% is below practical threshold. "
                     "Current XRT/MLIR-AIE stack cannot amortize fixed overhead for llama.cpp shapes. "
                     "XDNA needs larger/batched resident workloads.")
    else:
        verdict = "XDNA_CAPABILITY_STOP"
        rationale = (f"Max compute utilization {max_util:.1f}% is too low even for large workloads. "
                     "Investigate XRT/MLIR-AIE kernel mapping, tile utilization, or driver overhead.")

    print(f"  VERDICT: {verdict}")
    print(f"  Max effective utilization: {max_util:.1f}% of {THEORETICAL_PEAK_GOPS_PER_TILE:.0f} GOPS peak")
    print(f"  Rationale: {rationale}")

    with open(f"{RESULTS}/phase33_verdict.txt", "w") as f:
        f.write(f"VERDICT: {verdict}\n")
        f.write(f"MAX_EFF_UTIL: {max_util:.2f}%\n")
        f.write(f"RATIONALE: {rationale}\n")

    # Write combined CSV
    if valid:
        fieldnames = list(valid[0].keys())
        with open(f"{RESULTS}/phase33_all.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in valid:
                writer.writerow(r)

    return 0

if __name__ == "__main__":
    sys.exit(main())
