import glob, json

raw_files = glob.glob("/tmp/p34_sweep/t*/raw/bench_*.json")
print("found", len(raw_files), "files")
rows = []
for path in sorted(raw_files):
    with open(path) as f:
        r = json.load(f)
    rows.append(r)
    print("loaded", path, "tiles=", r.get("tiles"), "npu_gops=", r.get("npu_gops"))

rows.sort(key=lambda r: r["tiles"])
baseline = rows[0]
base_gops = baseline["npu_gops"]

print("")
print("=== Scaling Analysis: 512x2048x2048 ===")
print("Tiles  Rows  Cols    NPU_GOPS   Scaling  Efficiency  Correct")
print("-" * 62)
for r in rows:
    t = r["tiles"]
    g = r["npu_gops"]
    scaling = g / base_gops if base_gops > 0 else 0
    efficiency = scaling / t if t > 0 else 0
    correct = "PASS" if r["correct"] else "FAIL"
    print(f"{t:>5} {r['rows']:>5} {r['cols']:>5} {g:>10.2f} {scaling:>10.2f} {efficiency:>10.2f} {correct:>8}")

best = max(rows, key=lambda r: r["npu_gops"])
best_tiles = best["tiles"]
best_scaling = best["npu_gops"] / base_gops
best_efficiency = best_scaling / best_tiles
print("")
print(f"Baseline (T=1): {base_gops:.2f} GOPS")
print(f"Best (T={best_tiles}): {best['npu_gops']:.2f} GOPS, {best_scaling:.2f}x scaling, {best_efficiency:.2f} efficiency")

if best_tiles >= 8 and best_scaling >= 5:
    verdict = "WHOLE_ARRAY_SCALING_PASS"
elif best_tiles >= 4 and best_scaling >= 2.5:
    verdict = "WHOLE_ARRAY_SCALING_PARTIAL"
elif best_scaling < 1.5:
    verdict = "XDNA_ARRAY_SCALING_FAIL"
else:
    verdict = "XDNA_ARRAY_SCALING_PARTIAL"
print(f"Verdict: {verdict}")
