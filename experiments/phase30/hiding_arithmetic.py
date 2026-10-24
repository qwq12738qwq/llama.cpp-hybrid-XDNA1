#!/usr/bin/env python3
# Phase 30 STEP 6-8 arithmetic: can an XDNA KV-tile preparation be hidden inside
# Vulkan's per-tile attention work?
#
# Inputs are all MEASURED:
#   - Vulkan FA per-ubatch time at each KV length : Phase 29 flash_ms_6ops / 6 ops
#   - XDNA per-op host-BO work time              : Phase 26 xdna_host_consume (~1.3-3.5 ms
#                                                  for a M512 K1024 N1792 i8 GEMM on a userptr BO)
#   - Vulkan/XDNA async overlap exists           : Phase 20G overlap_proof, 20/20 overlap>0
#
# Hiding is trivially possible if XDNA_prepare(tile) <= Vulkan_compute(tile).
# But hiding a preparation is only MEANINGFUL if Vulkan actually consumes it and the
# critical path shrinks. Step 4 established there is no such useful preparation:
#   - layout is already contiguous/coalesced (no transform needed)
#   - KV byte reduction is neutralized (Vulkan re-expands quant KV to f16 scratch)
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
geo = os.path.join(HERE, "results", "phase30_kv_access_geometry.csv")

# XDNA representative per-op processing time on a host-BO (Phase 26, measured iter1/2)
XDNA_OP_US = [1500.0, 1810.0, 3650.0]   # userptr host BO GEMM, warm
XDNA_OP_TYP = 1800.0

rows = []
with open(geo) as f:
    for r in csv.DictReader(f):
        kv = int(r["kv_len"])
        if kv not in (512, 4096, 16384, 32768, 65536):
            continue
        flash6 = float(r["flash_ms_6ops"])          # 6 layers, one ubatch
        per_layer_us = flash6 * 1000.0 / 6.0
        # Vulkan per-layer "tile" time = per_layer / Tr, Tr=32 query tiles
        per_tile_us = per_layer_us / 32.0
        # XDNA could prepare one KV tile concurrently; assume the whole per-layer KV
        # is split into Tr tiles to match (optimistic: XDNA handles 1/32 of a layer)
        hiding_ratio = XDNA_OP_TYP / per_tile_us if per_tile_us > 0 else 0.0
        rows.append({
            "kv_len": kv,
            "vk_per_layer_us": round(per_layer_us, 2),
            "vk_per_tile_us": round(per_tile_us, 2),
            "xdna_prep_us": round(XDNA_OP_TYP, 1),
            "xdna_prep_hidden": "yes" if XDNA_OP_TYP <= per_tile_us else "no",
            "prep_vs_tile_ratio": round(hiding_ratio, 3),
        })

dest = os.path.join(HERE, "results", "phase30_hiding_arithmetic.csv")
with open(dest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("wrote", dest)
for r in rows:
    print(r)
print()
print("overlap_positive: VERIFIED (Phase 20G overlap_proof, 20/20, best 350us)")
print("prep_hidden: mechanically yes from 16K up (Vulkan per-tile >> XDNA prep)")
print("useful_prep_exists: NO (step 4) -> hiding is vacuous -> FAILURE-D/Failure-A")
