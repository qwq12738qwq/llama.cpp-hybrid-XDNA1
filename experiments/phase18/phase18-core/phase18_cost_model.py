#!/usr/bin/env python3
"""Phase 18-K: Vulkan vs XDNA Cost Model - generates decision data for any MUL_MAT shape."""
import json, os

OUT = "/home/ubuntu12738/phase18"

# Load all capability data
cap_files = [
    os.path.join(OUT, f) for f in os.listdir(OUT) if f.startswith("phase18_cap_M") and f.endswith(".json")
]
all_cap = []
for cf in sorted(cap_files):
    with open(cf) as f:
        all_cap.extend(json.load(f))

# Also load K optimum data
kopt_file = os.path.join(OUT, "phase18_k_optimum.json")
if os.path.exists(kopt_file):
    with open(kopt_file) as f:
        kopt = json.load(f)
else:
    kopt = []

print(f"Loaded {len(all_cap)} capability data points")
print(f"Loaded {len(kopt)} K-optimum data points")

# Build cost model
cost_model = {
    "description": "XDNA vs Vulkan cost model for GGML MUL_MAT",
    "version": "1.0",
    "reference": "M=512, K=1024, N=3072, Q8_0 x F32 -> F32",
    "vulkan_reference_ms": 34.0,
    "xdna_kernel_gops": 50.0,  # K=1024 baseline
    "xdna_kernel_gops_k4096": 108.0,  # K=4096
    "findings": {
        "xdna_k1024_ratio": "~1.8x slower than Vulkan",
        "xdna_k2048_ratio": "~1.25x slower than Vulkan",
        "xdna_k4096_ratio": "~0.86x (XDNA 14% faster than Vulkan)",
        "xdna_gops_stable": "50 GOPS regardless of M/N (DMA-bound at K=1024)",
        "xdna_gops_k4096": "108 GOPS (compute-bound at K=4096)",
        "aie_utilization_k1024": "0.65% (massive room for improvement)",
        "bottleneck": "DMA bandwidth at K=1024, compute at K=4096",
    },
    "decision_rules": [
        {
            "condition": "K >= 4096",
            "action": "XDNA preferred",
            "reason": "XDNA is 10-15% faster than Vulkan at K=4096",
            "heterogeneous": "N-partition beneficial if both devices have work",
        },
        {
            "condition": "K == 1024",
            "action": "Vulkan preferred",
            "reason": "XDNA is 1.8x slower at K=1024 due to DMA bottleneck",
            "heterogeneous": "Not beneficial - XDNA slower on critical path",
        },
        {
            "condition": "K == 2048",
            "action": "Borderline",
            "reason": "XDNA is 1.2x slower, approaching parity",
            "heterogeneous": "May benefit from N-partition for load balancing",
        },
    ],
    "capability_data": all_cap,
    "n_partition_data": {
        "M": 512, "K": 1024, "N": 3072,
        "xdna_times_ms": {
            "N=3072": 61.5,
            "N=2304": 47.1,
            "N=1536": 31.9,
            "N=768": 16.1,
        },
        "conclusion": "XDNA scales linearly with N, but absolute time exceeds Vulkan for K=1024",
    },
    "k_optimum": kopt,
}

jp = os.path.join(OUT, "phase18_cost_model.json")
with open(jp, "w") as f:
    json.dump(cost_model, f, indent=2)
print(f"Cost model saved: {jp}")

# Print summary
print(f"\n{'='*70}")
print("PHASE 18 COST MODEL SUMMARY")
print(f"{'='*70}")
print(f"\nXDNA kernel performance:")
print(f"  K=1024: ~50 GOPS (DMA-bound, 1.8x slower than Vulkan)")
print(f"  K=2048: ~79 GOPS (1.25x slower than Vulkan)")
print(f"  K=4096: ~108 GOPS (0.86x, XDNA 14% faster than Vulkan)")
print(f"\nN-partition heterogeneous execution:")
print(f"  K=1024: NOT beneficial (XDNA slower on critical path)")
print(f"  K=4096: BENEFICIAL (XDNA faster, split N for parallelism)")
print(f"\nDecision: For Qwen3.5 FFN (K=1024), Vulkan-only is optimal.")
print(f"         For models with K>=4096 layers, XDNA offload is beneficial.")
print(f"\nRemaining optimization opportunity:")
print(f"  AIE utilization at K=1024 is only 0.65%")
print(f"  If kernel optimized to 200 GOPS (2.5% utilization),")
print(f"  XDNA would be 0.36x of Vulkan (64% faster)")
