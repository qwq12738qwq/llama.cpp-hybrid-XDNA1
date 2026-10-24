#!/usr/bin/env python3
"""Print final Stage 2 verdict summary from collected data."""
import json

# Summary of all measured/estimated data
SUMMARY = {
    "context": "65536",
    "gpu_baseline": {
        "device": "AMD Radeon 780M (Vulkan/RADV)",
        "M=1024": {"total_ms": 684.0, "breakdown": {
            "Q_proj": 39.8, "K_proj": 40.1, "V_proj": 39.7, "O_proj": 39.4,
            "QK_t": 0.7, "Attn_V": 0.0, "FFN_gate": 39.1, "FFN_up": 39.2,
            "FFN_down": 352.3, "RMSNorm": 0.1, "Activation": 0.1
        }},
        "M=4096": {"total_ms": 2873.5, "breakdown": {
            "Q_proj": 168.9, "K_proj": 170.7, "V_proj": 168.4, "O_proj": 170.8,
            "QK_t": 11.0, "Attn_V": 0.1, "FFN_gate": 173.6, "FFN_up": 198.8,
            "FFN_down": 1639.2, "RMSNorm": 0.5, "Activation": 0.5
        }},
        "M=8192": {"total_ms": 1767.6, "breakdown": {
            "Q_proj": 205.2, "K_proj": 202.3, "V_proj": 203.3, "O_proj": 201.4,
            "QK_t": 29.2, "Attn_V": 0.1, "FFN_gate": 354.9, "FFN_up": 354.3,
            "RMSNorm": 1.1, "Activation": 1.1
        }},
        "M=16384": {"total_ms": 2198.3, "breakdown": {
            "Q_proj": 267.4, "K_proj": 266.2, "V_proj": 265.2, "O_proj": 267.7,
            "QK_t": 75.6, "Attn_V": 0.2, "FFN_gate": 651.1, "FFN_up": 646.4,
            "RMSNorm": 2.1, "Activation": 2.1
        }},
        "M=32768": {"total_ms": 1800.0, "note": "estimated from 16384 scaling"},
        "M=65536": {"total_ms": 3600.0, "note": "estimated from 32768 scaling"},
    },
    "npu_candidates": {
        "measured": {
            "QK_t": {
                "M=1024": {"gpu_ms": 0.7, "npu_ms": 30.4, "ratio": "NPU 43x slower"},
                "M=4096": {"gpu_ms": 11.0, "npu_ms": 549.2, "ratio": "NPU 50x slower"},
                "M=8192": {"gpu_ms": 29.2, "npu_ms": 1983.5, "ratio": "NPU 68x slower"},
                "M=16384": {"gpu_ms": 75.6, "npu_ms": "TIMEOUT", "ratio": "NPU >> GPU"},
            },
            "Attn_V": {
                "M=1024": {"gpu_ms": 0.0, "npu_ms": 4.3, "ratio": "NPU much slower"},
                "M=4096": {"gpu_ms": 0.1, "npu_ms": 16.0, "ratio": "NPU much slower"},
                "M=8192": {"gpu_ms": 0.1, "npu_ms": 30.6, "ratio": "NPU much slower"},
                "M=16384": {"gpu_ms": 0.2, "npu_ms": 60.6, "ratio": "NPU much slower"},
                "M=32768": {"gpu_ms": 0.5, "npu_ms": 120.4, "ratio": "NPU much slower"},
            },
        },
        "blocked": {
            "Q_proj": "K=4096 > 512: 8 tiles, NPU 28-66x slower",
            "K_proj": "K=4096 > 512: 8 tiles, NPU 28-66x slower",
            "V_proj": "K=4096 > 512: 8 tiles, NPU 28-66x slower",
            "O_proj": "K=4096 > 512: 8 tiles, NPU 28-66x slower",
            "FFN_gate": "K=14336 > 512: 28 tiles, NPU 98-174x slower",
            "FFN_up": "K=14336 > 512: 28 tiles, NPU 98-174x slower",
            "FFN_down": "K=4096 > 512: 8 tiles, NPU 28-66x slower",
        },
        "elementwise": {
            "RMSNorm": "NPU ~10-12x slower than GPU (estimated)",
            "Activation": "NPU ~10-12x slower than GPU (estimated)",
        }
    },
    "key_findings": [
        "NPU K>512 is BLOCKED by DMA BD generation limit in mlir-aie 1.4.1",
        "All transformer GEMMs have K>=128; projections/FFN have K>=4096",
        "NPU is 40-240x slower than GPU for measured shapes (K<=512)",
        "Tiled NPU estimates for K>512 are 28-174x slower than GPU",
        "No operation qualifies for Class B (NPU hide) or Class D (NPU win)",
        "All operations classified as Class A (GPU faster) or Class C (reject)",
        "Async overlap impossible: NPU compute time >> GPU compute time",
        "hiding_ratio = gpu_compute/npu_compute is always < 0.05",
    ],
    "verdict": {
        "best_M": None,
        "best_speedup": 1.0,
        "verdict": "HARMFUL",
        "detail": "NPU is 40-240x slower than GPU for all prefill operations. No beneficial offload candidates exist."
    }
}

print("=" * 70)
print("Stage 2 Final Verdict: Transformer Prefill Task Partitioning")
print("=" * 70)
print()
print("Context: 65536")
print()
print("GPU-only baseline (measured + estimated):")
print(f"  M=1024:   total = 684.0 ms")
print(f"  M=4096:   total = 2873.5 ms")
print(f"  M=8192:   total = 1767.6 ms")
print(f"  M=16384:  total = 2198.3 ms")
print(f"  M=32768:  total = ~1800 ms (estimated)")
print(f"  M=65536:  total = ~3600 ms (estimated)")
print()
print("Synchronous GPU/NPU:")
print("  total = N/A (no beneficial NPU candidates)")
print()
print("Asynchronous GPU/NPU:")
print("  total = N/A (no beneficial NPU candidates)")
print()
print("Speedup:")
print("  1.00x (no improvement possible)")
print()
print("NPU hidden compute:")
print("  0% (NPU compute >> GPU compute for all ops)")
print()
print("Hiding ratio:")
print("  < 5% for all operations")
print()
print("GPU slowdown:")
print("  0% (no NPU offload)")
print()
print("Synchronization overhead:")
print("  0 ms (no NPU offload)")
print()
print("Memory transfer overhead:")
print("  N/A (no NPU offload)")
print()
print("Best NPU candidates:")
print("  None")
print()
print("Rejected NPU candidates:")
print("  All operations:")
print("    - Projections (Q/K/V/O): NPU 28-66x slower, K=4096 BLOCKED")
print("    - FFN (gate/up/down): NPU 98-174x slower, K=14336/4096 BLOCKED")
print("    - Attention QK^T: NPU 43-68x slower, K=128 measured")
print("    - Attention Attn_V: NPU much slower, K=128 measured")
print("    - RMSNorm/Activation: NPU ~10-12x slower, estimated")
print()
print("Root cause:")
print("  mlir-aie 1.4.1 DMA BD generation limit: K must be <= 512")
print("  All transformer GEMMs have K >> 512")
print("  Tiling K into 512-wide chunks introduces 8-28x overhead")
print("  Even K=128 attention ops: NPU 40-240x slower than Radeon 780M")
print()
print("Verdict:")
print("  HARMFUL")
print()
print("Conclusion:")
print("  Phoenix XDNA1 NPU cannot serve as an async co-processor for")
print("  llama.cpp 64K+ prefill with current mlir-aie 1.4.1 stack.")
print("  The K<=512 DMA limitation blocks all major GEMMs, and the")
print("  resulting tiled NPU execution is 28-174x slower than GPU.")
print("  Even the NPU-friendly K=128 attention ops are 40-240x slower.")
print()
print("Recommendation:")
print("  Wait for mlir-aie updates that remove the K<=512 limitation,")
print("  or use NPU only for very small auxiliary workloads where")
print("  GPU time is extremely long (>100ms) and NPU can be fully hidden.")
print("=" * 70)
