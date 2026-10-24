# Phase 57 results: Scheduler-aware FFN offload investigation

## 1. Scheduler execution path investigation

### Graph structure
The ggml scheduler splits the full graph into small sub-graphs (1-2 nodes each).
For PP512: 1715 nodes split into ~96 XDNA + 126 GPU + 54 CPU splits.
The FFN pattern (gate, up, GLU, down) is never in a single sub-graph,
making backend-level fusion impossible without scheduler changes.

### Copy analysis
- 88 V->X copies (303MB) per pass: activation tensors copied Vulkan->XDNA
- 24 X->V copies (50MB) per pass: FFN output copied XDNA->Vulkan
- 63 backend transitions, 113 splits, 155 copy buffers
- sync_ms: ~9s per pass

### Copy deduplication
Scheduler already deduplicates per (src_id, backend_id). But gate and up
MUL_MATs use the same activation tensor in different splits, causing
separate copies. The scheduler's copy logic checks tensor_id_copy() per
split, not globally.

## 2. Changes attempted

1. silu/mul on XDNA (P55): No improvement (GLU already on XDNA)
2. Pre-packed weights (P55): No improvement (memcpy into BO still needed)
3. Fused FFN in graph_compute (P56): Dead code (scheduler splits graph)
4. Pending threshold 2->4 (P55): Marginal (17.06 vs 17.12)
5. Copy deduplication investigation: Scheduler already deduplicates per
   (src_id, backend_id); reuse not happening due to split boundaries

## 3. Performance results

| config | tok/s | XDNA ops | v2x (MB) | x2v (MB) | sync (s) |
|---|---:|---:|---:|---:|---:|
| Vulkan-only | 1248.2 | 0 | 0 | 0 | 0 |
| c4_mmul_v2 (P49) | 15.35 | 72 | 452 | 403 | ~9 |
| int8_requant_v2 (P54) | 17.12 | 72 | 303 | 50 | ~9 |
| +P55/56/57 | 17.06 | 72 | 303 | 50 | ~9 |

## 4. Fundamental analysis: can hybrid ever beat Vulkan?

### Per-GEMM time
| backend | gate (M512 K1024 N3584) | down (M512 K3584 N1024) |
|---|---:|---:|
| Vulkan (GPU) | ~2 ms | ~3 ms |
| CPU | ~16 ms | ~11 ms |
| XDNA (int8 pure) | ~110 ms | ~210 ms |

NPU is 55x slower than GPU per GEMM.

### Theoretical minimum hybrid time
Even with zero overhead: 72 * 110ms = 7.9s = 64 tok/s max.
Vulkan: 1248 tok/s. Hybrid max is 19x slower.

## 5. Conclusion

The Phoenix XDNA1 NPU cannot beat the Radeon 780M GPU for Qwen3.5-0.8B Q8_0 PP.
This is a hardware limitation: per-GEMM NPU is 55x slower than GPU.

### What was achieved
- Per-GEMM device compute improved 3x (359ms -> 110ms)
- Correctness verified (max err 0.13, token-identical)
- 72/72 FFN GEMMs execute on NPU, zero CPU fallback
- Identified all software bottlenecks (copies, sync, scheduler)

### What was NOT achieved
- PP512 Hybrid > Vulkan-only (17 vs 1248 tok/s)
- Long PP improvement (1024: 12 tok/s, worse)
- Copy elimination (requires major scheduler changes)

### Recommendation
The hybrid path is not viable for this model/hardware. The NPU is
fundamentally too slow for dense GEMM workloads. Future work should focus
on smaller models, lower-bit quantization, or NPU-specific architectures.