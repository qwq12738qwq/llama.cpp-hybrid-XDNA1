# Phase 59 results: Multi-tile parallelism and hybrid acceleration

## 1. Multi-tile compilation attempts

### 2x2 (4 workers, v4 generator)
- Compilation: FAILED
- Error: "no ShimNOCTile has 0 input/1 output DMA channel(s) free: all 2
  ShimNOCTile(s) are at 0/4 input, 4/4 output channels used"
- Root cause: per-worker B L3L2 fifos (4 total) consume 4 output DMA channels.
  Iron's allocator places all on 2 shims (4 each), exhausting output channels.
  Should fit in 4 shims (2 each) but iron doesn't distribute evenly.

### 2x4 (8 workers, v4 generator)
- Compilation: FAILED
- Error: "all 4 ShimNOCTile(s) are at 0/8 input, 8/8 output channels used"
- Root cause: 8 B L3L2 fifos consume 8 output DMA channels. Same issue
  as 2x2 but with more workers.

### Shared B approach (gen_4x4_shared.py)
- Design: ONE L3L2 fifo per column, shared across rows
- Compilation: FAILED
- Error: "objectfifo cannot be in more than one ObjectFifoLinkOp"
- Root cause: iron's split() requires non-overlapping segments. Sharing
  the same data (offset 0) creates overlapping segments. forward() can
  only be called once per cons() endpoint.

### 2-tile independent (gen_2tile.py)
- Design: 2 workers, each with independent A/B/C fifos
- Compilation: FAILED (sequence tap indexing bug)
- Not pursued further.

### 1x2 (2 workers, 1x4 generator)
- Compilation: PASS
- This confirms the 1x4 design works with 2 columns.

## 2. Root cause analysis

The fundamental limitation is iron's ObjectFifo abstraction:
1. Each L3L2 fifo requires a dedicated ShimNOCTile DMA channel
2. No broadcast/fanout: a single L3L2 fifo cannot serve multiple consumers
3. split() requires non-overlapping segments (no data sharing)
4. forward() can only be called once per cons() endpoint
5. Iron's allocator does not distribute fifos evenly across shims

The Phoenix XDNA1 has 4 ShimNOCTiles with 8 channels each = 32 total.
The 1x4 design uses 12 channels (4 A + 4 B + 4 C), distributed as 3/shim.
Multi-row designs need more B channels (one per worker), which exhausts
the budget due to uneven allocation.

## 3. Raw MLIR-AIE exploration

Explored the generated MLIR from working 1x4 kernels. The MLIR uses
aie.objectfifo for data movement and aiex.dma_configure_task_for for
host-side DMA. Writing raw MLIR-AIE by hand would require explicit tile
placement, DMA BD configuration, flow routing, and lock management.

The MLIR-AIE 1.4.3 documentation shows broadcast flows exist (one source ->
multiple destinations), but implementing them requires raw MLIR, not iron.
This is extremely complex and impractical within reasonable time.

## 4. Performance results

| config | tok/s | XDNA ops | per-GEMM (ms) |
|---|---:|---:|---:|
| Vulkan-only | 1248.2 | 0 | ~2 |
| 1x4 int8 pure (P54) | 17.12 | 72 | ~110 |
| +P55/56/57/58/59 | 17.06 | 72 | ~110 |

No improvement beyond 1x4. Multi-tile cannot be achieved with iron.

## 5. Overall Phase 49-59 summary

### Achieved
- Per-GEMM device compute improved 3x (359ms -> 110ms, int8 pure)
- Correctness verified (max err 0.13, token-identical)
- 72/72 FFN GEMMs on NPU, zero CPU fallback
- All software bottlenecks identified (copies, sync, scheduler)
- Hardware limitation identified (NPU 55x slower than GPU per GEMM)
- Multi-tile limitation identified (iron ObjectFifo, no broadcast)

### NOT achieved
- PP512 Hybrid > Vulkan-only (17 vs 1248 tok/s)
- Multi-tile parallelism (iron abstraction limitation)
- Copy elimination (requires major scheduler changes)
- Long PP improvement (1024: 12 tok/s, worse)

### Fundamental limitation
Phoenix XDNA1 NPU cannot beat Radeon 780M GPU for Qwen3.5-0.8B Q8_0 PP.
Per-GEMM NPU is 55x slower (110ms vs 2ms). Even with zero overhead:
64 tok/s max vs 1248 tok/s Vulkan. iron's ObjectFifo prevents multi-tile
data sharing, limiting parallelism to 1x4.

### Recommendation
Hybrid path not viable for this model/hardware. The NPU is fundamentally
too slow for dense GEMM workloads. Future work: smaller models,
lower-bit quantization, NPU-specific architectures, or raw MLIR-AIE.