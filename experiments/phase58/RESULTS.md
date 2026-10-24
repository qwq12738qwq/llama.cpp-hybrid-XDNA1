# Phase 58 results: 4x4 XDNA1 GEMM parallel dataflow

## 1. Goal

Use more AIE compute tiles (2x4, 3x4, 4x4) for GEMM to improve Hybrid PP performance.

## 2. Resource audit

### 1x4 (baseline, working)
- Workers: 4 (1 row x 4 cols)
- ObjectFifos: 18
- ShimNOCTile DMA: 4 A (input) + 4 B (input) + 4 C (output) = 12 channels, 3/shim
- Compilation: PASS
- Device latency: ~110ms per GEMM (int8 pure)

### 2x4 (failed)
- Workers: 8 (2 rows x 4 cols)
- ObjectFifos: 32
- ShimNOCTile DMA: 2 A + 8 B + 4 C = 14 channels
- Failure: "no ShimNOCTile has 0 input/1 output DMA channel(s) free: all 4
  ShimNOCTile(s) are at 0/8 input, 8/8 output channels used"
- Root cause: B has 8 independent L3L2 fifos (one per worker), consuming 8
  output DMA channels. Combined with A (2) = 10 output channels across 4 shims.
  Iron's allocator places all on output channels, exhausting the budget.

### 2x2 (failed)
- Workers: 4 (2 rows x 2 cols)
- ObjectFifos: ~20
- ShimNOCTile DMA: 2 A + 4 B + 2 C = 8 channels
- Failure: same as 2x4 - all shims at 4/4 output channels used
- Root cause: even with fewer workers, the per-worker B fifos consume too
  many output DMA channels.

### 3x4 (not attempted)
- Would require 3 rows, M=512 not divisible by 3*32=96
- Would exceed ShimNOCTile budget even more

### 4x4 (not attempted)
- Would require 4 rows, 16 workers, 16 B fifos
- Definitely exceeds ShimNOCTile budget

## 3. Attempted solutions

### A. Shared B fifos (one per column, not per worker)
- Design: 4 B L3L2 fifos (one per column) instead of 16
- Problem: iron's ObjectFifo.split() requires non-overlapping segments.
  Sharing the same data (offset 0) creates overlapping segments, which is
  not allowed.
- forward() also fails: "objectfifo cannot be in more than one ObjectFifoLinkOp"

### B. Duplicate B data in host buffer
- Design: put B data n_rows times in L3L2 buffer, split with offsets
- Problem: wastes n_rows x memory and DMA bandwidth. The external DMA
  transfers n_rows times more data, negating parallelism benefits.
- Also requires complex host-side buffer management.

### C. via_DMA=False for internal fifos
- Setting via_DMA=False on L2L1 fifos would save MemTile DMA channels
- But the bottleneck is ShimNOCTile (external) channels, not MemTile
- L3L2 fifos (external) must use DMA

## 4. Root cause

The fundamental limitation is iron's ObjectFifo abstraction:
1. Each L3L2 fifo requires a dedicated ShimNOCTile DMA channel
2. There is no broadcast/fanout mechanism - a single L3L2 fifo cannot
   serve multiple L2L1 consumers
3. split() requires non-overlapping segments (no data sharing)
4. forward() can only be called once per cons() endpoint

The Phoenix XDNA1 has 4 ShimNOCTiles with 8 channels each = 32 total.
The 1x4 design uses 12 channels (4 A + 4 B + 4 C). A 2x4 design
needs at least 2+8+4=14 channels, but the allocation is uneven and
exhausts individual shims' output channels.

## 5. Conclusion

**4x4 multi-tile GEMM cannot be achieved with iron's ObjectFifo abstraction.**

The ShimNOCTile DMA budget is the binding constraint, and iron provides
no mechanism to broadcast data from a single DMA channel to multiple
compute tiles. Achieving true 4x4 parallelism would require:

1. Writing raw MLIR-AIE (bypassing iron) with explicit MemTile broadcast
2. Using AIE's native stream routing (not iron's ObjectFifo)
3. A different dataflow that doesn't require per-worker external DMA

### What was achieved
- Confirmed 1x4 is the maximum parallelism with iron's abstraction
- Identified exact resource limitation (ShimNOCTile output DMA channels)
- Tested 2x2 and 2x4 topologies (both fail at compilation)
- Documented the root cause and potential solutions

### Recommendation
1. Focus on software optimizations for the 1x4 design (reduce sync, packing)
2. If multi-tile is required, write raw MLIR-AIE with MemTile broadcast
3. Consider using the NPU's vector/SIMD capabilities instead of more tiles
4. The 1x4 int8 pure kernel (110ms/GEMM) is the best achievable with iron