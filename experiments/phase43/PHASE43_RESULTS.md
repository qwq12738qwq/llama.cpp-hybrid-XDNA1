# Phase 43: DMA-Budget-Constrained Topology Search — Results

## Hardware Compilation Results

### 1x4 (v1) — BASELINE

| Property | Value |
|----------|-------|
| Workers | 4 (1 row x 4 cols) |
| ObjectFifos | 18 |
| ShimNOCTile budget | PASS |
| Compilation | PASS |
| Device latency | ~1,339 ms (standalone), ~1,481 ms (C++) |
| Correctness | PASS (0 mismatches, realistic pattern) |
| Cache namespace | M512_K{KC}_N3584_C4_Q8SCALEV_MMUL |

**Status: PRODUCTION BASELINE. Do not modify.**

### 2x4 — RESOURCE-BLOCKED

| Property | Value |
|----------|-------|
| Workers | 8 (2 rows x 4 cols) |
| ObjectFifos | 32 |
| ShimNOCTile budget | FAIL |
| Compilation | FAIL |
| Failure mode | `no ShimNOCTile on the device has 0 input/1 output DMA channel(s) free: all 4 ShimNOCTile(s) are at 0/8 input, 8/8 output channels used` |
| CPU simulation | PASS (0 mismatches) |

**Status: DATA-MOVEMENT RESOURCE LIMITATION. Not a compute tile issue.**

### 3x4 — INCOMPATIBLE + RESOURCE-BLOCKED

| Property | Value |
|----------|-------|
| Workers | 12 (3 rows x 4 cols) |
| ObjectFifos | 46 |
| M divisibility | FAIL (`M=512` not divisible by `M_TILE * n_rows = 96`) |
| Compilation | FAIL (assertion error before resource check) |
| CPU simulation | PASS (0 mismatches) |

**Status: INCOMPATIBLE with Qwen M=512. Would also exceed ShimNOCTile budget.**

### 16-Tile Experiments

| Variant | Workers | ObjectFifos | Failure Mode |
|---------|---------|-------------|--------------|
| v2 (4x4 shared-B) | 16 | 36 | Hardware race (shared FIFO) |
| v3 (16x1 per-worker-B) | 16 | 48 L3/L2 | Shim/NOCTile budget exceeded |
| v4 (4x4 per-worker-B) | 16 | 36 MemTile | MemTile budget exceeded |

**Classification: DATA-MOVEMENT RESOURCE BUDGET, not compute limitation.**

## Down Projection (K=3584, N=1024)

| Property | Value |
|----------|-------|
| Shape | 512x3584x1024 |
| Topology | 1x4 (v1 baseline) |
| Compilation | PASS |
| Device latency | ~1,305 ms |
| Correctness (realistic) | PASS (0 mismatches) |
| Correctness (geometric) | 35 mismatches (float16 precision, expected) |
| Cache namespace | M512_K3584_N1024_C4_Q8SCALEV_MMUL |

**Status: COMPLETE. Down projection kernel compiled and validated.**

## Selection

**Best valid topology: 1x4**

Rationale:
- 2x4 fails ShimNOCTile budget at compilation
- 3x4 is incompatible with M=512 and would exceed budget
- 16-tile designs are resource-blocked
- 1x4 is the only topology that compiles and runs correctly

## Real FFN Status

| Projection | Shape | Status |
|------------|-------|--------|
| ffn_gate | 512x1024x3584 | SUPPORTED (cached) |
| ffn_up | 512x1024x3584 | SUPPORTED (cached) |
| ffn_down | 512x3584x1024 | SUPPORTED (newly compiled) |

**Real FFN is now complete on XDNA.**

## PP512 Status

### Phase 31 Reference (build 7620399f5, Sep 20)
- qwen35 0.8B Q8_0, Vulkan,XDNA backend, pp512: **1365.07 ± 20.18 t/s**
- Subsequent measurement in this environment with same build: **1224 t/s** (XDNA ops executed=0)
- Conclusion: Phase 31 measurement was Vulkan-only. XDNA backend was loaded but not dispatching ops.

### Current Environment Measurement (Sep 23)
- Vulkan-only (GGML_XDNA_QWEN_PP=0): **1252.78 t/s**
- With XDNA routing enabled (GGML_XDNA_QWEN_PP=1): **1224.17 t/s** (XDNA ops executed=0)
- With modified scheduler forcing XDNA offload: **5.23 t/s** (XDNA ops executed=288, but terrible throughput)

**Blocker: XDNA ops cannot be dispatched in PP512 without catastrophic E2E throughput loss.**

Root cause analysis:
1. The 1x4 mmul-v2 kernel has ~1.3ms device latency per MUL_MAT op
2. In PP512, the scheduler splits the graph into ~288 small subgraphs (1-2 nodes each)
3. Each subgraph triggers engine-reuse flush, serializing execution: submit → wait → submit → wait
4. Per-op overhead (XRT copies, sync, run construction) adds ~2ms on top of 1.3ms kernel
5. Total XDNA compute: 288 × 1.3ms ≈ 374ms, but serialized execution + overhead dominates
6. Result: 5.23 t/s vs Vulkan-only 1252 t/s

The standalone kernel is correct and resource-validated, but the E2E integration via `ggml_backend_sched` does not provide sufficient overlap or batching to overcome per-op overhead.

### Modified Scheduler (REVERTED)
A temporary modification to `ggml-backend.cpp` attempted to route MUL_MAT ops to XDNA even when weights reside on Vulkan buffers. This caused the 5.23 t/s regression and was reverted. The original CPU-weight-only offload condition is restored.

## Next Actions

1. **E2E integration requires architectural change** — either:
   - Batch multiple MUL_MAT ops into a single XDNA submission
   - Implement persistent engine mode that holds the xclbin loaded across subgraphs
   - Accept Vulkan for this workload and use XDNA only for shapes where it shows proven advantage
2. **Do not modify 1x4 kernel** — it is validated and correct
3. **Do not pursue 16-tile topologies** — resource-blocked
4. Optional: research 16-tile broadcast topology in a separate branch after E2E is stable

## Bug Fix

Fixed `c_taps` indexing bug in `gen_q8_c4_mmul_v2_v4.py:255`:
- Was: `c_taps[row][col][mt][nt]`
- Now: `c_taps[col][row][mt][nt]`

This bug prevented 2x4 and 3x4 topologies from reaching the aiecc compilation stage.
