# Phase 43: DMA-Budget-Constrained Topology Search

## Status

Topology search in progress. Hardware compilation pending.

## 1. Baseline: 1x4 (v1)

| Property | Value |
|----------|-------|
| Workers | 4 (1 row x 4 cols) |
| ObjectFifos | 18 |
| Status | VALIDATED (Phase 40M) |
| Device latency | ~1.3 ms (Phase 43 context) / ~5.78 ms (Phase 40M) |
| Correctness | PASS (0 mismatches) |

**v1 is the production baseline. Do not modify until an alternative passes independent validation.**

## 2. Topology: 2x4

| Property | Value |
|----------|-------|
| Workers | 8 (2 rows x 4 cols) |
| ObjectFifos | 32 |
| Resource budget | PASS (under 36 limit) |
| CPU simulation | PASS (0 mismatches) |
| Hardware compilation | PENDING |

**2x4 is the first candidate for higher tile utilization.**

## 3. Topology: 3x4

| Property | Value |
|----------|-------|
| Workers | 12 (3 rows x 4 cols) |
| ObjectFifos | 46 |
| Resource budget | LIKELY FAIL (exceeds 36 limit) |
| CPU simulation | PASS (0 mismatches) |
| Hardware compilation | PENDING (expected to fail) |

**3x4 is expected to be resource-blocked. Document as DATAFLOW RESOURCE LIMITATION.**

## 4. 16-Tile Experiments

| Variant | Workers | ObjectFifos | Failure Mode |
|---------|---------|-------------|--------------|
| v2 (4x4 shared-B) | 16 | 36 | Hardware race (shared FIFO) |
| v3 (16x1 per-worker-B) | 16 | 48 L3/L2 | Shim/NOCTile budget exceeded |
| v4 (4x4 per-worker-B) | 16 | 36 MemTile | MemTile budget exceeded |

**Classification: DATA-MOVEMENT RESOURCE BUDGET, not compute limitation.**

## 5. Down Projection (K=3584, N=1024)

| Property | Value |
|----------|-------|
| Shape | 512x3584x1024 |
| CPU simulation | PASS (0 mismatches) |
| C++ backend | Already whitelisted |
| Compiled kernel | PENDING |

**The down projection kernel is already supported by the C++ backend. Only needs hardware compilation.**

## 6. Selection Criteria

Rank by:
1. Correctness PASS
2. Resource allocation PASS
3. Stable repeated execution
4. Device latency
5. Effective GOPS

## 7. Next Actions

1. Compile and benchmark 2x4 on hardware
2. Attempt 3x4 compilation; if it fails, document as DATAFLOW RESOURCE LIMITATION
3. Compile down projection (K=3584, N=1024)
4. Integrate real FFN (gate/up/down)
5. Run PP512
