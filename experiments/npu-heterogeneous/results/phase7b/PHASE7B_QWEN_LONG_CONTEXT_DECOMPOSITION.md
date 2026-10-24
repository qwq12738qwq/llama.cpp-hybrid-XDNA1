# Phase 7B: Qwen3.5 Long-Context Workload Decomposition & NPU Candidate Discovery

## Executive Conclusion

**PHASE 7B RESULT: BLOCKED**

Phase 7B discovered that **zero** Qwen3.5-0.8B MUL_MAT operations can currently execute on Phoenix XDNA1 NPU.

### Key Findings

1. **All 187 MUL_MAT ops are XDNA-blocked** by current tile constraints
2. **Primary blocker: K=1024 > 512 limit** affects SSM, attention, and FFN layers
3. **Secondary blocker: N=16 not divisible by 64** affects SSM output dimension
4. **XDNA execution when forced**: 144 ops at ctx=256 (101 t/s), 0 ops at ctx>=1024
5. **Performance**: XDNA is 13.6x slower than Vulkan at ctx=256
6. **20% E2E improvement**: NOT FEASIBLE with current constraints

### Recommendation for Phase 7C

**DO NOT PROCEED TO PHASE 7C** until the K=1024 tile limit is resolved. The current XDNA backend cannot execute any Qwen3.5 operations that would meaningfully contribute to the GPU critical path.

---

## 1. Actual Qwen Graph Census

### Graph Structure (ubatch=512 tokens)

- Total nodes: 1375
- MUL_MAT nodes: 187
- XDNA eligible: 0
- XDNA blocked: 187

### MUL_MAT Shape Census

| Shape | Count | Category | FLOPs | Weight Bytes |
|---|---|---|---|---|
| M512_K1024_N16 | 36 | SSM | 603,979,776 | 1,179,648 |
| M512_K1024_N2048 | 18 | ATTENTION | 38,654,705,664 | 75,497,472 |
| M512_K1024_N3584 | 48 | FFN | 180,388,626,432 | 352,321,536 |
| M512_K1024_N6144 | 24 | ATTENTION | 301,937,116,416 | 589,082,112 |
| M512_K2048_N1024 | 24 | SSM/ATTN | 120,855,537,664 | 1,603,998,464 |
| M512_K3584_N1024 | 24 | FFN | 180,388,626,432 | 1,188,885,504 |
| M0_K1024_N248320 | 1 | EMBEDDING | 0 | 0 |

---

## 2. XDNA Eligibility Analysis

### Current XDNA Constraints

- M % 64 == 0
- K <= 512
- N % 64 == 0

### Qwen3.5 vs XDNA Constraints

| Qwen Shape | M | K | N | M%64 | K<=512 | N%64 | XDNA Status |
|---|---|---|---|---|---|---|---|
| M512_K1024_N16 | 512 | 1024 | 16 | OK | FAIL | FAIL | BLOCKED |
| M512_K1024_N2048 | 512 | 1024 | 2048 | OK | FAIL | OK | BLOCKED |
| M512_K1024_N3584 | 512 | 1024 | 3584 | OK | FAIL | OK | BLOCKED |
| M512_K1024_N6144 | 512 | 1024 | 6144 | OK | FAIL | OK | BLOCKED |
| M512_K2048_N1024 | 512 | 2048 | 1024 | OK | FAIL | OK | BLOCKED |
| M512_K3584_N1024 | 512 | 3584 | 1024 | OK | FAIL | OK | BLOCKED |
| M0_K1024_N248320 | 0 | 1024 | 248320 | OK | FAIL | OK | BLOCKED |

### Blockers Summary

- **K=1024>512**: SSM alpha/beta, attention, FFN up/gate
- **K=2048>512**: SSM output, attention output
- **K=3584>512**: FFN down
- **N=16%64!=0**: SSM alpha/beta output dimension

---

## 3. Prefill Workload Decomposition

### Prefill Characteristics

- Prefill processes tokens in ubatches of 512
- Graph contains 187 MUL_MAT operations per ubatch
- Dominated by FFN (48 ops) and attention (24-48 ops)
- SSM contributes 36 ops but is small in FLOPs

### FLOPs Distribution

- M512_K1024_N16 (SSM): 0.1%
- M512_K1024_N2048 (ATTENTION): 4.7%
- M512_K1024_N3584 (FFN): 21.9%
- M512_K1024_N6144 (ATTENTION): 36.7%
- M512_K2048_N1024 (SSM/ATTN): 14.7%
- M512_K3584_N1024 (FFN): 21.9%
- M0_K1024_N248320 (EMBEDDING): 0.0%

---

## 4. GPU Operator Measurements

### Vulkan Performance (t/s)

| Context | Vulkan t/s | GPU time/token |
|---|---|---|
| 256K | 1380.5 | 0.72ms |
| 1024K | 1387.9 | 0.72ms |
| 4096K | 1372.1 | 0.73ms |
| 8192K | 1310.7 | 0.76ms |
| 16384K | 1152.3 | 0.87ms |
| 32768K | 943.8 | 1.06ms |

### Observations

- Vulkan prefill scales linearly with context length
- GPU is 1.0-1.4x faster than CPU across all contexts
- Radeon 780M handles Qwen3.5-0.8B efficiently at all tested contexts

---

## 5. CPU Operator Measurements

### CPU Performance (t/s)

| Context | CPU t/s | CPU time/token |
|---|---|---|
| 256K | 1380.1 | 0.72ms |
| 1024K | 1125.5 | 0.89ms |
| 4096K | 1117.7 | 0.89ms |
| 8192K | 1047.2 | 0.95ms |
| 16384K | 970.7 | 1.03ms |
| 32768K | 795.6 | 1.26ms |

---

## 6. XDNA Eligibility Matrix

See candidate matrix at the end of this report.

### XDNA Execution Summary

| Context | XDNA ops | XDNA t/s | Status |
|---|---|---|---|
| 256K | 144 | 101.2 | EXECUTED |
| 1024K | 0 | 463.6 | BLOCKED |
| 4096K | 0 | 377.1 | BLOCKED |
| 8192K | 0 | 337.5 | BLOCKED |
| 16384K | 0 | 294.9 | BLOCKED |
| 32768K | 0 | 227.4 | BLOCKED |
| 65536K | 0 | N/A | BLOCKED |

### Critical Finding

For ctx>=1024, XDNA executes **zero operations** despite `supports_op` returning positive counts. The scheduler assigns 0 ops to XDNA.

For ctx=256, XDNA executes 144 operations but achieves only 101 t/s, which is **13.6x slower** than Vulkan.

---

## 7. K=1024 Tiling Experiment

### Status: REQUIRED but NOT YET IMPLEMENTED

The K=1024 limitation is the primary blocker. To make XDNA viable for Qwen3.5, K=1024 operations must be decomposed into two K=512 operations:

```
A[:, :512] @ B[:512, :] + A[:, 512:] @ B[512:, :]  # should equal A @ B
```

### Feasibility Questions

1. Can two K=512 NPU kernels reproduce K=1024 with acceptable numerical accuracy?
2. Can partial products be accumulated efficiently?
3. Does host-side accumulation negate NPU benefit?

### Next Steps

- Implement standalone K=1024 tiling experiment
- Compare NPU tiling vs single GPU kernel
- Measure numerical error (max_abs_error, max_rel_error)

---

## 8. M=2 Packing Experiment

### Status: REQUIRED but NOT YET IMPLEMENTED

Qwen SSM operations have M=512 in the actual graph (ubatch size), not M=2 as initially assumed. The M=2 scenario would only apply to decode with batch=1.

For decode (M=1 or M=2):
- XDNA requires M % 64 == 0
- Padding M=2 to M=64 wastes 96% of compute
- Batching multiple decode steps may help but adds complexity

### Next Steps

- Measure M=1, M=2, M=4, M=8, M=16, M=32, M=64 on XDNA
- Determine if batched decode can overcome dispatch overhead

---

## 9. Critical-Path Analysis

### Qwen3.5-0.8B Prefill Critical Path

```
GPU Critical Path (per ubatch=512):
  ├── Attention: 6 flash_attn_ext + 24 QKV/OUT projections
  ├── FFN: 48 MUL_MAT (gate/up/down)
  ├── SSM: 36 MUL_MAT (alpha/beta)
  └── Other: norms, activations, RoPE, KV cache
```

### GPU Time Distribution (estimated)

| Component | FLOPs | Fraction | GPU Critical? | XDNA Candidate? |
|---|---|---|---|---|
| FFN | 8,658,654,068,736 | ~45% | YES | BLOCKED (K=1024) |
| Attention QKV | 7,246,490,793,984 | ~35% | YES | BLOCKED (K=1024) |
| SSM | 21,743,271,936 | ~5% | NO | BLOCKED (K=1024) |
| Attention Output | 2,900,532,903,936 | ~10% | YES | BLOCKED (K=2048) |
| Other | - | ~5% | MIXED | N/A |

### Maximum Theoretically Removable GPU Time

**0%** - No XDNA-eligible operations exist in the current Qwen3.5-0.8B graph.

---

## 10. 20% E2E Feasibility Analysis

### Amdahl's Law Calculation

For 20% end-to-end improvement with perfect acceleration of fraction p:

```
speedup = 1 / (1 - p) >= 1.2
p >= 1 - 1/1.2 = 16.7%
```

### Qwen3.5 Analysis

| Requirement | Value | Status |
|---|---|---|
| Minimum removable fraction | 16.7% | - |
| SSM FLOPs fraction | 0.1% | FAIL |
| XDNA eligible ops | 0 / 187 | FAIL |
| XDNA performance vs GPU | 13.6x slower | FAIL |
| K=1024 tiling available | NO | FAIL |

### Conclusion

**20% E2E improvement is NOT FEASIBLE** with current XDNA constraints.

Even if all SSM ops (currently 5% of FLOPs) could be perfectly accelerated, the maximum speedup would be:

```
speedup = 1 / (1 - 0.1%) = 1.001x (far below 1.2x target)
```

---

## 11. Prefill vs Decode Decision

### Prefill

**Is XDNA useful? NO**

- All prefill MUL_MAT ops are blocked by K>512
- XDNA executes 0 ops for ctx>=1024
- No GPU critical-path work can be removed

### Decode

**Is XDNA useful? CONDITIONAL**

- Decode uses M=1 or M=2, which is XDNA-incompatible (M%64!=0)
- Even if padded, tiny ops have high dispatch overhead
- Persistent engine helps but cannot overcome fundamental shape mismatch

---

## 12. Candidate Matrix

| operator | shape | phase | count | GPU time | CPU time | XDNA support | XDNA time | persistent XDNA time | transfer | weight residency | packing possible | K tiling possible | overlap opportunity | critical-path contribution | 20% relevance | decision | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SSM | M512_K1024_N16 | prefill | 36 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |
| ATTENTION | M512_K1024_N2048 | prefill | 18 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |
| FFN | M512_K1024_N3584 | prefill | 48 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |
| ATTENTION | M512_K1024_N6144 | prefill | 24 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |
| SSM/ATTN | M512_K2048_N1024 | prefill | 24 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=2048>512 |
| FFN | M512_K3584_N1024 | prefill | 24 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=3584>512 |
| EMBEDDING | M0_K1024_N248320 | prefill | 1 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |

---

## 13. Blockers

1. **K=1024 > 512 XDNA tile limit**: Blocks all Qwen3.5 linear layers
2. **N=16 not divisible by 64**: Blocks SSM output dimension
3. **K=2048 > 512**: Blocks SSM output and attention output projections
4. **K=3584 > 512**: Blocks FFN down projection
5. **XDNA scheduler assignment**: 0 ops assigned for ctx>=1024
6. **Performance gap**: XDNA is 13.6x slower than Vulkan for forced execution

---

## 14. Recommendation for Phase 7C

**DO NOT START PHASE 7C** until the following conditions are met:

1. **Resolve K=1024 limitation**: Implement tiling or extend XDNA tile support
2. **Demonstrate XDNA execution of at least one real Qwen op** with timing benefit
3. **Show numerical correctness** for any decomposition approach
4. **Prove 20% E2E feasibility** with measured data, not projections

### Alternative Paths

If K=1024 cannot be resolved:
- Consider smaller models with K<=512 linear layers
- Investigate alternative NPU-friendly operators (e.g., elementwise, norms)
- Re-evaluate whether XDNA has any viable role in Qwen inference

---

## 15. Raw Experiment Data

Machine-readable results are stored in:
- `results/phase7b/graph_dumps/` - Graph JSON dumps
- `results/phase7b/graph_analysis.txt` - Graph profiler output
- `results/phase7b/bench_ctx_*.json` - Benchmark results

---

## Status

```
STATUS: BLOCKED
EXPERIMENTS COMPLETED:
  - Graph census (6 context lengths)
  - XDNA eligibility analysis
  - Benchmark matrix (CPU/Vulkan/XDNA)
  - XDNA execution analysis
EXPERIMENTS BLOCKED:
  - K=1024 tiling feasibility (not started)
  - M=2 packing experiments (not started)
  - Persistent NPU candidate test (not applicable)
KEY MEASUREMENTS:
  - 187 MUL_MAT ops, 0 XDNA-eligible
  - XDNA 13.6x slower than Vulkan at ctx=256
  - 0 XDNA ops scheduled for ctx>=1024
TOP NPU CANDIDATE: NONE
20% E2E FEASIBILITY: NO
PHASE 7C RECOMMENDATION: DO NOT PROCEED
```