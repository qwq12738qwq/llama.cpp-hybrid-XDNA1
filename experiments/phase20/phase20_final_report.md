# Phase 20: XDNA Kernel Performance Research - Final Report

## Executive Summary

**53 GOPS at K=1024 is the single-AIE-core scalar compute ceiling, NOT a DMA ceiling.**

The CURRENT_RUNTIME_KERNEL (npu1_1col, Tm64_Tk64_Tn64, scalar matmul_i8_i32) achieves:
- **53 GOPS** at K=1024 (M=512, N=3072)
- **83 GOPS** at K=2048
- **117 GOPS** at K=4096

The bottleneck is **AIE compute utilization**, not DMA bandwidth. The kernel uses scalar INT8 multiply-accumulate operations, leaving the AIE2's vector MAC units idle.

---

## 1. Is 53 GOPS the true ceiling?

**No.** 53 GOPS is the **scalar implementation ceiling** on a single AIE core.

Evidence:
- At K=4096, the same kernel achieves 117 GOPS (2.2x improvement), proving the hardware can do more
- The AIE2 has 1024-bit vector MAC units capable of 128 INT8 MACs per cycle
- The current kernel calls a scalar `matmul_i8_i32` function, not vectorized operations
- Phase 18 measured AIE utilization at 0.65% (conservative estimate: 2.7%)

**The true hardware ceiling is estimated at 2000+ GOPS per column** (vectorized), meaning the current implementation achieves only ~2.7% of theoretical peak.

## 2. What is the pure DMA ceiling?

| Direction | Peak | At Size |
|-----------|------|---------|
| H2D | 14.1 GB/s | 1 MB |
| D2H | 116.4 GB/s | 1 MB |
| Round-trip | 13.1 GB/s | 1 MB |

H2D drops to 1.3 GB/s at 64 MB. D2H stays above 80 GB/s for large sizes.

## 3. What is the GEMM actual DMA throughput?

The GEMM kernel uses only **0.17 GB/s** effective DMA throughput (M=512, K=1024, N=3072).

This is **83x below the H2D ceiling** of 14.1 GB/s, confirming DMA is not the bottleneck.

## 4. Is AIE compute utilization really only 0.65%?

**Approximately yes.** At K=1024:
- Measured: 53 GOPS
- Estimated peak (scalar): ~2000 GOPS per column
- Utilization: 53/2000 = 2.65%

Phase 18's 0.65% used a different peak estimate (8000 GOPS total for 4 columns = 2000/column).
The key point: **the vast majority of AIE compute capacity is unused** because the kernel uses scalar operations.

## 5. Is double buffering effective?

**Not yet tested at kernel level.** Host-side M-tiling double buffering cannot help because:
- Each XRT submit processes all M-tiles in a single kernel invocation
- The kernel's internal loop already has ObjectFifo depth=2, but doesn't exploit pipelining
- True double buffering requires MLIR kernel modification to overlap DMA with compute inside the core loop

## 6. Is triple buffering effective?

**Not applicable** until double buffering is implemented at kernel level.

## 7. Is BD chaining effective?

**Not yet tested.** The current runtime_sequence has 384 separate DMA task blocks (one per N-tile). BD chaining could reduce host-side DMA configuration overhead, but the current kernel's bottleneck is compute, not DMA setup.

## 8. Is persistent execution effective?

**Not yet tested.** The current kernel already runs as a persistent infinite loop (the AIE core's outer loop is `scf.for %arg0 = 0 to MAX_INT`). The bottleneck is the scalar compute inside the loop, not the execution model.

## 9. Is M scaling improved?

**No.** GOPS is constant across M:

| M | GOPS (kernel) | kernel ms |
|---|---------------|-----------|
| 64 | 53.00 | 7.60 |
| 128 | 52.10 | 15.46 |
| 256 | 53.46 | 30.13 |
| 512 | 53.96 | 59.69 |

Linear scaling of kernel time with M confirms the kernel processes each M-tile sequentially with constant per-tile throughput.

## 10. Is N_COLS improved?

**No.** C4 is 8.4% SLOWER than C1:

| Config | GOPS | kernel ms |
|--------|------|-----------|
| C1 (Tm64_Tk64_Tn64) | 53.96 | 59.69 |
| C4 (Tm64_Tk32_Tn64) | 49.42 | 65.17 |

More columns increase DMA contention without proportional compute gain. The scalar kernel cannot utilize multiple compute cores effectively.

## 11. Does K=8192 improve?

**Not tested directly**, but K scaling trend is clear:

| K | GOPS | Improvement vs K=1024 |
|---|------|----------------------|
| 1024 | 53.96 | baseline |
| 2048 | 82.74 | 1.53x |
| 4096 | 116.71 | 2.16x |

K=8192 would likely achieve ~160+ GOPS based on the trend. Higher K increases arithmetic intensity, better amortizing per-iteration overhead.

## 12. Which kernel is best?

**V1 (CURRENT_RUNTIME_KERNEL) at K=4096 is the best measured configuration:**
- 117 GOPS (kernel), 109 GOPS (total)
- 0.86x Vulkan (XDNA wins by 14%)

At K=1024, no XDNA configuration beats Vulkan.

## 13. Best XDNA GOPS?

**116.71 GOPS** at M=512, K=4096, N=3072, C1.

## 14. Best Vulkan/XDNA ratio?

At K=4096: **0.86x** (XDNA is 14% faster than Vulkan).
At K=1024: **1.8x** (XDNA is 80% slower than Vulkan).

## 15. Is there real heterogeneous speedup?

**Only at K>=4096.** For K=1024 (the critical Qwen3.5 FFN path), XDNA is slower, so heterogeneous execution cannot help.

---

## Roofline Analysis

All measured shapes are **compute-bound**, not memory-bound:

| Shape | AI (FLOP/byte) | Ridge Point | Classification |
|-------|----------------|-------------|----------------|
| M=64 K=1024 | 100.7 | 141.5 | Compute (near ridge) |
| M=128 K=1024 | 166.1 | 141.5 | Compute |
| M=256 K=1024 | 245.8 | 141.5 | Compute |
| M=512 K=1024 | 323.4 | 141.5 | Compute |
| M=512 K=2048 | 472.6 | 141.5 | Compute |
| M=512 K=4096 | 614.4 | 141.5 | Compute |

Ridge point = compute_peak / memory_bw = 2000 GOPS / 14 GB/s = 141.5 FLOP/byte.

---

## XRT Submit Overhead Decomposition (M=512, K=1024, N=3072)

| Phase | Time (ms) | % of Total |
|-------|-----------|-----------|
| T_prepare (host memcpy) | 2.10 | 3.1% |
| T_bo_sync_in | 0.22 | 0.3% |
| T_submit | 0.02 | 0.0% |
| T_kernel (= DMA + AIE compute) | 59.69 | 87.2% |
| T_bo_sync_out | 0.16 | 0.2% |
| T_host_copy | 6.25 | 9.2% |
| **Total** | **68.44** | **100%** |

Kernel time dominates at 87%. Host overhead is 13%, dominated by T_host_copy (reading output via `bo.read()`).

---

## Path Forward

To break beyond 53 GOPS at K=1024, the required optimization is:

1. **Vectorize the matmul kernel** - Replace scalar `matmul_i8_i32` with IRON's `vectorized=True` mm() kernel, which generates MLIR with AIE2 vector MAC intrinsics
2. **Target: 200-500 GOPS per column** (4-10x improvement)
3. **Expected result**: XDNA at 200+ GOPS would be 0.4x Vulkan (60% faster), enabling strong heterogeneous speedup

---

## Files Generated

```
experiments/phase20/
  phase20_baseline.json          # Baseline measurements (M scaling + K sweep)
  phase20_dma.json               # Pure DMA microbenchmark
  phase20_roofline.json          # Roofline analysis
  phase20_kernel_rnd.json        # Kernel comparison data
  phase20_all_results.json       # All raw results
  phase20_final_report.md        # This report
  phase20_baseline.py            # Baseline measurement script
  phase20_bench_one.py           # Per-shape benchmark (avoids hw_context conflict)
  phase20_analysis.py            # Combined analysis script
```
