# Phase 53 results: NPU kernel and mixed-precision architecture redesign

Model `models/Qwen3.5-0.8B-Q8_0.gguf`, AMD Phoenix XDNA1 NPU, 1x4 topology.

## 1. Baseline (this tree)

- Vulkan-only PP512: 1077.31 tok/s (this run); historical 1248-1277 tok/s.
- XDNA hybrid (c4_mmul_v2) PP512: 16.09 tok/s (72/72 FFN GEMMs on NPU, token path runs).
- Current kernel: `c4_mmul_v2`, 1x4, full-K single submit, device-side per-32 scale apply.
- NPU per-GEMM device wait ~359 ms (Phase 50 gate). Q8_0 scale-aware kernel ~8-9 GFLOP/s.

## 2. New engine: int8_requant (added)

Switch: `XDNA_Q8_ENGINE=int8_requant`. Old `c4_mmul_v2` remains available for comparison.

Implementation:
- Weight preprocessing (cached by tensor pointer): dequant Q8_0 -> per-output-column int8 with one scale per column (sb[n]).
- Activation: per-input-row int8 with one scale per row (sa[m]).
- Scales are packed as constants into the existing mmul-v2 per-32 scale slots (all 8 blocks of a 256-chunk share the same sa/sb), so the existing full-K `Q8SCALEV_MMUL` kernel (one submit per GEMM) is reused without recompiling a kernel.
- Kernel output is float (scales applied on device); host does not re-scale.
- Reuses cached xclbin `M*_K*_N*_C4_Q8SCALEV_MMUL` via the registry, so no new compilation is required and the proven full-K path is kept.

## 3. Correctness

- Bench PP512 completed: 144 XDNA ops executed, 0 missing kernels, no NaN/abort during the run.
- No "failed to decode prompt batch" beyond the bench warmup issue seen only when device memory was exhausted (recovered after freeing memory).

## 4. End-to-end performance (PP512, ubatch512, 1 run)

| engine | tok/s |
|---|---|
| Vulkan-only | 1077.31 |
| c4_mmul_v2 (baseline) | 16.09 |
| int8_requant | 14.30 |

The int8_requant path is slightly slower than c4_mmul_v2 because of host requantization (per-GEMM activation re-quantization over M*K values) and unchanged device float scale path. It does not yet remove the per-32 float scale computation from the device, because the reused kernel still applies scales on device.

## 5. Pure-INT8 full-K kernel (not yet integrated)

- A pure int8 (Phase 36 style) chunked kernel (64x64x64) was compiled for the down shape (K=3584, N=1024) into `experiments/phase53/kernels/M512_K3584_N1024_C1/`. The gate/up shape reuses the existing cached pure-int8 xclbin.
- The chunked pure-int8 path (synchronous, 16 submits per K=1024 GEMM) was attempted but is far slower (single GEMM ~1.2 s) and risks device wedges; it is not competitive.
- A full-K pure-int8 kernel (no scale application on device, int32 output, host applies sa*sb) still needs a new MLIR generator adapted from `experiments/phase40/gen_q8_c4_mmul_v2.py`. That is the remaining step to remove the float scale path from the device and reach the projected speedup.

## 6. Decision

- The int8_requant representation (per-row/per-column scales) runs end-to-end without crashes and with plausible numerics, but does not yet beat c4_mmul_v2 because the device still applies scales.
- The device memory (host-mem bank) is the binding constraint: n_cols=4 mmul-v2 BOs (~4 MB weight BO) push the NPU over its mmap limit under memory pressure. Running fewer columns or smaller shapes would be safer.
- Remaining bottleneck: device-side float scale application (93% of cycles in Phase 50 trace) and host requantization cost. Removing it requires the full-K pure-int8 kernel, which is the next verifiable step.

## 7. Files changed

- `ggml/src/ggml-xdna/ggml-xdna.cpp`: added `ggml_xdna_requant_*` helpers and `ggml_xdna_submit_requant` (mmul-v2-based, constant scales), dispatch branch for `XDNA_Q8_ENGINE=int8_requant`, `sys/stat.h` include.
- `ggml/src/ggml-xdna/xdna-kernel-registry.cpp`: added `want_requant` flag routing to the mmul-v2 cache namespace.
- `experiments/phase53/kernels/M512_K3584_N1024_C1/`: compiled pure-int8 (chunked) kernel for the down shape.
