# Phase 54 results: pure INT8 GEMM (int8_requant_v2)

Model `models/Qwen3.5-0.8B-Q8_0.gguf`, AMD Phoenix XDNA1 NPU, 1x4 topology.

## 1. Audit of Phase 52/53

- Phase 52 re-quantization error (1.12% relative) measured on real weight
  `blk.0.ffn_gate` (3584x1024 Q8_0), dequantized to float, then re-quantized to
  int8 with per-output-column scale (sb[n]) and per-input-row activation scale
  (sa[m]); pure int8 MAC with scales applied once per output.
- Phase 53 `int8_requant` reused the mmul-v2 (Q8SCALEV_MMUL) scale-aware kernel
  by packing constant scales into per-32 slots. It did NOT remove the per-32 float
  scale path; device still applied scales. PP512 14.30 tok/s.
- Phase 53's "pure int8" xclbin (M*_C4_PURE) was compiled but never integrated
  with correct host packing; the chunked path was slow.

## 2. New engine: int8_requant_v2

- Added env `XDNA_Q8_ENGINE=int8_requant_v2`. Keeps `c4_mmul_v2` and
  `int8_requant` available for comparison. Default behavior unchanged.
- New MLIR generator: `experiments/phase54/gen_int8_pure_mmul_v2.py`, based on
  `gen_q8_c4_mmul_v2.py` but removes per-32 scale arrays and float conversion;
  output int32 accumulated across K chunks. Tile-major int8 packing matching the
  mmul-v2 kernel indexing.
- Compiled xclbins in `experiments/npu-int8-gemm/kernel_cache/M*_C4_PURE/`.
- Host: reuses cached per-column weight requantization (`g_requant_wcache`),
  per-row activation quantization, tile-major packing (no scales in buffers),
  host applies sa[m]*sb[n] after device returns int32.

## 3. Correctness

- Gate A single GEMM (M=512,K=1024,N=3584): max error 0.13 vs Q8_0 CPU ref
  (comparable to c4_mmul_v2's 0.12). 144 XDNA ops executed, 0 missing kernels.
- Gate D PP512: completed, no NaN/abort/crash, 144/144 NPU GEMMs.

## 4. Performance (PP512, ubatch512, 1 run)

| engine | tok/s | per-GEMM device time |
|---|---:|---:|
| c4_mmul_v2 (baseline) | 15.35 | ~359 ms |
| int8_requant_v2 (new)  | 17.12 | ~110 ms (K=1024), ~210 ms (K=3584) |

Per-GEMM device time dropped ~3x. End-to-end PP improved only 11.5% because of
host packing and 88 vk->xdna copies (303 MB) plus scheduler transitions.

## 5. Answers

1. New engine runs pure INT8 GEMM: yes (int8 operands, int32 accumulator, no
   per-32 float scale on device).
2. Per-32 float scale hot path removed from device: yes (kernel source has no
   scale arrays or to_float).
3. Single GEMM / FFN / PP improvements: GEMM ~3x; FFN/PP ~1.1x (copy-bound).
4. Numerical error vs Q8_0: max abs ~0.13 on M=512 GEMM; token path runs.
5. Weight preprocess + activation quant cost: included in host_prep (~50-120ms/
   GEMM); weight cached by tensor pointer.
6. NPU coverage: 72/72 FFN GEMMs (144 ops incl. warmup pass).
7. Faster than Vulkan-only? No (Vulkan PP512 ~1077-1277 tok/s); gap is host
   packing + DMA copies + scheduler transitions.
8. Software vs device limits: per-GEMM compute is now ~3x faster; remaining gap
   is host-side packing (tile-major repack of M*K + K*N per GEMM) and the
   Vulkan<->XDNA buffer copies, which are software/scheduler issues, not
   device-level limits.