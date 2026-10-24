# Phase 52 results: exploring new NPU compute paths for FFN

Model `models/Qwen3.5-0.8B-Q8_0.gguf`, AMD Phoenix XDNA1 NPU, 1x4 topology.
Baseline (Phase 50 verified): Vulkan-only PP512 1276.63 tok/s; Hybrid s12 PP512
17.86 tok/s (72/72 NPU, token-identical). NPU per-GEMM device wait 359 ms (gate).

## 1. Audit of real FFN math (Q8_0)

- Block size QK8_0 = 32; scale d is ggml_half (fp16); qs int8.
- Dequant: y[j] = qs[j] * fp16_to_fp32(d), per 32-block.
- GEMM shapes: gate/up (K=1024, N=3584), down (K=3584, N=1024), M up to 512.
- CPU vec_dot accumulates int32 per 32-block then scales by d_a*d_b (fp32).
- Current XDNA kernel converts int32->fp32 per 32-block (25 bundles) + fp32 scale
  ops (half rate). Phase 50 trace: MAC 7 cycles/block, float scale path 214
  cycles/block (93%).

## 2. Candidate matrix

| candidate | status | reason |
|---|---|---|
| A: INT32 accum + delayed scale (per-32 preserved) | not feasible | AIE2 has no int64 vector type; per-32 scales cannot be merged without re-quantization (Phase 51) |
| B: re-quantize to per-row/per-col scale (int8 pure MAC) | feasible, error 1.12% | see section 3 |
| C: 4-bit weights | not tested | decode overhead likely exceeds savings; deferred |
| D: grouped scale dataflow (keep per-32) | no gain | Phase 50 exhausted float-path restructurings |
| E: full FFN execution rework | not started | gated on kernel improvement |

## 3. Candidate B: re-quantization to coarser scales (measured)

Experiment: experiments/phase52/requant_test.py + requant_gemm_test.py.
Loaded real weight blk.0.ffn_gate (3584x1024 Q8_0), dequantized to float, then
re-quantized to int8 with per-row scale (one scale per output row of weight = per
output column of GEMM). Activation re-quantized to int8 per-row (one scale per
input row). GEMM computed as pure int8->int32, scale applied once per output.

Result (M=512, K=1024, N=3584):
- Per-row re-quantization of weight alone: max_err 0.000714 (relative 0.86%).
- Full GEMM (both A and W re-quantized) vs Q8_0-ref: max_err 0.0207,
  mean_err 0.0017, relative 1.12%.

The error is small and comparable to Q8_0's own quantization error (~0.8%).
For 24 layers the error compounds, but greedy decode may still match; requires
end-to-end verification.

Benefit: the NPU kernel becomes pure int8->int32 (Phase 36 style), eliminating
the per-32 float conversion and scale application from the hot path. Phase 36
measured 1876 GOPS on Qwen shapes for pure int8->int32 (vs 8-9 GOPS current).
Even with host re-quantization (~1.8M values per GEMM, <5 ms) and host scale
application (<2 ms), the NPU path could drop from 359 ms to ~5-10 ms per GEMM,
making it competitive with CPU (15 ms) and possibly Vulkan.

## 4. Integration plan (not yet implemented)

1. Add a new engine "int8row" (or reuse persistent int8 kernel) that:
   - Input A: int8 tiles (M, K) + float per-row scales (M).
   - Input B: int8 tiles (K, N) + float per-col scales (N).
   - Kernel: pure int8 MAC -> int32 C (Phase 36 persistent style, 1x4).
   - Host post-pass: C_int32 -> float, multiply by sa[m]*sb[n].
2. Weight preprocessing at model load: dequant Q8_0 -> re-quantize per-column
   (one scale per output feature), store int8 tiles + scales. Cache by tensor.
3. Activation re-quantization at GEMM submit time: float activation -> int8 per
   row + scale. (Host, ~5 ms for M=512.)
4. Keep Vulkan/CPU fallback; dispatch by shape (NPU only for large GEMMs).
5. Correctness gate: single GEMM, full FFN, PP512 greedy decode, token compare.

## 5. Blockers / risks

- Phase 36 persistent kernel timed out on this device during this session
  (device wedge risk). Needs careful single-run testing via npu_guard.
- Re-quantization changes model numerics; must verify greedy decode matches
  Vulkan-only on the full model before claiming success.
- Weight preprocessing memory: int8 weights same size as Q8_0 (1 byte/val vs
  Q8_0 1.0625 byte/val) — slightly smaller. Scales: N floats per weight matrix
  (3584*4 = 14KB) negligible.
- Activation re-quantization is per-GEMM (3 per layer x 24 = 72 per PP512 pass).
  Host cost ~72 * 5 ms = 360 ms total — could offset gains. Need to measure.

## 6. What was NOT done (time/device constraints)

- Did not build or run the int8-row kernel (reuses Phase 36 design, needs
  verification on current device).
- Did not integrate into ggml backend.
- Did not run PP512 with the new path.
- Did not test 4-bit or other formats.

## 7. Recommendation

The most promising path is Candidate B (re-quantization to per-row/per-col
scales + pure int8 NPU kernel). The error is small (1.12% relative on GEMM),
and the potential speedup is large (8-9 -> ~1800 GOPS, ~40-70x). The main
risk is host re-quantization cost and numerical compounding across 24 layers.
Next phase should implement and measure end-to-end.
PYEOF
echo written