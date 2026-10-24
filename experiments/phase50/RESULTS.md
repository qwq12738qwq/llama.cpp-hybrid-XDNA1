# Phase 50 results: XDNA FFN kernel dataflow redesign (s12) with verified end-to-end gain

Model `models/Qwen3.5-0.8B-Q8_0.gguf` (Q8_0, 24 layers, n_embd 1024),
`Vulkan0` = AMD Radeon 780M (UMA), `AMD Phoenix XDNA1 NPU`, 1x4 topology.
Physical ubatch 512. All NPU runs via `../phase48/npu_guard.sh`, memory capped
(`ulimit -v 4194304` unless noted). Baseline files in `BASELINE_*`.

Verdict: **SUCCESS (limited)** — a correctness-verified kernel/dataflow
improvement (s12) gives a reproducible **1.18x PP512 and 1.18x PP1024**
end-to-end gain with token-identical outputs. The NPU remains far slower than
Vulkan-only; the report quantifies why and what was tried.

## 1. End-to-end results (PP512, 5 reps, median)

| cfg | engine | tok/s (median) | wall | NPU GEMMs | verify md5 |
|---|---|---|---|---|---|
| Vulkan-only | - | 1276.63 | 0.40 s | 0/72 | `0ec3a71b13cb7372` reference |
| Hybrid prod | `c4_mmul_v2` | 15.41 | 33.2 s | 72/72 | `0ec3a71b13cb7372` identical |
| Hybrid s12 | `s12` | 18.18 | 28.2 s | 72/72 | `0ec3a71b13cb7372` identical |

s12 5 reps: 18.24 / 18.18 / 18.13 / 18.18 / 18.15 (range 0.11, stable).
Prod 5 reps: 15.40 / 15.41 / 15.31 / 15.43 / 15.85 (median 15.41).
**PP512 speedup s12 vs prod: 18.18 / 15.41 = 1.18x.**

## 2. PP1024 regression

| cfg | tok/s | wall | NPU GEMMs | verify |
|---|---|---|---|---|
| Vulkan-only | 1316.75 | 0.78 s | 0/144 | `1f57c384ad7252ea` reference |
| Hybrid prod | 15.44 | 66.3 s | 144/144 | (not re-verified, phase49 covers PP512) |
| Hybrid s12 | 18.20 | 56.3 s | 144/144 | `1f57c384ad7252ea` identical |

**PP1024 speedup s12 vs prod: 18.20 / 15.44 = 1.18x**, token-identical to
Vulkan-only. The gain scales linearly with M (same per-GEMM win, twice the
GEMMs), as expected.

## 3. Microbench (single GEMM, 5 reps, correctness-gated vs fp64 ref)

| shape | prod wait | s12 wait | prod max_err | s12 max_err | CPU same-shape |
|---|---|---|---|---|---|
| M512 K1024 N3584 (gate) | 433.0 ms | 358.9 ms + 9 ms scatter (1.21x) | 3.8e-05 PASS | 3.8e-05 PASS | 15.5 ms (XDNA 23x slower) |
| M512 K3584 N1024 (down) | 414.2 ms | 325.6 ms + 2 ms scatter (1.26x) | 0.42 PASS | 1.1e-04 PASS | 12.7 ms (XDNA 26x slower) |

CPU numbers re-measured today with this binary (`bench对照`, 9 iters,
median): gate 15.47 ms, down 12.69 ms — consistent with phase49 (16.0/11.4).
Effective rates: CPU gate ~243 GFLOP/s, XDNA s12 gate ~10.5 GFLOP/s (wait).
Even after the s12 win, XDNA per-GEMM is >20x slower than CPU on identical
shapes and identical timing boundaries (device wait only, no host packing).
Vulkan per-GEMM was not separately timed; end-to-end Vulkan-only (0.40 s
for PP512) already bounds it far below both.

Host pack (Python reference path, not production C++): s10 pack adds the
scale broadcast/duplication; device wait dominates. Host scatter after drain:
gate ~9 ms, down ~2 ms (vectorized numpy; C++ production scatter is a
single permuted copy, negligible vs GEMM time).

## 4. What s12 changes (and why it wins)

Production `c4_mmul_v2` per (bb,i,j) block (32 outputs):
- `to_float` int32->fp32 (API, ~25 bundles of bf16 magic: vconv/vmsc/vadd).
- scale built by 4 broadcasts + 4 muls + 32-wide `insert<8>` construction
  (~20 bundles) + 32-wide `p*scale` + 4x 8-lane C RMW.
- Block loop = 206 bundles (measured from ELF disassembly).

s12 (files `gen_s10.py` kernel source, `gen_s12.py` runtime):
- Host pre-broadcasts A scales to 32 floats per (bb,i) =
  `[a(i*4+0)x8, ..., a(i*4+3)x8]` and duplicates B scales to 16 floats per
  (bb,j) = `[bsv, bsv]`.
- Per block the scale products are single 16-lane muls (`t01`, `t23`);
  no in-kernel broadcast/insert; C RMW is two 16-lane load/add/store pairs
  on the contiguous `[i][j][4][8]` tile layout.
- Block loop = 152 bundles (26% fewer): vmac.f 32 (was 48), vmov 31
  (was 53), spill loads 27 (was 31). Conversion unchanged (~25).

## 5. Why not faster: measured limits

- AIE trace (phase46 gate3/down, re-analyzed): MAC phase 7.0 cycles/block,
  float scale path 214.0 cycles/block (93%), inter-block gap 4-9 cycles.
  LOCK_STALL present (DMA waits) but DMA-only microbench (57.7 ms) is
  almost fully hidden under compute (full 433 vs float-only 421 ms).
- AIE2 fp32 vector ops run at half rate on the bf16 datapath: 24 fp32
  8-lane ops per block lower to 48 `vmac.f` (2 cycles each). This is a
  hardware rate, not a compiler bug.
- The remaining ~58 shuffle/spill bundles come from register-bank pressure
  (16 vector regs; bf16-pair ops need specific banks). Every restructuring
  tried to cut live values made it worse (s1 245, s3 251, s4 301, s9 354,
  s7c 490 bundles) except s10/s12 (152).
- DMA redundancy is 8.8x (106 MB actual vs 12 MB unique) but DMA is hidden
  under compute, so DMA restructuring cannot improve wall time now.

## 6. Candidates tried and eliminated (each: source kept, 1 variable)

- `dma_only` (57.7 ms): DMA+setup isolation. `mmul_only` (57.8 ms): MACs
  hide under DMA. `scale_only` (421 ms): float path dominates. All PASS/zeros.
- `creg` (441 ms, max_err 1.5e-05 PASS): C-resident loop interchange; 8x
  fewer C RMW gives nothing -> C traffic is not the bottleneck.
- `orig` (1323 ms): pre-P32 kernel, much slower; confirms P32 was already
  a big step.
- `s1`/`s3` (245/251 bundles): row-wise 8-float without inserts; worse
  (double 32-lane conversion machinery + spills).
- `s4` (301): manual filter-based hi/lo conversion; worse.
- `s6` (206, identical to P32): scalar-scale muls; compiler canonicalizes
  to the same shape.
- `s7c` (977 ms, max_err 4.8e9 FAIL): manual accum bias-trick conversion;
  numerically wrong and slow. Abandoned.
- `s8`/`s11` (225 bundles, s11 505 ms correct-but-slower): pre-broadcast
  scales with production C layout; the 8-lane extracts + C RMW eat the win.
- `s9` (354): store/load bias conversion; aliasing spills. Abandoned.
- `accfloat` (250): `to_float` direct on MMUL accum; worse.
- `s10` (152 bundles, 358 ms device): the fast kernel, but its
  `[i][j][4][8]` C layout needs a 4D drain while the shim DMA BD supports
  3 access dims + repeat (a 4th dim silently becomes a repeat count:
  verified in `aie.mlir`, `len=256` instead of 2048). Per-i partial drains
  fight ObjectFifo object semantics (only first group correct). An 8-object
  C fifo deadlocked (61 s). **s12 = s10 kernel + linear drain + host
  scatter** is the shippable form.
- `-O3` on the kernel: identical 206 bundles (Peano already at ceiling).

## 7. Production integration (all gated, production path untouched by default)

- `ggml/src/ggml-xdna/xdna-kernel-registry.cpp`: `XDNA_Q8_ENGINE=s12`
  loads `M.._K.._N.._C.._Q8SCALEV_S12` xclbins into the existing
  `q8_c4_mmul_v2` engine with `s12=1`.
- `xdna-q8-c4-mmul-v2.h/.cpp`: `s12` flag; s12 chunk sizes
  (A 16384, B 20480 per chunk).
- `ggml-xdna.cpp`: `n_cols=4` for s12; s12 pre-broadcast packing in
  `ggml_xdna_submit_q8_c4_mmul_v2`; s12 host scatter in the wait/copyback
  path; submit dispatch accepts s12.
- Kernels staged in `experiments/npu-int8-gemm/kernel_cache/`:
  `M512_K1024_N3584_C4_Q8SCALEV_S12` and `M512_K3584_N1024_C4_Q8SCALEV_S12`.
- With `XDNA_Q8_ENGINE` unset/`c4_mmul_v2`, all s12 branches are off and the
  binary reproduces phase49 numbers (verified: prod 5-rep median 15.41).

## 8. Stopping assessment

SUCCESS (limited): a verified 1.18x end-to-end improvement with identical
outputs and a clean production-gated integration. The NPU (18 tok/s) remains
~70x slower than Vulkan-only (1277 tok/s) on this UMA part; the quantified
reasons are the fp32 half-rate datapath (~48 cycles/block of irreducible
scale math), the ~25-cycle int32->fp32 conversion, and ~58 cycles of
bank-shuffle/spill overhead the compiler cannot remove. No unexplained
large software overhead remains: device wait (359 ms) ~= 152 bundles x
~1.9 ns + DMA/lock stalls, matching the trace within measurement noise.
Further 2x+ gains would need either a native int32->fp32 conversion
instruction (absent on AIE2), a different accumulation domain, or
restructuring the model math (out of scope).
