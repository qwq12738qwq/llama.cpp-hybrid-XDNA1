# Phase 51 audit: Phase 36 vs Phase 50 kernel differences

## 1. Phase 36 high-throughput path (source: experiments/phase35/source/gen_phase35.py, build_persistent; reused by experiments/phase36/source/bench_phase36.py)

- Input dtype: int8, output dtype: int32 (no scales at all).
- MAC: `kernels.mm(dim_m=64, dim_k=64, dim_n=64, input_dtype=i8, output_dtype=i32)` -> AIE `aie::mmul` with int8 operands and int32 accumulator (`accauto`).
- Accumulation: pure integer, int32 output, no conversion, no scale application.
- Reported: T16 2120 GOPS on 512x2048x2048; Qwen shapes 1876-2368 GOPS. Correctness checked as exact int32 equality (max_mm == 0).
- No per-32 scales, no fp32 path, no host dequant/pack beyond tile-major int8.

## 2. Phase 50 current kernel (c4_mmul_v2 / s12) (source: experiments/phase40/gen_q8_c4_mmul_v2.py + experiments/phase50/gen_s12.py)

- Input dtype: int8 (Q8_0 tiles) + fp32 per-32 scales appended to A and B buffers.
- MAC: `aie::mmul<4,8,8,int8_t,int8_t,accauto>` -> int32 accumulator (same fast int path as Phase 36).
- After each 32-element K block (4 MACs): `aie::to_float(acc.to_vector<int32_t>())` (25 bundles per 32 values), then per-output scale product multiply (float, half-rate) and fp32 accumulate into C (float).
- Output dtype: float32.
- Reported: 8-9 GFLOP/s device, 433 ms (gate) / 414 ms (down) per GEMM.

## 3. Difference matrix

| aspect | Phase 36 | Phase 50 (s12) | equivalent? |
|---|---|---|---|
| MAC instruction | aie::mmul int8->int32 | aie::mmul int8->int32 | yes, same |
| Accumulator domain | int32 | int32 per 32-block, then float | no (conversion added) |
| Scale application | none | per-32-block fp32 scale product | extra cost |
| int32->fp32 conversion | none | 25 bundles per 32 values per block | extra cost |
| fp32 scale multiply | none | 16-32 lanes, half rate | extra cost |
| Output dtype | int32 | float32 | different |
| Correctness reference | exact int32 | fp64 reference, max_err threshold | different gate |
| Effective compute | int8 MAC only | int8 MAC + float scale path | Phase 36 omits scales |

## 4. Key conclusion from audit

Phase 36's high GOPS comes from OMITTING per-32 Q8_0 scales entirely. It is not numerically equivalent to the Q8_0 GEMM. The int8 MAC itself is already the fast path in Phase 50 (same `aie::mmul`). The slowness is the per-32-block scale application: int32->fp32 conversion (25 bundles) + fp32 scale multiply/accumulate (half rate on AIE2 bf16 datapath). AIE trace (Phase 50): MAC 7 cycles/block, float scale path 214 cycles/block (93%).

## 5. Feasibility probe (Phase 51)

- int64 vectors compile on AIE2: `aie::vector<int64_t,8>`, `aie::to_vector<int64_t>`, `aie::mul(int32,int32)->int64`, `aie::to_float(int64_vec)` all compile (gen_int64_test.py, gen_int64_mul_test.py, gen_int64_tofloat_test.py).
- However, AIE2 has no native int64 MAC; int64 mul/add/convert are not MAC-rate. Fixed-point integer accumulation would still need a float conversion at output (int64->float) and per-block scale handling; the per-block scale count cannot be reduced without changing Q8_0 per-32 semantics (scales differ per 32-element K block along K).
- No candidate was found that both preserves Q8_0 semantics and removes the per-block float conversion. Fixed-point path trades conversion frequency for int64 ops that are not accelerated on AIE2.
PYEOF
echo "written"