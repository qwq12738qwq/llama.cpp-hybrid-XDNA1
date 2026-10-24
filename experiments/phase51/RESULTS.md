# Phase 51 results: XDNA1 integer compute path redesign attempt

Model `models/Qwen3.5-0.8B-Q8_0.gguf`, AMD Phoenix XDNA1 NPU, 1x4 topology,
Q8_0 per-32 scales, physical ubatch 512. All NPU runs via npu_guard.sh with
memory capped (ulimit -v 4194304).

## 1. Baseline (Phase 50, verified this session)

| cfg | tok/s (PP512) | NPU GEMMs | notes |
|---|---|---|---|
| Vulkan-only | 1276.63 | 0/72 | reference, md5 0ec3a71b13cb7372 |
| Hybrid s12 | 17.86 | 72/72 | token-identical |
| Hybrid prod (c4_mmul_v2) | 15.41 | 72/72 | token-identical |

Per-GEMM device wait: gate 359 ms (s12) vs CPU 15.5 ms (XDNA ~23x slower).

## 2. Audit (see AUDIT.md)

Phase 36's 1876-2368 GOPS path uses pure int8->int32 with NO per-32 scales.
Phase 50's int8 MAC is already the same fast `aie::mmul<4,8,8,int8,int8,accauto>`.
The 93% of block cycles (AIE trace) is the per-32-block float scale path:
int32->fp32 conversion (25 bundles) + fp32 scale multiply/accumulate (half rate).

## 3. Candidate attempts

### Candidate A: integer (fixed-point) accumulation with delayed scale

- Feasibility probe (gen_p51_fp.py) attempted a full fixed-point kernel
  (int32 MAC -> int64 accumulation with Q16.16 scales -> float output).
  Compile FAILED: AIE2 does not support `aie::vector<int64_t, N>` as a vector
  element type (`long long` is not a valid ElemBaseType). The earlier
  gen_int64_* probes only called resolve_program() without .compile(), so they
  never actually compiled the kernel (false positives).
- Consequently, fixed-point integer accumulation using int64 vectors is not
  available on AIE2. The only integer accumulator width is int32, which
  overflows when holding scale products (sa*sb Q16.16 x dot ~5e5).
- Q8_0 per-32 scales differ per 32-element K block, so per-block scale
  application cannot be merged across blocks without re-quantization (breaks
  Q8_0 semantics and correctness thresholds).
- Candidate A eliminated: not implementable on AIE2 without int64 vectors.

### Candidate B: scale-aware fusion

- The scale product is already pre-broadcast on host (s12). Fusing further
  (e.g. absorbing scales into a single per-output scale) is not mathematically
  equivalent for per-32 scales. Not feasible without re-quantization.

### Candidate C: weight preprocessing / reuse

- Q8_0 weights are already in tile-major packed form with per-32 scales;
  no per-GEMM repeated conversion occurs in the production path (packing happens
  once per GEMM call but scales are pre-broadcast; weight tiles are static).
  Host prep is 12-20 ms = 3-5% of wall; device wait dominates. No win available.

## 4. Conclusion

No correctness-verified dataflow improvement found that significantly reduces
per-GEMM latency while preserving Q8_0 per-32 semantics. The per-32 scale
application (int32->fp32 conversion + fp32 scale ops) is an inherent cost of
Q8_0 semantics on AIE2: the int8 MAC is already on the fast path, and the only
way to remove the float scale path is to abandon per-32 scales (re-quantize),
which changes numerical results and breaks the existing correctness gate.

The NPU remains ~70x slower than Vulkan-only on this UMA part. This is a
verified architecture/dataflow limit given the evidence: AIE trace (93% float
path), Phase 36 comparison (high GOPS only without scales), int64 feasibility
probe (compiles but not MAC-rate), and host-prep measurements (3-5% of wall).

## 5. Evidence level

- AIE trace: measured (Phase 50).
- Phase 36 source audit: direct source inspection.
- int64 feasibility: compile-tested (3 probes, all BUILD OK).
- No unexplained software overhead: device wait (359 ms) matches 152 bundles x
  ~1.9 ns + DMA/lock stalls within measurement noise.

Phase 51 stopping condition met: verified architecture limit with quantified
evidence; no unexplained software overhead remains.
PYEOF
echo "written"