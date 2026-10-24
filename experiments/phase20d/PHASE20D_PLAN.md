# Phase 20D: Long-Context Heterogeneous PP Research - Execution Plan

## Research Question

> On Phoenix XDNA1 + Radeon 780M, can N-dimension partition, multi-column XDNA kernels,
> async concurrency, and long-context operator offloading make llama.cpp PP degradation
> significantly lower than Vulkan-only baseline as context grows 4K -> 8K -> 16K -> 32K -> 64K?

## Hypotheses

- **H0**: Heterogeneous PP does not significantly change long-context scaling
- **H1**: Heterogeneous PP reduces long-context PP growth rate
- **H2**: Heterogeneous PP achieves near-linear scaling in some context ranges
- **H3**: Only specific operators/partition strategies are effective

## Execution Order

### Step 1: Freeze Vulkan Baseline
```bash
./bench_vulkan_baseline.sh
```
- Run PP at 4K/8K/16K/32K/64K
- Record tokens/s, latency, scaling ratios
- This is the reference for all comparisons

### Step 2: Trace Qwen PP Graph
- Use `GGML_XDNA_DEBUG=1` to log operator dispatch
- Record operator contribution vs context length
- Identify which ops dominate at each context length

### Step 3: Generate Missing Multi-column Kernels
```bash
python3 gen_multicolumn_kernels.py
```
- Generate C2/C3/C4 MLIR designs for P0 shapes
- Compile with Peano: `aiecc.py design.mlir -o design.xclbin`
- Verify correctness with test_xdna_q8

### Step 4: Heterogeneous PP Benchmark
```bash
./bench_hetero.sh
```
- Sweep XDNA partition thresholds
- Measure PP at each context length

### Step 5: Scaling Analysis
```bash
python3 analyze_scaling.py results/
```
- Compute empirical alpha for Vulkan-only
- Compute empirical alpha for each heterogeneous config
- Check STOP conditions

### Step 6: Critical-Path Analysis (if Step 5 shows improvement)
- Use Chrome-trace profiling from qwen-hetero-pp
- Measure GPU busy, XDNA busy, overlap, dependency, transfer, idle
- Answer: is improvement from XDNA compute or GPU/XDNA overlap?

## STOP Conditions

### SUCCESS
If ALL of:
- 64K heterogeneous PP < 64K Vulkan-only
- 32K/64K scaling exponent decreases
- CPU fallback = 0
- Real GPU/XDNA overlap confirmed

Then: `LONG_CONTEXT_HETERO_PP_VALIDATED`

### PARTIAL
If:
- Microbenchmark improved but E2E PP no improvement

Then: `LONG_CONTEXT_HETERO_PP_PARTIAL` with critical-path bottleneck identified

### NEGATIVE
If ALL of:
- Completed 4K/8K/16K/32K/64K
- CPU fallback = 0
- Multi-column 1/2/3/4 tested
- Partition sweep completed
- Timeline/critical-path analyzed
- Still no improvement

Then: `LONG_CONTEXT_HETERO_PP_NOT_VALIDATED`

## Kernel Cache Status (Phase 20D, measured)

Production cache: `/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache`
Kernel name: `M{M}_K32_N{N}_C{C}_Tm64_Tk32_Tn64` (engine K=32; host chunks over full K)

Production N whitelist (`is_mul_mat_supported`, ggml-xdna.cpp):
512 768 1024 1536 2048 2304 3072 3584 4096 4608 5120 6144

| M | C1 | C2 | C3 | C4 |
|---|----|----|----|----|
| 512 | 12/12 | 12/12 | 6/6 | 12/12 |
| 448 | 12/12 | 12/12 | 6/6 | 12/12 |

C3 is bounded by `N % 192 == 0`, so only N in {768, 1536, 2304, 3072, 4608, 6144}
are feasible. All 84 kernels verified bit-exact against a numpy
`int8 x int8 -> int32` reference by `experiments/phase20d/verify_kernels.py`.

### Generator Fix: single-BD B tap

`gen_xdna_gemm.py` used one BD for a whole column slice when
`K * N_per_col <= MAX_BD_BYTES` (K=32, so `N_per_col <= 2048`). Every shape on
that path was numerically wrong: only 20/84 kernels were bit-exact. The runtime
issues one B fill per (M-row, N-tile) step, so the oversized tap over-feeds the
B FIFO. Fix: always split B per N-tile. Result: 20/84 -> 84/84 bit-exact.

This is why the earlier heterogeneous PP runs were unusable: most production
shapes returned wrong values from the cached kernels.

## Registry Changes (Phase 20D)

- `n_cols`: expanded from 1-2 to 1-4
- `K`: changed from {1024,2048,4096} to any K%32==0
- `M`: changed from {64,128,256,448,512} to any M>=64

## Key Files

- `ggml/src/ggml-xdna/ggml-xdna.cpp` - XDNA backend
- `ggml/src/ggml-xdna/xdna-kernel-registry.cpp` - Kernel registry
- `experiments/npu-int8-gemm/gen_xdna_gemm.py` - Kernel generator (IRON + Peano)
- `experiments/phase20d/gen_missing_kernels.py` - Batch kernel generation
- `experiments/phase20d/verify_kernels.py` - Kernel numeric verification
- `experiments/phase20d/bench_vulkan_baseline.sh` - Baseline benchmark
- `experiments/phase20d/bench_hetero.sh` - Heterogeneous benchmark
- `experiments/phase20d/analyze_scaling.py` - Scaling analysis

Note: `experiments/phase20d/gen_multicolumn_kernels.py` is a placeholder that
emits non-compilable MLIR (empty core program). Use `gen_missing_kernels.py`
instead.
