# Phase 20G E2E status

## Model

`/Storage/Cache/llama.cpp/models/Qwen3.5-0.8B-Q8_0.gguf` (arch qwen35, loads fine).

Q8_0 MUL_MAT weight shapes (K,N):
```
K=1024 N=512 / 2048 / 3584 / 4096 / 6144
K=2048 N=1024
K=3584 N=1024
```
(Q8_0 K=1024 N=16 and the embedding N=248320 are not MUL_MAT scale-aware shapes.)

## Kernels

Vectorized kernels in `kernel_cache/M512_K{K}_N{N}_C1_Q8SCALEV/` (see
`gen_q8_scale_vec.py`: aie::mac vector MAC + TaskGroup, tile m=32 n=64 k_chunk=256).

The earlier scalar kernels in `..._Q8SCALE` are superseded.

## How XDNA gets work

The Phase 20D sched patch offloads a MUL_MAT to an accelerator backend when its
weight sits in a host buffer:

```
-ot 'blk\.\d+\.ffn_down\.weight=CPU'
```

`-dev 'Vulkan0/AMD Phoenix XDNA1 NPU'` alone does NOT offload (XDNA's buffer type
is the CPU buffer type, so weights default to Vulkan).

## E2E result (llama-bench pp512, ng18 Qwen3.5-0.8B Q8_0)

```
Vulkan-only:          905.85 t/s
Vulkan+XDNA (hetero):  36.88 t/s   (XDNA ops executed=144, missing_kernel=0)
```

Vectorization vs the scalar kernel: the scalar hetero run did not finish in > 4
minutes; the vectorized run completes but is still ~24x SLOWER than Vulkan-only.

## Kernel timing

`q8_scale_vec_smoke` (kernel submit+wait only, 100 iters):
```
M=512 K=1024 N=1024: 157 ms
M=512 K=1024 N=3584: 553 ms
```

That is ~3 GOPS, far below what an AIE core should reach. The vector MAC helps
(~20x vs the scalar loop, consistent with the earlier Phase 20E note) but a single
core doing scalar-blocked vector MAC is not enough.

## Verdict

`critical_path_hetero < critical_path_Vulkan_only` is NOT met. To get there the
device GEMM must be ~100-500x faster, which needs:
- the AIE matrix MAC unit (mm/dense intrinsics), not per-element vector MAC, and
- multiple AIE columns (n_cols > 1) executing independent N partitions.

Combined those are the only plausible path; micro-tuning the current kernel cannot
close a 24x gap.
