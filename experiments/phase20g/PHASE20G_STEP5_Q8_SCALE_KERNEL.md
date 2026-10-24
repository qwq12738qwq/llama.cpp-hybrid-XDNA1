# Phase 20G Step 5 - scale-aware Q8_0 full-K kernel (PASS)

Goal: make a Q8_0 MUL_MAT run the whole K in ONE XRT submit, with the per-32-block
scales applied on the device, instead of host K-chunking into 32 submits.

## Kernel

Generator: `experiments/phase20g/gen_q8_scale_gemm.py`

Device math:
```
C[m,n] = sum_blk ( sum_{k in blk} A[m,k]*B[k,n] ) * As[m,blk] * Bs[blk,n]
```
The AIE core loops over K/32 blocks inside the device and applies both scales.
Tile shape m=64, n=64, k_block=32.

Host-visible operands are tile-major packed (the packing adapter):
```
A_packed: (M/64)*nb tiles, each = [64*32 int8 data][64 f32 scale]
B_packed: (N/64)*nb tiles, each = [32*64 int8 data][64 f32 scale]
C       : f32 [M, N]
```
`nb = K/32`. A single `fill` per M-row pushes all nb tiles, so the DMA uses only
a handful of BDs (TaskGroup reuse); the earlier per-block fills hit the 16-BD limit.

## Layout verification

Harness: `experiments/phase20g/q8_scale_smoke.cpp` (native XRT, no IRON runtime).

With uniform scales (As = Bs = 1) the result is bit-exact against the CPU int64
reference, which proves the tile-major packing has no layout error:

```
Q8_SCALE PASS M=64  K=1024 N=64   max_rel=0.000e+00 bad=0/4096
Q8_SCALE PASS M=64  K=1024 N=1024 max_rel=0.000e+00 bad=0/65536
Q8_SCALE PASS M=64  K=1024 N=2048 max_rel=0.000e+00 bad=0/131072
Q8_SCALE PASS M=128 K=1024 N=2048 max_rel=0.000e+00 bad=0/262144
```

With non-uniform scales the remaining error is float32 accumulation only (bad=0):
```
Q8_SCALE PASS M=64  K=1024 N=1024 max_rel=8.6e-04 bad=0/65536
Q8_SCALE PASS M=64  K=1024 N=2048 max_rel=1.9e-02 bad=0/131072
Q8_SCALE PASS M=128 K=1024 N=1024 max_rel=1.2e-03 bad=0/131072
Q8_SCALE PASS M=128 K=1024 N=2048 max_rel=1.9e-02 bad=0/262144
```

## Backend integration

New: `ggml/src/ggml-xdna/xdna-q8-scale.{h,cpp}` - its own xrt context + BOs +
`submit` (xrt::run.start, no wait) / `wait` (copies the f32 C).

- `xdna-kernel-registry.{h,cpp}`: Q8_0 with `k_tile == K` and `dtype_a == Q8_0`
  loads from the dedicated cache namespace `M{M}_K{K}_N{N}_C1_Q8SCALE`.
  I8 full-K (`dtype_a == I8`) keeps the old namespace.
- `ggml-xdna.cpp`: `tensor_to_kernel_key` sets `k_tile = K` for Q8_0;
  `ggml_xdna_submit_q8_scalefullk` dequantizes Q8_0 to int8 + f32 scales, packs
  A/B tile-major and issues one submit; `drain_pending` waits and writes f32.

## Results

Backend Q8_0 MUL_MAT (M=512, K=1024, N=4096), via `test_xdna_q8`:
```
[XDNA-Q8S] engine ready M=512 K=1024 N=4096 a=589824 b=4718592 c=8388608
[XDNA-REGISTRY] Loaded scale-aware Q8_0 kernel: M=512 K=1024 N=4096
[XDNA] XDNA ops executed=1 missing_kernel=0      <- one submit, no CPU fallback
CPU:  266.02 ...   XDNA: 266.14 ...   Max error: 0.1204  (~4.5e-4 relative)
```

Regression: int8 async path unchanged.
```
ASYNC_BACKEND PASS mismatches=0 early_unwritten=1 event_ok=1
```

## Cache artifacts

```
kernel_cache/M512_K1024_N1024_C1_Q8SCALE/{design.xclbin,design.insts.bin}
kernel_cache/M512_K1024_N4096_C1_Q8SCALE/{design.xclbin,design.insts.bin}
```

## Not done yet

- Production N sweep (768, 1536, 2304, 3072, 3584, 4608, 5120, 6144) + K in {2048, 4096}
- Section 14 async partition sweep, section 15 E2E PP
