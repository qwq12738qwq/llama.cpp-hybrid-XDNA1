# Phase 20G step 11 - Vulkan + XDNA N-partition of one GEMM

Status: **PASS**. The heterogeneous output equals the full Vulkan GEMM and the CPU
reference exactly, and the two N-partitions run concurrently (overlap > 0 in 30/30).

## Goal (step 11)

One GEMM `C[M x N] = A[M x K] x B[K x N]`, split along N:

```
       M x K x N
            │
     ┌──────┴──────┐
  Vulkan N0     XDNA N1
            │
       N0 + N1 = N
```

The concatenated output `[C0 | C1]` must equal the full Vulkan GEMM.

## Setup

- Shape: `M=64, K=1024, N=256`, `N0=128` (Vulkan), `N1=128` (XDNA). `N0 + N1 = N`.
- XDNA kernel: `kernel_cache/M64_K1024_N128_C1` (full K=1024, single submit).
- Data type: int8 x int8 -> int32 on both paths. The Vulkan path widens the int8
  operands into int32 buffers and accumulates in int32; this is numerically
  identical to the XDNA device int8 GEMM (exact integers), so a column-for-column
  comparison is exact.
- `B` is split into `B0 = B[:, 0:N0]` and `B1 = B[:, N0:N]` by a CPU row-major
  repack *before* timing (step 13 allows CPU packing off the critical path).
- Both partitions share the same `A`.

## Correctness

```
[full-vulkan] mismatches vs CPU 0 / 16384
[correctness] hetero vs CPU 0 / 16384, vs full-vulkan 0 / 16384
```

- `hetero vs CPU = 0`: `[C0 | C1]` equals the exact int32 reference.
- `hetero vs full-vulkan = 0`: `[C0 | C1]` equals the full Vulkan GEMM.
- `full-vulkan vs CPU = 0`: the int32 Vulkan shader itself is correct.

No tolerance was touched; any mismatch returns `RESULT FAIL layout-mismatch`.

## Concurrency

`--iters 30`, parallel mode (`results/step11_parallel_n256.csv`):

```
iter  xdna_submit  vk_submit  xdna_done  vk_done   overlap   critical
29      116.6 us   134.9 us    538.9 us   865.2 us  404.0 us  748.6 us
```

| mode | mean critical path | overlap > 0 |
|------|-------------------|-------------|
| parallel (async) | **847 us** | 30/30 |
| serial (baseline) | **1232 us** | 0/30 (expected) |

Parallel critical path is **~31% lower** than the serial Vulkan-then-XDNA baseline.
`xdna_state = 4` (`ERT_CMD_STATE_COMPLETED`) on every iteration.

The serial run also reports `mismatches 0/16384`; its `RESULT FAIL` only reflects the
script's overlap criterion (`overlap_positive=0/30`), which is the expected outcome for
a deliberately serialized baseline.

## Constraints honored

- One XRT submit per XDNA partition (`N1=128`, full `K=1024`), no K=32 loop.
- `xrt::run.start()` with no immediate wait; the two partitions are waited on
  separate threads.
- CPU only does packing / copy-in before `t0`; it is not on the compute critical path.
- No IRON runtime: XRT is driven directly, Vulkan is a native compute pipeline.
- Vulkan and XDNA are not serialized in the measured mode.

## Not yet done (next)

- Step 8: ggml backend async contract (declare `async`/`events`, real
  `graph_compute_async`, `event_record`/`event_wait`).
- Step 10: correctness sweep over `M in {64,128,256,512}` x `K=1024` x `N in {1024,2048}`.
- Step 14: async partition-ratio sweep (10/25/50/75%).
- Step 15: E5 PP end-to-end (`PP4096..65536`, CPU fallback = 0, per-MUL_MAT submit = 1).
- Step 16: structured per-MUL_MAT overlap logging.
