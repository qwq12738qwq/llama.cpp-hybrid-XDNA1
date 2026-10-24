# Phase 48: Vulkan + XDNA heterogeneous PP scaling, PP512 -> PP5K

Baseline carried over from PHASE47B (frozen, not re-derived here):

| | |
|---|---|
| model | `models/Qwen3.5-0.8B-Q8_0.gguf`, Q8_0, 24 layers, n_embd 1024 |
| GPU | `Vulkan0` (AMD Radeon 780M, UMA) |
| NPU | `AMD Phoenix XDNA1 NPU`, opt-in via `GGML_XDNA_QWEN_PP=1` |
| NPU topology | 1 row x 4 cols, `q8_c4_mmul_v2`, m_tile 32, n_tile 64, k_tile 256, Q8_0 per-32 scales |
| PHASE47B result | Vulkan PP512 1292.67 tok/s, Hybrid PP512 1241.73 tok/s |

## What is measured

Two independent controls at every PP point, because the question in this phase
is whether a degradation comes from the NPU or from the batch size itself:

* `vk` - Vulkan only, `GGML_XDNA_DISABLE=1` so the XDNA backend is never even
  registered.
* `hybrid` - Vulkan + XDNA with `GGML_XDNA_QWEN_PP=1`.

For each (tag, pp) the driver runs three separate processes, so the headline
throughput is never measured with instrumentation turned on:

1. `bench` - 2 warmup + 5 measured prompt phases. Prints `RESULT,...` with
   median pp tok/s, its spread, total ms and peak RSS.
2. `trace` - 1 warmup + 1 measured prompt phase with `GGML_SCHED_TRACE_STATS=1`.
   Prints `[SCHED-SUMMARY]`, `[VK-SUMMARY]`, `[XDNA-SUMMARY]` and
   `[LLAMA-UBATCH]` lines; `collect_pp_scaling.py` folds them into one row.
3. `verify` - greedy decode of 32 tokens after the PP, printing the token ids
   and their md5. Hybrid is compared against the Vulkan md5 at the same PP.

## Files

| file | purpose |
|---|---|
| `bench_pp_scaling.cpp` | the bench / verify / ubatch-sweep driver |
| `run_pp_scaling.sh` | runs the PP matrix for one or both controls |
| `collect_pp_scaling.py` | folds all logs into `results/pp_scaling.csv` |
| `compile_kernels.sh` | builds the Q8SCALEV-C4 kernels for each needed M |
| `npu_guard.sh` | blocks until no other process holds `/dev/accel/accel0` |
| `results/` | raw logs plus the collected CSV |

## Two things this phase changed in the product code

1. **`GGML_XDNA_QWEN_PP` no longer implies NPU weight placement.**
   `ggml_xdna_device_get_memory()` reported 1 GB whenever the routing opt-in was
   set, so the documented flag silently moved 6 of 24 layers into NPU buffers:
   72 splits, 38 transitions, 197 MB of copies, 12 of 72 FFN GEMMs routed, and
   87 tok/s against 1315 tok/s Vulkan-only, reproducible over 4 identical runs.
   Placement now needs its own `GGML_XDNA_WEIGHTS=1`; the routing opt-in lands
   deterministically on the 2-split, ~1310 tok/s state. The NPU-routed
   configuration stays reachable and is measured as the `hybrid_npu` tag.
2. **Leftover allocator logging is gated.** `ggml_gallocr_alloc_graph` printed
   two lines per node unconditionally (~2800 lines per graph). Now behind
   `GGML_ALLOC_TRACE`.
3. **Per-GEMM layer/op classification no longer needs the verbose trace.**
   `ggml_xdna_gemm_record_metadata` only ran under `XDNA_GEMM_PROFILE`, so
   `[XDNA-SUMMARY]` reported every GEMM as `other`. It now also runs under
   `GGML_SCHED_TRACE_STATS`, which is what makes the gate/up/down coverage
   counts in the CSV meaningful.

## Environment variables

| variable | meaning |
|---|---|
| `PP_LIST` | PP points, default `512 768 1024 1536 2048 3072 4096 5120` |
| `UBATCH` | physical ubatch, default `512` (see below); `pp` means "one ubatch per PP" |
| `MODE` | `ab` (default), `full`, `vk`, `hybrid`, `hybrid_npu` |
| `ROUNDS` | interleaved vk/hybrid rounds per PP for `MODE=ab`, default 2 |
| `WARMUP`, `MEASURE` | repetitions per point, default 2 and 5 |
| `MODEL`, `BUILD`, `OUT` | override the model, build dir and results dir |
| `XDNA_Q8_ENGINE` | default `c4_mmul_v2`, the frozen 1x4 kernel |

`MODE=ab` interleaves `vk` and `hybrid` per PP and the collector takes the
median of the paired per-round ratios, because absolute tok/s on this host
drifts up to 7% between processes and a sequential sweep turns that drift into a
fake 9% hybrid overhead.

## Why the physical ubatch is pinned to 512

The Vulkan compute buffer scales with the *physical* ubatch, and this host has
7 GB of RAM shared with the UMA iGPU. Measured, Vulkan-only, PP3072:

| ubatch | compute buffer / result |
|---|---|
| 3072 (= PP) | `failed to allocate Vulkan0 buffer of size 3323719684`, 1130 tok/s never runs |
| 1024 | 981.50 tok/s, 3129.909 ms |
| 512 | 1130.08 tok/s, 2718.386 ms |

So at high PP a single big ubatch does not fit, and where it does fit it is
slower than 512 on UMA. PP therefore grows through `n_batch` while the physical
ubatch stays at the frozen baseline of 512, which also keeps the NPU GEMM shape
`M = ubatch = 512` identical at every PP point and makes the curve comparable
to PP512.

The cost of that choice is reported explicitly: `n_ubatch` in the CSV is the
number of ubatches per PP, so PP3072 at ubatch 512 issues six times as many NPU
GEMMs as PP512 at the same M.

## Instrumentation

Added for this phase, all gated by `GGML_SCHED_TRACE_STATS`:

* `[SCHED-SUMMARY]` in `ggml-backend.cpp` - node counts per backend, MUL_MAT
  count per backend (the CPU fallback count is `n_mm_cpu`), split count,
  backend transitions, copy count and bytes, and the V->X / X->V / C->X copy
  count, bytes and wall time, graph allocation time, submit time, scheduler
  synchronization time.
* `[VK-SUMMARY]` in `ggml-vulkan.cpp` - exact GPU busy time from the first
  `graph_compute` to the `synchronize` that retires it, split into submit and
  wait.
* `[XDNA-SUMMARY]` in `ggml-xdna.cpp` - NPU GEMM count split into gate / up /
  down / other, the observed M/K/N and whether it was uniform across the graph,
  plus host packing, XRT submit, kernel wait, device sync and copyback time
  summed over the GEMMs.
* `[LLAMA-UBATCH]` in `llama-context.cpp` - one line per ubatch with the
  physical token count, whether the graph was reused, graph build time,
  `ggml_backend_sched_alloc_graph` time and `set_inputs` time.

Per-copy and per-split detail stays behind the separate, much noisier
`GGML_SCHED_TRACE_COPY=1`, so the aggregate path stays cheap.

## Running it

```sh
# build first (Vulkan + XDNA)
cmake -B build -DGGML_VULKAN=ON -DGGML_XDNA=ON .. && make -C build -j8 bench_pp_scaling

# the XDNA kernel cache is keyed by M; compile every M the matrix will use
experiments/phase48/compile_kernels.sh

# interleaved control + heterogeneous, then the NPU-routed configuration
MODE=ab          experiments/phase48/run_pp_scaling.sh
MODE=hybrid_npu PP_LIST="512 1024 2048" experiments/phase48/run_pp_scaling.sh
experiments/phase48/collect_pp_scaling.py
```

Results and analysis are in [RESULTS.md](RESULTS.md).

`run_pp_scaling.sh` routes every NPU invocation through `npu_guard.sh`, which
waits until nothing else has `/dev/accel/accel0` open. The driver has no compute
queue sharing, so two processes on the NPU at once can wedge the device. Never
run two NPU benchmarks concurrently, and do not run `compile_kernels.sh` while a
benchmark is in flight: aiecc saturates all cores and the throughput numbers
become meaningless (it produced a spurious 87 tok/s reading before this was
caught).