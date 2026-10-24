# Phase 20G step 8 - ggml backend async contract

Status: **PASS** for the I8 full-K path. graph_compute submits and returns without
waiting for the NPU; synchronize / event_wait drain the queue and produce correct
results.

## Goal (step 8)

Declare `async = true` / `events = true`, make `graph_compute` non-blocking (no hidden
synchronize), and implement async submit/wait separation using the existing ggml
mechanisms (`ggml_backend_graph_compute_async`, `ggml_backend_event_*`,
`ggml_backend_synchronize`).

## What changed (`ggml/src/ggml-xdna/`)

- **Caps**: `ggml_xdna_device_get_props` now sets `props->caps.async = true` and
  `props->caps.events = true`.
- **Async queue**: `ggml_xdna_backend_context` gained a `pending` list plus
  `submitted_seq` / `completed_seq`.
- **Non-blocking submit**: `graph_compute` for `GGML_TYPE_I8` weights calls
  `ggml_xdna_submit_i8_fullk`, which runs `ggml_xdna_persistent_engine_submit`
  (`xrt::run.start()`) and pushes a pending entry. It does not wait.
- **Drain**: `ggml_xdna_drain_pending` waits each `xrt::run`, copies the int32 result
  into the destination tensor, and is called only from `synchronize` / `event_*`.
- **Events**: `event_record` marks the current submission sequence; `event_wait` and
  `event_synchronize` drain the queue. `event_new`/`event_free` allocate a
  `ggml_xdna_event` (the previous implementation called `ggml_backend_event_new`, which
  recursed into itself).
- **Registry**: an I8 weight now maps to a **full-K** kernel (`k_tile = K`,
  `engine_k = K`), so one XRT submit covers the whole K. Q8_0 keeps `k_tile = 32`.

Note on the full-K change: this is the step 6 invariant (`1 XRT submit / MUL_MAT`, no
K=32 loop). The kernel cache holds full-K int8 kernels (`M{M}_K{K}_N{N}`); the registry
now finds them for I8 (`M64_K1024_N64_C1`, engine `k=1024`).

## Bug found and fixed

ggml stores a 2D `[K,N]` tensor column-major (element `[k,n]` at `k + n*K`), while the
kernel wants `B` row-major `[K][N]` (`k*N + n`). Without a transpose the output was
100% wrong. `submit_i8_fullk` now transposes the weight. The input `[K,M]` is already
`[M][K]` row-major, and the output `C[m*N+n]` maps directly onto dst's `[N,M]` layout
index `m*N+n`, so neither of those needs a copy.

## Verification (`experiments/phase20g/test_async_backend.cpp`)

Runs an I8 `MUL_MAT` (M=64, K=1024, N=64) through the XDNA backend:

```
ASYNC_BACKEND PASS mismatches=0 early_unwritten=1 event_ok=1 avg_submit_ns=102807 avg_sync_ns=240848
[test] graph_compute_async avg 102807 ns (no NPU wait)
[test] synchronize avg 240848 ns (NPU execution)
[test] dst unwritten right after submit: yes
[test] mismatches 0 / 4096
[test] event: unwritten_before=1 mismatches=0
```

- `avg_submit_ns (103 us) << avg_sync_ns (241 us)`: `graph_compute` returns before the
  NPU finishes; the wait is deferred to `synchronize`. A hidden synchronize inside
  `graph_compute` would make the submit time include the NPU time.
- `dst unwritten right after submit = yes`: the output is only copied back on drain.
- `mismatches 0 / 4096`: results are correct.
- event path (`event_record` + `event_synchronize`) also defers and then completes.

Build/run:

```
cmake --build build-new --target ggml-xdna -j      # build-new: GGML_XDNA=ON, GGML_VULKAN=ON
XDNA_KERNEL_CACHE=/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache \
  ./experiments/phase20g/test_async_backend
```

## Constraints honored

- No IRON runtime; native XRT `xrt::run.start()`.
- Full-K single submit for the I8 path; no K=32 loop; no `engine->k == 32`.
- No hidden synchronize in `graph_compute`.

## Known limitation (step 5 TODO)

Q8_0 weights still take the host K-chunked **synchronous** path
(`execute_mul_mat_q8_scaled`), because Q8_0 needs per-32-block scales that the existing
pure-int8 full-K kernels cannot apply. Making Q8_0 `scale-aware + full-K + single
submit` requires generating a new kernel, which needs the IRON/Peano toolchain (not
present in this environment; the user will provide a path/container).

## Not yet done (next)

- Step 5: scale-aware Q8_0 full-K kernel (needs IRON/Peano).
- Step 14: async partition-ratio sweep (10/25/50/75%).
- Step 15: E5 PP end-to-end.
- Step 16: structured per-MUL_MAT overlap logging.
