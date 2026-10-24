# Phase 20G step 9 - minimal Vulkan + XDNA async overlap proof

Status: **PASS** (overlap > 0 in 30/30 measured iterations; parallel critical path < serial critical path).

## Goal (step 9)

With two independent execution paths in one process:

```
G = Vulkan GEMM
N = XDNA GEMM
```

capture submit/completion timestamps for both and prove

```
overlap_ns = min(vk_done, xdna_done) - max(vk_submit, xdna_submit) > 0
```

i.e. the two backends really run concurrently, not `Vulkan end <= XDNA start`.

## Environment

- GPU: `AMD Radeon 780M Graphics (RADV PHOENIX)`, Vulkan 1.4, `timestampPeriod=10.02 ns`
- NPU: `RyzenAI-npu1` (`/dev/accel0`), XRT 2.21.75
- Host: linux/amd64

## Method

Single process, no IRON runtime anywhere.

- **Vulkan path** (`overlap_proof.cpp::VkGemm`): compute GEMM (`gemm.spv`) recorded into a
  command buffer, submitted with `vkQueueSubmit`, waited with `vkWaitForFences`. The submit
  and wait are split so XDNA can be dispatched in between.
- **XDNA path** (`overlap_proof.cpp::XdnaGemm`): native XRT. `xrt::run::start()` returns
  immediately (no wait). `run.wait()` runs strictly later. Kernel is the cached
  `M64_K1024_N64` design (`design.xclbin` + `design.insts.bin`), 8-arg ABI, `K=1024` in a
  single submit - no K=32 loop.
- **Wait separation**: each path is waited on its own thread so both completion instants
  are observed independently; neither wait blocks the other. This avoids the earlier
  measurement bug where a serial `vk.wait()` followed by `xdna.wait()` inflated `xdna_done`.
- **CPU off the critical path**: the XDNA host copy-in (`memcpy` + BO `sync`) is done
  *before* `t0`; timestamps cover only device dispatch -> completion.

## Timeline (representative parallel iteration, `vk 320^3`, `--iters 30`)

```
iter  xdna_submit  vk_submit  xdna_done  vk_done   overlap   critical
29       7.4 us     30.3 us    278.9 us   826.7 us  248.6 us  819.3 us
```

- XDNA submit precedes Vulkan submit by ~23 us.
- XDNA completes at ~279 us, well inside the Vulkan window that ends at ~827 us.
- `overlap = xdna_done - vk_submit = 248.6 us > 0`.
- `xdna_state = 4` (`ERT_CMD_STATE_COMPLETED`).

## Aggregate (`vk 320^3`, 30 iterations each)

| mode | mean critical path | overlap > 0 | xdna_dur (mean) | vk_dur (mean) |
|------|-------------------|-------------|-----------------|---------------|
| parallel (async) | **819 us** | 30/30 | ~260 us | ~790 us |
| serial (baseline) | **1207 us** | 0/30 | ~253 us | ~946 us |

Parallel critical path is **~32% lower** than the serial Vulkan-then-XDNA baseline, and
every parallel iteration shows positive overlap. Raw CSV:
`results/step9_parallel_vk320.csv`, `results/step9_serial_vk320.csv`.

## XDNA correctness (same kernel / ABI)

`xdna_smoke.cpp` runs the identical native XRT path (M=64 K=1024 N=64, single submit) and
checks against a CPU int32 reference:

```
[smoke] start_ns=28382 wait_ns=633368 state=4
[smoke] mismatches 0 / 4096
SMOKE PASS mismatches=0 state=4
```

So the async `start()`/`wait()` separation produces correct results, not just "returned
without waiting".

## Constraints honored

- No IRON runtime on the execution path - XRT calls are made directly.
- `xrt::run.start()` with no immediate `wait()`; wait is separate.
- Single XRT submit per XDNA GEMM; no `K=32` per-block loop; no `engine->k == 32`.
- CPU copy-in is outside the measured critical path.
- Vulkan and XDNA are not serialized.

## Not yet done (next steps)

- Vulkan and XDNA run *different* GEMMs here. Step 11 requires the same `M x K x N` split
  into N-partitions (`N0` Vulkan, `N1` XDNA, `N0 + N1 = N`) whose output equals the full
  Vulkan GEMM.
- The benchmark harness is a standalone program, not yet the ggml backend async contract
  (step 8: declare `async`/`events`, real `graph_compute_async`, `event_record`/`event_wait`).
- Per-MUL_MAT structured logging (step 16) not yet emitted.
