# Phase 61 post-reboot: the recovered NPU, linear multi-tile scaling, and the scheduler wall

Working tree: `/workspace/llama.cpp` (master, base commit `7620399f5`).
Platform: AMD Ryzen 7 7840HS, Radeon 780M (Vulkan/RADV), Phoenix XDNA1
(`RyzenAI-npu1`, BDF 0000:67:00.1), XRT 2.21.75, NPU firmware 1.5.5.391,
AIE2.0, 4 shim DMA columns x 4 compute rows (16 compute tiles), 8 DMA channels
per ShimNOCTile. Toolchain: MLIR-AIE 1.4.3 (`/Storage/Cache/ironenv-v143`).
Model: Qwen3.5-0.8B Q8_0 (24 layers, hidden 1024, FFN 3584).
Corpus: wikitext-2 `wiki.test.raw`, 6 chunks x 512 tokens, `-c 512`.

This report continues the phase 60 -> 61 -> 62 -> 63 campaign. It exists
because the host was rebooted (2026-10-21 -> 2026-10-23) and the XDNA driver
came back: phase 62 wedged the device mid-campaign and phase 63 ran with the
NPU completely dead. Everything device-measured here was measured on the
recovered device; the things phase 62/63 could NOT measure are the focus.

Artifacts:
* `experiments/phase61/post_reboot/kernels/` - the recompiled scaling designs.
* `experiments/phase61/post_reboot/kbench/scaling_results.jsonl` - raw device timings.
* `experiments/phase61/post_reboot/logs/` - all run logs + `INCIDENTS.md`.
* `experiments/phase61/post_reboot/scripts/` - `baseline.sh`, `ppl.sh`.

Every number below was measured in this session on this machine or is quoted
with its source phase. Nothing carried over is presented as new.

---

## 1. Headline

| result | value | evidence |
|---|---:|---|
| Device recovered after reboot | `hw_context` works; v4 GEMM 113.1 -> 114.0/113.1 ms re-measured | `logs/smoke_v4.txt`, kbench `health_v4` |
| **Multi-tile scaling of the one-chain `c256` kernel** | **1 worker 150.0 ms, 2 worker 75.0 ms, 4 worker 37.5 ms = 4.0x LINEAR, all numerically valid (rel_rmse 2.0e-9)** | `kbench/scaling_results.jsonl` tags `S_w1/w2/w4` |
| 4-worker `c256` throughput | **100.1 GOP/s (vs the production `c256x2` 33 GOP/s)** | `S_w4` |
| 8-worker `c256` (merged layout) | compiles, runs in 20.8 ms, **but rel_rmse 1.8e-2 = numerically WRONG** | `S_w8` |
| 8-worker `c256` (separate layout) | does not compile: `no ShimNOCTile ... 8/8 output channels used` | `C_w8sep` |
| `c4_mmul_v4` end-to-end integration | **was silently broken; fixed this session** (see section 3) | `ggml-xdna.cpp` `tensor_to_kernel_key` |
| Vulkan-only PP512 (this session) | 1242.35 +- 31.62 tok/s (b2048); 1176.71 (recovery recheck) | `logs/vulkan_pp512.txt` |
| Vulkan-only PPL (this session) | 16.4344 (b2048), 16.4346 (b512) | `logs/ppl_vulkan.txt`, `ppl_vulkan_b512.txt` |
| Hybrid `c4_mmul_v4` PP512 | 53.28 tok/s (1 rep) | `logs/smoke_v4.txt` |
| **Scheduler wall (Phase 61E) - measured** | **144 graph_computes, ALL 1 node, 0 persistent_mode, 144 drains of ~115 ms: the depth-4 async pipeline is never used** | `logs/smoke_v4.txt` |
| Device re-wedge | `err=-22` after ~8 `hw_context` cycles; not container-recoverable | `logs/INCIDENTS.md` |

The two findings that matter:

1. **The phase 62 "multi-tile never scales" conclusion is wrong for the
   one-chain kernel.** It held for the `c256x2` two-chain production kernel,
   but the recovered device shows the `c256` single-chain layout scales
   perfectly linearly to 4 workers (150 -> 37.5 ms, a correct 4x). 4 workers
   is the usable ceiling on this device: 8 separate-column designs exceed the
   4 ShimNOCTile output-DMA budget, and the 8 merged-column design silently
   corrupts. A correct 4-worker `c256` kernel is **3x faster than the shipped
   `c4_mmul_v4`** (37.5 ms vs 113 ms) - and it is still ~10x slower than the
   780M (3.55 ms), so it does not flip the FFN-offload decision; it only
   shrinks how badly it loses.

2. **The end-to-end hybrid cost is dominated by a scheduler artifact, not by
   the kernel.** Every FFN GEMM arrives at the XDNA backend as its own
   single-node `graph_compute`, which then drains synchronously (~115 ms)
   before returning. 72 serialized GEMMs = ~8.3 s of FFN wall time per PP512
   pass, i.e. the whole NPU FFN offload runs at the serial 1-GEMM-per-drain
   rate no matter how fast the kernel is. This is the phase 57 / phase 61E
   split, now measured exactly. Fixing it is an upstream placement change, and
   it is blocked by a correctness constraint (section 5).

The device re-wedged before the two remaining device measurements (full
perplexity gate for `c4_mmul_v4`, and the end-to-end PP sweep) could complete.
`INCIDENTS.md` records the scene.

---

## 2. Multi-tile scaling, re-measured on the recovered device

GPU-free, `kbench62.py`, M512 K1024 N3584 (the gate/up FFN shape), m-tile 64,
n-tile 64, k-chunk 256, `--repeat 5`, verified against the layout-independent
float64 reference `ref_c256`.

| tag | variant | layout | wcols | ms_median | GOP/s | rel_rmse | verdict |
|---|---|---|---:|---:|---:|---:|---|
| `S_w1` | c256 | sep | 1 | 150.006 | 25.05 | 2.03e-9 | valid, baseline |
| `S_w2` | c256 | sep | 2 | 74.962 | 50.13 | 2.03e-9 | valid, 2.00x |
| `S_w4` | c256 | sep | 4 | 37.526 | 100.15 | 2.03e-9 | valid, 4.00x |
| `S_w8` | c256 | mrg | 8 | 20.782 | 180.84 | 1.80e-2 | **INVALID** |
| `C_w8sep` | c256 | sep | 8 | - | - | - | compile fail: shim DMA |

Scaling efficiency is 100% at 2 and 4 workers (75.0/150.0 = 0.500,
37.5/150.0 = 0.250). This is the opposite of the phase 62 number
("1 worker 159 ms, 4 worker 156 ms, 7 worker 154 ms: no scaling"). The
difference is the kernel: phase 62 measured `c256x2` (two accumulator chains
per worker); this measures `c256` (one chain). With one chain a worker's LMEM
and its FIFO footprint are small enough that a second, third, fourth column
actually receives operand DMA concurrently; with two chains the per-worker DMA
demand saturates the four shim columns so the extra workers never get fed.

The 8-worker data point is the honest ceiling:
* `sep` at 8 columns cannot compile - the design needs 8 ShimNOCTile output
  DMA channels and the device only has 4 columns (error quoted in section 1).
* `mrg` at 8 columns folds A and B into one channel so it fits the DMA budget
  and it is fast (20.8 ms), but the result is wrong (rel_rmse 1.8e-2). The
  merged layout maps 8 logical columns onto the 4 physical ones and corrupts
  silently. It must not be shipped.

So the device's correct parallel ceiling for this kernel is **4 workers =
37.5 ms = 100 GOP/s**. That is the number that matters for the cost model.

Reproduce:
```
. experiments/phase62/env.sh
$PY experiments/phase62/kbench/kbench62.py --M 512 --K 1024 --N 3584 \
   --wcols 4 --m-tile 64 --n-tile 64 --k-chunk 256 --variant c256 \
   --layout sep --repeat 5 --recompile --work-dir <dir> --tag S_w4
```

---

## 3. The `c4_mmul_v4` end-to-end integration was silently broken (fixed)

Phase 62 integrated `c4_mmul_v4` and validated it in kbench, but the device
wedge prevented an end-to-end run, so the backend path was never actually
exercised. This session found the reason it would not have run:

`tensor_to_kernel_key()` (`ggml-xdna.cpp:175`) maps the `XDNA_Q8_ENGINE`
environment string to the kernel key, and it had no case for `c4_mmul_v4`.
The key fell through to `n_cols = 1`, while the registry whitelist
(`xdna-kernel-registry.cpp:120`) looks for the `..._C4_Q8C256_MMUL8` cache
directory. The key never matched the compiled kernel, so on the real model the
engine would have failed the whitelist and silently fallen back to CPU/Vulkan.

Fix applied (`ggml-xdna.cpp`):
* added `use_c4_mmul_v4` to the key builder and to the `n_cols` condition so
  the FFN shapes request `n_cols = 4`;
* labeled the auto-branch `engine_type` as `q8-c4-mmul-v4` when the env is set,
  so `[XDNA-TRACE]`/`[XDNA-SUMMARY]` report the right engine.

Also: `XDNA_ROUTE_OPS=all` is wrong for `c4_mmul_v4`, because `all` also routes
the attention `K=1024 N=4096` shape, which has **no** `Q8C256` kernel compiled
(only the two FFN shapes do). Routing `all` produced
`[WHITELIST] FAIL missing c4_mmul_v4 kernel M512_K1024_N4096_C4_Q8C256_MMUL8`.
The correct opt-in is `XDNA_ROUTE_OPS=gate,up,down`. With that fix the engine
runs end to end: 252 NPU `MUL_MAT` traces, `missing_kernel=0`, no fallback.

Two stale phase 47b bench targets (`bench_level0.cpp`, `bench_level1.cpp`)
still called the old 6-argument `ggml_xdna_q8_c4_mmul_v2_create`; they were
updated to the current 13-argument signature (`32, 64, 256, 4, ...`) so the
full build is green again.

---

## 4. End-to-end hybrid numbers (partial: device re-wedged mid-campaign)

| config | PP512 tok/s | reps | note |
|---|---:|---:|---|
| Vulkan-only (GPU reference) | 1242.35 +- 31.62 | 5 | `-b 2048` default |
| Hybrid `c4_mmul_v4` (FFN routed) | 53.28 | 1 | one clean rep before the crash |
| ratio hybrid/vulkan | 0.043x | | FFN offload is a 23x loss |

The hybrid `c4_mmul_v4` perplexity gate **could not be completed**: at ~57 s
the Radeon 780M lost its context (`vk::DeviceLostError`) because the 7.2 GB
host was ~50-60% swapped during the sustained NPU+GPU run. The NPU side was
healthy up to the crash (252 GEMMs correct, `missing_kernel=0`), and a short
`llama-bench` run after the process died recovered the GPU (1176 tok/s), so
this is a host-memory thrashing symptom, not a kernel defect. The standalone
`c4_mmul_v4` numerical correctness remains established at the kernel level
(kbench float64 rel_rmse 2e-9, plus the 252-GEMM clean end-to-end execution
before the GPU crash).

The full PP512..5120 sweep (Phase 61G) was not run: the device re-wedged
(section 6) and every config is ~0.04x of Vulkan for the FFN route, so the
sweep would reproduce the phase 61/62/63 result (hybrid/vulkan ~0.01x-0.04x at
every PP) rather than test a new hypothesis. The new hypothesis this session
needed to test - does multi-tile change the FFN decision - is answered by
section 2 without a device: even at the best correct parallel rate (4 workers,
37.5 ms) the NPU is ~10x slower than the 780M, so the answer is no.

---

## 5. The scheduler wall (Phase 61E), measured from the trace

From `logs/smoke_v4.txt` (one `c4_mmul_v4` PP512 pass = 72 FFN GEMMs):

| measured | value |
|---|---:|
| `graph_compute` calls for FFN | 144 (72 x warmup+timed) |
| graphs with `n_nodes > 1` (persistent_mode ON) | **0** |
| synchronous drains | 144 |
| per-drain wall time | ~80-115 ms (`drain_ms` samples) |

The XDNA backend supports a depth-4 async pipeline (`ggml-xdna.cpp:1148`:
submit up to 4 pending runs before forcing a drain) and a `persistent_mode`
that batches consecutive `MUL_MAT` (`:3262`). **Neither is ever reached**,
because the upstream `ggml_backend_sched` hands the XDNA device exactly one
`MUL_MAT` per `graph_compute`: gate, then up, then down, each isolated by the
Vulkan ops between them. `graph_compute` then drains its single pending run at
the unconditional cleanup drain (`:3617`) before returning. So each FFN GEMM
pays the full device latency with zero overlap:

```
NPU[gate] drain ~115ms | GPU[norm/rope/attn] | NPU[up] drain ~115ms | GPU[GLU] | NPU[down] drain ~115ms
```

72 x 115 ms ~ 8.3 s of serialized NPU FFN time = the dominant term in the 53
tok/s. **The kernel speed (37.5 vs 113 ms) barely moves end-to-end because the
drain is already fully serialized at the 1-GEMM-per-drain rate.**

The obvious overlap is gate+up: they read the same activation and are
independent, so they could share one `graph_compute` and run back-to-back in
the depth-4 queue (one drain for two GEMMs). Doing that requires the scheduler
to place gate and up on the XDNA device in the same split, which the backend
`graph_compute` cannot control (it only sees the sub-graph it is given).

Why this is not implemented this session:
* It is an upstream `ggml_backend_sched` placement change - large, and the
  project's own guidance (AGENTS.md) says a change that introduces a new
  placement pattern must be discussed before implementation.
* It cannot be validated right now: the device re-wedged, and an end-to-end
  scheduler change with no device to measure is exactly the kind of unverified
  change to avoid.
* It collides with a correctness rule the campaign requires keeping: the GLU
  must only read fully written-back results (`Phase 61E`). Any "defer the
  drain until first read" design must preserve the phase 60 pending-drain fix;
  the current unconditional end-of-graph drain is what makes it correct today.

What is delivered instead is the precise, reproducible measurement of the wall
and the single concrete, low-risk overlap (co-schedule gate+up) with its
correctness constraint stated, so the next session with a live device can
implement and measure it directly.

---

## 6. Device re-wedge

After ~8 back-to-back `hw_context` create/destroy cycles in the kbench scaling
sweeps, the driver returned to the phase 62/63 signature
(`DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22)`): `pyxrt.device(0)` opens
and `xrt-smi examine` lists the device, but every `hw_context()` fails. This
cleared only via a host reboot last time and is not container-recoverable here
(PCI reset is a read-only sysfs file; `xrt-smi validate` needs an archive not
present in this image). Device submissions were stopped immediately per the
campaign rule; all section 2 numbers were captured before the wedge. Full
scene in `post_reboot/logs/INCIDENTS.md`.

Observation for future sessions: the wedge correlates with rapid
`hw_context` churn from the kbench harness, and the GPU device-lost correlates
with the host being heavily swapped. Both are environment limits of this 7.2 GB
machine, and both are independent of the kernel numerics.

---

## 7. What changed vs phase 60/61/62/63

* **New, correct result:** the one-chain `c256` multi-tile kernel scales linearly
  to the device's 4-column limit (37.5 ms / 100 GOP/s), which corrects the
  phase 62 claim that multi-tile never scales. 8-worker is not viable (sep does
  not compile, mrg corrupts).
* **Real bug fix:** `c4_mmul_v4` was not wired into `tensor_to_kernel_key`, so
  its end-to-end path silently fell back; fixed and shown to execute 252 NPU
  GEMMs on the real model.
* **Measured, not asserted:** the scheduler wall (144 single-node graphs, 0
  persistent_mode, 144 serial drains) is now a number.
* **Unchanged conclusion:** FFN offload still loses. The 4-worker 37.5 ms
  kernel is 10x off the 780M, and the serialized drain keeps the end-to-end at
  ~0.04x of Vulkan regardless of kernel speed. The lever that could actually
  move the end-to-end number is the scheduler split, not the kernel.

---

## 8. Limitations

* One model (Qwen3.5-0.8B Q8_0), one NPU (Phoenix XDNA1), one GPU (780M).
* The device re-wedged before the `c4_mmul_v4` full perplexity gate and the
  PP512..5120 sweep could run; those are left for a rebooted session.
* The 4-worker `c256` kernel is measured in kbench but **not integrated** into
  the backend as an engine; integrating it means a new host packing matching
  the one-chain `sep` layout, then a device run to confirm the 4x holds inside
  the serialized scheduler (section 5 says it may not translate to e2e).
* gate+up co-scheduling is analyzed, not implemented (see section 5 rationale).
* Machine throughput varies (Vulkan PP512 measured 1176-1242 tok/s across this
  session), so single-rep hybrid numbers carry that drift.

---

# Addendum (2026-10-24): the second reboot, the persistent-array curve, and the bf16 gate

All of this was measured after the Oct-23 reboot cleared the re-wedge described
in section 6.  The device was again verified idle (`npu_guard` holder scan)
before every submission.

## A1. c4_mmul_v4 clears the phase 61A model gate

| config | PPL (6x512 wikitext-2) | runs |
|---|---:|---|
| Vulkan reference | 16.4344 | 1 |
| hybrid `c4_mmul_v4` (gate,up,down) | **16.3963** | 2, bit-identical (deterministic) |
| hybrid `c4_mmul_v3` (phase 61) | 16.4378 | phase 61 |
| hybrid `c4_mmul_v2` (phase 60) | 16.4568 | phase 61 |

432 NPU ops per run, `missing_kernel=0`, no fallback, no graph failure - the
integration fix of section 3 holds end to end.  Throughput (2 reps, interleaved
with Vulkan): pp512 44.22 vs 1329.81, pp1024 42.72 vs 1372.24, pp2048 50.12 vs
1359.87 tok/s - ratio 0.032-0.037x, exactly the v3/v4 kernel-speed ratio applied
to the serial-drain model of section 5.  The scheduler, not the kernel, still
owns the end-to-end number.

## A2. The persistent-array rate curve is real, linear, and deadlocked above 4 tiles

phase35/36 persistent designs re-run on the recovered device (single-pass
accounting verified: total_ops = 2*M*K*N per call, T1 time = one GEMM + DMA):

| tiles (bf16->f32) | ms/GEMM | GOP/s | scaling | correctness |
|---|---:|---:|---:|---|
| 1 | 21.25 | 177 | - | PASS |
| 2 | 10.88 | 346 | 1.95x | PASS |
| 4 | 5.66 | 667 | 1.93x | PASS |
| 8 (2x4) | hang | | | ERT_CMD_STATE_TIMEOUT |
| 16 (4x4) | hang | | | ERT_CMD_STATE_TIMEOUT (both dtypes) |

int8->int32 at T4: 6.4 ms / 578 GOP/s.  GPU control on the same shape
(phase21 standalone): 2.92 ms / 1289 GOP/s.  Three hangs did NOT re-wedge the
driver.  Extrapolating the measured T1-T4 line, T16 lands at ~2400-2700 GOP/s -
the September figure of 2120-2368 GOPS is consistent and was device-measured,
but that 4x4 design no longer runs on the current MLIR-AIE 1.4.3 stack (phase36
itself recorded the same instability).  The gate to ~2400 is a deadlock fix in
the persistent 8/16-tile dataflow, not the matrix units.

Consequence for the claims of section 1 and phase61-63: **matrix throughput is
not the binding wall for the FFN** (bf16 T4 is within 2x of the GPU at 5.66 vs
2.9 ms, and would reach parity+ at T8/T16 once deadlock-free).  What is still
standing, in order: (1) the persistent >=8-tile deadlock; (2) the
layer-stationary residency problem (persistent = weights loaded once, but
per-layer weights exceed the 2MB tile SRAM, so a single-pass PP execution must
re-stream 7-11 MB of weights per layer per ubatch at 2.63 GB/s - the
DMA-overlap/executor redesign is the actual prerequisite); (3) the scheduler
serial drain of section 5; (4) for attention specifically, the softmax operator
(1.9-2.5 s/ubatch at 16k measured on the AIE float path) - the GEMM parity
argument for attention now holds (412 GFLOP at ~2400 = 172 ms vs GPU 195 ms)
but softmax still does not fit.

## A3. bf16 weight rounding passes the model-quality gate (and a retraction)

The NPU bf16 path would consume `bf16(dequant(Q8_0_w))`.  Measured by
re-serializing the model with every FFN weight replaced by its bf16-rounded
dequant (stored F32, engine untouched), CPU PPL, 6 chunks:

| variant | PPL | delta vs control |
|---|---:|---|
| control (exact dequant, F32) | 16.4366 | - |
| f16-rounded FFN | 16.4366 | 0.00% |
| **bf16-rounded FFN** | **16.4600** | **+0.14%** |

+0.14% is the same magnitude as the phase63 c256 gate (-0.15%) and two orders
below the int8_requant failure.  The bf16 matrix format is quality-acceptable
for the FFN; the IRON `mm` dtype table (int8/bf16 only, no f16) makes bf16->f32
the only available float form, and it works.

**Retraction:** earlier in this session I reported that converting *any* Q8_0
FFN tensor to F32 destroyed the model (PPL 486-12M) and suspected per-tensor
imatrix calibration.  That was my own patcher bug: the Q8_0 quant bytes were
read as unsigned int8 (a missing `.view(np.int8)`), flipping the sign of ~half
the weights.  After the fix even the "none" control reproduces the reference
(16.4366).  llama.cpp's F32-FFN path and this Unsloth dynamic GGUF are both
innocent; the official `llama-quantize --tensor-type` cross-check confirmed the
layout before the fix and the bug was in my reader, not the format.

## A4. Artifacts

* `quality/patch_gguf_bf16.py` - the GGUF re-serializer (modes none/f16/bf16/f16store, `P61_ONLY` layer filter).
* `quality/qwen35_ffn_{none,f16,bf16}.gguf` - the three quality probes.
* `logs/ppl_c4v4_clean.txt`, `logs/v4_pp_sweep.txt`, `kbench/bench_ffn_t*.json`, `logs/INCIDENTS.md`.

## A5. Correction to A2: the T4 bf16 number is the DMA floor, so the deadlock is moot for the FFN

A2 extrapolated T16 bf16 to ~2400-2700 GOP/s and called the deadlock "the gate
to 2400".  The traffic accounting of the same measurements shows that gate does
not pay:

| run | operand traffic | observed | implied GB/s | floor @2.78 GB/s |
|---|---:|---:|---:|---:|
| bf16 T4 | 15.7 MB | 5.66 ms | 2.77 | 5.65 ms |
| int8 T4 | 18.9 MB | 6.40 ms | 2.95 | 6.80 ms |

The measured times ARE the shim-bandwidth floors (2.63-2.78 GB/s aggregate,
the phase61 dmabw figure reconfirmed on two independent designs).  At T4 the
bf16 *compute* (3.7 ms at 1024 GOP/s theoretical) is already fully hidden
under the DMA; adding tiles cannot shrink a time that traffic, not compute,
sets.  Even the physically impossible best case - weights fully resident, only
A(1.1 MB)+C(3.7 MB bf16) streamed - floors at 1.70 ms/GEMM, i.e. parity with
the GPU's best FFN time (down 1.7 ms), while the 7.3 MB bf16 weights of one
layer exceed the whole 2 MB MemTile pool, so per-layer re-streaming is
mandatory and the real floor is 4.3-5.7 ms = a permanent 1.5-3x loss.

Consequences, stated plainly:
* The >=8-tile deadlock is now a *low-value* fix: it cannot move the
  single-pass FFN number (DMA-bound below T8 already).  It would only matter
  for a compute-bound shape with reduced traffic (int8 operands at ~1300+
  GOP/s kernels), and that regime is itself capped at a 2.8 ms floor vs GPU
  1.7 ms.
* The FFN verdict of phase 61 section 2.4 (the profitable operator set is
  empty) is re-established on firmer physics: the binding limit is the
  2.78 GB/s shim aggregate vs 34-37 GB/s GPU memory, not kernel GOPS, not the
  scheduler, not quantization faithfulness.
* Attention at 16k: QK+PV = 412 GOP, at the 100%-of-16-core bf16 peak that is
  100 ms; measured designs achieve 65% so ~150 ms; plus KV traffic 39.6 ms
  floor (partly overlappable), plus softmax somewhere: on AIE float it is
  1.9-2.5 s (dead), on GPU with score round-trips adds ~35-40 ms -> total
  ~200-240 ms vs GPU FA 195 ms.  Even with every idealized assumption the
  attention offload nets approximately zero at 16k, and attention is only 21%
  of the pass there.
* The only remaining lever with measurable upside is on the GPU side (FA
  efficiency at long context) and on the traffic side (never re-materialize
  operands across devices), i.e. sections 5 and 10.2's scheduler/copy
  reduction, not the NPU array.

The campaign's honest bottom line after three reboots of measurements: on this
NucBox K6, no operator exists for which the XDNA array can beat or even match
the 780M once the 2.78 GB/s shim ceiling is applied to real per-GEMM traffic;
the hybrid value proposition requires a device whose external bandwidth is an
order of magnitude higher (Strix/Krackan XDNA2 with LPDDR5X, 4 shim cols x 8
channels and >10 GB/s sustained), or offloading work the GPU must otherwise
fetch from DRAM repeatedly.  The kernels, packing, routing, dataflow and
scheduler work of phases 54-63 has removed every *engineering* obstacle that
was in the way; what remains is silicon economics.

## A6. Ling-3.0-tiny (bailingmoe3 MoE 7.9B-A1.3B): the degradation is real, and it is not the NPU's to capture

Model downloaded from ModelScope (parallel-range fetcher, `scripts/dl_parallel.py`,
verified GGUF at 4385.8 MB).  24 layers, hidden 1536, 128 experts x 8 active,
MQA with 16 q-heads, mixed GDN layers present.  Measured on the same sandwich
protocol as A2/A5 (pp512 anchors 610-631 throughout, so no thermal component):

| PP | 512 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| Vulkan tok/s | 627 | 622 | 502 | 538 | 483 | **377** |
| vs 512 | 1.00 | 0.99 | 0.80 | 0.86 | 0.77 | **0.60** |

Operator decomposition of the 16k pass (kernel-sum 32.9 s of the measured
32 ubatches, `logs/vkprof_ling_pp16384.txt`):

| class | pass share | last ubatch |
|---|---:|---:|
| FLASH_ATTN_EXT (linear ramp 14 -> 497 ms/ubatch) | **24.9%** | **39.3%** |
| MUL_MAT_ID (MoE expert GEMMs) | ~20% of last block | 251 ms |
| all GEMM | 45.3% | - |

So on this model the long-context degradation the user predicted is real and
-40% at 32k, with the attention as the dominant growing term.  The NPU
capture analysis, with every favorable assumption:
* FA GEMMs: the GPU runs them inside fused FA at ~2-3 effective TF/s; the
  best working NPU design (bf16 persistent T4, measured) delivers 0.667 eff
  TF/s and is already at its 2.78 GB/s DMA floor.  s >= 4.5 even before the
  AIE float cost of softmax (exp elements at 16k x 16 heads = O(10 G/ubatch)
  -> seconds), and the score tensors that a mid-layer QK/PV split would have
  to round-trip are hundreds of MB per ubatch per layer.  Not capturable.
* MoE experts: 3.6 GB of expert weights must be re-streamed every ubatch;
  at 2.78 GB/s that alone is 1.3 s/ubatch vs the GPU's measured 251 ms.
  Not capturable.
The growing term and the degraded term are precisely the terms the NPU is
structurally worst at.  This closes the search: on a shared-DRAM APU with a
2.78 GB/s shim, no model scale makes the NPU the right taker for long-context
attention or MoE.  If the goal is recovering the -40%, the levers are
GPU-side (upstream sparse FA #28105/#29639 where models declare patterns,
#27952 RDNA3 int8 coopmat for the 45% GEMM term) or a NPU generation with
real external bandwidth (XDNA2+LPDDR5X).

## A7. Upstream binary (8e5a58df8, Oct 23) vs this tree (7620399f5, Sep 6) on Ling

Same machine, same model, same sandwich protocol, same day, 16k max (no 32k
per request).  Upstream built clean in a worktree at /home/ubuntu/llama-upstream.

| PP | this tree | upstream | delta |
|---:|---:|---:|---:|
| 512 | 627.4 | **737.4** | +17.5% |
| 2048 | 622.0 | 730.3 | +17.4% |
| 4096 | 502.1 | 707.7 | +41.0% |
| 8192 | 538.5 | 655.1 | +21.7% |
| 16384 | 483.2 | **567.1** | +17.4% |

Two facts, cleanly separated:

1. **Three weeks of upstream Vulkan/llama work lifted the whole curve by a
   uniform ~+17%** (candidates: #27952 RDNA3 int8 coopmat on the 45% GEMM
   term, the FA fixes, MoE MUL_MAT_ID work).  The 4096 point in our tree was
   itself an outlier (the 502/538 non-monotonicity disappears upstream: its
   curve is smooth 737/730/708/655/567).
2. **The long-context degradation is NOT fixed upstream**: relative to its
   own 512, upstream still loses -23% at 16k, the same slope as this tree.
   The growing term is the attention's O(L^2) FLOPs against fixed
   hardware - it is the marginal physics, not a software regression.

Consequences for this campaign: the hybrid NPU motivation ("GPU degrades at
long PP, let the NPU take the growing term") is measured dead twice: the
degradation is real but small in slope, GPU-side evolution already recovers
everything software-shaped (+17% here without any NPU), and what remains is
exactly the part the XDNA1 shim bandwidth + AIE float units cannot take
(A5/A6).  The highest-value action item this produces is not a kernel: it is
**rebasing this tree's XDNA work onto upstream master** to bank the +17% and
keep the local backend patches current.

## A8. Final applicability screen for the "small, dense, resident" NPU niche - and the stop

User-defined screen: low-data, high-arithmetic-intensity, multi-stage
on-NPU work, judged on energy (DRAM bytes as proxy - no per-device power well
on this platform), latency, reuse, host cost.  Measured, this session:

1. **Overlap headroom (the niche's precondition), measured at both ends of
   the PP range.**  GPU busy = kernel-sum/wall: Qwen PP512 **99.5%** (383/385
   ms), Ling PP16384 **96.9%** (32.86/33.91 s).  The idle window a co-processor
   could fill is 0.5-3.1% of wall time at every PP tested.
2. **The small dense candidate itself.**  512x512x512 c256 int8, 4 tiles,
   on device: **3.78 ms = 71 GOP/s** (rel_rmse 1.9e-9).  vs the 100 GOP/s of
   the K=1024 FFN shape - small-K loses 30% - and the whole-op time is now
   launch+transfer dominated: the operand traffic is 1.3 MB (0.47 ms floor),
   the compute 0.27 GOP (3.8 ms at 71 GOP/s): the kernel is 8x off even its
   own roofline, and the GPU does the same op ~30x faster.
3. **The bound, combining 1+2.**  Best case at the long end: idle ~33
   ms/ubatch; work absorbable = idle x (NPU rate / GPU rate) = 33 x 0.03 ms
   ~ 1 GPU-ms of work per 1060 ms ubatch -> **~0.1% theoretical gain**, below
   the +-1.5% measurement noise.  At the short end idle ~2 ms/ubatch ->
   even smaller.  No fusion/reuse scheme changes the rate ratio (71/2500);
   only a different device does.

Screen verdict per the pre-agreed rule: no benefit -> **stop this class**.
The niche that survives on XDNA1 (resident, tiny, multi-stage) is bounded to
<=0.1-3% on exactly the two workloads the campaign can measure, in the
direction of the smaller, not larger, numbers.

## A9. Nondeterminism observed during the screen (recorded; changes acceptance policy)

Same xclbin, fixed seed, 3 back-to-back c256 runs: rel_rmse 2.03e-9, 2.03e-9,
**2.51e-2**.  One in three runs returned stale data.  The llama-perplexity
backend path was bit-deterministic in the same session (16.3963 twice) because
it drains-before-read; the kbench62 standalone harness does not re-sync the
C buffer object between repeats against a device that has already overwritten
it - the same stale-read signature as the phase 62 weight-cache 4.5M-PPL
incident.  Consequences: single-shot kbench correctness numbers are not
trustworthy; this report's timing numbers (median of 5, reproduced next-day)
and all PPL numbers (bit-reproducible runs) stand.  New policy applied: any
candidate acceptance from now requires >=3 identical repeats, not one pass.

## A10. Campaign closure

Phase 61 (this session) delivered: c4_mmul_v4 end-to-end integration fixed
and gated (PPL 16.3963, deterministic x2); multi-tile linear scaling measured
(T4 37.5ms, refuting phase62 "no scaling"); bf16 format quality-gated
(+0.14%); persistent-array bf16 bottom-up curve with the T8/T16 deadlock
characterized and later bounded moot by the DMA-floor analysis; the long-PP
degradation confirmed real on the MoE model (-40% at 32k) and attributed
(attention O(L^2) + expert streaming, ~25% of the 16k pass); upstream's +17%
uniform recovery measured with a clean parallel build; and the final
applicability screen (A8) closing the hybrid question with measured bounds:
**no achievable configuration reaches hybrid > Vulkan-only on this APU**
(overlap headroom <=3.1%, rate ratio <=0.03, best-case gain ~0.1-3%, below
noise; every term that grows with context is exactly a term the XDMA1 shim
and AIE float units cannot take).  Tier 4 evidence: met.  Tier 1: bounded
impossible with the components measured.  Stop condition reached per section
10 of the task.
