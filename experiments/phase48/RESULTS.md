# Phase 48 results: heterogeneous PP scaling, PP512 -> PP5K

Model `models/Qwen3.5-0.8B-Q8_0.gguf` (Q8_0, 24 layers, n_embd 1024),
`Vulkan0` = AMD Radeon 780M (UMA), `AMD Phoenix XDNA1 NPU` with
`XDNA_Q8_ENGINE=c4_mmul_v2` (1 row x 4 cols, m_tile 32, n_tile 64, k_tile 256,
Q8_0 per-32 scales). Physical ubatch 512 at every point, PP grows through
`n_batch`. Machine: 16 cores, 7 GB RAM shared with the UMA iGPU.

Raw logs in `results/`, collected tables in `results/pp_scaling.csv` (full PP
matrix, 2 interleaved rounds) and `results/precise/pp_scaling.csv` (4
interleaved rounds, 9 measured reps, at PP512/1024/2048/5120).

## 1. Headline

The precise run is the authoritative throughput measurement: 4 interleaved
vk/hybrid rounds per point, 2 warmup + 9 measured prompt phases each, medians
below. The ratio is the median of the 4 paired per-round ratios.

| PP | Vulkan tok/s | Hybrid tok/s | Ratio | NPU FFN GEMMs | splits | trans | copies | copy MB | GPU ms/ubatch | NPU ms/ubatch | CPU fb | correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 1314.5 +-10.9 | 1310.2 +-7.1 | 0.983 | 0 / 72 | 2 | 1 | 8 | 2.52 | 436.7 | 0 | 0 | ok |
| 1024 | 1308.7 +-15.7 | 1289.2 +-4.1 | 0.983 | 0 / 72 | 2 | 1 | 8 | 3.02 | 444.5 | 0 | 0 | ok |
| 2048 | 1314.1 +-5.7 | 1294.9 +-16.9 | 0.980 | 0 / 72 | 2 | 1 | 8 | 4.02 | 460.6 | 0 | 0 | ok |
| 5120 | 1315.7 +-2.5 | 1307.4 +-1.4 | 0.993 | 0 / 72 | 2 | 1 | 8 | 7.02 | 472.7 | 0 | 0 | ok |

The full 8-point matrix (`results/pp_scaling.csv`, 2 rounds, 5 reps) is flat and
structurally identical everywhere:

| PP | Vulkan tok/s | Hybrid tok/s | Ratio | splits | trans | copies | copy MB | GPU ms/ubatch | CPU fb | correct |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 1240.2 | 1232.1 | 0.99 | 2 | 1 | 8 | 2.52 | 436.7 | 0 | ok |
| 768 | 1201.8 | 1152.5 | 0.96 | 2 | 1 | 8 | 1.38 | 324.4 | 0 | ok |
| 1024 | 1152.6 | 1198.3 | 1.04 | 2 | 1 | 8 | 3.02 | 444.5 | 0 | ok |
| 1536 | 1164.4 | 1192.3 | 1.02 | 2 | 1 | 8 | 3.52 | 445.0 | 0 | ok |
| 2048 | 1166.8 | 1182.0 | 1.01 | 2 | 1 | 8 | 4.02 | 460.6 | 0 | ok |
| 3072 | 1144.6 | 1141.9 | 1.00 | 2 | 1 | 8 | 5.02 | 410.5 | 0 | ok |
| 4096 | 1157.4 | 1243.3 | 1.07 | 2 | 1 | 8 | 6.02 | 450.7 | 0 | ok |
| 5120 | 1171.9 | 1093.9 | 0.93 | 2 | 1 | 8 | 7.02 | 472.7 | 0 | ok |

The 2-round sweep scatters by +-7% run to run on this host (absolute tok/s
drifts 1254 -> 1097 across the sweep), which is why the ratio there jumps
around 1.00. The precise run, with 9 reps and 4 paired rounds, resolves it:
over 16 pairs the hybrid/vulkan ratio has median 0.985, mean 0.992, range
0.967-1.053, and it does **not** trend with PP (0.983 at PP512, 0.993 at
PP5120).

Hybrid = Vulkan + XDNA registered with `GGML_XDNA_QWEN_PP=1`, the PHASE47B
frozen baseline.

**Conclusion: the heterogeneous architecture is flat from PP512 to PP5K. There is
no cliff.** Vulkan throughput is 1314 tok/s at PP512 and 1316 tok/s at PP5120.
Splits (2), backend transitions (1), copy count (8), node count (1391), MUL_MAT
routing (187 on the GPU) and CPU fallback (0) are identical at every PP. GPU
busy time per ubatch is 324-473 ms with no trend against PP. Peak RSS is flat
at 175 MB (Vulkan) and 243 MB (hybrid) from PP512 to PP5120. Registering the
NPU costs 1.5%, and that cost does not amortize or grow with batch.

## 2. The finding that matters: the baseline runs zero NPU work

The PHASE47B baseline number, 1241.73 tok/s, is reproduced here as
1232.1 tok/s at PP512 (full matrix) and 1310.2 tok/s (precise run) - and it
executes **no NPU GEMM at all**:

```
[SCHED-SUMMARY] n_mm_gpu=187 n_mm_xdna=0 n_mm_cpu=0 n_splits=2 n_transitions=1 copy_bytes=2637832
```

Zero `[XDNA-SUMMARY]` lines, `n_nodes_xdna=0`. The "27.5x over old hybrid"
improvement was the removal of NPU interference from the critical path, not NPU
acceleration. The NPU is registered and idle.

Why: `ggml_backend_sched_backend_id_from_cur()` only sends a MUL_MAT to
`op_offload` when its weight lives in a **host** buffer. With all weights on
Vulkan the FFN weights are device buffers, `op_offload` never fires, and no FFN
GEMM is routed to the XDNA. That is why the whole PP512->PP5K curve above has
zero NPU coverage: it is Vulkan throughput.

## 3. Routing the FFN to the NPU: measured, and it costs 14x

`GGML_XDNA_QWEN_PP=1` alone also gates a 1 GB weight budget, which puts 6 of 24
layers in NPU buffers. Reproducible over 4 identical runs:

| config | tok/s | NPU FFN GEMMs | splits | trans | copies | copy MB | correct |
|---|---|---|---|---|---|---|---|
| Vulkan only | 1314.5 | 0 / 72 | 2 | 1 | 8 | 2.5 | reference |
| `QWEN_PP=1`, gated shapes | 1310.2 | 0 / 72 | 2 | 1 | 8 | 2.5 | ok |
| `QWEN_PP=1`, 1 GB budget | 87.9 | 12 / 72 | 72 | 38 | 87 | 197.3 | ok |

Two separate costs, attributed from `[XDNA-SUMMARY]`:

* **Placement, 14.2x of the regression is structural.** 70 extra splits and
  190 MB of extra copies appear because every weight in an NPU buffer is copied
  to Vulkan at graph build time. 68 non-GEMM nodes additionally fall to the CPU
  backend (`n_nodes_cpu` 71 vs 3).
* **Kernel, the dominant term.** Per GEMM at M=512, measured on the device:

| shape | `run_wait_ms` | GOP/s |
|---|---|---|
| K1024 N3584 (gate, up) | 425.8 | 8.8 |
| K3584 N1024 (down) | 396.5 | 9.5 |

  12 GEMMs x 426 ms = 5000 ms of `npu_run_wait_ms` per ubatch against 508 ms of
  GPU busy: **91% of the ubatch wall is the NPU kernel**. One NPU FFN GEMM
  costs more than a whole Vulkan PP512 (405-450 ms).

Both costs scale linearly with M, and so does Vulkan's, so the ratio is
PP-independent. Measured directly with the NPU FFN routed:

| PP | NPU-routed tok/s | wall ms | tok/s ratio to Vulkan |
|---|---|---|---|
| 512 | 88.19 | 5805.8 | 0.067 |
| 1024 | 88.23 | 11606.4 | 0.073 |
| 2048 | 88.39 | 23169.8 | 0.067 |

Flat tok/s, exactly linear wall time, all three verified against the Vulkan
md5. There is no batch size at which the current NPU kernel wins, and no batch
size at which it stops scaling cleanly.

### Software fix applied

`ggml_xdna_device_get_memory()` tied the 1 GB NPU weight budget to
`GGML_XDNA_QWEN_PP`, so the documented routing opt-in silently moved a third of
the model onto a device whose kernel is 200x too slow. The two decisions are
now separate: `GGML_XDNA_QWEN_PP=1` routes ops, `GGML_XDNA_WEIGHTS=1` places
weights. With only the routing opt-in, the flag is deterministic and lands on
the 2-split, 1232 tok/s state instead of the 72-split, 88 tok/s state, and the
NPU-routed configuration stays reachable for measurement.

Note this does not restore NPU coverage in the baseline; coverage there was and
remains 0 / 72, and the row above reports it explicitly.

## 4. Why the ubatch is pinned to 512

The Vulkan compute buffer scales with the physical ubatch and this host has 7 GB
shared with the UMA iGPU. Measured, Vulkan only, PP3072:

| ubatch | result |
|---|---|
| 3072 (= PP) | `failed to allocate Vulkan0 buffer of size 3323719684` |
| 1024 | 981.50 tok/s |
| 512 | 1130.08 tok/s |

Larger ubatches are slower where they fit, so PP grows through `n_batch`. The
cost is reported rather than hidden: `n_ubatch` in the CSV is 1, 2, 2, 3, 4, 6,
8, 10, i.e. PP5120 issues 10 NPU-sized work units where PP512 issues one. The
NPU GEMM shape stays `M=512` at every PP point, which is what makes the curve
comparable to the PP512 baseline.

## 5. FFN locality is already correct

No change was needed. `[XDNA-SUMMARY]` per graph compute:

```
graph=4 n_nodes=2 n_gemm=2 gate=1 up=1  M=512 K=1024 N=3584
graph=5 n_nodes=1 n_gemm=1 down=1       M=512 K=3584 N=1024
```

gate and up are consecutive in one NPU split, down in the next, one V->X in and
one X->V out per layer: 8 V->X copies (37.7 MB) and 12 X->V copies (67.1 MB)
per ubatch, not a per-GEMM round trip. The measured V->X asymmetry from PHASE47B
(8.55 ms for 2 MB vs 0.21 ms the other way) is a fixed per-copy cost and is
irrelevant here because the copy volume per layer does not grow with PP.

## 6. Overlap

Not pursued, per PHASE47B: the NPU command stream has no usable concurrency.
`[SCHED-SUMMARY]` reports `overlap_ms` for completeness; with the FFN on the
NPU the copy boundary is a full `ggml_backend_synchronize()` of the source
backend, so submit-side overlap is 0 and cannot be improved in software without
changing what runs where.

## 7. What is left

Software-side, measured and not fixable here:

* The NPU kernel runs at 8.8-9.5 GOP/s against ~1900 GOP/s on the 780M. This
  is the Phase40/47 kernel and hardware dataflow, not scheduling. Both sides
  of the GEMM scale linearly with M, so no PP makes it competitive.
* NPU weight placement costs 197 MB of copies per ubatch and pushes 68 nodes to
  the CPU. Fixing the placement would remove ~9% of the 14.2x regression and
  leave the kernel term, which is 91% of it.
* `ggml_xdna_backend_graph_compute` drains at the end of every graph and
  `cpy_tensor_async` is null, so copies degrade to
  `synchronize(src) + blocking copy`. That is inherent to the current kernel
  contract (host packs Q8_0, device returns f32 through a host-mapped BO) and
  costs nothing measurable at these copy volumes.

Device: one `ERT_CMD_STATE_TIMEOUT` (`wait_state=8`) was hit at the end of a
41-consecutive-GEMM-graph configuration with CPU-resident weights and all 72
FFN GEMMs routed to the NPU. `xrt-smi examine` was clean afterwards and a
subsequent 4-round run reproduced normally; the offending configuration was
abandoned rather than repeated.