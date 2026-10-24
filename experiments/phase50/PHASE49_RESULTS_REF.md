# Phase 49 results: real NPU FFN path, routing to 72/72 with exact correctness

Model `models/Qwen3.5-0.8B-Q8_0.gguf` (Q8_0, 24 layers, n_embd 1024),
`Vulkan0` = AMD Radeon 780M (UMA), `AMD Phoenix XDNA1 NPU` with
`XDNA_Q8_ENGINE=c4_mmul_v2` (1x4, m32 n64 k256, Q8_0 per-32 scales).
Physical ubatch 512, PP grown through `n_batch`. Host: 16 cores, 7 GB RAM
shared with the UMA iGPU. All NPU runs serialized via `npu_guard.sh`;
memory capped with `ulimit -v 4194304` after one uncapped run OOMed the host.

Raw logs in `results/`. Binary rebuilt 3x during the phase; numbers below
are from the final binary unless marked `oldbin` (pre-fix).

## 1. Four configurations (PP512, warmup 0-1 / measure 1)

| cfg | flags | tok/s | NPU GEMMs | splits | trans | copies | copy MB | CPU nodes | correct |
|---|---|---|---|---|---|---|---|---|---|
| A Vulkan-only | `GGML_XDNA_DISABLE=1` | 1248.2 | 0/72 | 2 | 1 | 8 | 2.5 | 3 | reference, md5 `0ec3a71b` |
| B routing, no placement | `QWEN_PP=1` | - (hang) | 72/72 assigned | 98 | 50 | 97+ | 333 V->X | - | not reached |
| C routing + placement | `QWEN_PP=1 WEIGHTS=1` | 15.5 | 72/72 executed | 157 | 127 | 259 | 1100 (452 V->X + 403 X->V) | 116 (1 GEMM) | md5 identical |
| D placement, no routing | `WEIGHTS=1` | 940.5 | 0/72 | 58 | 40 | 91 | 284 | 106 | ok, placement overhead only |

C final (`final512`, 1 graph): wall 33.0 s, `n_mm 126 gpu / 72 xdna / 1 cpu`,
`n_nodes 1735 / 72 / 116`, peak RSS 366 MB (capped runs).
`[XDNA] XDNA ops executed=72 missing_kernel=0`.
Verify mode (`--mode verify --n-predict 8`): hybrid md5 `0ec3a71b13cb7372`,
token-identical to Vulkan-only. Single-GEMM gates also pass
(max_err 0.12 gate / 0.42 down, threshold 1.0).

D has no NPU work and only placement cost (284 MB staged copies, +106 CPU
nodes): no further D tests. B routes without placement and is pathological
by construction (every weight copied per ubatch, then hang): no further B
execution. PP1024/2048 were not run: gated on a PP512 improvement (§20),
which did not happen (15 vs 1248 tok/s); both sides scale linearly with M,
so no PP changes the ratio (proven PHASE48).

## 2. Routing: two real bugs found and fixed, 60/72 -> 72/72

Starting point (fresh binary, before fixes): C assigned 60/72, always the
last 4 layers (20-23) back to Vulkan, deterministic across 3 runs
(`C_assign`, `C_noglu` with `XDNA_GLU=0`, `pure`). GLU exonerated.

Bug 1 - `supports_op` had HW side effects (`ggml-xdna.cpp`).
`is_mul_mat_supported()` called `get_or_create_kernel()`, which opens a
device context per new shape. Routing then depended on transient device
state (one `Failed to init` observed live); late-queried shapes fail once
contexts exhaust. Fix: supports is now a pure function of shape + cached
artifacts on disk; the compute path still opens on demand and aborts
loudly on failure. Attention/rope/norm/sampling untouched (MUL_MAT-only
paths, xdna `offload_op` already ffn-gated).

Bug 2 - generic claim reclaimed XDNA-placed weights (`ggml-backend.cpp`).
Backends are `[Vulkan0, XDNA, XDNA(dup), CPU]`; Vulkan reports `offload_op`
for everything above batch 32. The claim loop ("first non-owner claimant
wins") sent XDNA-resident FFN weights to Vulkan: decision log at assignment
time shows `blk.20-23 wbuft=XDNA backend=0(Vulkan0) sup=1 off=1` for all 12.
Fix (`keep_owner`): when the weight owner is an ACCEL device that
supports+offloads the op, the claim is skipped and the weight preference
keeps it with zero copies. Vulkan-owned FFN still claims to XDNA (explicit
routing per §5); non-FFN MUL_MATs keep the old behavior (host->device
offload intact).

After both fixes: 72/72 assigned XDNA, 0 Vulkan, 0 CPU GEMMs per pass,
4/4 passes identical; `[XDNA] XDNA ops executed=72`.

Also found, not fixed (out of scope, harmless for assignment): the XDNA
backend registers twice (`backend[1]=XDNA backend[2]=XDNA`, same buft).
Scheduler consistently picks b=1.

Diagnostics added (env-gated, zero behavior change): `[FFN-ASSIGN]`
extended with `wbuft/wusage/abuft/ausage/supoff/claim/reason`, plus
`[FFN-CLAIM]` decision log at assignment time with original tensors
(post-hoc traces only see scheduler copies, which misled analysis twice:
`XDNA#` prefixes are V->X copies, not placement).

## 3. Scheduler / locality / copy / async audit

- Assignment chain reviewed: `backend_id_from_cur` (pre-alloc, view_src,
  input, claim, weight preference) + `backend_from_buffer` + pass-2
  expansion. Claim outranks weight preference for FFN only.
- Locality NOT ideal: 157 splits for 24 layers (6.5/layer), 127 backend
  transitions, 259 copies, 1.10 GB staged per ubatch (104 V->X 452 MB,
  72 X->V 403 MB). Per-layer single-split (V->X, gate, up, down, X->V)
  not achieved; mixed placement fragments every layer. Graph math
  untouched (no fusion, per §8).
- `graph_compute_async` == `graph_compute` (drains at end: fake async);
  `cpy_tensor_async`/`set/get_async` are null, so copies degrade to
  `synchronize(src+dst) + blocking host-staged copy`; events coarse-drain
  (correct, no hazard). Inherent to the kernel contract (host packs Q8_0,
  device returns f32 via host-mapped BO). Not changed: correctness risk
  with no measurable win (copies live inside per-GEMM host prep, §4).
- Serialization confirmed: `pending>=2` drain + single device queue, no
  overlap from submit/submit/wait/wait. Backend drains every graph; the
  copy boundary is a full source synchronize (overlap 0 by construction).
- CPU fallback is NOT 0: 116 CPU nodes incl. 1 CPU MUL_MAT (spill GEMM,
  exact op not attributed; neighbors are mask/cache-shaped elementwise).
  Follow-up, does not affect FFN verdict.
- `v2x_ms/x2v_ms/submit_ms` read 0.000 in this build (timer gap); wall and
  per-GEMM benches are the reliable clocks.

## 4. Kernel microbench (1x4 current, 5x, correctness-gated)

Standalone single GEMM, `c4_mmul_v2`, host-placed tensors:

| shape | median | min | max | avg | GFLOP/s | max_err | CPU same shape |
|---|---|---|---|---|---|---|---|
| M512 K1024 N3584 (gate) | 451.7 | 447.1 | 454.8 | 450.8 | 8.3 | 0.12 PASS | 16.0 ms (28x) |
| M512 K3584 N1024 (down) | 414.2 | 413.5 | 416.8 | 414.5 | 9.1 | 0.42 PASS | 11.4 ms (36x) |

Host prep (dequant + pack + submit) 12-20 ms = 3-5% of wall; device wait
~96%. Scale path host cost is minor; the bottleneck is on-device
(execute + DMA in/out, inseparable without device counters). Matches
PHASE48 (425.8/396.5 ms, 8.8/9.5 GOP/s) within noise. No 2x4/3x4/4x4
retested (excluded). No kernel code changed: 33 s wall ~= 72 x 0.46 s,
i.e. ~100% of PP time is the kernel; there is nothing left to schedule
around it.

## 5. Classification

- Kernel gap (8-9 vs ~1900 GOP/s, CPU 30x faster same shape, host 4%):
  HARDWARE/DATAFLOW LIMIT. AIE/DMA/MemTile/MMUL/scale-store split needs
  device counters - not done here.
- Routing/determinism (supports purity, keep_owner, 72/72, md5): fixed,
  SOFTWARE DONE.
- Remaining SOFTWARE: staged-copy robustness (1 GB+ mixed graphs OOM the
  7 GB host once and device-lost Vulkan once; 33 s/72-GEMM graphs are the
  operable ceiling), 1 CPU spill GEMM unattributed, duplicate XDNA
  registration, copy-timer gap. Locality/copies only become optimizable
  after the kernel stops dominating 100% of wall.

## 6. Repro (all NPU runs via guard, capped memory)

```
export LD_LIBRARY_PATH=build/bin XDNA_KERNEL_CACHE=experiments/npu-int8-gemm/kernel_cache
M=../models/Qwen3.5-0.8B-Q8_0.gguf; ulimit -v 4194304
# A: GGML_XDNA_DISABLE=1 ./bin/bench_pp_scaling --model $M --dev Vulkan0 --pp 512 --batch 512 --ubatch 512 --warmup 0 --measure 1
# C: GGML_XDNA_QWEN_PP=1 GGML_XDNA_WEIGHTS=1 XDNA_Q8_ENGINE=c4_mmul_v2 XDNA_GLU=0 \
#    npu_guard.sh --timeout 120 -- ./bin/bench_pp_scaling --model $M --dev "Vulkan0,AMD Phoenix XDNA1 NPU" \
#    --pp 512 --batch 512 --ubatch 512 --warmup 0 --measure 1
# verify: add --mode verify --n-predict 8, compare md5 lines
# microbench: /tmp/test_xdna_q8_latency (gate), /tmp/test_down_latency (down)
```

Do NOT run uncapped PP512 B/C executions on small hosts: the staged-copy
volume (0.3-1.3 GB) plus host packing can exhaust RAM+swap and wedge the
box. Allocation/assignment harvesting (decision logs print pre-exec) is
the safe way to study routing.
