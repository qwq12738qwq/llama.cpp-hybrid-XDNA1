# llama.cpp XDNA1 - Phoenix NPU x Radeon 780M hybrid inference research (SEALED)

Sealed archive of a multi-month investigation (phases 16-63, final campaign
"phase61") into whether the **AMD Phoenix XDNA1 NPU** can provide real
prompt-processing (PP) speedup next to the **Radeon 780M iGPU** inside
llama.cpp, on `Qwen3.5-0.8B Q8_0` (dense) and `Ling-3.0-tiny`
(bailingmoe3, MoE 7.9B-A1.3B).

- Working tree this snapshot was cut from: `/workspace/llama.cpp`,
  `master @ 7620399f5` (upstream Sep 6 2026), local XDNA backend under
  `src/ggml-xdna/`.
- Platform: Ryzen 7 7840HS (Vega, 16MB L3), Radeon 780M (RADV, UMA),
  Phoenix XDNA1 (AIE2.0, 4 shim NoC columns x 4 compute rows = 16 tiles,
  MemTile L2 pool ~2.5MB), XRT 2.21.75, FW 1.5.5.391, MLIR-AIE 1.4.3 (ironenv
  at `/Storage/Cache/ironenv-v143`).
- Nothing in this archive is pushed to `ggml-org/llama.cpp`; the research is
  sealed here only. The upstream sync record is in `docs/upstream_sync.md`.

---

## 1. Final verdict (measured, not theorized)

**No achievable configuration makes hybrid (GPU+NPU) PP faster than
Vulkan-only on this APU, at any PP tested (512-32768), for either the dense
or the MoE model, with or without fidelity-preserving quantization.** The
bound is composed from measured components:

| quantity | value | how measured |
|---|---|---|
| GPU busy during PP (kernel-sum / wall) | 99.5% @Qwen PP512, 96.9% @Ling PP16k | `GGML_VK_PERF_LOGGER` totals |
| overlap headroom (idle fraction) | 0.5% - 3.1% | the two rows above |
| XDNA1 host<->device effective bandwidth | 2.63-2.95 GB/s, does NOT scale with shim count or tiles | `dmabw.py`, T4/T1 bf16+int8 GEMM floors |
| 780M DRAM bandwidth | 34-37 GB/s | vocab-projection and profiler |
| best correct NPU GEMM rate | 100 GOP/s (c256 4-worker) / 667 GOP/s (bf16 T4) | kbench62, phase35/36 persistent |
| GPU FFN GEMM rate, same shapes | 2.2-3.0 TFLOP/s coopmat | profiler |
| NPU per-GEMM floor incl. traffic | 5.66 ms (bf16, = traffic / 2.78GB/s) | A5 |
| GPU FFN GEMM time in-model | 1.7-2.9 ms | profiler |
| theoretical best-case hybrid gain (idle x rate-ratio) | 0.1-3%, below the +-1.5% noise | A8 |

Both structural traps were verified:
- the NPU **cannot be faster on any operator** that grows with context
  (attention: softmax on AIE float = 1.9-2.5 s/ubatch @16k vs GPU entire FA
  195 ms; MoE: expert re-streaming 1.3 s/ubatch at shim bw vs 251 ms GPU);
- and the NPU **cannot fill idle time** because at 97-99% GPU-busy there is
  almost none, and anything data-dependent puts NPU latency on the critical
  path (drain-before-read is required for correctness, see D9).

Acceptance achieved: **Tier 4** (multi-tile kernel correctness + scaling
efficiency + DMA/host attribution, all measured). Tier 1-3: bounded
impossible with the components measured. Stopping rule reached.

---

## 2. NPU paths researched (full inventory)

`XDNA_Q8_ENGINE` selects the Q8_0 matmul engine; `XDNA_ROUTE_OPS` selects ops.
Status is the final, re-measured position of the phase61 campaign.

| path | data path | best measured (M512 K1024 N3584 unless noted) | verdict |
|---|---|---|---|
| `scale` q8scalev | 1 shim col, per-block scales host-side | 348 ms | superseded |
| `c4` (`q8_scale_c4`) | 4 cols, per-block scales applied in kernel | 446 ms | superseded |
| `c4_ds` device-scale | scales streamed separately | ~450 ms | superseded |
| `c4_mmul_v2` | 4x64x64x256 mmul, faithful Q8_0 | 433 ms | fidelity baseline (phase60) |
| `c4_mmul_v3` | v2 + 16-lane vector scale-update | 298 ms (1.50x) | faithful, +7.4% e2e (phase61) |
| `c4_mmul_v4` (c256) | per-K-chunk requantized weights, int32 chunk accumulate | **113-115 ms; 4-worker 37.5 ms / 100 GOP/s in kbench** | faithful-enough (PPL 16.3963, deterministic), shipped in backend |
| `int8_requant` v1 | per-column requant int8, integer-only | 161 ms (v3) | quality FAIL: PPL 3e6-13M, single-GEMM error hides 24-layer compounding |
| `int8_requant_v4` | pure int8->int32 T32x64K512_M8 | ~40 ms range | quality FAIL (same family) |
| `s12` | pre-broadcast rank-1 scale, 16-lane, linear C | 321-382 ms variants | ties/slower than v3, dead |
| fp256 | f16 in / f32 out mmul | 667 GOP/s @4 tiles (Oct24) | **quality PASSES** (+0.14% PPL) but DMA-floor 5.66ms/GEMM |
| persistent 4x4 array (phase35/36) | resident weights, int8/bf16, n_iters replay | T16 2120-2368 GOP/s Sep, but **T8/T16 ERT timeout on current stack**; T1/T2/T4 measured Oct: 177/346/667 linear | single-pass = DMA floor 5.66 ms regardless of tiles |
| MemTile relay 2-16 workers (phase63) | per-column shim + MemTile L2 relay | compiles for 16 workers, 8 shim ch, rel-RMS 3.6e-9 offline; **never ran on device (wedged)** | relay does not remove B-replication; moot given floor |
| MoE MUL_MAT_ID, FLASH_ATTN on NPU | - | never servable (K=128..576 skinny, dynamic N, softmax) | ruled out by A6/A8 numbers |
| M-split GPU/NPU overlap of any class | concurrent | ceiling = idle(3.1%) x rate-ratio(0.03) ~ 0.1% | A8: closed without building the executor |

## 3. End-to-end results (all re-measured on current binaries, interleaved)

| config | PP512 | PP1024 | PP2048 | PP16384 |
|---|---:|---:|---:|---:|
| Vulkan-only (0.8B) | 1242-1331 tok/s | ~1360 | ~1378 | - |
| hybrid v2 (all FFN) | 15.26 | - | - | - |
| hybrid v3 (all FFN) | 16.42 (0.0118x) | flat ratio to 5120 | - | - |
| hybrid v4 (gate,up,down) | 44.2 (0.033x) | 42.7 | 50.1 | - |
| PPL Qwen (ref Vulkan 16.4344) | v3 16.4378 / v4 **16.3963** (x2 deterministic) | | | |
| Ling-3.0-tiny Vulkan (this tree, Sep 6 code) | 627 | 622 | 502(4k) | 483 (16k); 377 @32k (-40%) |
| Ling on upstream master (Oct 23 build) | 737 | 730 | 708(4k) | **567 (+17.4% uniform)**; same -23% slope |

PP-curve conclusion: the long-PP degradation is real (-40% @32k on MoE),
dominated by attention O(L^2) (~25% of the 16k pass, 39% of the last block)
plus expert re-streaming; it is NOT software-rot, and upstream already recovers
the software part (+17%). The growing part is exactly what XDNA1 cannot take
(section 1).

---

## 4. Defect catalog (bugs found; several fixed; kept for future trees)

Backend / integration:
- **D1 (fixed)** `tensor_to_kernel_key()` had no `c4_mmul_v4` case -> key
  `n_cols=1` -> whitelist looked for `C1_Q8C256_MMUL8`, engine silently fell
  back on the real model; v4 had never run end-to-end before Oct 23.
  (`src/ggml-xdna/ggml-xdna.cpp`, see commit in `patches/`)
- **D2** `XDNA_ROUTE_OPS=all` mis-routes attention q (K1024 N4096) which has
  no c256 kernel -> `[WHITELIST] FAIL` / `missing_kernel=1` / graph ABORT.
  Correct opt-in is `gate,up,down`.
- **D3 (landmine)** the phase44 "Fused FFN" path inside `graph_compute`
  (requires gate/up/glu/down adjacent in one cgraph) never fires (scheduler
  splits at 1-2 nodes) **and** calls `submit_int8_pure` - the numerically
  invalid engine (PPL 13M). Dormant, but wrong-by-construction.
- **D4** XDNA-resident GLU (phase49) computes with a host scalar loop
  (~120 ms/ubatch incl. the forced pending-drain) - the "keep GLU on NPU to
  avoid copies" design no longer produces a single split (measured 3 splits/
  layer) while still paying the drain and the scalar loop.
- **D5** `graph_compute` drains unconditionally at cleanup
  (`ggml-xdna.cpp:3617`) -> with 1-node graphs the depth-4 async pipeline is
  dead; 144 graphs, 0 persistent_mode, ~115 ms serial drains - measured twice
  (v3, v4). The e2e ratio equals `kernel_speedup x serialized_drain`.
- **D6 (fixed earlier, root cause now explained)** the weight cache returned
  wrong-tensor weights once (phase62 PPL 4.5M incident); cache off by default,
  bounded 128MB; the Oct-24 kbench nondeterminism (D9) suggests stale-read,
  not hash-collision as in phase62 - both share the "read before flushed"
  signature.
- **D7** A weight-cache allocation pushed the host to swap -> `ERT_CMD_STATE_
  TIMEOUT` wait_state=8 (phase61 incident). On a 7.2GB machine the cache is a
  hazard; keep it off.
- **D8** Two stale phase47b bench targets (`bench_level0/1`) broke the full
  build after the engine signature grew 6->13 args; fixed this session.
- **D9 (NEW, Oct 24) harness nondeterminism**: same xclbin + fixed seed, 3
  runs -> rel_rmse `2e-9, 2e-9, 2.5e-2`. kbench62-style loops do not re-sync
  the C buffer object between repeats; one run in three reads stale data.
  Acceptance policy from now: >=3 identical repeats required, never a single
  shot. Backend PPL runs (drain-before-read) are bit-stable.

Toolchain / driver / environment:
- **D10** XRT amdxdna driver wedges as `DRM_IOCTL_AMDXDNA_GET_INFO err=-22`
  after ~8 rapid `hw_context` create/destroy cycles or repeated kernel ERT
  timeouts; not recoverable inside the container (read-only sysfs reset);
  host reboot is the only observed cure. Two full research phases were lost
  to this (62, 63). Avoid hw_context churn; probe before campaigns.
- **D11** RADV loses the Vulkan context under sustained NPU+GPU load while the
  host is ~50-60% swapped (device-lost mid-PPL). GPU recovers for short runs;
  campaigns must watch free memory and keep runs bursty.
- **D12** sustained long tests thermally throttle the 780M enough to fake a
  "long-context cliff" (pp8192 782 vs 1229 burst - 36%). All curves here use
  cooldown + pp512 anchors; historical single-pass curves (phase60's 840)
  should be distrusted on this machine.
- **D13** aiecc: `Stride 3 exceeds range` blocks whole-array kernels at
  N=6144 (attn_qkv of Qwen) - the 18/24 GDN qkv shape is unservable.
- **D14** 8 separate-column workers: "no ShimNOCTile has ... free DMA"; the
  merged-column variant compiles but returns rel_rmse 1.8e-2 (silent
  corruption at 8 logical columns on 4 physical).
- **D15** IRON `kernels.mm` dtype table: int8/{i16,i32}, bf16/{bf16,f32} only
  - no f16. bf16 is the only float matrix form (Q8_0 -> bf16 costs 0.4% per
  element, measured +0.14% PPL; f16 would have been lossy-free but is absent).
- **D16** (mine, corrected) Python Q8_0 readers MUST `.view(np.int8)`; an
  unsigned read silently flips ~half the weights and produced a false
  "llama.cpp can't do F32 FFN" finding (PPL 486/9M artifacts).
- **D17** phase36's famous "1.41x/1.68x/1.37x faster than Vulkan" numbers:
  (a) integer-equality only correctness (no block scales -> model-invalid),
  (b) the "vulkan" baseline was 1318-1407 GFLOP/s = ~half the real rate
  (throttled/stale), (c) the numbers were resident-weight replays. This
  triple-inflated claim is what seeded the stale FFN-offload allowlist the
  whole campaign kept. Treat historical GOPS as upper bounds of a different
  execution model, never as single-pass throughput.

Methodology rules adopted (worked): `npu_guard.sh` holder-check before every
submission; probe device health after any hang; interleaved A/B with anchors;
>=3 repeats for correctness claims; memory floor before GPU runs; no
assumption that old numbers match the current binary.

---

## 5. What would change the verdict (conditions, not plans)

1. Shim-class host bandwidth >= 10 GB/s (XDNA2+LPDDR5X parts) turns the 5.66
   ms traffic floor on FFN GEMMs into compute-bound at bf16 rates - then, and
   only then, does M-split overlap with a fixed scheduler (D4/D5/D9 class
   fixes) become worth building.
2. An NPU-attached memory with residency > per-layer working set (~7 MB int8
   for Qwen FFN; 153+ MB for MoE experts) - none exists on Phoenix.
3. A softmax-capable vector unit at >= 1/4 the matrix rate (AIE2.0 is ~1/30
   for fp32 chains; measured 19.8 ns per 16-lane float op).
None of these are software; the software layer here already extracts ~all of
what XDNA1 can do (correctness gates passed, scaling linear, DMA attributed).

## 6. Upstream sync record

`docs/upstream_sync.md` (in this repo) records: upstream tip
`8e5a58df8` (Oct 23), measured on this machine with a clean parallel build at
`/home/ubuntu/llama-upstream`, **+17% uniform** over our base commit on
Ling-3.0-tiny (737/730/708/655/567 vs 627/622/502/538/483 tok/s), same
long-PP slope; the RDNA3 int8-coopmat (#27952) + FA fixes as prime movers.
Rebasing this tree onto upstream is the single highest-value open action
(+17%, no NPU involvement); it was deliberately NOT done inside the sealed
tree.

## 7. Archive map

```
experiments/            all phase reports + harnesses + results.jsonl (binaries stripped)
  phase61/post_reboot/  THIS session: RESULTS.md (A1-A10 addenda), logs/, kbench/, quality/ probes
  phase61|62|63/        kbench/skbench/relay generators, campaign scripts, logs, incidents
  phase49..60/          the engine family development, whole-array, persistent designs
  docs-xdna/, gpu-npu-* , npu-*  earlier phases (16-48) reports and kernels
src/ggml-xdna/          full backend + engine sources (38 files)
src/ggml-xdna.h         device/backend header
patches/                npu-local-tracked-changes.patch - the 11 tracked-file
                        diffs (CMake, backend-reg, sched trace hooks, vulkan,
                        kv-cache, memory-recurrent, tools, tests)
```
Model weights (.gguf), compiled artifacts (.xclbin/.pdi/.insts.bin,
design.prj/*) and the 638MB kernel cache are excluded by policy; regenerate
kernels with `scripts/` + `env.sh` + aiecc.

## 8. Sealed state facts (for any future unseal)

- Device: healthy at seal time (hw_context + 432-op PPL + kbench runs after
  the last reboot); wedge history twice, each needed a reboot.
- The c4_mmul_v4 engine is the recommended starting point (37.5 ms 4-worker
  kernel exists in kbench form; 113 ms shipped 4-column form; PPL gate green).
- Every claim in section 1/3/4 is traceable to a file under
  `experiments/phase61/post_reboot/` or the older phase dirs.
- This repository is the closure of the XDNA1 hybrid line. Reopening it
  requires changing a condition from section 5.
