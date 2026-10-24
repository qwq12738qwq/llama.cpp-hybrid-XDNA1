# Phase 25 - Is the Vulkan readback pathology a memory-type choice?

Date: 2026-09-20.  Host: RyzenAI-npu1 (XDNA1) + Radeon 780M RADV PHOENIX iGPU,
XRT 2.21.75.  Model: Qwen3.5-0.8B-Q8_0.gguf.  Build: build-vk.  PP512, ngl 99.

## Question
Phase 23 §3 isolated the in-graph hybrid tax to `ggml_backend_tensor_get(ct)` on
the Vulkan slice result: 3.67 MB read back at ~230 MB/s (~13-16 ms/op), while
reading a long-lived main-graph tensor via the same API runs at ~7-13 GB/s.
Phase 23 §6 proposed the Vulkan readback path as the next lever.  The Vulkan
backend has an explicit switch for UMA buffer placement
(`GGML_VK_PREFER_HOST_MEMORY=1` -> allocate activation buffers in
HOST_VISIBLE|HOST_COHERENT instead of DEVICE_LOCAL|HOST_VISIBLE).  Hypothesis:
the slow readback is the RADV/Phoenix DEVICE_LOCAL+host-visible mapping, and
host-memory buffers fix it.

## Experiment
Same split op (ffn_gate, M=512 K=1024 N=3584, n_gpu=1792) with the in-graph
timing (`XDNA_WA_TIMING=1`), baseline vs `GGML_VK_PREFER_HOST_MEMORY=1`, plus a
repeated E2E comparison for the PP512 gate.

## Result 1 - the readback does NOT move (per-op in-graph ms)

  run        [XDNA-WA-VK] get   (3.67 MB)
  baseline      14.18            <= ~260 MB/s
  baseline      15.43
  baseline      13.55
  prefer_host   13.19            <= ~280 MB/s
  prefer_host   12.07
  prefer_host   13.14

`GGML_VK_PREFER_HOST_MEMORY=1` leaves `get` at ~12-13 ms.  The memory-type
choice is NOT the root cause.

## Result 2 - E2E PP512 gate still loses (r=4)

  vulkan_only      1391.18 t/s   1.000x
  split            1186.56 t/s   0.853x
  split+prefer     1248.98 t/s   0.898x

PREFER_HOST_MEMORY buys ~5% on the split but the E2E is still ~0.90x.  The
hybrid does not flip.

## Result 3 - where the asymmetry actually is (same process, same API)

  get_a  (src1, main-graph activation, 2 MB)        0.15 ms   ~13 GB/s
  get_w0 (src0, main-graph weight slice, ~3.7 MB)   0.12-0.73 ms  ~5-30 GB/s
  get    (ct, mini-graph buffer, 3.67 MB)           12-15 ms   ~0.26 GB/s

The slow readback is a property of the **throwaway mini-graph buffer** (a fresh
`ggml_init` + `ggml_backend_alloc_ctx_tensors` + `ggml_backend_buffer_free` every
op), not of Vulkan buffers in general and not of the memory type.  Reading a
long-lived buffer through the identical `ggml_backend_tensor_get` path is 45x
faster.

## Result 4 - it is NOT first-touch either (second read of the same buffer)

A probe was added to `ggml_xdna_wa_execute_split` that reads `ct` twice in the
same op (same buffer, same API).  If the cost were first-touch / fresh mapping,
the second read would be fast.

  op    get1 (3.67 MB)   get2 (same buffer)
  1       13.70 ms          5.97 ms
  2        6.66 ms         11.57 ms
  3        7.74 ms         11.90 ms

Both reads are slow and the second is not reliably faster.  The readback is a
persistent ~280-600 MB/s host read of this Vulkan buffer on RADV/Phoenix, not a
one-time mapping/fault cost.  (With the extra read inserted, the reported
`free=` field now includes get2, so it is not a real free cost.)

## Result 5 - force the non-host-visible (staging) path: also no change

`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` makes the activation buffers
device-local (non host-visible), so the readback goes through the transfer-queue
staging-buffer path instead of a direct mapped memcpy.

  op    get1       get2
  1     13.89      11.56
  2     12.37       6.49
  3     13.12       5.88      (split E2E 1253 t/s, 0.90x)

Same ~12-14 ms.  So it is not the direct-mapping vs staging choice either.

## Interpretation
The ~13 ms readback for the mini-graph buffer is invariant under every
Vulkan-side memory lever tried: host-memory placement, non-host-visible
(staging) path, and repeated reads.  Reading `ct` (mini-graph buffer, 3.67 MB)
stays at ~0.3-0.6 GB/s while reading the main-graph `src0`/`src1` through the
identical API runs at 5-13 GB/s.  The difference is tied to the freshly-created
mini-graph context/buffer, but not to first-touch, memory type, or mapping mode.
The mechanism is not fully pinned down; what is clear is that no Vulkan buffer
flag removes it.

Paths that remove the tax are therefore not Vulkan-flags but data-flow changes:
(1) never read the Vulkan buffer back on the critical path - have XDNA
consume/produce Vulkan memory directly via a cross-device shared buffer; or
(2) have the Vulkan graph write the slice result into a cached host buffer
instead of reading a device tensor back.

## Verdict
NEGATIVE for the memory-type / mapping levers.  `GGML_VK_PREFER_HOST_MEMORY=1`
and `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` do not remove the ~13 ms/op Vulkan
readback tax; split E2E stays ~0.85-0.90x vs Vulkan-only.  Phase 23/24
conclusions hold and are strengthened: the tax is on the Vulkan-backend /
graph-scheduling side (throwaway slice buffer readback + CPU quant/merge), not
on NPU DMA/sync or host<->NPU staging.

Artifacts: timing_raw.txt (grep of the raw harness output: XDNA-WA* timing lines
and the pp512 t/s rows) plus this RESULTS.md.
