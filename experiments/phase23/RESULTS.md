# Phase 23 - Resident Hybrid Graph: RESULTS

Date: 2026-09-20.  Host: RyzenAI-npu1 (XDNA1), Radeon 780M RADV PHOENIX iGPU,
XRT 2.21.75.  Model: Qwen3.5-0.8B-Q8_0.gguf.  Build: build-vk (Vulkan + XDNA).

Artifacts:
- zero_copy_probe.cpp / zero_copy_probe   (raw BO sync + kernel sync/no-sync)
- resident_probe.cpp  / resident_probe    (resident / batched / shared activation)
- results/phase23_cost_model.md           (detailed cost model)
- results/phase23_resident.csv            (resident probe output)

In-graph timing added behind XDNA_WA_TIMING in
ggml/src/ggml-xdna/ggml-xdna.cpp (ggml_xdna_wa_execute_split):
  [XDNA-WA-SPLIT] and [XDNA-WA-VK] lines.

--------------------------------------------------------------------------------
## 1. Phase 22 cost model is WRONG (host<->NPU staging is NOT the bottleneck)

Phase 22 (results/phase22_dma_floor.csv) assumed 2 GB/s staging.  Measured
(zero_copy_probe, host_only BO, median over 20 reps):

  sync(TO_DEVICE)  16/32/64/128 MB -> 111 / 118 / 116 / 123 GB/s
  sync(FROM_DEVICE)16/32/64/128 MB -> 111 / 113 / 115 / 125 GB/s
  memcpy into map  16/32/64/128 MB ->  12 /  11 /  12 /  15 GB/s
  memcpy out of map16/32/64/128 MB -> 7.8/ 7.3 / 7.2 / 7.0 GB/s

A no-sync control (sync=0 rows) shows sync time ~0, so sync is real work but
fast.  xrt::bo flags normal / device_only are UNSUPPORTED on this driver
(err 95 / err 22); cacheable + host_only work.  The mapped host-memory copy is
the slow part (~7-15 GB/s) but is still sub-ms per ffn_gate op.

=> Host<->NPU staging costs ~0.05-0.5 ms/op, NOT ~3.9 ms.  There is almost
   nothing to amortize.  The Phase 22 break-even premise (N x ingress/egress)
   does not hold.

## 2. Zero-copy activation (shared buffer) works but gains nothing

xrt::ext::bo(device, userptr, size, read_write|local) wraps a page-aligned
host buffer as a host_only BO.  The WA GEMM reading A directly from that shared
buffer is CORRECT (mismatches=0) and avoids the activation memcpy entirely.

resident_probe, M=512 K=1024 N=3584, per-GEMM ms (median):

  mode              K=1     K=2     K=3     K=4
  perop (restage)  6.61    6.60    6.63    6.62
  resident (1 stg) 6.59    6.66    6.67    6.62
  shared-resident  6.74    6.72    6.70    6.61

=> resident staging, one-sync batching, and shared activation all give ZERO
   per-GEMM improvement.  saved_transfer ~= 0.  Experiments B / D / E / 11 /
   break-even (goal sections 7/10/11/15/16) are all negative.

## 3. Where the in-graph tax actually is (real ffn_gate, PP512, split)

llama-bench -p 512 -ngl 99 -ot blk.0.ffn_gate.weight=CPU, XDNA_WA=1 + timing.

  n_gpu=1792 (steady-state op), ms:
    get_a  0.17   (2 MB activation Vulkan->host)
    pack_w 0.18   (weight int8 pack, cached)
    get_w0 0.29
    quant  1.84   (CPU fp32->int8 of A, 512*1024)
    submit 0.38
    vk    19.89   <-- Vulkan slice mini-graph, rebuilt per op
    wait   1.07   (NPU kernel N=1792)
    merge  1.77   (CPU merge + scale)
    set    0.85   (7.3 MB host->Vulkan dst)
    TOTAL 26.44   vs Vulkan-only whole op 2.73 ms

  Vulkan slice breakdown [XDNA-WA-VK]:
    alloc=0.10  set=0.49  compute=1.44  get=15.82  free=0.15

  The killer is `ggml_backend_tensor_get(ct)` of the GPU slice result:
    n_gpu= 896 -> get  8.33 ms (1.84 MB)  ~220 MB/s
    n_gpu=1792 -> get 15.82 ms (3.67 MB)  ~230 MB/s
    n_gpu=2688 -> get 30.33 ms (5.51 MB)  ~180 MB/s
  It scales with n_gpu.  By contrast reading the *original* graph activation
  (src1, 2 MB) via the same API costs 0.17 ms (~12 GB/s).  Reading a tensor
  allocated by ggml_backend_alloc_ctx_tensors in a throwaway context is ~50x
  slower.  This is a Vulkan-backend-buffer readback pathology.

=> The split is expensive NOT because of NPU DMA/sync, but because:
   (a) it forces a GPU->host readback of the GPU slice at ~200 MB/s (16-30 ms),
       a cost the pure-Vulkan graph never pays (its tensors stay on device), and
   (b) CPU fp32<->int8 quantize (~1.8 ms) + merge/scale (~1.8 ms) sit on the
       graph critical path.

## 4. E2E gate (PP512, this session; frozen baseline 1400.6 t/s)

  vulkan_only      1369 t/s       1.000x   (fresh baseline)
  whole-op async   1320 t/s       0.964x
  split n_gpu=1792 1241 t/s       0.907x

  Both hybrid variants lose.  Consistent with Phase 22 (0.95-0.99x).

## 5. Levers verified (goal sections 4-11, 15-17)

  zero-copy activation (shared buffer)   VERIFIED, correctness PASS, no gain
  persistent XDNA context                VERIFIED (hw_context + engine cached), no gain
  batched submission (N=1..4, one sync)  VERIFIED, no gain
  resident activation                    VERIFIED, no gain
  multi-GEMM fusion (via batching)       VERIFIED equivalent, no gain
  dma-buf Vulkan<->XDNA end-to-end       NOT tested (needs C++ export/import + Vulkan
                                         external memory; the measured binding cost
                                         is on the Vulkan readback + CPU side, which
                                         dmabuf does not address)

## 6. Verdict: STOP

Under the current Phoenix XDNA1, XRT/MLIR-AIE and the current llama.cpp backend
memory model, the XDNA whole-array GEMM in operator-level N-split hybrid does
produce a real, repeatable reduction of the operator critical path (NPU
N=1792 kernel 1.07-1.16 ms vs ~1.4 ms Vulkan slice vs 2.73 ms full-Vulkan op),
but the cross-backend data path (Vulkan slice result readback at ~200 MB/s plus
CPU fp32<->int8 conversion and merge) costs ~16-20 ms/op on the graph critical
path and cannot be amortized by resident/batched execution, because NPU
ingress/egress is already ~0.

This does NOT mean XDNA1 is unsuitable for LLM inference, nor that the NPU is
useless.  It means the profit is eaten by the cross-backend data movement and
the GPU-side readback/buffer-boundary, not by the NPU.

Next route if pursued: fix the Vulkan backend's readback path (~200 MB/s for a
fresh compute-context buffer) and move the fp32<->int8 quantize/scale off the
host critical path.  Those are Vulkan-backend / graph-scheduling fixes, not NPU
fixes.
