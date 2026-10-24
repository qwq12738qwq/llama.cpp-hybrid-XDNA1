# Phase 23 - measured staging cost model (E1 + zero-copy probe)

## Correction of the Phase 22 cost model
Phase 22 (results/phase22_dma_floor.csv) ASSUMED host<->NPU staging runs at
2 GB/s and concluded the ~3.9 ms/op staging floor exceeds the Vulkan op.
That assumption is wrong. Measured directly (experiments/phase23/zero_copy_probe.cpp):

  host_only BO sync(TO_DEVICE) / sync(FROM_DEVICE) bandwidth:
     16 MB  -> ~111 / ~111 GB/s
     32 MB  -> ~118 / ~113 GB/s
    128 MB  -> ~123 / ~125 GB/s
  (median, XDNA_WA/no-sync control: sync=0 rows show ~0, so sync IS the work.)

  memcpy into/out of the BO map:
     in  ->  12-15 GB/s
     out ->   7-8  GB/s   (mapped host memory read is the slow side)

  So the BO staging is NOT a 2 GB/s shim. It is a normal host-memory copy.

## Cross-device / flag feasibility
- xrt::bo flags normal / device_only: UNSUPPORTED on this driver
  (normal -> err 95 unsupported; device_only -> err 22 invalid).
  cacheable works. host_only works.
- xrt::ext::bo(dev, userptr, size, access_mode) WORKS: it wraps a page-aligned
  user host buffer as a host_only BO. sync behaves identically (~120 GB/s).
  This is the "shared allocation" path (CPU-owned pointer visible to NPU).
- amdxdna kernel driver supports dma-buf prime import/export; pyxrt has no
  export/import API, so cross-device dma-buf needs C++.

## Kernel (whole-array C4_WA, i8 x i8 -> i32) scaling, M=512 K=1024
  N= 896 : 0.59 ms  (~1.59 TOPS)
  N=1792 : 1.16 ms  (~1.62 TOPS)   <- best regime
  N=2688 : 3.14 ms  (~0.90 TOPS)   <- nonlinear jump (MemTile/BD wall)
  N=3584 : 3.72 ms  (~1.01 TOPS)
Vulkan-only full ffn_gate op (phase21) = 2.73 ms.

## In-graph per-op breakdown (one real ffn_gate, PP512, N-split n_gpu=1792)
llama-bench, XDNA_WA=1 XDNA_WA_NGPU=1792 XDNA_WA_TIMING=1, -ot blk.0.ffn_gate.weight=CPU
steady-state op:
  get_a  = 0.17 ms   (2 MB activations Vulkan->host)
  pack_w = 0.18 ms   (weight int8 pack, cached)
  get_w0 = 0.29 ms   (Vulkan weight slice)
  quant  = 1.84 ms   (CPU fp32->int8 of A, 512*1024 elems)
  submit = 0.38 ms   (NPU start)
  vk     = 19.89 ms  <-- Vulkan mini-graph slice: ggml_init + alloc_ctx_tensors
                          + tensor_set(wt) + tensor_set(bt) + graph_compute
                          + tensor_get(ct) + buffer_free, rebuilt EVERY op
  wait   = 1.07 ms   (NPU kernel N=1792, matches standalone 1.16 ms)
  merge  = 1.77 ms   (CPU merge c0 + Cn*scale)
  set    = 0.85 ms   (7.3 MB host->Vulkan dst)
  TOTAL  = 26.44 ms  vs Vulkan-only whole op 2.73 ms

### Conclusion for experiment 2/3
The tax is NOT DMA (fast) and NOT synchronization (fast). It is:
  (a) ~20 ms/op building a fresh Vulkan mini-graph for the GPU slice, and
  (b) ~4-5 ms/op of CPU fp32<->int8 quantize + merge + host tensor copies.
The ideal split critical path max(vk_slice ~1.4 ms, npu 1.07 ms) is NOT what
runs. Overhead dominates by ~10x.

Phase 23 levers therefore are:
  L1 - stop rebuilding the Vulkan slice mini-graph per op (persistent vk ctx/buffers)
  L2 - remove CPU quant/merge from the critical path (layout/scale on device)
  L3 - shared activation buffer so get_a/set disappear (xrt::ext::bo userptr path)
