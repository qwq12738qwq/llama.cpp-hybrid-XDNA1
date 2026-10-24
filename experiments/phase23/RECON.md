# Phase 23 - Resident Hybrid Graph: recon notes

## Frozen from Phase 22
- E2E hybrid: PP512 0.993x, PP2048 0.957x, PP8192 0.953x (all <= Vulkan-only).
- Operator-level N-split hybrid: 1.54-1.74x (Phase 21).
- Per-op accounting (results/phase22_op_accounting.csv):
    ffn_gate 512x1024x3584 Vulkan full = 2.73 ms
    XDNA whole-array run->wait = 4.69 ms, + staging/merge 7.61 ms = 12.30 ms
- DMA floor (results/phase22_dma_floor.csv): host<->NPU staging ~3.93 ms/op at
  assumed 2 GB/s > Vulkan op 2.73 ms. Conclusion recorded: only dmabuf/UMA
  zero-copy avoids it.

## Current backend staging path (ground truth)
- ggml/src/ggml-xdna/xdna-whole-array.cpp::ggml_xdna_wa_submit
    memcpy(a_bo->map(), a) ; memcpy(b_bo->map(), b)
    a_bo->sync(TO_DEVICE) ; b_bo->sync(TO_DEVICE)
    run->start()
  ggml_xdna_wa_wait: run->wait() ; c_bo->sync(FROM_DEVICE) ; memcpy(c_out, c_bo->map())
- Same pattern in xdna-q8-scale.cpp.
- All BOs are `xrt::bo::flags::host_only` (group_id 3). So "staging" = local
  memcpy + XRT sync DMA.

## Cross-device zero-copy feasibility (evidence gathered)
- Devices present: /dev/dri/renderD128 (amdgpu, Radeon 780M RADV PHOENIX iGPU),
  /dev/accel/accel0 (amdxdna NPU).
- XRT 2.21.75 installed (/usr/include/xrt, libxrt_*). Driver source at
  /tmp/xdna-driver.
- amdxdna kernel driver SUPPORTS dma-buf:
    amdxdna_gem.c: amdxdna_gem_prime_export / amdxdna_gem_prime_import,
    amdxdna_gem_dmabuf_mmap, DRM prime export. (ubuf obj export returns
    -EOPNOTSUPP; the prime path is the usable one.)
- xrt::ext::bo (experimental/xrt_ext.h) provides:
    bo(device, userptr, sz, access)  -> host_only buffer wrapping a user
        (page-aligned) host buffer, access = read|write|local|shared|process|hybrid
    bo(device, pid, export_handle)   -> import an exported buffer from process
  No direct "import a dmabuf fd" constructor is exposed in the header.
- pyxrt (python bindings) exposes only: bo(device/ctx, size, flags, group),
  sub-buffer bo(parent,size,offset), write/read/sync/map/address.
  NO export/import handle API -> zero-copy test needs C++.

## Open decisive questions (Phase 23 core)
1. Is the ~2 GB/s "shim DMA" a real data movement, or just cache/IOMMU
   bookkeeping? Decisive test: write host_only BO map, SKIP sync, run kernel,
   compare correctness + timing vs with-sync. If results stay correct without
   sync, sync is flush only; the real cost is the NPU's shim read/write of host
   RAM at ~2 GB/s.
2. Can a Vulkan (amdgpu/RADV) buffer be shared with amdxdna so the NPU DMAs
   directly from iGPU-owned memory (UMA, no host bounce)?
   Path A: Vulkan exports dma-buf fd (VK_KHR_external_memory_fd), amdxdna
   imports it (prime_import). Need to confirm XRT can attach an imported
   dma-buf as a kernel arg (via xrt::ext::bo pid+handle, or a raw BPF/ioctl).
   Path B: xrt::ext::bo(userptr) wrapping a host-visible Vulkan mapping
   (avoids the memcpy but not the sync DMA).
3. Resident multi-GEMM: N GEMMs, 1 ingress/1 egress. Needs a multi-op xclbin
   (dynamic runtime sequence, MLIR-AIE 1.4) or N kernels sharing one BO set.

## Blocker for the next step
- pyxrt cannot test zero-copy; need to compile a C++ probe against XRT
  (/usr/include/xrt, -lxrt_coreutil) and run against the cached WA xclbin.
- Need the WA xclbin/inst paths (registry builds them under XDNA_KERNEL_CACHE,
  dir name "M%d_K%d_N%d_C4_WA"). Locate the cached artifact on disk first.

## Planned experiment order (goal doc sections)
E1 timeline of one real ffn_gate (sec 2/3) -> E-zerocopy microbench (sec 5/6)
-> E-B two GEMMs one staging (sec 7) -> E-D/E persistent + batched submit
(sec 10/11/16) -> E-C/F Qwen resident chain (sec 8/12) -> E2E gate (sec 23).
