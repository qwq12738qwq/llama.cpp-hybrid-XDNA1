# Phase 21-23 final status (frozen)

Phase 21  Native / hybrid GEMM
         PASS, 1.54-1.74x operator-level

Phase 22  llama.cpp E2E integration
         PARTIAL
         PP512  0.993x
         PP2048 0.957x
         PP8192 0.953x

Phase 23  Resident / backend-boundary investigation
         Timeline                    DONE
         Zero-copy/shared probe      DONE
         Two-GEMM residency          DONE
         Submission batching         DONE
         PP512 E2E                   DONE
         DMA-BUF E2E                 NOT TESTED
         Full Qwen resident FFN      NOT TESTED
         Higher-PP E2E               NOT TESTED

Phase 24  Persistent runtime sequence feasibility (minimal)
         xrt::runlist N-GEMM single submission: PASS
         removes ~0.3-1.0 ms/GEMM host overhead, not the ~3.3 ms kernel floor
         true MLIR-AIE 1.4 dynamic runtime sequence: NOT compiled in the cached design

Phase 25  Vulkan readback root cause (is it a memory-type choice?)
         GGML_VK_PREFER_HOST_MEMORY=1: NO effect on the ~13 ms readback
         second read of the same buffer: ALSO slow -> not first-touch
         E2E PP512: vulkan_only 1391 / split 1187 (0.85x) / split+prefer 1249 (0.90x)
         verdict: NEGATIVE (memory-type + first-touch levers ruled out)
         remaining cost = persistent ~0.3-0.6 GB/s host read of the Vulkan buffer

## Final research conclusion

Under the current Phoenix XDNA1 + XRT/MLIR-AIE + llama.cpp execution model, the
XDNA whole-array GEMM has demonstrated real operator-level heterogeneous
acceleration, but the data movement, DMA and synchronization overhead at the
existing Vulkan<->XDNA backend boundary cancels that gain, so through PP512/2048/
8192 no E2E PP acceleration was obtained.

Limitation: the real Vulkan external-memory / dma-buf zero-copy path was NOT
verified, and a full multi-op resident Qwen FFN xclbin was NOT built.  Therefore
we cannot claim every cross-device memory architecture has been ruled out.

Corrected Phase 23 measurements (see phase23/RESULTS.md): host<->NPU BO sync is
~110 GB/s (NOT the assumed 2 GB/s); the in-graph per-op tax is dominated by a
Vulkan slice readback at ~200 MB/s plus CPU fp32<->int8 quantize/merge, neither
of which resident/batched execution can amortize.

The problem has moved from a kernel-optimization problem to a cross-device
memory-architecture problem: can Vulkan and XDNA share activation/weight lifetime
so that XDNA is a real co-processor rather than an external accelerator behind a
host API?
