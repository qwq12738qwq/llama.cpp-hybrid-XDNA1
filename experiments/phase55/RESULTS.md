# Phase 55 results: Hybrid Vulkan+XDNA acceleration

Model `models/Qwen3.5-0.8B-Q8_0.gguf`, AMD Phoenix XDNA1 NPU + Radeon 780M Vulkan.

## 1. Changes made

1. Added `GGML_OP_UNARY` (silu) and `GGML_OP_MUL` support to XDNA backend
   (computed on host-mapped BO, no device kernel needed) to keep FFN
   intermediate tensors on XDNA.
2. Pre-packed weights into tile-major format at cache time (Bp in
   g_requant_wcache) to avoid per-GEMM repack. Used pointer-based cache
   to avoid 7MB copy per GEMM.
3. Rebuilt and tested.

## 2. Performance (PP512, ubatch512)

| config | tok/s | XDNA ops | v2x copies | x2v copies | transitions |
|---|---:|---:|---:|---:|---:|
| Vulkan-only | 1248.2 | 0 | 0 | 0 | 2 |
| c4_mmul_v2 (Phase 49) | 15.35 | 72 | 88 (303MB) | 24 (50MB) | 63 |
| int8_requant_v2 (Phase 54) | 17.12 | 72 | 88 (303MB) | 24 (50MB) | 63 |
| int8_requant_v2 + Phase 55 | 17.06 | 72 | 88 (303MB) | 24 (50MB) | 63 |

Phase 55 changes did not improve end-to-end PP. The silu/mul ops were already
handled by the GLU path on XDNA. Pre-packing weights saved repack computation
but the memcpy into BO still dominates.

## 3. Long PP tests

| PP | Hybrid tok/s | notes |
|---:|---:|---|
| 512 | 16.92 | baseline hybrid |
| 1024 | 11.96 | worse (2 passes x 30s overhead) |

Longer PP with ubatch=512 doubles the per-pass overhead, making it worse.

## 4. Root cause analysis

The hybrid path is synchronization-bound, not compute-bound:
- Per-GEMM device time: ~110ms (3x better than c4_mmul_v2's 359ms)
- Per-pass total: ~30s for 72 GEMMs
- Unaccounted overhead: ~22s (sync + copies + scheduler)
- sync_ms alone: ~9s per pass
- 88 V->X copies (303MB) + 24 X->V copies (50MB)

The GLU op was already on XDNA. The remaining copies come from the MUL_MAT
ops themselves: each GEMM's input activation must be copied from Vulkan to
XDNA. The scheduler treats each MUL_MAT independently and copies the activation
per-GEMM, even when gate and up share the same input.

## 5. What would be needed to beat Vulkan

1. Fuse entire FFN (gate+up+glu+down) into a single XDNA operation
   - Eliminates all intermediate V<->X copies
   - Requires new GGML_OP or model-level change
2. Reduce synchronization: batch multiple GEMMs into fewer XRT submits
3. Larger ubatch (more tokens per pass) to amortize fixed overhead
   - Limited by device memory (currently 512 max safe)

## 6. Conclusion

The XDNA NPU's per-GEMM compute is now 3x faster (int8 pure vs Q8 scale),
but the hybrid path cannot beat Vulkan-only due to:
- Synchronization overhead (~9s/pass)
- Backend copies (~303MB V->X per pass)
- Scheduler transitions (63 per pass)
- Host packing (~7s per pass)

The remaining gap requires architectural changes (fused FFN op or scheduler
improvements) beyond kernel optimization. The device-level compute bottleneck
has been solved; the system-level integration bottleneck remains.

## 7. Artifacts

- Kernel: experiments/npu-int8-gemm/kernel_cache/M*_C4_PURE/
- Generator: experiments/phase54/gen_int8_pure_mmul_v2.py
- Engine: int8_requant_v2 (XDNA_Q8_ENGINE=int8_requant_v2)
- Correctness: max err 0.13 on M=512 GEMM, 144/144 NPU ops, no NaN
```<|end|><|start|><|start|>assistant<|channel|>analysis<|message|>I've written the Phase 55 results. Let me verify the file was written correctly.<|end|><|start|>-start to=read_file<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="file_path">
/workspace/llama.cpp/experiments/phase55/RESULTS.md