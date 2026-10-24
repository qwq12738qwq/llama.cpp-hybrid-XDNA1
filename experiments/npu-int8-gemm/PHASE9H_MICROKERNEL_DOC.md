# Phase 9h Microkernel Documentation

## Status: FROZEN AS MICROKERNEL

This phase is preserved as a validated microkernel specialization. It is NOT the generic backend.

## Specialization Parameters

```
M = 1
K = 32 (chunked for larger K via K=32 chunks)
N = 16 (also validated for N=192 via tiling)
```

## Artifacts

| Artifact | Path | Size |
|----------|------|------|
| xclbin | `/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_test_new/design.xclbin` | 10345 bytes |
| insts.bin | `/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_test_new/design.insts.bin` | 420 bytes |
| main.pdi | `/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_test_new/main.pdi` | 3888 bytes |

## Generator Script

`gen_m1_k32_n16_v2.py` - uses CompilableDesign.compile() with scalar kernel (vectorized=False)

## Kernel Characteristics

- **Scalar MAC kernel**: Uses `-DSCALAR_ONLY` compilation path
- **Single core**: M=1, K=32, N=16 tile
- **K-chunking**: Larger K handled by host-side chunking (execute_chunked)
- **N-tiling**: Larger N handled by host-side output channel tiling (192+192+128 for N=512)
- **Async pipelining**: M-tiling with XDNA_M_TILE=64 for large M

## Validated Capabilities

| Capability | Status |
|------------|--------|
| M=1, K=32, N=16 | ✅ PASS (0 mismatches) |
| K-chunking (K=1024 = 32×32) | ✅ Implemented in execute_chunked |
| N=512 via output tiling | ✅ Implemented in execute_mul_mat_n512_tiled |
| M>1 via async pipelining | ✅ Implemented with M_TILE=64 |
| Q8_0 weight + Q8_0/F32 input | ✅ Supported |
| Real NPU execution | ✅ Verified on Phoenix XDNA1 |

## Limitations (Intentional - Not Bugs)

1. **Only N=16/192/512 supported** - hardcoded in supports_op()
2. **Only K multiple of 32** - required by kernel
3. **Only Q8_0 weight** - quantization format fixed
4. **Single xclbin** - compile-time specialized for M=1,K=32,N=16
5. **M=1 kernel** - host loops over M for larger M

## Role in Generic Backend

This microkernel becomes **one entry** in the specialization registry:

```
KernelKey {
    op_type: "MUL_MAT",
    dtype_a: Q8_0,
    dtype_b: Q8_0/F32,
    dtype_c: F32,
    M_tile: 1,
    K_tile: 32,
    N_tile: 16,
    quantization: Q8_0
}
```

The generic backend will select this specialization when the shape matches, or invoke the MLIR-AIE generator to produce new specializations for other shapes.

## Do Not Modify

- The xclbin/insts/pdi artifacts
- The gen_m1_k32_n16_v2.py generator
- The Phase 9h directory structure

This microkernel is complete and validated. Further development proceeds by building the generic registry around it, not by modifying it.