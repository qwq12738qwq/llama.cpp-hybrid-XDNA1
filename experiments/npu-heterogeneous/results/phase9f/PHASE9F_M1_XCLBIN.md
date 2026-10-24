# Phase 9-F: Generate and Validate M=1 Qwen SSM XCLBIN - Report

## Executive Summary

**Classification: BLOCKED**

The M=1, K=1024, N=16 XCLBIN was successfully generated (artifacts: `design.xclbin`, `design.insts.bin`, `main.pdi`), but standalone correctness testing reveals **100/100 iterations with numerical mismatches**. The kernel produces completely incorrect results due to tensor tiling/signature mismatch between the MLIR design and the scalar matmul kernel.

## Step 1: Audit of Existing M=16 Kernel

### M Dimension Source
- MLIR `aie.mlir`: `memref<16x1024xi8>` for A, `memref<16x16xi32>` for C in runtime_sequence
- DMA BD for A: `sizes = [1, 16, 16, 64]` (16 = M)
- DMA BD for C: `sizes = [1, 1, 16, 16]` (16 = M)
- Core loop: `scf.for %arg2 = %c0 to %c16` (16 iterations = M)
- C L2L3 objectfifo: `memref<256xi32>` (16×16 = M×N)

### K Dimension Source
- MLIR `aie.mlir`: `memref<1024xi8>` for A and B in objectfifo declarations
- C++ kernel template: `DIM_K` compile-time constant (1024 for full K, 64 for tile_k)
- Core inner loop: `scf.for %arg2 = %c0 to %c16` (16 iterations = K/tile_k = 1024/64)

### N Dimension Source
- MLIR `aie.mlir`: `memref<1024x16xi8>` for B, `memref<16x16xi32>` for C
- DMA BD for B: `sizes = [1, 1, 1024, 16]` (16 = N)
- DMA BD for C: `sizes = [1, 1, 16, 16]` (16 = N)

### Output Tile Size
- C L1L2 objectfifo: `memref<256xi32>` = 16×16 int32 (M×N)
- DMA output length: 256 int32 elements = 1024 bytes

### Input Tile Size
- A L2L1 objectfifo: `memref<1024xi8>` = 16×64 int8 (M_tile × K_tile)
- B L2L1 objectfifo: `memref<1024xi8>` = 64×16 int8 (K_tile × N_tile)

### DMA Output Length
- C L2L3 DMA: `len = 256` (int32 elements) = 1024 bytes

### Runtime Argument Shape
- `memref<16x1024xi8>` (A), `memref<1024x16xi8>` (B), `memref<16x16xi32>` (C)

## Step 2: M=1 Kernel Generation

### Configuration
- **M=1, K=1024, N=16**
- Tile sizes: M_tile=1, K_tile=32, N_tile=16
- 32 K-chunks (1024/32)
- Scalar matmul kernel (vectorized=False, requires M % 16 == 0 for vectorized)
- Tile M=1 does not satisfy vectorized requirement (M % 16 == 0)

### Generated Artifacts
```
k1024_m1_n16/
├── design.xclbin       (10,873 bytes)
├── design.insts.bin    (420 bytes)
└── main.pdi            (4,416 bytes)
```

### Compilation Pipeline
- Iron API `mm()` with `vectorized=False` → scalar `matmul_scalar_i8_i32`
- Custom scalar kernel `matmul_scalar_05f2508a.cc` compiled with clang++
- MLIR generated via Iron API, compiled via `aiecc` with `--get-xclbin --xclbin-name`
- Scalar matmul kernel: `05f2508a_matmul_scalar_i8_i32` (hash: 05f2508a)

## Step 3: Standalone Correctness Test - FAILED

### Test Configuration
- 100 iterations with random int8 inputs
- CPU reference: `C = A @ B` (int8 @ int8 → int32 accumulation)
- NPU execution via `NPUKernel` with pre-compiled XCLBIN

### Results
| Metric | Value |
|--------|-------|
| Iterations | 100 |
| Mismatches | 100/100 (100%) |
| Max abs error | ~687,811 |
| Latency (mean) | ~1,591 us |
| Latency (median) | ~362 us |

### Failure Analysis
The kernel produces completely incorrect results (output values bear no relation to CPU reference). Root cause: **Tensor tiling/signature mismatch** between:
1. MLIR design tensor shapes (tap_A, tap_B, tap_C tiling patterns)
2. Scalar matmul kernel signature (`memref<32xi8>`, `memref<512xi8>`, `memref<16xi32>`)
3. DMA BD sizes/strides for A (1024 bytes), B (16384 bytes), C (64 bytes)

The tensor tiling patterns (tap_A, tap_B, tap_C) and DMA BD sizes/strides do not correctly map the full M=1, K=1024, N=16 matrices to the 32 K-chunks expected by the scalar kernel.

## Root Cause
The Iron API's `TensorTiler2D.group_tiler` and `TensorAccessPattern` for M=1, K=1024, N=16 with tile sizes (1, 32, 16) do not correctly generate the DMA BD sizes/strides and objectfifo `dims_to_stream` for the scalar kernel's expected buffer layout. The MLIR design expects specific buffer shapes and access patterns that don't match the scalar kernel's expectations.

## Blocker Classification
**BLOCKED**: M=1 kernel cannot be correctly executed with current tensor tiling approach. The tensor tiling/signature mismatch is fundamental to the MLIR design and requires redesign of the tiling patterns for M=1 scalar path.

## Recommendations
1. **Redesign tensor tiling** for M=1 scalar path: Use 1D flat access patterns instead of 4D tiled patterns
2. **Verify DMA BD sizes/strides** match the scalar kernel's buffer expectations exactly
3. **Add MLIR verification pass** to validate tensor shapes match kernel signatures
4. **Consider alternative**: Use M=4 with tile_m=4 (minimum for vectorized path) and mask unused rows