# Phase 20V2: XDNA Kernel V2 Adaptation Report

## Environment

| Component | Version | Status |
|-----------|---------|--------|
| OS | Ubuntu 26.04.1 (Podman) | OK |
| Kernel | 7.0.0-31 | OK |
| XRT | 2.21.75 | OK |
| NPU Firmware | 1.5.5.391 | OK |
| mlir-aie | 1.4.3 | OK (in ironenv-v143) |
| llvm-aie | 22.0.0 (LLVM 24.0.0) | OK |
| aiecc | ELF binary | OK |
| Peano | aie2p-none-unknown-elf | OK |
| Vitis aietools | NOT INSTALLED | BLOCKER for xclbin gen |

## What Was Accomplished

### 1. API Migration Matrix (mlir-aie 1.4.3)

**Critical change**: `!aie.objectfifosubview<memref<...>>` is REMOVED.
- Old: `aie.objectfifo.acquire @name(Port, count) : !aie.objectfifosubview<memref<...>>`
- New: `aie.objectfifo.acquire @name(Port, count) : memref<...>`
- `aie.objectfifo.subview.access` is REMOVED

All other APIs (ObjectFifo create, link, runtime_sequence, dma_*) are UNCHANGED.

### 2. Vectorized AIE2 Kernel

**File**: `ggml/src/ggml-xdna/matmul_i8_i32_vec.cpp`

- Uses 64-wide INT8 MAC: `mac_8x8_8x8(v64int8, v64int8, v64acc32)`
- Compiles with `clang++ --target=aie2p-none-unknown-elf -O2`
- Generates proper AIE2 vector instructions: `vmac`, `vbcst.8`, `vldb`
- Expected: 4-10x improvement over 53 GOPS scalar ceiling
- C linkage `matmul_i8_i32` matches MLIR `link_with` reference

### 3. MLIR Generator

**File**: `experiments/xdna-v2/gen_mlir.py`

- Parameterized: M, K, N, n_cols
- Generates valid MLIR for mlir-aie 1.4.3
- Supports n_cols=1 and n_cols=2 (NPU1 has 4 columns max)
- n_cols=4 requires >4 columns (not available on NPU1)

### 4. Build Pipeline

**File**: `experiments/xdna-v2/build_xdna_kernels.sh`

Verified end-to-end:
1. Kernel compilation (aie2p target) - PASS
2. MLIR generation - PASS
3. MLIR validation (aie-opt) - PASS
4. aiecc ELF compilation - PASS (21/21 steps)

### 5. Kernel Registry Updates

**File**: `ggml/src/ggml-xdna/xdna-kernel-registry.cpp`

- Whitelist expanded: n_cols in {1, 2}
- V2 kernel path search added: `experiments/xdna-v2/M{}_K{}_N{}_C{}`

## Known Blockers

### xclbin Generation

aiecc generates ELF but NOT xclbin. The CDO/xclbin generation requires Vitis aietools which is NOT installed.

**Impact**: Cannot generate new xclbin files for V2 kernels.

**Mitigation**: Use existing xclbin files from kernel cache for runtime testing. The 161 cached xclbin files work with XRT 2.21.75.

**Resolution path**:
1. Install Vitis aietools (full Vitis AI installation)
2. OR: Use IRON Python API (requires mlir-aie with iron module)
3. OR: Manual CDO generation using bootgen + xclbinutil

### Multi-Column xclbin

n_cols=2 MLIR compiles to 2 ELFs but no xclbin is generated. Need Vitis aietools for xclbin packaging.

## Files Modified/Created

### Modified
- `ggml/src/ggml-xdna/xdna-kernel-registry.cpp` - whitelist + V2 path

### Created
- `ggml/src/ggml-xdna/matmul_i8_i32_vec.cpp` - vectorized kernel
- `experiments/xdna-v2/gen_mlir.py` - MLIR generator
- `experiments/xdna-v2/build_xdna_kernels.sh` - build script
- `experiments/xdna-v2/API_MIGRATION_MATRIX.md` - migration docs
- `experiments/xdna-v2/M512_K1024_N3072_C1/` - compiled artifacts (n_cols=1)
- `experiments/xdna-v2/M512_K1024_N3072_C2/` - compiled artifacts (n_cols=2)

## Next Steps

1. **Install Vitis aietools** to enable xclbin generation
2. **Runtime test** with existing xclbin + new vectorized kernel
3. **Multi-column runtime** once xclbin generation works
4. **NPU performance benchmark** to measure vectorized kernel GOPS
5. **Heterogeneous PP** with Vulkan + XDNA concurrent execution
