# MLIR-AIE 1.4.3 API Migration Matrix

## Critical API Changes

| Feature | Old (pre-1.4.3) | New (1.4.3) | Impact |
|---------|-----------------|-------------|--------|
| ObjectFifo subview | `!aie.objectfifosubview<memref<...>>` | REMOVED - acquire returns `memref<...>` directly | HIGH - all core code must change |
| subview.access | `aie.objectfifo.subview.access` | REMOVED | HIGH - remove all subview.access ops |
| acquire syntax | `aie.objectfifo.acquire @name(Port, count) : !aie.objectfifosubview<...>` | `aie.objectfifo.acquire @name(Port, count) : memref<...>` | HIGH - type annotation changes |
| release syntax | `aie.objectfifo.release @name(Port, count)` | Same | NONE |
| ObjectFifo creation | `aie.objectfifo @name(producer, {consumers}, depth) : !aie.objectfifo<...>` | Same | NONE |
| ObjectFifo link | `aie.objectfifo.link [@src] -> [@dst]([] [0])` | Same | NONE |
| dimensionsToStream | Same | Same (part of objectfifo.create) | NONE |
| runtime_sequence | Same | Same | NONE |
| dma_bd | Same | Same | NONE |
| dma_configure_task_for | Same | Same | NONE |
| dma_start_task | Same | Same | NONE |
| dma_await_task | Same | Same | NONE |
| dma_free_task | Same | Same | NONE |

## Toolchain

| Tool | Version | Status |
|------|---------|--------|
| aie-opt | AOMP-23.0-60, LLVM 24.0.0 | WORKING |
| aie-translate | same | WORKING |
| aiecc | ELF binary | WORKING (generates ELF, not xclbin) |
| bootgen | v2026.1 | AVAILABLE |
| xclbinutil | XRT 2.21.75 | AVAILABLE |
| clang/clang++ | Peano aie2p-none-unknown-elf | WORKING |

## Kernel Compilation

| Target | Status | Notes |
|--------|--------|-------|
| aie2-none-unknown-elf | FAIL | No v64acc32 type |
| aie2p-none-unknown-elf | PASS | Correct target for Phoenix XDNA1 |

## Vectorized Kernel

| Function | Intrinsic | Status |
|----------|-----------|--------|
| INT8 MAC | `mac_8x8_8x8(v64int8, v64int8, v64acc32)` | WORKING |
| INT8 broadcast | `broadcast_to_v64int8(int8_t)` | WORKING |
| INT8 vector load | `*(v64int8*)ptr` | WORKING |
| acc32 vector store | `*(v64acc32*)ptr` | WORKING |

## What Works

1. MLIR generation with new acquire/release syntax (memref directly)
2. aie-opt MLIR validation and ObjectFifo lowering
3. aiecc ELF compilation pipeline (steps 0-21)
4. Vectorized kernel compilation with aie2p target

## What Needs Work

1. **xclbin generation**: aiecc generates ELF but not xclbin
   - Need CDO generation from ELF
   - Need xclbin packaging with xclbinutil
   - OR: find correct aiecc flags for xclbin generation
   
2. **Multi-column support**: MLIR generator supports n_cols parameter
   - Core functions need per-column FIFOs
   - Runtime sequence needs per-column DMA
   
3. **insts.bin generation**: Required for XRT runtime
   - Need to extract from CDO or generate separately

## Next Steps

1. Investigate xclbin generation path (bootgen + xclbinutil)
2. OR: Check if aiecc needs `--aietools` flag pointing to full Vitis install
3. Generate multi-column MLIR (2col, 4col)
4. Adapt ggml-xdna backend for new kernel loading
