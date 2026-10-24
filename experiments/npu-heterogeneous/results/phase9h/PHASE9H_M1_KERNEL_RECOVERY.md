# PHASE 9H — M=1 Scalar XDNA Kernel Recovery Report

## Executive Summary

**Status: STOP-2 — Kernel compiles and links but runtime output is numerically incorrect**

The M=1 K=32 N=16 scalar XDNA kernel was successfully recovered through the `CompilableDesign.compile()` path (Phase 9H-A/B). The `-DSCALAR_ONLY` compile flag patch to `mm()` in `linalg.py` enables clean compilation. However, the kernel produces numerically incorrect output with a consistent **6.0x scaling factor** across all output elements (16/16 mismatches).

## Toolchain Versions (Phase 9H-A)

| Component | Version |
|-----------|---------|
| Python | 3.14.4 |
| mlir-aie | 1.4.1 |
| IRON | Available (via `aie.iron`) |
| aiecc | Fails without `PYTHONPATH`; works via Python API |
| clang (Peano) | 21.0.0 (Xilinx llvm-aie c9c5ecb7) |
| XRT | 2.21.75 |
| NPU Device | Phoenix XDNA1 (`/dev/accel/accel0`) |

## Compilation Path Analysis (Phase 9H-B)

### Path A: ExternalFunction via CompilableDesign.compile() ✓
- `mm(vectorized=False)` now emits `-DSCALAR_ONLY` compile flag (patched in `linalg.py`)
- `CompilableDesign.compile()` invokes `compile_external_kernel()` automatically
- Object file `matmul_scalar_i8_i32_<hash>.o` generated in `design.prj/`
- aiecc links successfully → XCLBIN produced

### Path B: Precompiled Kernel (not tested)
- Not needed since Path A works for compilation

## Numerical Correctness Gate (Phase 9H-C) — FAILED

### Test Configuration
- M=1, K=32, N=16, tile=(1, 32, 16)
- Input: random int8 (seed=42), 100 iterations
- CPU reference: `int32(A) @ int32(B)`

### Results
| Metric | Value |
|--------|-------|
| Mismatches | 16/16 (100%) |
| Max Absolute Error | ~394,595 |
| Error Pattern | Consistent 6.00x scaling across all 16 outputs |

### Diagnostic Patterns Tested
| Pattern | Result |
|---------|--------|
| Random int8 (seed=42) | 6.00x error |
| All ones | 6.00x error |
| A[k]=k, B=1 | 6.00x error |
| A=1, B[k,n]=k*N+n | 6.00x error |

The consistent 6.00x scaling across all patterns and all 16 output elements indicates a **systematic scaling error**, not a data corruption or layout mismatch.

## Root Cause Analysis

### Compilation Path (Working)
1. `mm(vectorized=False)` → ExternalFunction with `-DSCALAR_ONLY`
2. `CompilableDesign.compile()` → `compile_external_kernel()` → `.o` file generated
3. aiecc links `.o` → XCLBIN produced
4. Runtime execution via `iron.jit` → NPU execution

### Numerical Failure Point
The kernel executes but produces `C_npu ≈ 6.0 × C_cpu` for all test patterns.

### Hypotheses Ruled Out
| Hypothesis | Status |
|------------|--------|
| TAP/stride mismatch | Ruled out — MLIR DMA BDs show correct row-major strides `[16, 1]` |
| C buffer not zeroed | Ruled out — MLIR shows explicit zeroing loop before matmul |
| K-accumulation error | Ruled out — K=32 single tile, no accumulation loop |
| Data corruption | Ruled out — Consistent 6.00x scaling, not random noise |
| B matrix column-major | Ruled out — TAP strides `[16, 1]` confirm row-major |

### Remaining Suspects (STOP-2 Classification)
| Category | Evidence |
|----------|----------|
| **Kernel Implementation** | Scalar kernel `matmul_scalar` in `mm.cc` may have M=1 bug |
| **Kernel ABI** | Template instantiation `DIM_M=1, DIM_K=32, DIM_N=16` may have edge case |
| **Data Layout** | Flat `memref<512xi8>` vs kernel's 2D `b[i*16+col]` indexing mismatch |

### Key Finding: mm.cc combos Macro Regression
The current `mm.cc` has a critical issue: the `combos` macro is redefined multiple times, with the **last definition (AIE2P emulated bf16) overriding all previous**. The int8→int32 vectorized combo was **removed from the final combos definition**, meaning:
- Vectorized int8→int32 is **unavailable** in current toolchain
- Only scalar path exists for int8→int32
- Historical M=16 vectorized kernel worked with **older mlir-aie version**

This explains why the scalar path is the only option and why it may be undertested.

## STOP-2 Determination

Per Phase 9H instructions:
> **STOP-2**: If manually compiled `.o` links but runtime output is incorrect: STOP. Classify between: kernel ABI, DMA, ObjectFIFO, TAP, runtime argument wiring, K accumulation, kernel implementation.

**Classification: KERNEL IMPLEMENTATION / KERNEL ABI**

The scalar kernel `matmul_scalar` in `aie_kernels/aie2/mm.cc` produces a consistent 6.00x scaling error for M=1. This is not a data movement issue (DMA/ObjectFIFO/TAP all verified correct via MLIR inspection) but a computation error in the kernel or its ABI binding for M=1.

## Artifacts Generated

```
/home/ubuntu12738/Storage/Cache/llama.cpp/experiments/npu-heterogeneous/results/phase9h/
├── PHASE9H_M1_KERNEL_RECOVERY.md       (this report)
├── phase9h_toolchain.json
├── phase9h_compile_paths.json
├── phase9h_correctness.json
└── phase9h_k_sweep.json                (not run — blocked at K=32)
```

## Next Steps Required

1. **Debug scalar kernel**: Inspect `matmul_scalar` assembly or add instrumentation
2. **Check M=1 template instantiation**: Verify `DIM_M=1` doesn't trigger undefined behavior
3. **Verify memref lowering**: Ensure `memref<512xi8>` → `int8*` pointer ABI matches kernel's `b[i*16+col]` indexing
4. **Consider workaround**: Use M=4 (minimum vectorized) with zero-padding if scalar path is fundamentally broken

## Phase 9H Compliance

| Gate | Status |
|------|--------|
| Phase 9H-A: Reproduce/isolate compilation | ✓ PASS |
| Phase 9H-B: Two compilation paths | ✓ Path A works |
| Phase 9H-C: K=32 correctness gate | ✗ FAIL (6.00x error) |
| Phase 9H-D: K-sweep | ✗ BLOCKED |
| Phase 9H-E: Memory safety | Not tested |
| Phase 9H-F: Performance | Not tested |
| Phase 9H-G: ggml-xdna integration | Not tested |
| Phase 9H-H: Qwen SSM validation | Not tested |
| Phase 9H-I: Qwen decode benchmark | Not tested |

**Final Status: STOP-2 — Kernel implementation/ABI issue blocks further progress**