# Phase 7D: GPU+NPU Critical-Path Cooperative Execution Root-Cause Audit

## Executive Summary

**ROOT CAUSE: BENCHMARK METHODOLOGY ERROR (E) compounded by RUNTIME/SCHEDULER INTERFERENCE (C)**

Phase 7C's reported 20–145× slowdown was **not** caused by architectural memory bandwidth contention, NPU DMA, or power management. The catastrophic slowdown originated from **flawed benchmark methodology** in Phase 7C's :

1. **Process startup overhead**: Each concurrent run spawned fresh GPU and NPU processes that performed full Vulkan/XRT initialization, compilation, and buffer allocation **inside the measured window**.
2. **No true synchronization**: The harness used  +  but measured wall time from Python's  around thread joins, capturing process startup, JIT compilation, and driver initialization rather than kernel overlap.
3. **No persistent contexts**: GPU () and NPU () recreated devices, buffers, and pipelines every iteration.
4. **Incorrect GPU timing**:  JSON lacks wall timestamps; Phase 7C used  (which includes host↔device copies) as GPU
