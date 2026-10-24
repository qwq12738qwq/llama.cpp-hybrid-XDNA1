# Phase 40M v2: Q8SCALEV-C4 mmul-v2 Multi-Row Progress

## Summary

Successfully created a multi-row version of the Q8SCALEV-C4 mmul-v2 kernel using IRON `Worker.grid`. The v2 generator compiles and produces correct results for 1x1 configuration. The 4x4 multi-row configuration compiles but experiences deadlock on hardware due to shared ObjectFifos.

## What Works

### v1 Kernel (4 workers, 4 compute tiles)
- **Status**: COMPLETE
- **Device latency**: ~5.78 ms (median)
- **Correctness**: 0 bad values (matches CPU reference)
- **Kernel cache**: `/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C4_WA/`

### v2 Kernel 1x1 (1 worker, 1 compute tile)
- **Status**: COMPLETE
- **Device latency**: ~1.34 s (slower than v1 due to different dataflow)
- **Correctness**: 0 bad values (matches CPU reference)
- **XCLBIN**: `/tmp/q8c4mmul_v2_r1c1/design.xclbin`

## Key Fixes in v2 Generator

1. **Use a_taps instead of A_tiles**: The original v2 generator used `A_tiles` which only specified int8 data without scales. Fixed by using `a_taps` which includes the full chunk (int8 + scales).

2. **Removed dims_to_stream**: The `dims_to_stream` parameter on ObjectFifos was causing incorrect data streaming. Removing it fixed the correctness issues.

3. **Fixed c_taps offsets**: Changed from `col * n_nt_c + nt` to `col * nb_n_per_col + nt` to correctly index into the C matrix for multi-row workers.

4. **Fixed A_L2L1 creation for n_rows=1**: The split operation with duplicate offsets caused compilation errors. Fixed by using forward instead of split when there's only one row.

## Remaining Issues

### Deadlock in Multi-Row Configuration (4x4)
- **Status**: BLOCKED
- **Issue**: Workers in the same row share the same A ObjectFifo. With `fifo_depth=1`, this causes deadlock when multiple workers try to acquire A simultaneously.
- **Attempted fix**: Increasing `fifo_depth` to 2 causes memory allocation failures on the device.
- **Root cause**: The C4_WA dataflow is designed for n_cols workers, not n_rows × n_cols workers. Extending it to multi-row requires rethinking the dataflow.

### Hardware State
- **Status**: The hardware is currently in a bad state (DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed)
- **Impact**: Cannot run further hardware tests until the device is reset
- **Recommendation**: Restart the container or reload the AMD XDNA driver

## Recommendations

1. **Short-term**: The v1 kernel already meets the performance goal (<10 ms). Use it as the production kernel.

2. **Medium-term**: To utilize all 16 compute tiles, redesign the dataflow from scratch for 16 workers. Consider:
   - Using MemTile fan-out to share A/B data among workers
   - Using a ping-pong buffer scheme to avoid deadlock
   - Using different ObjectFifo topologies (e.g., tree, butterfly)

3. **Long-term**: Consider using a different kernel or dataflow that's specifically designed for the Phoenix XDNA1 architecture's 4×4 compute tile grid.

## Files Modified

- `/workspace/llama.cpp/experiments/phase40/gen_q8_c4_mmul_v2_v2.py`: Multi-row v2 generator
- `/tmp/compile_v2_jit.py`: JIT compilation script for v2
- `/tmp/compile_v2_r1c1.py`: 1x1 compilation script
- `/tmp/compile_v2_r2c2.py`: 2x2 compilation script
- `/tmp/compile_v2_r4c1.py`: 4x1 compilation script
- `/tmp/bench_mmul_v2_v2.py`: Hardware benchmark script for v2

## Next Steps

1. Reset the hardware (restart container or reload driver)
2. Verify v1 kernel still works
3. Test v2 1x1 kernel to confirm it still works
4. Redesign multi-row dataflow to avoid deadlock
5. Benchmark multi-row kernel once deadlock is fixed
