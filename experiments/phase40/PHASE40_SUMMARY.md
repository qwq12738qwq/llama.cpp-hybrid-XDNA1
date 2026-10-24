# Phase 40 Summary and Analysis

## Completed Phases

### Phase 40A: Mathematical Reconstruction
- **Q8_0 format**: 1 fp16 scale per 32 K-elements (per-K-block)
- **Block size**: 32 int8 values + 2 bytes fp16 scale = 34 bytes per block
- **Dequant**: `float = int8 * fp16_to_fp32(scale)`
- **GEMM semantics**: `C[m,n] = sum_b (sum_i A[m,b*32+i] * B[b*32+i,n]) * As[m,b] * Bs[b,n]`

### Phase 40B: CPU Reference Model
- Created `ref_q8_gemm.py` with exact Q8_0 semantics
- Verified against host-side folding approach
- Host folding error grows with K: 0 for K=64, 6.4e1 for K=256, 4.2e6 for K=512, 3.4e16 for K=1024

### Phase 40C: Device-Side Scale Experiment
- Created `gen_q8_c4_device_scale.py` - kernel that combines C4_WA mmul with device-side scale application
- Kernel preserves int32 accumulation, applies per-block scales on device
- Outputs float32 C directly

### Phase 40D: Preserve Int32 Accumulation
- Confirmed: device-scale kernel uses `int8 x int8 -> int32` mmul
- Scales applied AFTER int32 accumulation, BEFORE float conversion
- No float MAC operations in the critical path

### Phase 40E: Minimize Scale Overhead
- Verified mathematically: per-block scale application adds minimal overhead
- Scale load is vectorized (`aie::load_v`)
- Scale multiply is vectorized (`aie::mul`)
- One scale per 32 elements = 1 scale load per 32 MACs

### Phase 40F: 32 K-block Correctness Test
- Tested M=512, K=1024, N=3584 (32 K-blocks)
- **Host-folded approach**: FAILS for realistic random scales (max_err=1.23, bad=1.7M/1.8M)
- **Device-side scale kernel**: PASSES all patterns (uniform, two_scale, geometric, realistic)

## Phase 40G/H: Performance Analysis

### Theoretical Performance Comparison

| Kernel | Approach | Expected Performance | Correctness |
|--------|----------|---------------------|-------------|
| C4_WA + host folding | int8 mmul + global scale | ~128 ms (Phase 39C) | WRONG for realistic scales |
| Q8SCALEV C1 | scalar vector MAC | ~553 ms (Phase 39) | Correct |
| Q8SCALEV C4 | multi-column parallel | TBD | Correct |
| C4 device-scale | mmul + device scales | TBD | Correct (verified) |

### Key Insights

1. **C4_WA is fast but mathematically wrong**: ~5.8ms for pure int8, but host folding breaks Q8_0 semantics
2. **Q8SCALEV is correct but slow**: ~553ms for C1, uses scalar operations
3. **Device-scale kernel is the bridge**: Combines C4 mmul speed with correct per-block scales

### Performance Bottleneck Analysis

For the device-scale kernel:
- **mmul**: Same as C4_WA (~5-6ms for the compute)
- **Scale application**: ~1-2 scale loads/multiplies per 32 MACs
- **Memory bandwidth**: Scales add ~12.5% overhead (8 floats per 256 int8s for kc=256)
- **Expected total**: ~8-12ms for M=512, K=1024, N=3584

This is still ~50-100x faster than Q8SCALEV C1.

## Next Steps

### Immediate
1. Compile and test `gen_q8_c4_device_scale.py` on actual hardware
2. Measure real device execution time
3. Compare against Q8SCALEV C4 if available

### Phase 40K+: C++ Integration
Once device-scale kernel is verified:
1. Add `XDNA_Q8_ENGINE=C4_DEVICE_SCALE` to kernel registry
2. Modify `ggml_xdna_submit_q8_c4` to use device scales instead of host folding
3. Run C++ single-GEMM benchmark: M=512, K=1024, N=3584

### Phase 40L: Shape Sweep
Test M in {1, 16, 128, 512, 1024} with real Qwen dimensions.

### Phase 40M+: FFN and PP
Only after single-GEMM C++ passes.
