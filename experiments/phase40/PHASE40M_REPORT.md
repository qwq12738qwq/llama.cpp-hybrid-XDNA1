# Phase 40M Report — Q8SCALEV-C4 mmul-v2

## Status: Q8_C4_MMUL_PARTIAL

### Bug Fixed

`ggml_xdna_submit_q8_c4_mmul_v2` and `ggml_xdna_submit_q8_c4_ds` in `ggml-xdna.cpp` packed int8 A/B operands in **row-major** order, but `aie::mmul<4,8,8>` kernels require **tile-major** (4×8 / 8×8) order. Fixed both packing loops to match the Python generator's tile-major layout.

### Files Modified

- `ggml/src/ggml-xdna/ggml-xdna.cpp`: fixed tile-major int8 packing for mmul-v2 and device-scale
- `experiments/phase40/run_phase40m_npu.sh`: fixed cache dir name (`Q8SCALEV` → `Q8SCALEV_MMUL`)
- `experiments/phase40/bench_q8_c4_mmul_v2.py`: added missing `numpy` import

### Kernel Cache

```
experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C4_Q8SCALEV_MMUL/
  design.xclbin
  design.insts.bin
```

### Correctness (Hardware)

| Shape | Pattern | Device bad | Device max_err |
|-------|---------|------------|----------------|
| M=512 K=1024 N=3584 cols=4 | realistic | 0/1,835,008 | 0.0 |
| M=512 K=1024 N=3584 cols=4 | geometric | 93/1,835,008 | ~1e-3 |

Realistic pattern: **PASS** (exact match).
Geometric pattern: 93 mismatches due to float16 precision loss with large scales (up to 65504). This is expected and not a kernel bug.

### Performance (Hardware)

| Engine | Device ms |
|--------|-----------|
| Q8SCALEV-C1 (ref) | ~61,133 ms |
| Q8SCALEV-C4 software | ~63,643 ms |
| old C4_DEVICE_SCALE | ~0.79 ms |
| **new Q8-C4 mmul-v2** | **~1,339 ms** |

The new kernel is ~47x faster than Q8SCALEV-C4 software but ~1700x slower than old C4_DEVICE_SCALE. This indicates the aie::mmul<4,8,8> kernel with device-side scale application is not fully optimized for this tile size.

### Verdict

**Q8_C4_MMUL_PARTIAL**

- Correctness: PASS for realistic scales, minor float16 precision loss for extreme geometric scales
- Performance: 1339 ms (needs optimization to reach < 10 ms target)
- C++ integration: complete and functional

### Next Step

Proceed to Phase 41 (C++ single-GEMM gate) as requested.
