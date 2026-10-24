# Phase 7A: MUL_MAT Operator Characterization Report

## Summary

- Total shapes tested: 6
- CPU+GPU both OK: 5
- NPU OK: 6
- NPU BLOCKED: 0

### Classification Summary

| Class | Meaning | Count |
|-------|---------|-------|
| A | GPU only (NPU too slow) | 5 |
| UNKNOWN | Insufficient data | 1 |

## Detailed Results

| Label | M | K | N | CPU(ms) | GPU(ms) | NPU(ms) | NPU wall(ms) | H2D(ms) | D2H(ms) | MM | Class |
|-------|---|---|---|---------|---------|---------|--------------|---------|---------|----|-------|
| M64_K512_N64 | 64 | 512 | 64 | 0.02 | 0.03 | 0.27 | 0.52 | 0.00 | 0.00 | PASS | A |
| M128_K256_N256 | 128 | 256 | 256 | 0.04 | 0.00 | 1.18 | 1.38 | 0.00 | 0.00 | PASS | UNKNOWN |
| M128_K512_N512 | 128 | 512 | 512 | 0.10 | 0.29 | 2.44 | 2.83 | 0.00 | 0.00 | PASS | A |
| M256_K256_N256 | 256 | 256 | 256 | 0.06 | 0.08 | 2.18 | 2.44 | 0.00 | 0.00 | PASS | A |
| M256_K512_N256 | 256 | 512 | 256 | 0.10 | 0.15 | 2.42 | 2.74 | 0.00 | 0.00 | PASS | A |
| M256_K512_N512 | 256 | 512 | 512 | 0.17 | 0.30 | 4.93 | 5.32 | 0.00 | 0.00 | PASS | A |

## Key Insights

### Transfer Costs (average)
- H2D: 0.003ms
- D2H: 0.001ms

### Scaling Observations

**K=256:**
- M= 256: GPU=0.08ms  NPU=2.44ms  ratio=29.40

**K=512:**
- M=  64: GPU=0.03ms  NPU=0.52ms  ratio=19.35
- M= 128: GPU=0.29ms  NPU=2.83ms  ratio=9.82
- M= 256: GPU=0.15ms  NPU=2.74ms  ratio=17.81
- M= 256: GPU=0.30ms  NPU=5.32ms  ratio=17.63
