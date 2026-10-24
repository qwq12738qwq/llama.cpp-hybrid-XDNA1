# experiments/npu-heterogeneous/

Phase 7A-R: GPU Baseline Recovery and GPU+NPU Overlap

## Status

Environment verified. GPU benchmarks recovered. Real hardware overlap confirmed.

## Files

| File | Purpose |
|------|---------|
| `cpu_bench.py` | CPU baseline (numpy BLAS) |
| `npu_bench.py` | NPU benchmark (iron framework) |
| `gpu_bench_wrapper.py` | GPU benchmark (wraps gpu_bench.cpp) |
| `transfer_bench.py` | Host<->NPU transfer cost |
| `overlap_bench.py` | GPU+NPU overlap test |
| `shape_sweep.py` | Systematic shape sweep |
| `analyze.py` | Analysis and report generation |
| `build_gpu_bench.sh` | Build script for GPU binaries |

## Environment

- Container: npu-dev (rootless Podman)
- GPU: AMD Radeon 780M Graphics (RADV PHOENIX) via Vulkan 1.4
- NPU: Phoenix XDNA1 via /dev/accel/accel0
- LD_LIBRARY_PATH: /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/vulkan

## Key Findings

1. **GPU standalone is 10-30x faster than NPU** for tested shapes
2. **Real hardware overlap confirmed** (65-70% efficiency)
3. **NPU tile constraints block Qwen SSM shapes** (M=2, K=1024, N=16)
4. **GPU shader has tiling limitations** (M<16 causes GPUVM fault for some K/N)
5. **Transfer cost is minimal** (< 6us for NPU, < 1ms for GPU)

## Deliverables

- `docs/xdna/PHASE7_OPERATOR_CHARACTERIZATION.md` — Phase 7A-R report
- `results/*.json` — Raw benchmark data
