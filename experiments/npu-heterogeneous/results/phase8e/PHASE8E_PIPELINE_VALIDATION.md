# Phase 8-E: Persistent Pipeline Validation Report

## Executive Summary

**VERDICT: PASS**

Double-buffered persistent pipeline (depth=2) achieves **1.69× speedup** over sequential execution, successfully hiding NPU host wait latency and GPU/NPU submission overhead. Triple buffering (depth=3) provides no additional benefit.

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| **Hardware** | Ryzen 7 7840HS (Phoenix XDNA1 NPU + Radeon 780M iGPU) |
| **Workload** | INT8 GEMM: C = A × B, M=1, K=1024, N=16 |
| **Precision** | INT8 inputs, INT32 accumulation (both GPU and NPU) |
| **Iterations** | 100 measured + 20 warmup per depth |
| **Pipeline Depths** | 1 (sequential), 2 (double buffer), 3 (triple buffer) |

## Pipeline Depth Results

| Depth | Median Latency (µs) | P95 Latency (µs) | Speedup vs Sequential | GPU Slowdown |
|-------|---------------------|------------------|----------------------|--------------|
| **1 (Sequential)** | 1072.9 | 1138.9 | 1.00× | +12.6% |
| **2 (Double Buffer)** | **635.4** | **669.7** | **1.69×** | **-21.5%** |
| **3 (Triple Buffer)** | 640.4 | 683.7 | 1.68× | -20.9% |

## Analysis

### Double Buffering (Depth=2) — Optimal

The pipeline executes as follows:
```
Frame i:     GPU submit(i) ───→ GPU exec ───→ GPU wait
             NPU start(i) ─────→ NPU exec ──→ NPU wait
Frame i+1:              GPU submit(i+1) ───→ GPU exec ───→ GPU wait
                         NPU start(i+1) ─────→ NPU exec ──→ NPU wait
```

**Key Observations:**
- NPU host wait (~145 µs) fully overlapped with GPU execution (~387 µs)
- GPU submission latency (~6 µs) hidden in previous frame's NPU execution
- Effective throughput: 1 frame per 635 µs vs 1073 µs sequential
- GPU slowdown is *negative* (-21.5%) because GPU host wait time is reduced when overlapped with NPU

### Triple Buffering (Depth=3) — No Benefit

- Marginally slower than depth=2 (640 vs 635 µs)
- Increased scheduling complexity without additional overlap opportunity
- NPU is single-stream; cannot process multiple frames in parallel
- GPU already has 4 frame resources; depth=2 fully utilizes them

## Throughput Comparison

| Configuration | Throughput (frames/sec) | Speedup |
|---------------|------------------------|---------|
| Sequential (depth=1) | 932 | 1.00× |
| **Double Buffer (depth=2)** | **1,574** | **1.69×** |
| Triple Buffer (depth=3) | 1,562 | 1.68× |

## GPU/NPU Utilization

| Metric | Depth=1 | Depth=2 | Depth=3 |
|--------|---------|---------|---------|
| GPU Device Utilization | 36% | 61% | 60% |
| NPU Device Utilization | 14% | 23% | 23% |
| Effective Overlap | 0% | **58%** | 58% |

*Utilization = device_time / pipeline_time*

## XRT Host Wait Latency Hidden

| Latency Component | Sequential (µs) | Double Buffer (µs) | Hidden |
|-------------------|-----------------|---------------------|--------|
| NPU Host Wait | 145 | ~0 (overlapped) | **100%** |
| GPU Host Wait | 422 | 238 (reduced) | 43% |
| GPU Submit | 6 | 6 (hidden in prev NPU) | 100% |

## Decision Gate Evaluation

| Criterion | Threshold | Depth=2 Actual | Status |
|-----------|-----------|----------------|--------|
| Speedup ≥ 1.10× | 1.10× | **1.69×** | ✅ PASS |
| GPU Slowdown ≤ 20% | 20% | **-21.5%** | ✅ PASS |
| Iterations ≥ 100 | 100 | 100 | ✅ PASS |
| Zero Mismatches | 0 | 0 | ✅ PASS |

**Overall: PASS** — Double-buffered pipeline meets all criteria.

## Conclusion

The persistent double-buffered pipeline (depth=2) successfully achieves the design goal:
- **Hides NPU host wait latency** (145 µs) completely by overlapping with GPU execution
- **Hides GPU submission latency** by pipelining submissions
- **Achieves 1.69× end-to-end speedup** over sequential execution
- **No GPU slowdown** (negative slowdown indicates reduced wait time)

Triple buffering provides no additional benefit for this single-stream NPU workload.

**Next Step**: The heterogeneous GPU||NPU execution model is validated for Phoenix XDNA1. Ready for ggml-xdna/llama.cpp integration targeting Qwen3.5 SSM operators (ssm_beta, ssm_alpha).

## Evidence Files

- `phase8e_raw_benchmarks.json` — Full statistical results for all 3 depths
- This report: `PHASE8E_PIPELINE_VALIDATION.md`