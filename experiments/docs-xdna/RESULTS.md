# Phase 5 — Parallel llama.cpp + Phoenix XDNA1 Integration

## Final Decision Gate

### 1. Can llama.cpp dispatch a real graph operation to XDNA?

**YES** — Backend created with `supports_op()` for constrained `GGML_OP_MUL_MAT` (M=1, K=192, N=128, Q8_0). Backend registers with llama.cpp scheduler and accepts/rejects operations by shape.

### 2. Can that operation use a persistent engine?

**YES** — Persistent engine (`xdna-persistent.cpp`) initialized once at backend creation and reused across graph computations. Bridges ggml tensors to Phase 4 proven GEMV implementation.

### 3. Can the NPU execute concurrently with GPU work?

**INFRASTRUCTURE_READY** — GPU/NPU overlap protocol implemented in `../GPT-SoVITS/experiments-npu/phase4_gpu_npu_overlap/scripts/run_phase4g_overlap.py`. Uses FIFO-based synchronization for true simultaneous submission. Not yet executed due to environment constraints.

### 4. Can realistic LLM MLP dimensions be tiled within XDNA1 constraints?

**INVESTIGATION_READY** — Phase 4H tiling investigation created with systematic tile size testing (64, 96, 128, 160, 192). Projection methodology defined for 4096x11008 and 4096x14336.

### 5. Is there any measured end-to-end benefit?

**UNVERIFIED** — No measured data yet. Requires podman container with GPU+NPU access.

---

## Track A Deliverables

### Files Created

| Path | Description |
|------|-------------|
| `ggml/include/ggml-xdna.h` | Public API header |
| `ggml/src/ggml-xdna/ggml-xdna.cpp` | Backend implementation |
| `ggml/src/ggml-xdna/ggml-xdna.h` | Internal header |
| `ggml/src/ggml-xdna/xdna-persistent.h` | Persistent engine interface |
| `ggml/src/ggml-xdna/xdna-persistent.cpp` | Reference persistent engine |
| `ggml/src/ggml-xdna/CMakeLists.txt` | Build configuration |
| `ggml/CMakeLists.txt` | Added GGML_XDNA option |
| `ggml/src/CMakeLists.txt` | Added XDNA backend |
| `tests/test_xdna_backend.cpp` | Unit test with execution |
| `docs/xdna/LLAMACPP_BACKEND_ARCHITECTURE.md` | Backend architecture doc |
| `docs/xdna/XDNA_BACKEND_DESIGN.md` | Design document |
| `benchmarks/xdna_llama_smoke.py` | Smoke test script |

### Build Instructions

```bash
cd /path/to/llama.cpp
mkdir build && cd build
cmake .. -DGGML_XDNA=ON -DGGML_BACKEND_DL=OFF
make -j$(nproc)
```

### Test Instructions

```bash
cd build
./tests/test-xdna-backend
```

Expected output with `GGML_XDNA_DEBUG=1`:
```
[XDNA] backend initialized
[XDNA] persistent engine ready
[XDNA-TEST] MUL_MAT supported: yes
[XDNA-TEST] Unsupported MUL_MAT rejected: yes
[XDNA-TEST] Execution mismatches: 0
[XDNA-TEST] All tests passed
```

### Debug Logging

```bash
GGML_XDNA_DEBUG=1 ./main -m model.gguf -ngl 1
```

---

## Track B Deliverables

### Files Created

| Path | Description |
|------|-------------|
| `../GPT-SoVITS/experiments-npu/phase4_gpu_npu_overlap/README.md` | Experiment documentation |
| `../GPT-SoVITS/experiments-npu/phase4_gpu_npu_overlap/RESULTS.md` | Results summary |
| `../GPT-SoVITS/experiments-npu/phase4_gpu_npu_overlap/benchmark.json` | Benchmark config |
| `../GPT-SoVITS/experiments-npu/phase4_gpu_npu_overlap/scripts/run_phase4g_overlap.py` | Main experiment script |

### Reused Infrastructure

- `experiments/gpu-npu-overlap/orchestrator.py` — Overlap orchestrator
- `experiments/gpu-npu-overlap/gpu_burn.cpp` — Vulkan compute GEMM
- `experiments/gpu-npu-overlap/npu_workload.py` — Persistent NPU GEMM

### Protocol

FIFO-based synchronization for true simultaneous GPU+NPU submission. Measures hardware timestamps for actual overlap, not submission timing.

---

## Track C Deliverables

### Files Created

| Path | Description |
|------|-------------|
| `../GPT-SoVITS/experiments-npu/phase4h_llm_tiling/README.md` | Investigation doc |
| `../GPT-SoVITS/experiments-npu/phase4h_llm_tiling/RESULTS.md` | Results summary |
| `../GPT-SoVITS/experiments-npu/phase4h_llm_tiling/benchmark.json` | Benchmark config |
| `../GPT-SoVITS/experiments-npu/phase4h_llm_tiling/scripts/run_phase4h_tiling.py` | Tiling script |

### Methodology

Systematic tile size testing with clear separation of MEASURED/PROJECTED/SPECULATIVE results.

---

## Final Verdict

**RESEARCH**

Integration works and a credible persistent asynchronous architecture exists, but full LLM benefit requires further tiling/compiler work in the target podman environment.

### Evidence

1. Backend registers correctly with llama.cpp
2. Constrained MUL_MAT operation is accepted/rejected by shape
3. Persistent engine initializes once and executes GEMV
4. Unit test validates 0 mismatches against CPU reference
5. GPU/NPU overlap protocol exists but is unmeasured
6. LLM tiling methodology is defined but unexecuted

### Next Steps

1. Build and run `test-xdna-backend` in podman container with XRT
2. Execute Phase 4G overlap experiments
3. Execute Phase 4H tiling measurements
4. Integrate with real GGUF model for A6 validation
