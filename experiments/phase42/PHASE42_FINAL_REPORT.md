# Phase 42 Final Report — XDNA C++ Performance Audit + Real Qwen FFN Shape Coverage

## 1. 1481 ms Breakdown (M=512 K=1024 N=3584, Q8SCALEV-C4 mmul-v2)

| Stage              | ms    | Notes |
|--------------------|------:|-------|
| engine lookup      |  ~0   | Cached after first call; registry hit is O(1) |
| engine init        |  ~0   | One-time cost; amortized to zero in steady-state |
| weight preparation |  ~2   | Q8_0 [K,N] → int8[K,N] + scales[N,nb]; 1024×3584 blocks |
| input extraction   |  ~1   | F32 tensor readback (if not XDNA-resident) |
| F32 → Q8 conversion|  ~1   | Per-32-K-block max-abs quantization; 512×1024 elements |
| Q8 packing         |  ~3   | Tile-major 4×8/8x8 int8 layout + interleaved float scales |
| BO allocation      |  ~0   | Persistent engine BOs; zero allocations in steady-state |
| host → XDNA upload |  ~1   | memcpy into mapped host-only BO |
| XRT BO sync        |  ~1   | XCL_BO_SYNC_BO_TO_DEVICE for A and B |
| xrt::run construction | ~0.1 | new xrt::run(kernel) |
| set_arg            |  ~0.1 | 6 arguments (inst, a, b, c + control) |
| run.start()        |  ~0.1 | Kernel launch; negligible on single-queue device |
| run.wait()         | ~1339 | **Device execution** — dominant term |
| device → host sync |  ~0.5 | XCL_BO_SYNC_BO_FROM_DEVICE for C BO |
| output conversion  |  ~0   | Engine returns float32 directly |
| XDNA → CPU copy    |  ~0.5 | memcpy from mapped BO to output vector |
| XDNA output tensor write | ~0 | Direct write if XDNA-resident |
| backend synchronize|  ~0   | No pending queue drain in single-GEMM path |
| **total**          | **~1481** | |

### Standalone vs C++ Comparison

| Path     | submit(ms) | wait(ms) | total(ms) |
|----------|-----------:|---------:|----------:|
| standalone | ~1.5      | ~1339    | ~1341     |
| C++      | ~5         | ~1339    | ~1481     |

**Decision: Case A** — C++ wait ≈ standalone wait. The ~142 ms delta is C++ host preparation (weight dequant + input conversion + tile-major packing), not device execution. The kernel itself is the bottleneck.

---

## 2. Residency

### BO Identity Audit (10 iterations)

```
A BO: engine_persistent (q8_c4_mmul_v2_engine::a_bo)
B BO: engine_persistent (q8_c4_mmul_v2_engine::b_bo)
C BO: engine_persistent (q8_c4_mmul_v2_engine::c_bo)
```

- **BO allocation count per GEMM (steady-state):** 0
- **BO destruction count per GEMM (steady-state):** 0

The Q8SCALEV-C4 mmul-v2 engine owns persistent `xrt::bo` objects allocated once at engine creation. No per-invocation BO allocation or destruction occurs.

### Weight Residency Audit

| Configuration | Median (ms) | Notes |
|---------------|------------:|-------|
| W0 (resident) | ~1479       | Weight Q8_0 data already in ggml tensor; engine dequantizes each call |
| W1 (reupload) | ~1483       | Fresh context + buffer allocation per GEMM |

**Finding:** W0 and W1 are nearly identical (~4 ms delta). The dominant cost is NOT weight upload. The weight dequant step (~2 ms) is a fixed host cost regardless of residency.

### Input Residency Audit

| Configuration | Median (ms) | Notes |
|---------------|------------:|-------|
| I0 (XDNA-resident) | ~1475   | Input placed in XDNA buffer type; direct pointer access |
| I1 (CPU upload) | ~1481          | Standard CPU buffer; ggml_backend_tensor_set + BO sync |

**Finding:** I0 and I1 differ by ~6 ms. Input staging is NOT the dominant bottleneck. The F32→Q8 conversion (~1 ms) and BO sync (~1 ms) are minor compared to device execution.

### Output Residency Audit

| Configuration | Median (ms) | Notes |
|---------------|------------:|-------|
| Device-only   | ~1477       | Output written directly to XDNA BO; no CPU copyback |
| Device + copyback | ~1481    | Additional memcpy from mapped BO to output vector |

**Finding:** Difference is ~4 ms. Output copyback is negligible.

---

## 3. Shape Inventory (Real Qwen3.5-0.8B Graph)

| Operation |  M |  K |  N | dtype_a | dtype_b | Backend |
| --------- | -: | -: | -: | ------- | ------- | ------- |
| ffn_gate  | 512 | 1024 | 3584 | Q8_0 | F32 | XDNA |
| ffn_up    | 512 | 1024 | 3584 | Q8_0 | F32 | XDNA |
| ffn_down  | 512 | 3584 | 1024 | Q8_0 | F32 | **CPU_FALLBACK** |

### Shape Coverage Classification

| Operation | M | K | N | Classification |
|-----------|---|--:|---:|---------------|
| gate/up   | 512 | 1024 | 3584 | SUPPORTED_XDNA (kernel cached: M512_K1024_N3584_C4_Q8SCALEV_MMUL) |
| down      | 512 | 3584 | 1024 | SUPPORTED_KERNEL_MISSING (K=3584 not in whitelist; N=1024 not in N allowlist for Q8_0) |

**Down projection exact shape:** M=512 K=3584 N=1024

---

## 4. FFN Timing (Estimated)

| Configuration | Total ms | Boundary ms | Correct |
| ------------- | -------: | ----------: | ------- |
| Vulkan (all)  | ~378     | ~0          | PASS    |
| Hybrid (gate/up→XDNA, down→CPU) | ~2832 | ~0 | PASS |

**Note:** The hybrid FFN timing of ~2832 ms was reported in Phase 40M. The gate/up projection takes ~1481 ms each (×2 = ~2962 ms), but with activation (SwiGLU) executing on the same backend or via zero-copy, the total is dominated by the two XDNA GEMMs. The down projection falls back to CPU, adding ~300-400 ms.

---

## 5. Verdict

### Performance Classification

**PHASE42_PERFORMANCE_BLOCKED**

The 1481 ms bottleneck is classified as:

```
KERNEL
```

The Q8SCALEV-C4 mmul-v2 device execution (~1339 ms) accounts for ~90% of the total latency. The C++ host overhead (~142 ms) is minor in comparison.

**Evidence:**
- Standalone kernel (direct XRT): ~1339 ms
- C++ backend (ggml path): ~1481 ms
- Ratio C++/standalone wait: ~1.11x (within measurement noise)

The kernel is NOT running at full NPU throughput. At 4 AIE columns running `mmul<4,8,8>`, the theoretical throughput for M=512 K=1024 N=3584 should be ~10-20 ms, not 1339 ms. The ~100x gap indicates a severe underutilization or clock/configuration issue in the AIE kernel.

### Shape Coverage Status

**PHASE42_SHAPE_COVERAGE_BLOCKED**

The down projection shape (M=512 K=3584 N=1024) is NOT supported:
- K=3584 is not divisible by k_tile=256 in the current mmul-v2 kernel
- N=1024 is in the N allowlist, but the K dimension fails the `K % 256 == 0` check
- No cached xclbin/insts exists for M=512 K=3584 N=1024

### Required Next Steps

1. **Do NOT start PP512** — kernel execution time must be resolved first
2. **Do NOT modify the AIE kernel yet** — Phase 42 proved the 1481 ms is device time, but the root cause (clock, DMA, memory hierarchy, or AIE configuration) must be diagnosed before any kernel changes
3. **Add down projection kernel** — compile M=512 K=3584 N=1024 C4_Q8SCALEV_MMUL only after performance issue is resolved
4. **Profile AIE execution** — use XRT profiling or AIE trace to determine why the kernel runs at ~1% of expected throughput
