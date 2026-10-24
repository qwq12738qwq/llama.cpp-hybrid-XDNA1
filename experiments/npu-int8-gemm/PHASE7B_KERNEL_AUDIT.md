# Phase 7B Kernel Audit — Recovered Capabilities

## Validated Multi-Shape XDNA GEMM Kernels

All kernels below use **real NPU execution** on Phoenix XDNA1 (npu1/aie2), validated with CPU reference correctness (0 mismatches for int32).

| Shape (M×K×N) | Directory | Status | Key Features |
|---|---|---|---|
| **128×1024×128** | `fixed_M128_K1024_N128` | ✅ PASS | Single column, M=128 tile |
| **128×1024×256** | `fixed_M128_K1024_N256` | ✅ PASS | Single column, M=128 tile |
| **128×1024×512** | `fixed_M128_K1024_N512` | ✅ PASS | Single column, M=128 tile |
| **128×1024×64** | `fixed_M128_K1024_N64` | ✅ PASS | Single column, M=128 tile |
| **256×1024×128** | `fixed_M256_K1024_N128` | ✅ PASS | Single column, M=256 tile |
| **64×1024×128** | `fixed_M64_K1024_N128` | ✅ PASS | Single column, M=64 tile |
| **64×1024×64** | `verify_verify_baseline_M64_N64` | ✅ PASS | Baseline correctness |
| **128×1024×6144** | `k1024_n6144_m128_run` | ✅ PASS | Multi-column (N=6144 = 96×64) |
| **256×1024×6144** | `k1024_n6144_m256_run` | ✅ PASS | Multi-column, M=256 |
| **64×1024×6144** | `k1024_n6144_m64_run` | ✅ PASS | Multi-column, M=64 |
| **128×1024×1024** | `k1024_n1024_m128_run` | ✅ PASS | Single column, N=1024 |
| **512×1024×1024** | `k1024_n1024_m512_run` | ✅ PASS | Large M, N=1024 |
| **128×1024×128** | `k1024_n128_m128_run` | ✅ PASS | Re-validated |
| **128×1024×256** | `k1024_n256_m128_run` | ✅ PASS | Re-validated |
| **128×1024×512** | `k1024_n512_m128_run` | ✅ PASS | Re-validated |
| **128×512×128** | `k512_n128_m128_run` | ✅ PASS | K=512 variant |
| **128×512×64** | `k512_n64_m128_run` | ✅ PASS | K=512 variant |
| **64×512×64** | `k512_n64_run` | ✅ PASS | K=512 variant |
| **512×1024×16** | `qwen_ssm_run` | ✅ PASS | Qwen SSM: M=512, N=16 (ssm_beta/alpha) |
| **512×1024×6144** | `qwen_qkv_run` | ✅ PASS | Qwen QKV: M=512, N=6144 |

## Key Technical Achievements (Phase 7B-R)

### 1. TensorTiler2D Fix (Critical)
**Problem**: `pattern_repeat=N//n` in A-tile TensorTiler2D caused BD aliasing when M//m > 1 AND N//n > 1 AND K > 512.
**Fix**: `pattern_repeat=1` in `int8_gemm_multicol_fixed.py:103`
```python
a_taps = TensorTiler2D.group_tiler(
    (M, K), (m, k), (1, K // k), pattern_repeat=1
)
```

### 2. BD Size Limit Handling
**Problem**: B matrix DMA descriptor limited to 65536 bytes (MAX_BD_BYTES).
**Fix**: Split B L3L2 taps by N-tiles when K×N > 65536 (`int8_gemm_multicol_fixed.py:114-131`)

### 3. N-Parallel Multi-Column Design
**Design**: `int8_gemm_nparallel.py` - spatial parallelism across columns
- Each column computes independent output slice: C[:, col_slice] = A × B[:, col_slice]
- No numerical reduction needed - pure spatial parallelism
- Validated up to N=6144 (96 tiles of N=64)

### 4. Qwen-Specific Shapes
- **SSM (ssm_beta/alpha)**: M=512, K=1024, N=16 — `qwen_ssm_run`
- **QKV**: M=512, K=1024, N=6144 — `qwen_qkv_run`
- Both use multi-column DMA with repeat_count for A matrix broadcast

### 5. Core Kernel Reuse
All designs share the same core MAC kernel:
- Tile: M=64, K=64, N=64 (matches aie2 int8 MAC dims 4×8×8)
- Vectorized: true
- Kernel: `mm(dim_m=64, dim_k=64, dim_n=64, input_dtype=np.int8, output_dtype=np.int32, vectorized=True)`
- Core loop: M//m × N//n iterations, each doing K//k MAC steps

## Generator Scripts

| Script | Purpose | Parameterized |
|---|---|---|
| `int8_gemm_multicol_fixed.py` | Canonical fixed multi-column | M, K, N, tile sizes |
| `int8_gemm_nparallel.py` | N-parallel multi-column | M, K, N, n_cols |
| `gen_m1_k32_n16_v2.py` | Scalar M=1 microkernel | M=1, K=32, N=16 only |

## What Prevents Direct ggml-xdna Use

1. **Hardcoded xclbin paths** in `ggml-xdna.cpp:98-99` — only Phase 9h xclbin
2. **Hardcoded N=512 restriction** in `is_mul_mat_supported()` — rejects all other N
3. **M=1 kernel assumption** — engine created with M=1, host loops over M
4. **No kernel registry** — no mechanism to select/compile specialization by shape
5. **No MLIR-AIE generator integration** — cannot compile new shapes on demand
6. **Quantization hardcoded** — only Q8_0 weight, Q8_0/F32 input

## Reusable Infrastructure

| Component | Location | Reusable? |
|---|---|---|
| Core MAC kernel (64×64×64) | `matmul_i8_i32.cc` | ✅ Yes |
| Multi-column DMA sequence | `int8_gemm_multicol_fixed.py` | ✅ Yes |
| N-parallel design | `int8_gemm_nparallel.py` | ✅ Yes |
| BD splitting logic | Both generators | ✅ Yes |
| TensorTiler2D pattern_repeat=1 fix | Both generators | ✅ Yes |
| Compile-time specialization via `@iron.jit` + `.specialize()` | All generators | ✅ Yes |
| xclbin/insts caching to work_dir | All generators | ✅ Yes |

## Next Steps for Generic Backend

1. Extract parameterized generator from `int8_gemm_multicol_fixed.py`
2. Create kernel registry keyed by (M_tile, K_tile, N_tile, quantization)
3. Implement on-demand compilation with caching to `/Storage/Cache/xdna_kernels/`
4. Replace hardcoded supports_op() with registry lookup + generator fallback
5. Generalize M-tiling to use M_TILE ≥ 32 inside kernel (not host loop)