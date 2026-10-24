// Phase 38B: XDNA Resident Execution Microbenchmark
//
// Measures the cost of data movement vs XRT execution for Qwen GEMM shapes.
// Independent of llama-bench - directly exercises the XDNA backend API.

#include "ggml-backend.h"
#include "ggml-xdna.h"
#include "xdna-buffer.h"
#include "ggml-alloc.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

// Qwen3.5-0.8B PP512 shape
static const int M = 512;
static const int K = 1024;
static const int N = 3584;

static double now_ms(void) {
    auto t = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

// Simple CPU reference GEMV for validation
static void cpu_gemv_f32(const float * A, const float * B, float * C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[n * M + m] = sum;
        }
    }
}

static int count_mismatches_f32(const float * a, const float * b, int n, float tol = 0.05f) {
    int mm = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) mm++;
    }
    return mm;
}

static void fill_random_f32(std::vector<float> & v, float scale = 1.0f) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

// Quantize F32 [n_rows, K] to Q8_0 (per-32-block along K)
static void quantize_f32_to_q8_0(const float * src, int n_rows, int K, std::vector<uint8_t> & out_q8) {
    int nb = K / 32;
    out_q8.resize((size_t)n_rows * nb * 34);
    for (int row = 0; row < n_rows; row++) {
        for (int b = 0; b < nb; b++) {
            size_t block_idx = (size_t)row * nb + b;
            const float * blk = src + (size_t)row * K + (size_t)b * 32;
            float max_abs = 0;
            for (int k = 0; k < 32; k++) max_abs = std::max(max_abs, fabsf(blk[k]));
            float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / 127.0f);
            ggml_fp16_t scale_h = ggml_fp32_to_fp16(scale);
            memcpy(&out_q8[block_idx * 34], &scale_h, sizeof(ggml_fp16_t));
            for (int k = 0; k < 32; k++) {
                int q = (int)llroundf(blk[k] / scale);
                q = std::max(-127, std::min(127, q));
                out_q8[block_idx * 34 + 2 + k] = (uint8_t)(q & 0xFF);
            }
        }
    }
}

// Dequantize Q8_0 back to F32 for comparison
static void dequantize_q8_0_to_f32(const uint8_t * q8, int n_rows, int K, std::vector<float> & out_f32) {
    int nb = K / 32;
    out_f32.resize((size_t)n_rows * K);
    for (int row = 0; row < n_rows; row++) {
        for (int b = 0; b < nb; b++) {
            size_t block_idx = (size_t)row * nb + b;
            float scale = ggml_fp16_to_fp32(*(const ggml_fp16_t *)&q8[block_idx * 34]);
            for (int k = 0; k < 32; k++) {
                int8_t q = (int8_t)q8[block_idx * 34 + 2 + k];
                out_f32[(size_t)row * K + b * 32 + k] = (float)q * scale;
            }
        }
    }
}

int main(void) {
    fprintf(stderr, "[BENCH] Phase 38B: XDNA Resident Execution Microbenchmark\n");
    fprintf(stderr, "[BENCH] Shape: M=%d K=%d N=%d\n", M, K, N);
    
    // Initialize XDNA backend
    ggml_backend_reg_t reg = ggml_backend_xdna_reg();
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        fprintf(stderr, "[BENCH] ERROR: XDNA backend not available\n");
        return 1;
    }
    
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_t backend = ggml_backend_xdna_init();
    if (!backend) {
        fprintf(stderr, "[BENCH] ERROR: XDNA backend init failed\n");
        return 1;
    }
    
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t xdna_buft = ggml_xdna_buffer_type();
    
    fprintf(stderr, "[BENCH] XDNA buffer type: %s\n", ggml_backend_buft_name(xdna_buft));
    fprintf(stderr, "[BENCH] CPU buffer type: %s\n", ggml_backend_buft_name(cpu_buft));
    
    // Prepare data
    srand(42);
    std::vector<float> act_f32(M * K);
    std::vector<float> wt_f32(N * K);  // weight in [N, K] row-major = [K, N] ggml col-major
    std::vector<float> ref_f32(N * M);
    fill_random_f32(act_f32, 0.5f);
    fill_random_f32(wt_f32, 0.5f);
    
    // CPU reference: matmul(act, wt) where act=[M,K], wt=[K,N]
    cpu_gemv_f32(act_f32.data(), wt_f32.data(), ref_f32.data(), M, K, N);
    
    // Quantize weight to Q8_0
    // weight is [N, K] row-major, quantize per row with K as inner dim
    std::vector<uint8_t> wt_q8;
    quantize_f32_to_q8_0(wt_f32.data(), N, K, wt_q8);
    
    // Verify quantization round-trip
    std::vector<float> wt_dequant;
    dequantize_q8_0_to_f32(wt_q8.data(), N, K, wt_dequant);
    int qmm = count_mismatches_f32(wt_dequant.data(), wt_f32.data(), N * K, 0.5f);
    fprintf(stderr, "[BENCH] Quantization mismatches: %d / %d\n", qmm, N * K);
    
    size_t act_bytes = act_f32.size() * sizeof(float);
    size_t wt_bytes = wt_q8.size();
    size_t dst_bytes = ref_f32.size() * sizeof(float);
    
    fprintf(stderr, "[BENCH] Data sizes: act=%.2f MB, wt=%.2f MB, dst=%.2f MB\n",
            act_bytes/1024.0/1024.0, wt_bytes/1024.0/1024.0, dst_bytes/1024.0/1024.0);
    
    // Build graph for MUL_MAT
    // In ggml: src0=[K,N] weight^T, src1=[K,M] input^T, dst=[N,M]=[M,N] row-major
    
    // Test 1: Current path - allocate in CPU buffer, copy data each iteration
    fprintf(stderr, "\n[BENCH] === Test 1: Current path (CPU buffer) ===\n");
    {
        struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
        struct ggml_context * ctx = ggml_init(params);
        
        struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);
        
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);
        
        ggml_backend_buffer_t in_buf = ggml_backend_buft_alloc_buffer(cpu_buft, act_bytes);
        ggml_backend_buffer_t wt_buf = ggml_backend_buft_alloc_buffer(cpu_buft, wt_bytes);
        ggml_backend_buffer_t out_buf = ggml_backend_buft_alloc_buffer(cpu_buft, dst_bytes);
        
        if (!in_buf || !wt_buf || !out_buf) {
            fprintf(stderr, "[BENCH] CPU buffer allocation failed\n");
            return 1;
        }
        
        ggml_backend_tensor_alloc(in_buf, src1, ggml_backend_buffer_get_base(in_buf));
        ggml_backend_tensor_alloc(wt_buf, src0, ggml_backend_buffer_get_base(wt_buf));
        ggml_backend_tensor_alloc(out_buf, dst, ggml_backend_buffer_get_base(out_buf));
        
        // Set input data (goes through ggml_backend_tensor_set -> CPU buffer)
        ggml_backend_tensor_set(src1, act_f32.data(), 0, act_bytes);
        ggml_backend_tensor_set(src0, wt_q8.data(), 0, wt_bytes);
        
        // Warmup
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        
        // Benchmark: measure per-GEMM overhead including data copies
        double t0 = now_ms();
        for (int i = 0; i < 5; i++) {
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
        }
        double t1 = now_ms();
        double avg_ms = (t1 - t0) / 5.0;
        fprintf(stderr, "[BENCH] Test 1 avg: %.2f ms per GEMM\n", avg_ms);
        
        // Read result
        ggml_backend_tensor_get(dst, ref_f32.data(), 0, dst_bytes);
        
        int mm = count_mismatches_f32(ref_f32.data(), ref_f32.data(), N * M, 0.05f);
        fprintf(stderr, "[BENCH] Test 1 self-check: %d mismatches\n", mm);
        
        ggml_free(ctx);
    }
    
    // Test 2: XDNA buffer path - allocate in XDNA buffer, reuse
    fprintf(stderr, "\n[BENCH] === Test 2: XDNA resident buffer ===\n");
    {
        struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
        struct ggml_context * ctx = ggml_init(params);
        
        struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);
        
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);
        
        ggml_backend_buffer_t in_buf = ggml_backend_buft_alloc_buffer(xdna_buft, act_bytes);
        ggml_backend_buffer_t wt_buf = ggml_backend_buft_alloc_buffer(xdna_buft, wt_bytes);
        ggml_backend_buffer_t out_buf = ggml_backend_buft_alloc_buffer(xdna_buft, dst_bytes);
        
        if (!in_buf || !wt_buf || !out_buf) {
            fprintf(stderr, "[BENCH] XDNA buffer allocation failed, skipping\n");
        } else {
            ggml_backend_tensor_alloc(in_buf, src1, ggml_backend_buffer_get_base(in_buf));
            ggml_backend_tensor_alloc(wt_buf, src0, ggml_backend_buffer_get_base(wt_buf));
            ggml_backend_tensor_alloc(out_buf, dst, ggml_backend_buffer_get_base(out_buf));
            
            // Set input data once (syncs to XRT BO)
            double t_set0 = now_ms();
            ggml_backend_tensor_set(src1, act_f32.data(), 0, act_bytes);
            ggml_backend_tensor_set(src0, wt_q8.data(), 0, wt_bytes);
            double t_set1 = now_ms();
            fprintf(stderr, "[BENCH] Data upload to XDNA: %.2f ms\n", t_set1 - t_set0);
            
            // Warmup
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            
            // Benchmark
            double t0 = now_ms();
            for (int i = 0; i < 5; i++) {
                ggml_backend_graph_compute(backend, gf);
                ggml_backend_synchronize(backend);
            }
            double t1 = now_ms();
            double avg_ms = (t1 - t0) / 5.0;
            fprintf(stderr, "[BENCH] Test 2 avg: %.2f ms per GEMM\n", avg_ms);
            
            // Read result
            ggml_backend_tensor_get(dst, ref_f32.data(), 0, dst_bytes);
        }
        
        ggml_free(ctx);
    }
    
    ggml_backend_free(backend);
    
    fprintf(stderr, "\n[BENCH] Complete\n");
    return 0;
}
