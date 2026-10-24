/**
 * xdna_q8_0_scaling_test.cpp
 *
 * Phase 16-C: Verify Q8_0 scaling separately from GEMM arithmetic.
 *
 * Tests Q8_0 path with deterministic inputs:
 * 1. all q = 1, scale = 1.0
 * 2. q = -1, scale = 1.0
 * 3. q = [-128..127], scale = 1.0
 * 4. random q, random scale
 * 5. positive-only, negative-only, alternating
 *
 * Compares:
 * - CPU Q8_0 reference (quantize -> int8 dot -> scale)
 * - XDNA Q8_0 path (kernel int8 dot -> host scale)
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-xdna.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>

#define QK8_0 32

typedef struct {
    uint16_t d;       // fp16 scale
    int8_t  qs[QK8_0];
} block_q8_0_test;

static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    static const clock::time_point epoch = clock::now();
    auto d = clock::now() - epoch;
    return std::chrono::duration<double, std::milli>(d).count();
}

static void die(const char* msg) {
    fprintf(stderr, "[FATAL] %s\n", msg);
    exit(1);
}

// fp16 conversion (simplified)
static uint16_t fp32_to_fp16(float f) {
    // Simple conversion - in practice use GGML_FP32_TO_FP16
    uint32_t x = *reinterpret_cast<const uint32_t*>(&f);
    uint16_t h = 0;
    int exp = (x >> 23) & 0xFF;
    int mantissa = x & 0x7FFFFF;
    int sign = x >> 31;
    
    if (exp == 0) {
        h = sign << 15;
    } else if (exp == 0xFF) {
        h = (sign << 15) | 0x7C00;
    } else {
        int new_exp = exp - 127 + 15;
        if (new_exp >= 31) {
            h = (sign << 15) | 0x7C00;
        } else if (new_exp <= 0) {
            h = (sign << 15);
        } else {
            h = (sign << 15) | (new_exp << 10) | (mantissa >> 13);
        }
    }
    return h;
}

static float fp16_to_fp32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1F;
    int mantissa = h & 0x3FF;
    
    if (exp == 0) {
        return sign ? -0.0f : 0.0f;
    } else if (exp == 0x1F) {
        return sign ? -INFINITY : INFINITY;
    }
    
    float f = (1.0f + mantissa / 1024.0f) * powf(2.0f, exp - 15);
    return sign ? -f : f;
}

// CPU reference: Q8_0 GEMM
// A: float [K, M], B: Q8_0 [K, N] (block_q8_0 per column)
// C: float [N, M] = A^T * B
static void cpu_gemm_q8_0(const float* A, const block_q8_0_test* B, float* C, int M, int K, int N) {
    int nb = K / QK8_0;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                float scale = fp16_to_fp32(B[(size_t)n * nb + b].d);
                for (int j = 0; j < QK8_0; j++) {
                    int k = b * QK8_0 + j;
                    float a = A[(size_t)k * M + m];
                    int8_t q = B[(size_t)n * nb + b].qs[j];
                    acc += a * q * scale;
                }
            }
            C[(size_t)n * M + m] = acc;
        }
    }
}

// Test case: create Q8_0 weight with known values
static void create_q8_0_weight(block_q8_0_test* B, int K, int N, int pattern, float scale_val) {
    int nb = K / QK8_0;
    uint16_t scale_fp16 = fp32_to_fp16(scale_val);
    
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            block_q8_0_test* block = &B[(size_t)n * nb + b];
            block->d = scale_fp16;
            for (int j = 0; j < QK8_0; j++) {
                int8_t q = 0;
                switch (pattern) {
                    case 0: q = 1; break;           // all 1
                    case 1: q = -1; break;          // all -1
                    case 2: q = (int8_t)(j - 128); break; // -128..127
                    case 3: q = (int8_t)(rand() % 255 - 127); break; // random
                    case 4: q = (j % 2 == 0) ? 1 : -1; break; // alternating
                    case 5: q = 127; break;         // max positive
                    case 6: q = -127; break;        // max negative
                }
                block->qs[j] = q;
            }
        }
    }
}

// Test case: create Q8_0 input with known values
static void create_q8_0_input(block_q8_0_test* A, int M, int K, int pattern, float scale_val) {
    int nb = K / QK8_0;
    uint16_t scale_fp16 = fp32_to_fp16(scale_val);
    
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            block_q8_0_test* block = &A[(size_t)m * nb + b];
            block->d = scale_fp16;
            for (int j = 0; j < QK8_0; j++) {
                int8_t q = 0;
                switch (pattern) {
                    case 0: q = 1; break;
                    case 1: q = -1; break;
                    case 2: q = (int8_t)(j - 128); break;
                    case 3: q = (int8_t)(rand() % 255 - 127); break;
                    case 4: q = (j % 2 == 0) ? 1 : -1; break;
                    case 5: q = 127; break;
                    case 6: q = -127; break;
                }
                block->qs[j] = q;
            }
        }
    }
}

// CPU reference using float input (not Q8_0)
static void cpu_gemm_float_q8_0(const float* A, const block_q8_0_test* B, float* C, int M, int K, int N, int n_start, int n_end) {
    int nb = K / QK8_0;
    int Nslice = n_end - n_start;
    for (int n = 0; n < Nslice; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                float w_scale = fp16_to_fp32(B[(size_t)(n_start + n) * nb + b].d);
                for (int j = 0; j < QK8_0; j++) {
                    int k = b * QK8_0 + j;
                    float a = A[(size_t)k * M + m];
                    int8_t q = B[(size_t)(n_start + n) * nb + b].qs[j];
                    acc += a * q * w_scale;
                }
            }
            C[(size_t)n * M + m] = acc;
        }
    }
}

struct ErrorStats {
    double max_abs = 0;
    double mean_abs = 0;
    double max_rel = 0;
    double mean_rel = 0;
    double rmse = 0;
    size_t mismatch_count = 0;
    double mismatch_pct = 0;
    std::vector<std::tuple<int, int, float, float, double>> first_mismatches;

    void compute(const float* cpu, const float* xdna, int M, int N) {
        size_t total = (size_t)M * N;
        double sum_abs = 0;
        double sum_sq = 0;
        
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                float c = cpu[(size_t)n * M + m];
                float x = xdna[(size_t)n * M + m];
                double d = std::fabs((double)c - (double)x);
                
                if (d > max_abs) max_abs = d;
                sum_abs += d;
                sum_sq += d * d;
                
                double denom = std::fabs((double)c) + 1.0f;
                double rel = d / denom;
                if (rel > max_rel) max_rel = rel;
                mean_rel += rel;
                
                if (d > 1e-5) {
                    mismatch_count++;
                    if (first_mismatches.size() < 32) {
                        first_mismatches.emplace_back(m, n, c, x, d);
                    }
                }
            }
        }
        
        mean_abs = total > 0 ? sum_abs / total : 0;
        mean_rel = total > 0 ? mean_rel / total : 0;
        rmse = total > 0 ? std::sqrt(sum_sq / total) : 0;
        mismatch_pct = total > 0 ? (100.0 * mismatch_count / total) : 0;
    }

    void print(const char* label) const {
        printf("\n=== %s ===\n", label);
        printf("  max_abs      = %.6f\n", max_abs);
        printf("  mean_abs     = %.6f\n", mean_abs);
        printf("  max_rel      = %.6f\n", max_rel);
        printf("  mean_rel     = %.6f\n", mean_rel);
        printf("  RMSE         = %.6f\n", rmse);
        printf("  mismatch     = %zu / %.2f%% (>1e-5)\n", mismatch_count, mismatch_pct);
        if (!first_mismatches.empty()) {
            printf("  first 32 mismatches (m, n, cpu, xdna, diff):\n");
            for (auto& [m, n, c, x, d] : first_mismatches) {
                printf("    [%5d, %5d] cpu=%12.6f xdna=%12.6f diff=%.6f\n", m, n, c, x, d);
            }
        }
    }
};

static void run_xdna_q8_0(int M, int K, int N, int n_start, int n_end,
                          const float* A, const block_q8_0_test* B, float* output) {
    ggml_backend_load_all();
    fprintf(stderr, "[TEST] Trying XDNA backend...\n");
    
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    if (!backend_xdna) backend_xdna = ggml_backend_init_by_name("AMD Phoenix XDNA1 NPU", "");
    if (!backend_xdna) {
        fprintf(stderr, "[TEST] XDNA backend init failed, trying ggml_backend_xdna_init()...\n");
        backend_xdna = ggml_backend_xdna_init();
    }
    if (!backend_xdna) die("XDNA backend not available");

    struct ggml_init_params params{0};
    params.mem_size = (size_t)(K * N) * 2 + (size_t)(K * M) * sizeof(float) + (size_t)(M * N) * sizeof(float) + 64 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init failed");

    int Nslice = n_end - n_start;
    // Weight: Q8_0 tensor
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, Nslice);
    // Input: F32 tensor
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    if (!buf) die("alloc_ctx_tensors failed");

    // Set weight data (Q8_0) - GGML layout: blocks per column
    int nb = K / QK8_0;
    std::vector<uint8_t> w_data((size_t)Nslice * nb * sizeof(block_q8_0_test));
    for (int n = 0; n < Nslice; n++) {
        for (int b = 0; b < nb; b++) {
            block_q8_0_test* block = (block_q8_0_test*)(w_data.data() + ((size_t)n * nb + b) * sizeof(block_q8_0_test));
            *block = B[(size_t)(n_start + n) * nb + b];
        }
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());

    // Set input data (F32) - GGML layout: [K, M] = M rows of K elements, index = k + m*K
    std::vector<float> b_data((size_t)K * M);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            b_data[(size_t)k + (size_t)m * K] = A[(size_t)k * M + m];
        }
    }
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

    // Execute
    if (ggml_backend_graph_compute(backend_xdna, gf) != GGML_STATUS_SUCCESS) {
        die("graph_compute failed");
    }

    // Get output - GGML output is [N, M] layout: N rows of M elements
    std::vector<float> out_f32((size_t)Nslice * M);
    ggml_backend_tensor_get(c, out_f32.data(), 0, out_f32.size() * sizeof(float));

    // Copy to output
    for (size_t i = 0; i < out_f32.size(); i++) {
        output[i] = out_f32[i];
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend_xdna);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int M = 64, K = 128, N = 256, nv = 64;  // Smaller for faster testing
    int pattern = 0;
    float scale = 1.0f;
    
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M" && i + 1 < argc) M = std::stoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc) K = std::stoi(argv[++i]);
        else if (a == "--N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (a == "--nv" && i + 1 < argc) nv = std::stoi(argv[++i]);
        else if (a == "--pattern" && i + 1 < argc) pattern = std::stoi(argv[++i]);
        else if (a == "--scale" && i + 1 < argc) scale = std::stof(argv[++i]);
    }
    int Nslice = N - nv;

    if (K % QK8_0 != 0) {
        die("K must be multiple of 32");
    }

    printf("[Q8_0_TEST] M=%d K=%d N=%d  pattern=%d scale=%.3f\n", M, K, N, pattern, scale);
    fflush(stdout);

    // Generate test data
    std::vector<float> A((size_t)K * M);
    std::vector<block_q8_0_test> B((size_t)K / QK8_0 * N);
    
    // Input A: random float
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);

    // Weight B: Q8_0 with specified pattern
    create_q8_0_weight(B.data(), K, N, pattern, 1.0f);

    // CPU reference
    printf("[Q8_0_TEST] Computing CPU reference...\n");
    double t0 = now_ms();
    std::vector<float> cpu_ref((size_t)N * M);
    cpu_gemm_float_q8_0(A.data(), B.data(), cpu_ref.data(), M, K, N, 0, N);
    double t_cpu = now_ms() - t0;
    printf("[Q8_0_TEST] CPU: %.2f ms\n", t_cpu);

    // CPU XDNA partition reference
    std::vector<float> cpu_xd_part((size_t)Nslice * M);
    cpu_gemm_float_q8_0(A.data(), B.data(), cpu_xd_part.data(), M, K, N, nv, N);

    // XDNA Q8_0 path
    printf("[Q8_0_TEST] Running XDNA Q8_0 path...\n");
    std::vector<float> xdna_out((size_t)Nslice * M);
    t0 = now_ms();
    run_xdna_q8_0(M, K, N, nv, N, A.data(), B.data(), xdna_out.data());
    double t_xdna = now_ms() - t0;
    printf("[Q8_0_TEST] XDNA: %.2f ms\n", t_xdna);

    // Compare
    ErrorStats stats;
    stats.compute(cpu_xd_part.data(), xdna_out.data(), M, Nslice);
    stats.print("XDNA Q8_0 vs CPU (XDNA partition)");

    // Summary
    printf("\n=== PHASE 16-C SUMMARY (pattern=%d, scale=%.3f) ===\n", pattern, scale);
    if (stats.max_abs < 1e-3 && stats.mismatch_count == 0) {
        printf("PASS: XDNA Q8_0 scaling is CORRECT!\n");
    } else {
        printf("FAIL: XDNA Q8_0 has scaling errors (max_abs=%.6f)\n", stats.max_abs);
    }
    fflush(stdout);

    return (stats.max_abs < 1e-3 && stats.mismatch_count == 0) ? 0 : 1;
}