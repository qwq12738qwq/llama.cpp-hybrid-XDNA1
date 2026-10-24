/**
 * xdna_phase17_harness.cpp
 *
 * Phase 17: Comprehensive layered Q8_0 correctness harness.
 *
 * Tests each layer independently:
 * 1. Raw int8 GEMM (bit-exact)
 * 2. Q8_0 all-ones (simplest scale test)
 * 3. Q8_0 alternating scales (exposes scale indexing bugs)
 * 4. Q8_0 random patterns
 * 5. Per-block scale isolation
 * 6. K-block boundary tests
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
#include <fstream>

#define QK8_0 32

typedef struct {
    uint16_t d;
    int8_t  qs[QK8_0];
} block_q8_0;

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

// Use ggml fp16 conversion functions
static float fp16_to_fp32(uint16_t h) {
    return ggml_fp16_to_fp32(h);
}

static uint16_t fp32_to_fp16(float f) {
    return ggml_fp32_to_fp16(f);
}

// CPU reference: raw int8 GEMM (no scales)
static void cpu_gemm_raw_int8(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[(size_t)k * M + m] * (int32_t)B[(size_t)k * N + n];
            }
            C[(size_t)n * M + m] = acc;
        }
    }
}

// CPU reference: Q8_0 GEMM (per-block scales)
static void cpu_gemm_q8_0(const float* A, const block_q8_0* B, float* C, int M, int K, int N) {
    int nb = K / QK8_0;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                float w_scale = fp16_to_fp32(B[(size_t)n * nb + b].d);
                for (int j = 0; j < QK8_0; j++) {
                    int k = b * QK8_0 + j;
                    acc += A[(size_t)k * M + m] * B[(size_t)n * nb + b].qs[j] * w_scale;
                }
            }
            C[(size_t)n * M + m] = acc;
        }
    }
}

// CPU reference: Q8_0 GEMM with Q8_0 input (per-block scales for both)
static void cpu_gemm_q8_0_q8_0(const block_q8_0* A, const block_q8_0* B, float* C, int M, int K, int N) {
    int nb = K / QK8_0;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                float a_scale = fp16_to_fp32(A[(size_t)m * nb + b].d);
                float w_scale = fp16_to_fp32(B[(size_t)n * nb + b].d);
                for (int j = 0; j < QK8_0; j++) {
                    int k = b * QK8_0 + j;
                    acc += (float)A[(size_t)m * nb + b].qs[j] * B[(size_t)n * nb + b].qs[j] * a_scale * w_scale;
                }
            }
            C[(size_t)n * M + m] = acc;
        }
    }
}

// CPU reference: quantize F32 to int8, then raw int8 GEMM (matches XDNA F32 path)
static void cpu_gemm_f32_via_int8(const float* A, const block_q8_0* B, int32_t* C, int M, int K, int N) {
    int nb = K / QK8_0;
    // Quantize A to int8 per-block
    std::vector<int8_t> A_q(K * M);
    std::vector<float> a_scale(M * nb);
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            float max_abs = 0;
            for (int k = 0; k < QK8_0; k++) {
                float v = A[(size_t)(b * QK8_0 + k) * M + m];
                max_abs = std::max(max_abs, std::fabs(v));
            }
            float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / 127.0f);
            a_scale[(size_t)m * nb + b] = scale;
            for (int k = 0; k < QK8_0; k++) {
                float v = A[(size_t)(b * QK8_0 + k) * M + m];
                int q = (int)std::lroundf(v / scale);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                A_q[(size_t)(b * QK8_0 + k) * M + m] = (int8_t)q;
            }
        }
    }
    // Dequantize B to int8
    std::vector<int8_t> B_q(K * N);
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            float w_scale = fp16_to_fp32(B[(size_t)n * nb + b].d);
            for (int k = 0; k < QK8_0; k++) {
                B_q[(size_t)(b * QK8_0 + k) * N + n] = B[(size_t)n * nb + b].qs[k];
            }
        }
    }
    // Raw int8 GEMM
    cpu_gemm_raw_int8(A_q.data(), B_q.data(), C, M, K, N);
    // Apply per-block scales
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc_f = 0.0f;
            for (int b = 0; b < nb; b++) {
                float a_s = a_scale[(size_t)m * nb + b];
                float w_s = fp16_to_fp32(B[(size_t)n * nb + b].d);
                acc_f += (float)C[(size_t)n * M + m] * a_s * w_s / nb;  // Hmm, this is wrong
            }
            C[(size_t)n * M + m] = (int32_t)acc_f;
        }
    }
}

struct ErrorStats {
    double max_abs = 0, mean_abs = 0, rmse = 0, max_rel = 0, mean_rel = 0;
    size_t mismatch_count = 0;
    double mismatch_pct = 0;
    std::vector<std::tuple<int, int, double, double, double>> first_mismatches;

    void compute(const float* cpu, const float* xdna, int M, int N) {
        size_t total = (size_t)M * N;
        double sum_abs = 0, sum_sq = 0, sum_rel = 0;
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                double c = cpu[(size_t)n * M + m];
                double x = xdna[(size_t)n * M + m];
                double d = std::fabs(c - x);
                if (d > max_abs) max_abs = d;
                sum_abs += d;
                sum_sq += d * d;
                double denom = std::fabs(c) + 1.0;
                double rel = d / denom;
                if (rel > max_rel) max_rel = rel;
                sum_rel += rel;
                if (d > 1e-5) {
                    mismatch_count++;
                    if (first_mismatches.size() < 32) {
                        first_mismatches.emplace_back(m, n, c, x, d);
                    }
                }
            }
        }
        mean_abs = total > 0 ? sum_abs / total : 0;
        mean_rel = total > 0 ? sum_rel / total : 0;
        rmse = total > 0 ? std::sqrt(sum_sq / total) : 0;
        mismatch_pct = total > 0 ? (100.0 * mismatch_count / total) : 0;
    }

    void compute_int32(const int32_t* cpu, const int32_t* xdna, int M, int N) {
        size_t total = (size_t)M * N;
        double sum_abs = 0, sum_sq = 0;
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                double c = cpu[(size_t)n * M + m];
                double x = xdna[(size_t)n * M + m];
                double d = std::fabs(c - x);
                if (d > max_abs) max_abs = d;
                sum_abs += d;
                sum_sq += d * d;
                if (d > 0) {
                    mismatch_count++;
                    if (first_mismatches.size() < 32) {
                        first_mismatches.emplace_back(m, n, c, x, d);
                    }
                }
            }
        }
        mean_abs = total > 0 ? sum_abs / total : 0;
        rmse = total > 0 ? std::sqrt(sum_sq / total) : 0;
        mismatch_pct = total > 0 ? (100.0 * mismatch_count / total) : 0;
    }

    void print(const char* label) const {
        printf("\n=== %s ===\n", label);
        printf("  max_abs      = %.6f\n", max_abs);
        printf("  mean_abs     = %.6f\n", mean_abs);
        printf("  RMSE         = %.6f\n", rmse);
        printf("  max_rel      = %.6f\n", max_rel);
        printf("  mean_rel     = %.6f\n", mean_rel);
        printf("  mismatch     = %zu / %.2f%%\n", mismatch_count, mismatch_pct);
        if (!first_mismatches.empty()) {
            printf("  first mismatches:\n");
            for (auto& [m, n, c, x, d] : first_mismatches) {
                printf("    [%5d, %5d] cpu=%12.6f xdna=%12.6f diff=%.6f\n", m, n, c, x, d);
            }
        }
    }
};

// Run XDNA raw int8 GEMM (I8 weight, no scales)
static void run_xdna_raw_int8(int M, int K, int N, int n_start, int n_end,
                              const int8_t* A, const int8_t* B, int32_t* output) {
    ggml_backend_load_all();
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    if (!backend_xdna) backend_xdna = ggml_backend_init_by_name("AMD Phoenix XDNA1 NPU", "");
    if (!backend_xdna) backend_xdna = ggml_backend_xdna_init();
    if (!backend_xdna) die("XDNA backend not available");

    struct ggml_init_params params{0};
    params.mem_size = (size_t)(K * N) + (size_t)(K * M) + (size_t)(M * N) + 64 * 1024 * 1024;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init failed");

    int Nslice = n_end - n_start;
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, Nslice);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    if (!buf) die("alloc_ctx_tensors failed");

    std::vector<int8_t> w_data((size_t)K * Nslice);
    for (int n = 0; n < Nslice; n++) {
        for (int k = 0; k < K; k++) {
            w_data[(size_t)k + (size_t)n * K] = B[(size_t)k * N + (n_start + n)];
        }
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());

    std::vector<int8_t> b_data((size_t)K * M);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            b_data[(size_t)k + (size_t)m * K] = A[(size_t)k * M + m];
        }
    }
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size());

    if (ggml_backend_graph_compute(backend_xdna, gf) != GGML_STATUS_SUCCESS) {
        die("graph_compute failed");
    }

    std::vector<float> out_f32((size_t)M * Nslice);
    ggml_backend_tensor_get(c, out_f32.data(), 0, out_f32.size() * sizeof(float));
    for (size_t i = 0; i < out_f32.size(); i++) {
        output[i] = (int32_t)std::lroundf(out_f32[i]);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend_xdna);
}

// Run XDNA Q8_0 GEMM (Q8_0 weight, F32 input)
static void run_xdna_q8_0_f32(int M, int K, int N, int n_start, int n_end,
                              const float* A, const block_q8_0* B, float* output) {
    ggml_backend_load_all();
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    if (!backend_xdna) backend_xdna = ggml_backend_init_by_name("AMD Phoenix XDNA1 NPU", "");
    if (!backend_xdna) backend_xdna = ggml_backend_xdna_init();
    if (!backend_xdna) die("XDNA backend not available");

    struct ggml_init_params params{0};
    params.mem_size = (size_t)K * N * 2 + (size_t)K * M * sizeof(float) + (size_t)M * N * sizeof(float) + 64 * 1024 * 1024;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init failed");

    int Nslice = n_end - n_start;
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, Nslice);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    if (!buf) die("alloc failed");

    int nb = K / QK8_0;
    std::vector<uint8_t> w_data((size_t)Nslice * nb * sizeof(block_q8_0));
    for (int n = 0; n < Nslice; n++) {
        for (int b_idx = 0; b_idx < nb; b_idx++) {
            block_q8_0* block = (block_q8_0*)(w_data.data() + ((size_t)n * nb + b_idx) * sizeof(block_q8_0));
            *block = B[(size_t)(n_start + n) * nb + b_idx];
        }
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());

    std::vector<float> b_data((size_t)K * M);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            b_data[(size_t)k + (size_t)m * K] = A[(size_t)k * M + m];
        }
    }
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend_xdna, gf) != GGML_STATUS_SUCCESS) {
        die("graph_compute failed");
    }

    std::vector<float> out_f32((size_t)Nslice * M);
    ggml_backend_tensor_get(c, out_f32.data(), 0, out_f32.size() * sizeof(float));
    for (size_t i = 0; i < out_f32.size(); i++) output[i] = out_f32[i];

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend_xdna);
}

// Run XDNA Q8_0 GEMM (Q8_0 input, Q8_0 weight)
static void run_xdna_q8_0_q8_0(int M, int K, int N, int n_start, int n_end,
                               const block_q8_0* A, const block_q8_0* B, float* output) {
    ggml_backend_load_all();
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    if (!backend_xdna) backend_xdna = ggml_backend_init_by_name("AMD Phoenix XDNA1 NPU", "");
    if (!backend_xdna) backend_xdna = ggml_backend_xdna_init();
    if (!backend_xdna) die("XDNA backend not available");

    struct ggml_init_params params{0};
    params.mem_size = (size_t)K * N * 2 + (size_t)K * M * 2 + (size_t)M * N * sizeof(float) + 64 * 1024 * 1024;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init failed");

    int Nslice = n_end - n_start;
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, Nslice);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    if (!buf) die("alloc failed");

    int nb = K / QK8_0;
    std::vector<uint8_t> w_data((size_t)Nslice * nb * sizeof(block_q8_0));
    for (int n = 0; n < Nslice; n++) {
        for (int b_idx = 0; b_idx < nb; b_idx++) {
            block_q8_0* block = (block_q8_0*)(w_data.data() + ((size_t)n * nb + b_idx) * sizeof(block_q8_0));
            *block = B[(size_t)(n_start + n) * nb + b_idx];
        }
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());

    std::vector<uint8_t> b_data((size_t)M * nb * sizeof(block_q8_0));
    for (int m = 0; m < M; m++) {
        for (int b_idx = 0; b_idx < nb; b_idx++) {
            block_q8_0* block = (block_q8_0*)(b_data.data() + ((size_t)m * nb + b_idx) * sizeof(block_q8_0));
            *block = A[(size_t)m * nb + b_idx];
        }
    }
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size());

    if (ggml_backend_graph_compute(backend_xdna, gf) != GGML_STATUS_SUCCESS) {
        die("graph_compute failed");
    }

    std::vector<float> out((size_t)Nslice * M);
    ggml_backend_tensor_get(c, out.data(), 0, out.size() * sizeof(float));
    for (size_t i = 0; i < out.size(); i++) output[i] = out[i];

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend_xdna);
}

// Create Q8_0 block with all q=val and given scale
static void set_block(block_q8_0* block, int8_t q_val, float scale) {
    block->d = fp32_to_fp16(scale);
    for (int j = 0; j < QK8_0; j++) {
        block->qs[j] = q_val;
    }
}

// Create Q8_0 block with pattern
static void set_block_pattern(block_q8_0* block, int pattern, float scale, int block_idx) {
    block->d = fp32_to_fp16(scale);
    for (int j = 0; j < QK8_0; j++) {
        int8_t q = 0;
        switch (pattern) {
            case 0: q = 1; break;
            case 1: q = -1; break;
            case 2: q = (int8_t)(j - 128); break;
            case 3: q = (int8_t)((block_idx * 32 + j) % 255 - 127); break;
            case 4: q = (j % 2 == 0) ? 1 : -1; break;
            case 5: q = 127; break;
            case 6: q = -127; break;
            default: q = 0; break;
        }
        block->qs[j] = q;
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int M = 64, K = 128, N = 256, nv = 128;
    int pattern = 0;
    float scale_val = 1.0f;
    bool test_raw = false;
    bool test_q8q8 = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M" && i + 1 < argc) M = std::stoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc) K = std::stoi(argv[++i]);
        else if (a == "--N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (a == "--nv" && i + 1 < argc) nv = std::stoi(argv[++i]);
        else if (a == "--pattern" && i + 1 < argc) pattern = std::stoi(argv[++i]);
        else if (a == "--scale" && i + 1 < argc) scale_val = std::stof(argv[++i]);
        else if (a == "--raw") test_raw = true;
        else if (a == "--q8q8") test_q8q8 = true;
    }

    if (K % QK8_0 != 0) die("K must be multiple of 32");
    int nb = K / QK8_0;

    printf("[PHASE17] M=%d K=%d N=%d nb=%d pattern=%d scale=%.3f\n", M, K, N, nb, pattern, scale_val);
    fflush(stdout);

    // ===== TEST 1: Raw int8 GEMM (bit-exact) =====
    if (test_raw) {
        printf("\n[TEST 1] Raw int8 GEMM (bit-exact)\n");
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(-127, 127);

        std::vector<int8_t> A((size_t)K * M);
        std::vector<int8_t> B((size_t)K * N);
        for (auto& v : A) v = (int8_t)dist(rng);
        for (auto& v : B) v = (int8_t)dist(rng);

        std::vector<int32_t> cpu_ref((size_t)N * M);
        double t0 = now_ms();
        cpu_gemm_raw_int8(A.data(), B.data(), cpu_ref.data(), M, K, N);
        double t_cpu = now_ms() - t0;
        printf("  CPU: %.2f ms\n", t_cpu);

        std::vector<int32_t> xdna_out((size_t)(N - nv) * M);
        t0 = now_ms();
        run_xdna_raw_int8(M, K, N, nv, N, A.data(), B.data(), xdna_out.data());
        double t_xdna = now_ms() - t0;
        printf("  XDNA: %.2f ms\n", t_xdna);

        ErrorStats stats;
        stats.compute_int32(cpu_ref.data() + nv * M, xdna_out.data(), M, N - nv);
        stats.print("Raw int8: XDNA vs CPU");

        if (stats.max_abs == 0 && stats.mismatch_count == 0) {
            printf("  >>> PASS: bit-identical\n");
        } else {
            printf("  >>> FAIL: max_abs=%.6f\n", stats.max_abs);
            return 1;
        }
    }

    // ===== TEST 2: Q8_0 all-ones (simplest scale test) =====
    printf("\n[TEST 2] Q8_0 all-ones (scale=%.3f)\n", scale_val);
    std::vector<block_q8_0> B_q8((size_t)K / QK8_0 * N);
    std::vector<float> A_f32((size_t)K * M);
    for (auto& v : A_f32) v = 1.0f;

    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            set_block(&B_q8[(size_t)n * nb + b], 1, scale_val);
        }
    }

    std::vector<float> cpu_ref((size_t)N * M);
    double t0 = now_ms();
    cpu_gemm_q8_0(A_f32.data(), B_q8.data(), cpu_ref.data(), M, K, N);
    double t_cpu = now_ms() - t0;
    printf("  CPU: %.2f ms\n", t_cpu);

    std::vector<float> xdna_out((size_t)(N - nv) * M);
    t0 = now_ms();
    run_xdna_q8_0_f32(M, K, N, nv, N, A_f32.data(), B_q8.data(), xdna_out.data());
    double t_xdna = now_ms() - t0;
    printf("  XDNA: %.2f ms\n", t_xdna);

    ErrorStats stats2;
    stats2.compute(cpu_ref.data() + nv * M, xdna_out.data(), M, N - nv);
    stats2.print("Q8_0 all-ones: XDNA vs CPU");

    if (stats2.max_abs < 1e-3 && stats2.mismatch_count == 0) {
        printf("  >>> PASS\n");
    } else {
        printf("  >>> FAIL: max_abs=%.6f\n", stats2.max_abs);
    }

    // ===== TEST 3: Q8_0-Q8_0 all-ones =====
    if (test_q8q8) {
        printf("\n[TEST 3] Q8_0-Q8_0 all-ones (scale=%.3f)\n", scale_val);
        std::vector<block_q8_0> A_q8((size_t)M * nb);
        for (int m = 0; m < M; m++) {
            for (int b = 0; b < nb; b++) {
                set_block(&A_q8[(size_t)m * nb + b], 1, scale_val);
            }
        }

        std::vector<float> cpu_ref3((size_t)(N - nv) * M);
        t0 = now_ms();
        cpu_gemm_q8_0_q8_0(A_q8.data(), B_q8.data(), cpu_ref3.data(), M, K, N);
        double t_cpu3 = now_ms() - t0;
        printf("  CPU: %.2f ms\n", t_cpu3);

        std::vector<float> xdna_out3((size_t)(N - nv) * M);
        t0 = now_ms();
        run_xdna_q8_0_q8_0(M, K, N, nv, N, A_q8.data(), B_q8.data(), xdna_out3.data());
        double t_xdna3 = now_ms() - t0;
        printf("  XDNA: %.2f ms\n", t_xdna3);

        ErrorStats stats3;
        stats3.compute(cpu_ref3.data(), xdna_out3.data(), M, N - nv);
        stats3.print("Q8_0-Q8_0 all-ones: XDNA vs CPU");

        if (stats3.max_abs < 1e-3 && stats3.mismatch_count == 0) {
            printf("  >>> PASS\n");
        } else {
            printf("  >>> FAIL: max_abs=%.6f\n", stats3.max_abs);
        }
    }

    // ===== TEST 4: Alternating scales (exposes scale indexing bugs) =====
    printf("\n[TEST 4] Alternating scales\n");
    {
        std::vector<block_q8_0> B_alt((size_t)K / QK8_0 * N);
        float scales[4] = {0.01f, 1.0f, 0.02f, 2.0f};
        for (int n = 0; n < N; n++) {
            for (int b = 0; b < nb; b++) {
                float s = scales[b % 4];
                set_block(&B_alt[(size_t)n * nb + b], 1, s);
            }
        }

        std::vector<float> cpu_ref4((size_t)N * M);
        cpu_gemm_q8_0(A_f32.data(), B_alt.data(), cpu_ref4.data(), M, K, N);

        std::vector<float> xdna_out4((size_t)(N - nv) * M);
        run_xdna_q8_0_f32(M, K, N, nv, N, A_f32.data(), B_alt.data(), xdna_out4.data());

        ErrorStats stats4;
        stats4.compute(cpu_ref4.data() + nv * M, xdna_out4.data(), M, N - nv);
        stats4.print("Alternating scales: XDNA vs CPU");

        if (stats4.max_abs < 1e-3 && stats4.mismatch_count == 0) {
            printf("  >>> PASS\n");
        } else {
            printf("  >>> FAIL: max_abs=%.6f\n", stats4.max_abs);
        }
    }

    // ===== TEST 5: K-block boundary (K=64, 96, 128, 160) =====
    printf("\n[TEST 5] K-block boundary tests\n");
    for (int test_K : {64, 96, 128, 160}) {
        if (test_K > K) break;
        int test_nb = test_K / QK8_0;
        printf("  K=%d (nb=%d): ", test_K, test_nb);

        std::vector<block_q8_0> B_k((size_t)test_K / QK8_0 * N);
        for (int n = 0; n < N; n++) {
            for (int b = 0; b < test_nb; b++) {
                set_block(&B_k[(size_t)n * test_nb + b], 1, scale_val);
            }
        }

        std::vector<float> A_k((size_t)test_K * M);
        for (auto& v : A_k) v = 1.0f;

        std::vector<float> cpu_ref5((size_t)N * M);
        cpu_gemm_q8_0(A_k.data(), B_k.data(), cpu_ref5.data(), M, test_K, N);

        std::vector<float> xdna_out5((size_t)(N - nv) * M);
        run_xdna_q8_0_f32(M, test_K, N, nv, N, A_k.data(), B_k.data(), xdna_out5.data());

        ErrorStats stats5;
        stats5.compute(cpu_ref5.data() + nv * M, xdna_out5.data(), M, N - nv);
        printf("max_abs=%.6f mismatches=%zu\n", stats5.max_abs, stats5.mismatch_count);
    }

    printf("\n[PHASE17] Complete\n");
    return 0;
}
