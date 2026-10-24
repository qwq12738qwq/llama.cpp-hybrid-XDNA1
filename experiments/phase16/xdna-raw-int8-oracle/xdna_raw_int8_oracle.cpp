/**
 * xdna_raw_int8_oracle.cpp
 *
 * Phase 16-A: Numerical oracle for raw int8 GEMM (bypassing Q8_0).
 *
 * Tests: M=512, K=1024, N=4096 with N-partition (Vulkan: 1024, XDNA: 3072)
 * Computes: int8 × int8 → int32 GEMM (no scales, no Q8_0)
 * Compares: CPU reference vs XDNA raw output
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

// CPU reference: int8 × int8 → int32 GEMM (no scales)
// A: [K, M] row-major (K rows, M cols)
// B: [K, N] row-major (K rows, N cols)
// C: [N, M] = A^T * B  (so C[n,m] = sum_k A[k,m] * B[k,n])
// Output layout matches GGML mul_mat: [N, M] (ne[0]=M, ne[1]=N)
static void cpu_gemm_int8_int32(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N) {
    // A is [K, M] - K rows, M cols
    // B is [K, N] - K rows, N cols
    // C is [N, M] - N rows, M cols (GGML mul_mat output layout)
    // C[n,m] = sum_{k=0}^{K-1} A[k,m] * B[k,n]
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int32_t a = static_cast<int32_t>(A[(size_t)k * M + m]);
                int32_t b = static_cast<int32_t>(B[(size_t)k * N + n]);
                acc += a * b;
            }
            C[(size_t)n * M + m] = acc;
        }
    }
}

// CPU reference for partitioned N dimension
// Output: [Nslice, M] layout (Nslice rows, M cols)
static void cpu_gemm_int8_int32_partition(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N, int n_start, int n_end) {
    int Nslice = n_end - n_start;
    for (int n = 0; n < Nslice; n++) {
        for (int m = 0; m < M; m++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int32_t a = static_cast<int32_t>(A[(size_t)k * M + m]);
                int32_t b = static_cast<int32_t>(B[(size_t)k * N + (n_start + n)]);
                acc += a * b;
            }
            C[(size_t)n * M + m] = acc;
        }
    }
}

// Comprehensive error statistics
struct ErrorStats {
    double max_abs = 0;
    double mean_abs = 0;
    double max_rel = 0;
    double mean_rel = 0;
    double rmse = 0;
    size_t mismatch_count = 0;
    double mismatch_pct = 0;
    std::vector<std::tuple<int, int, int32_t, int32_t, double>> first_mismatches; // m, n, cpu, xdna, diff
    std::vector<double> per_column_max_abs;
    std::vector<double> per_kblock_max_abs;

    void compute(const int32_t* cpu, const int32_t* xdna, int M, int N, int K, int n_offset = 0) {
        // Both cpu and xdna are in [N, M] layout (N rows, M cols): index = n * M + m
        size_t total = (size_t)N * M;
        double sum_abs = 0;
        double sum_sq = 0;
        per_column_max_abs.assign(N, 0);
        
        // K-block size is 32
        int nb = K / 32;
        per_kblock_max_abs.assign(nb, 0);

        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                int32_t c = cpu[(size_t)n * M + m];
                int32_t x = xdna[(size_t)n * M + m];
                double d = std::fabs((double)c - (double)x);
                
                if (d > max_abs) max_abs = d;
                sum_abs += d;
                sum_sq += d * d;
                
                double denom = std::fabs((double)c) + 1.0;
                double rel = d / denom;
                if (rel > max_rel) max_rel = rel;
                mean_rel += rel;
                
                if (d > 0) {
                    mismatch_count++;
                    if (first_mismatches.size() < 32) {
                        first_mismatches.emplace_back(m, n + n_offset, c, x, d);
                    }
                }
                
                if (d > per_column_max_abs[n]) per_column_max_abs[n] = d;
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
        printf("  mismatch     = %zu / %.2f%%\n", mismatch_count, mismatch_pct);
        if (!first_mismatches.empty()) {
            printf("  first 32 mismatches (m, n, cpu, xdna, diff):\n");
            for (auto& [m, n, c, x, d] : first_mismatches) {
                printf("    [%5d, %5d] cpu=%12d xdna=%12d diff=%.3f\n", m, n, c, x, d);
            }
        }
        // Per-column summary
        double col_max = 0, col_mean = 0;
        for (double v : per_column_max_abs) {
            if (v > col_max) col_max = v;
            col_mean += v;
        }
        col_mean /= per_column_max_abs.size();
        printf("  per-column max_abs: max=%.6f mean=%.6f\n", col_max, col_mean);
    }
};

static void run_xdna_raw_int8(int M, int K, int N, int n_start, int n_end,
                              const int8_t* A, const int8_t* B, int32_t* output) {
    // Initialize XDNA backend
    ggml_backend_load_all();
    
    // Debug: list available backends
    fprintf(stderr, "[ORACLE] Trying XDNA backend...\n");
    
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    if (!backend_xdna) backend_xdna = ggml_backend_init_by_name("AMD Phoenix XDNA1 NPU", "");
    if (!backend_xdna) {
        fprintf(stderr, "[ORACLE] XDNA backend init failed, trying ggml_backend_xdna_init()...\n");
        backend_xdna = ggml_backend_xdna_init();
    }
    if (!backend_xdna) die("XDNA backend not available");

    // Create ggml context and tensors
    struct ggml_init_params params{0};
    params.mem_size = (size_t)(K * N) + (size_t)(K * M) + (size_t)(M * N) + 64 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init failed");

    int Nslice = n_end - n_start;
    // Use I8 tensors (raw int8, not Q8_0)
    // GGML tensor layout: ne[0]=K, ne[1]=Nslice for weight (row-major: Nslice rows of K elements, index = k + n*K)
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, Nslice);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    if (!buf) die("alloc_ctx_tensors failed");

    // Set weight data (slice of B) - GGML row-major: [K, Nslice] = Nslice rows of K elements, index = k + n*K
    std::vector<int8_t> w_data((size_t)K * Nslice);
    for (int n = 0; n < Nslice; n++) {
        for (int k = 0; k < K; k++) {
            w_data[(size_t)k + (size_t)n * K] = B[(size_t)k * N + (n_start + n)];
        }
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());

    // Set input data (A) - GGML row-major: [K, M] = M rows of K elements, index = k + m*K
    std::vector<int8_t> b_data((size_t)K * M);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            b_data[(size_t)k + (size_t)m * K] = A[(size_t)k * M + m];
        }
    }
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size());

    // Execute
    if (ggml_backend_graph_compute(backend_xdna, gf) != GGML_STATUS_SUCCESS) {
        die("graph_compute failed");
    }

    // Get output - GGML mul_mat output is [N, M] in ne, stored as [M, N] row-major in memory
    // We need to transpose to [N, M] to match the CPU reference layout
    std::vector<float> out_f32((size_t)M * Nslice);
    ggml_backend_tensor_get(c, out_f32.data(), 0, out_f32.size() * sizeof(float));

    // Transpose from [M, Nslice] to [Nslice, M]
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < Nslice; n++) {
            output[n * M + m] = static_cast<int32_t>(std::lroundf(out_f32[m * Nslice + n]));
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend_xdna);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int M = 512, K = 1024, N = 4096, nv = 1024;
    int iters = 1;
    bool test_ones = false;
    
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M" && i + 1 < argc) M = std::stoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc) K = std::stoi(argv[++i]);
        else if (a == "--N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (a == "--nv" && i + 1 < argc) nv = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else if (a == "--ones") test_ones = true;
    }
    int nx = N - nv;

    printf("[ORACLE] M=%d K=%d N=%d  Vulkan N=[0,%d)  XDNA N=[%d,%d)\n", M, K, N, nv, nv, N);
    fflush(stdout);

    // Generate test data
    std::vector<int8_t> A((size_t)K * M);
    std::vector<int8_t> B((size_t)K * N);
    
    if (test_ones) {
        // Test with all ones: C = K * ones = 1024 * ones
        for (auto& v : A) v = 1;
        for (auto& v : B) v = 1;
        printf("[ORACLE] Test mode: all ones (expected output = %d)\n", K);
    } else {
        // Fixed seed for reproducibility
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(-127, 127);
        for (auto& v : A) v = static_cast<int8_t>(dist(rng));
        for (auto& v : B) v = static_cast<int8_t>(dist(rng));
    }

    // CPU full reference
    printf("[ORACLE] Computing CPU full reference...\n");
    double t0 = now_ms();
    std::vector<int32_t> cpu_full((size_t)N * M);  // [N, M] layout
    cpu_gemm_int8_int32(A.data(), B.data(), cpu_full.data(), M, K, N);
    double t_cpu = now_ms() - t0;
    printf("[ORACLE] CPU full: %.2f ms\n", t_cpu);

    // CPU XDNA partition reference
    printf("[ORACLE] Computing CPU XDNA partition reference...\n");
    std::vector<int32_t> cpu_xd_part((size_t)nx * M);
    cpu_gemm_int8_int32_partition(A.data(), B.data(), cpu_xd_part.data(), M, K, N, nv, N);

    // XDNA raw int8 GEMM
    printf("[ORACLE] Running XDNA raw int8 GEMM...\n");
    std::vector<int32_t> xdna_out((size_t)nx * M);
    t0 = now_ms();
    run_xdna_raw_int8(M, K, N, nv, N, A.data(), B.data(), xdna_out.data());
    double t_xdna = now_ms() - t0;
    printf("[ORACLE] XDNA: %.2f ms\n", t_xdna);

    // Compare CPU XDNA partition vs XDNA raw
    ErrorStats stats_xdna;
    stats_xdna.compute(cpu_xd_part.data(), xdna_out.data(), M, nx, K, nv);
    stats_xdna.print("XDNA raw int32 vs CPU int32 (XDNA partition)");

    // Also test CPU full vs CPU XDNA partition (should be exact)
    std::vector<int32_t> cpu_full_xd_part((size_t)nx * M);
    for (int n = 0; n < nx; n++) {
        for (int m = 0; m < M; m++) {
            cpu_full_xd_part[(size_t)n * M + m] = cpu_full[(size_t)(nv + n) * M + m];
        }
    }
    ErrorStats stats_cpu_part;
    stats_cpu_part.compute(cpu_full_xd_part.data(), cpu_xd_part.data(), M, nx, K, nv);
    stats_cpu_part.print("CPU full slice vs CPU XDNA partition (sanity check)");

    // Summary
    printf("\n=== PHASE 16-A SUMMARY ===\n");
    if (stats_xdna.max_abs == 0 && stats_xdna.mismatch_count == 0) {
        printf("PASS: XDNA raw int8 GEMM is BIT-IDENTICAL to CPU reference!\n");
        printf("  -> GEMM arithmetic, accumulation, N-partition, K-chunking are ALL correct\n");
        printf("  -> The 3.78 error comes from Q8_0 scaling, not GEMM\n");
    } else {
        printf("FAIL: XDNA raw int8 GEMM has errors (max_abs=%.6f)\n", stats_xdna.max_abs);
        printf("  -> Error source is in GEMM arithmetic/accumulation/N-partition/K-chunking\n");
        printf("  -> Must debug kernel before addressing Q8_0\n");
    }
    fflush(stdout);

    return (stats_xdna.max_abs == 0 && stats_xdna.mismatch_count == 0) ? 0 : 1;
}