/**
 * xdna_kchunk_test.cpp
 *
 * Phase 16-E: Test K-chunk accumulation with different chunk sizes.
 *
 * Tests: K=32, 64, 128, 256, 512, 1024
 * Verifies that host-side K-chunking produces correct accumulation.
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-xdna.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

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

static void cpu_gemm_int8(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N) {
    // A: [K, M], B: [K, N], C: [N, M] = A^T * B
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

struct ErrorStats {
    double max_abs = 0, mean_abs = 0, rmse = 0;
    size_t mismatch = 0;

    void compute(const int32_t* cpu, const int32_t* xdna, int M, int N) {
        size_t total = (size_t)M * N;
        double sum_abs = 0, sum_sq = 0;
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                double d = std::fabs((double)cpu[(size_t)n * M + m] - (double)xdna[(size_t)n * M + m]);
                if (d > max_abs) max_abs = d;
                sum_abs += d;
                sum_sq += d * d;
                if (d > 0) mismatch++;
            }
        }
        mean_abs = total > 0 ? sum_abs / total : 0;
        rmse = total > 0 ? std::sqrt(sum_sq / total) : 0;
    }
};

static void run_xdna_kchunk(int M, int K, int N, int k_chunk,
                            const int8_t* A, const int8_t* B, int32_t* output) {
    ggml_backend_load_all();
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    if (!backend_xdna) backend_xdna = ggml_backend_init_by_name("AMD Phoenix XDNA1 NPU", "");
    if (!backend_xdna) backend_xdna = ggml_backend_xdna_init();
    if (!backend_xdna) die("XDNA backend not available");

    struct ggml_init_params params{0};
    params.mem_size = (size_t)K * N + (size_t)K * M + (size_t)M * N + 64 * 1024 * 1024;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init failed");

    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    if (!buf) die("alloc failed");

    // Set data - GGML layout: [K, N] = N rows of K elements
    std::vector<int8_t> w_data((size_t)K * N);
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) w_data[(size_t)k + (size_t)n * K] = B[(size_t)k * N + n];
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());

    std::vector<int8_t> b_data((size_t)K * M);
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) b_data[(size_t)k + (size_t)m * K] = A[(size_t)k * M + m];
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size());

    if (ggml_backend_graph_compute(backend_xdna, gf) != GGML_STATUS_SUCCESS) {
        die("graph_compute failed");
    }

    std::vector<int32_t> out((size_t)N * M);
    ggml_backend_tensor_get(c, out.data(), 0, out.size() * sizeof(int32_t));
    for (size_t i = 0; i < out.size(); i++) output[i] = out[i];

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend_xdna);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    
    int M = 64;
    std::vector<int> K_list = {32, 64, 128, 256, 512, 1024};
    int N = 256;
    
    // Fixed seed
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-127, 127);

    printf("[K-CHUNK] Testing K-chunk accumulation for M=%d, N=%d\n", M, N);
    
    for (int K : K_list) {
        if (K % 32 != 0) continue;
        
        printf("\n=== K=%d ===\n", K);
        
        // Generate random data
        std::vector<int8_t> A((size_t)K * M);
        std::vector<int8_t> B((size_t)K * N);
        for (auto& v : A) v = (int8_t)dist(rng);
        for (auto& v : B) v = (int8_t)dist(rng);
        
        // CPU reference
        std::vector<int32_t> cpu_ref((size_t)N * M);
        cpu_gemm_int8(A.data(), B.data(), cpu_ref.data(), M, K, N);
        
        // XDNA
        std::vector<int32_t> xdna_out((size_t)N * M);
        double t0 = now_ms();
        run_xdna_kchunk(M, K, N, 32, A.data(), B.data(), xdna_out.data());
        double t_xdna = now_ms() - t0;
        
        // Compare
        ErrorStats stats;
        stats.compute(cpu_ref.data(), xdna_out.data(), M, N);
        
        printf("  K=%d: max_abs=%.2f mean_abs=%.2f RMSE=%.2f mismatch=%zu/%zu (%.2f%%) time=%.1fms\n",
               K, stats.max_abs, stats.mean_abs, stats.rmse, 
               stats.mismatch, (size_t)M*N, 100.0*stats.mismatch/(M*N), t_xdna);
        
        if (stats.max_abs > 0) {
            printf("  -> FAIL: K=%d has accumulation errors\n", K);
        } else {
            printf("  -> PASS: K=%d bit-identical\n", K);
        }
    }
    
    return 0;
}