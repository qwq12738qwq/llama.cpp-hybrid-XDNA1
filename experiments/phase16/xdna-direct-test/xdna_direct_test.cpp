/**
 * xdna_direct_test.cpp
 *
 * Direct XDNA engine test with known inputs for debugging.
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-xdna.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

static void die(const char* msg) {
    fprintf(stderr, "[FATAL] %s\n", msg);
    exit(1);
}

// CPU reference: int8 x int8 -> int32 GEMM
// A: [K, M] row-major, B: [K, N] row-major, C: [M, N] row-major
static void cpu_gemm_int8(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[(size_t)k * M + m] * (int32_t)B[(size_t)k * N + n];
            }
            C[(size_t)m * N + n] = acc;
        }
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int M = 64, K = 128, N = 64;
    bool test_ones = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M" && i + 1 < argc) M = std::stoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc) K = std::stoi(argv[++i]);
        else if (a == "--N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (a == "--ones") test_ones = true;
    }

    printf("[DIRECT] M=%d K=%d N=%d\n", M, K, N);
    fflush(stdout);

    // Generate test data
    std::vector<int8_t> A((size_t)K * M);
    std::vector<int8_t> B((size_t)K * N);

    if (test_ones) {
        for (auto& v : A) v = 1;
        for (auto& v : B) v = 1;
        printf("[DIRECT] Test mode: all ones\n");
    } else {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(-127, 127);
        for (auto& v : A) v = static_cast<int8_t>(dist(rng));
        for (auto& v : B) v = static_cast<int8_t>(dist(rng));
    }

    // CPU reference
    printf("[DIRECT] Computing CPU reference...\n");
    std::vector<int32_t> cpu_ref((size_t)M * N);
    cpu_gemm_int8(A.data(), B.data(), cpu_ref.data(), M, K, N);

    // Print first few CPU values
    printf("[DIRECT] CPU first 16 values (row-major M*N):\n");
    for (int i = 0; i < 16; i++) {
        printf("  cpu[%d] = %d\n", i, cpu_ref[i]);
    }

    // Initialize XDNA backend
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

    // Create tensors - ggml mul_mat(src0, src1) where src0 is weight [K,N] and src1 is input [K,M]
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    if (!buf) die("alloc_ctx_tensors failed");

    // Set weight data - GGML layout: [K, N] = N rows of K elements, index = k + n*K
    std::vector<int8_t> w_data((size_t)K * N);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            w_data[(size_t)k + (size_t)n * K] = B[(size_t)k * N + n];
        }
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());

    // Set input data - GGML layout: [K, M] = M rows of K elements, index = k + m*K
    std::vector<int8_t> b_data((size_t)K * M);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            b_data[(size_t)k + (size_t)m * K] = A[(size_t)k * M + m];
        }
    }
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size());

    // Execute
    printf("[DIRECT] Running XDNA backend...\n");
    if (ggml_backend_graph_compute(backend_xdna, gf) != GGML_STATUS_SUCCESS) {
        die("graph_compute failed");
    }

    // Get output - GGML mul_mat output is [N, M] in ne, stored as [M, N] row-major
    std::vector<float> out_f32((size_t)M * N);
    ggml_backend_tensor_get(c, out_f32.data(), 0, out_f32.size() * sizeof(float));

    printf("[DIRECT] XDNA first 16 values (row-major M*N):\n");
    for (int i = 0; i < 16; i++) {
        printf("  xdna[%d] = %.0f (cpu=%d)\n", i, out_f32[i], cpu_ref[i]);
    }

    // Compare
    double max_abs = 0, mean_abs = 0;
    size_t mismatches = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double d = std::fabs((double)cpu_ref[m * N + n] - (double)out_f32[m * N + n]);
            if (d > max_abs) max_abs = d;
            mean_abs += d;
            if (d > 0.5) mismatches++;
        }
    }
    mean_abs /= (size_t)M * N;

    printf("\n[DIRECT] max_abs=%.2f mean_abs=%.2f mismatches=%zu/%zu\n",
           max_abs, mean_abs, mismatches, (size_t)M * N);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend_xdna);

    return (max_abs == 0) ? 0 : 1;
}
