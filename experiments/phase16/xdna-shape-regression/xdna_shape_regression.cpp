/**
 * xdna_shape_regression.cpp
 *
 * Phase 16-J: Verify all key Qwen shapes with the 128->127 fix.
 * Tests M=448/512, N=2048/4096/6144, K=1024/2048/3072/3584
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
#include <algorithm>

#define QK8_0 32

typedef struct {
    uint16_t d;
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

static uint16_t fp32_to_fp16(float f) {
    uint32_t x = *reinterpret_cast<const uint32_t*>(&f);
    uint16_t h = 0;
    int exp = (x >> 23) & 0xFF;
    int mantissa = x & 0x7FFFFF;
    int sign = x >> 31;
    if (exp == 0) h = sign << 15;
    else if (exp == 0xFF) h = (sign << 15) | 0x7C00;
    else {
        int new_exp = exp - 127 + 15;
        if (new_exp >= 31) h = (sign << 15) | 0x7C00;
        else if (new_exp <= 0) h = (sign << 15);
        else h = (sign << 15) | (new_exp << 10) | (mantissa >> 13);
    }
    return h;
}

static float fp16_to_fp32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1F;
    int mantissa = h & 0x3FF;
    if (exp == 0) return sign ? -0.0f : 0.0f;
    if (exp == 0x1F) return sign ? -INFINITY : INFINITY;
    float f = (1.0f + mantissa / 1024.0f) * powf(2.0f, exp - 15);
    return sign ? -f : f;
}

static void cpu_gemm_q8_0(const float* A, const block_q8_0_test* B, float* C, int M, int K, int N, int n_start, int n_end) {
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

struct ShapeResult {
    int M, K, N, Nv, Nx;
    double cpu_ms, xdna_ms;
    double max_abs, mean_abs, rmse;
    size_t mismatch;
    bool pass;
};

static void run_shape_test(int M, int K, int N, int nv, std::vector<ShapeResult>& results) {
    int nx = N - nv;
    int nb = K / QK8_0;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A((size_t)K * M);
    for (auto& v : A) v = dist(rng);

    std::vector<block_q8_0_test> B((size_t)K / QK8_0 * N);
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            block_q8_0_test* block = &B[(size_t)n * nb + b];
            float amax = 0;
            for (int j = 0; j < QK8_0; j++) amax = std::max(amax, std::fabs(dist(rng) * 10.0f));
            float scale = amax / 127.0f;
            if (scale == 0) scale = 1.0f;
            block->d = fp32_to_fp16(scale);
            for (int j = 0; j < QK8_0; j++) {
                float v = dist(rng) * 10.0f;
                int q = (int)std::lroundf(v / scale);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                block->qs[j] = (int8_t)q;
            }
        }
    }

    ggml_backend_load_all();
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    if (!backend_xdna) backend_xdna = ggml_backend_init_by_name("AMD Phoenix XDNA1 NPU", "");
    if (!backend_xdna) backend_xdna = ggml_backend_xdna_init();
    if (!backend_xdna) die("XDNA backend not available");

    struct ggml_init_params params{0};
    params.mem_size = (size_t)K * N * 2 + (size_t)K * M * sizeof(float) + (size_t)M * nx * sizeof(float) + 64 * 1024 * 1024;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init failed");

    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, nx);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    if (!buf) die("alloc failed");

    std::vector<uint8_t> w_data((size_t)nx * nb * sizeof(block_q8_0_test));
    for (int n = 0; n < nx; n++) {
        for (int b = 0; b < nb; b++) {
            block_q8_0_test* block = (block_q8_0_test*)(w_data.data() + ((size_t)n * nb + b) * sizeof(block_q8_0_test));
            *block = B[(size_t)(nv + n) * nb + b];
        }
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());

    std::vector<float> b_data((size_t)K * M);
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) b_data[(size_t)k + (size_t)m * K] = A[(size_t)k * M + m];
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

    // CPU reference
    double t0 = now_ms();
    std::vector<float> cpu_xd_part((size_t)nx * M);
    cpu_gemm_q8_0(A.data(), B.data(), cpu_xd_part.data(), M, K, N, nv, N);
    double t_cpu = now_ms() - t0;

    // XDNA
    t0 = now_ms();
    if (ggml_backend_graph_compute(backend_xdna, gf) != GGML_STATUS_SUCCESS) {
        die("graph_compute failed");
    }
    double t_xdna = now_ms() - t0;

    std::vector<float> xdna_out((size_t)nx * M);
    ggml_backend_tensor_get(c, xdna_out.data(), 0, xdna_out.size() * sizeof(float));

    // Compare
    ShapeResult r{M, K, N, nv, nx, t_cpu, t_xdna, 0, 0, 0, 0, false};
    double sum_abs = 0, sum_sq = 0;
    for (int n = 0; n < nx; n++) {
        for (int m = 0; m < M; m++) {
            double d = std::fabs((double)cpu_xd_part[(size_t)n * M + m] - (double)xdna_out[(size_t)n * M + m]);
            if (d > r.max_abs) r.max_abs = d;
            sum_abs += d;
            sum_sq += d * d;
            if (d > 1e-3) r.mismatch++;
        }
    }
    r.mean_abs = sum_abs / ((size_t)M * nx);
    r.rmse = std::sqrt(sum_sq / ((size_t)M * nx));
    r.pass = (r.max_abs < 3.78);  // Stage 15 baseline

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend_xdna);

    results.push_back(r);
    printf("  M=%d K=%d N=%d: max_abs=%.3f mean_abs=%.3f RMSE=%.3f mismatch=%.1f%% %s\n",
           M, K, N, r.max_abs, r.mean_abs, r.rmse, 100.0 * r.mismatch / ((size_t)M * nx),
           r.pass ? "PASS" : "FAIL");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    
    printf("=== PHASE 16-J: QWEN SHAPE REGRESSION ===\n");
    
    std::vector<ShapeResult> results;
    
    // Qwen PP shapes
    const int shapes[][3] = {
        {448, 1024, 2048}, {448, 1024, 4096}, {448, 1024, 6144},
        {512, 1024, 2048}, {512, 1024, 4096}, {512, 1024, 6144},
        {512, 2048, 4096}, {512, 3072, 4096}, {512, 3584, 4096},
    };
    
    for (auto& s : shapes) {
        int M = s[0], K = s[1], N = s[2];
        int nv = std::min(1024, N/4);  // 25% Vulkan partition
        if (N < 1024) nv = N/4;
        printf("\nTesting M=%d K=%d N=%d (nv=%d nx=%d)...\n", M, K, N, nv, N-nv);
        run_shape_test(M, K, N, nv, results);
    }
    
    // Summary
    printf("\n=== PHASE 16-J SUMMARY ===\n");
    int pass_count = 0;
    for (auto& r : results) {
        printf("  M=%d K=%d N=%d: max_abs=%.3f %s\n", r.M, r.K, r.N, r.max_abs, r.pass ? "PASS" : "FAIL");
        if (r.pass) pass_count++;
    }
    printf("\nPassed: %d/%zu\n", pass_count, results.size());
    
    return (pass_count == (int)results.size()) ? 0 : 1;
}