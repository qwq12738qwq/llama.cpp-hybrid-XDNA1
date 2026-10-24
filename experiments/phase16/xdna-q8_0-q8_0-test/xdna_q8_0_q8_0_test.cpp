/**
 * xdna_q8_0_input_test.cpp
 *
 * Test XDNA Q8_0 path with Q8_0 input (not F32).
 * This avoids F32->int8 quantization error.
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

static void cpu_gemm_q8_0_q8_0(const block_q8_0_test* A, const block_q8_0_test* B, float* C, int M, int K, int N, int n_start, int n_end) {
    int nb = K / QK8_0;
    int Nslice = n_end - n_start;
    for (int n = 0; n < Nslice; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                float a_scale = fp16_to_fp32(A[(size_t)m * nb + b].d);
                float w_scale = fp16_to_fp32(B[(size_t)(n_start + n) * nb + b].d);
                for (int j = 0; j < QK8_0; j++) {
                    int k = b * QK8_0 + j;
                    int8_t qa = A[(size_t)m * nb + b].qs[j];
                    int8_t qb = B[(size_t)(n_start + n) * nb + b].qs[j];
                    acc += qa * qb * a_scale * w_scale;
                }
            }
            C[(size_t)n * M + m] = acc;
        }
    }
}

struct ErrorStats {
    double max_abs = 0, mean_abs = 0, rmse = 0;

    void compute(const float* cpu, const float* xdna, int M, int N) {
        size_t total = (size_t)M * N;
        double sum_abs = 0, sum_sq = 0;
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                double d = std::fabs((double)cpu[(size_t)n * M + m] - (double)xdna[(size_t)n * M + m]);
                if (d > max_abs) max_abs = d;
                sum_abs += d;
                sum_sq += d * d;
            }
        }
        mean_abs = total > 0 ? sum_abs / total : 0;
        rmse = total > 0 ? std::sqrt(sum_sq / total) : 0;
    }
};

static void run_xdna_q8_0_q8_0(int M, int K, int N, int n_start, int n_end,
                                const block_q8_0_test* A, const block_q8_0_test* B, float* output) {
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
    std::vector<uint8_t> w_data((size_t)Nslice * nb * sizeof(block_q8_0_test));
    for (int n = 0; n < Nslice; n++) {
        for (int b = 0; b < nb; b++) {
            block_q8_0_test* block = (block_q8_0_test*)(w_data.data() + ((size_t)n * nb + b) * sizeof(block_q8_0_test));
            *block = B[(size_t)(n_start + n) * nb + b];
        }
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());

    std::vector<uint8_t> b_data((size_t)M * nb * sizeof(block_q8_0_test));
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            block_q8_0_test* block = (block_q8_0_test*)(b_data.data() + ((size_t)m * nb + b) * sizeof(block_q8_0_test));
            *block = A[(size_t)m * nb + b];
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

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    
    int M = 512, K = 1024, N = 4096, nv = 1024;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M" && i + 1 < argc) M = std::stoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc) K = std::stoi(argv[++i]);
        else if (a == "--N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (a == "--nv" && i + 1 < argc) nv = std::stoi(argv[++i]);
    }
    int nx = N - nv;

    if (K % QK8_0 != 0) die("K must be multiple of 32");

    printf("[Q8_0xQ8_0] M=%d K=%d N=%d Nv=%d Nx=%d\n", M, K, N, nv, nx);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int nb = K / QK8_0;
    std::vector<block_q8_0_test> A((size_t)M * nb);
    std::vector<block_q8_0_test> B((size_t)N * nb);

    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            block_q8_0_test* block = &A[(size_t)m * nb + b];
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

    printf("[Q8_0xQ8_0] Computing CPU reference...\n");
    double t0 = now_ms();
    std::vector<float> cpu_ref((size_t)nx * M);
    cpu_gemm_q8_0_q8_0(A.data(), B.data(), cpu_ref.data(), M, K, N, nv, N);
    double t_cpu = now_ms() - t0;
    printf("[Q8_0xQ8_0] CPU: %.2f ms\n", t_cpu);

    printf("[Q8_0xQ8_0] Running XDNA Q8_0xQ8_0 path...\n");
    std::vector<float> xdna_out((size_t)nx * M);
    t0 = now_ms();
    run_xdna_q8_0_q8_0(M, K, N, nv, N, A.data(), B.data(), xdna_out.data());
    double t_xdna = now_ms() - t0;
    printf("[Q8_0xQ8_0] XDNA: %.2f ms\n", t_xdna);

    ErrorStats stats;
    stats.compute(cpu_ref.data(), xdna_out.data(), M, nx);
    printf("\n=== Q8_0 x Q8_0 ERROR ANALYSIS ===\n");
    printf("  max_abs = %.6f\n", stats.max_abs);
    printf("  mean_abs = %.6f\n", stats.mean_abs);
    printf("  RMSE = %.6f\n", stats.rmse);

    if (stats.max_abs < 1.0) {
        printf(">>> PASS: Q8_0 x Q8_0 path is accurate! <<<\n");
    } else {
        printf(">>> FAIL: Q8_0 x Q8_0 has error %.3f <<<\n", stats.max_abs);
    }

    return 0;
}