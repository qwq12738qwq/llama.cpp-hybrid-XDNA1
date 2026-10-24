// Phase 40L: Q8_0 shape sweep benchmark.
//
// Tests M in {1, 16, 128, 512, 1024} with K=1024, N=3584 (Qwen3.5-0.8B FFN shapes).
// Measures host preparation, BO upload, XRT submit, device execution, sync, copyback.
//
// Build:
//   g++ -O2 -std=c++17 -I. -Iggml/include -Iggml/src/ggml-xdna \
//       experiments/phase40/cpu_q8_reference.cpp \
//       experiments/phase40/bench_q8_sweep.cpp \
//       ggml/src/ggml-xdna/ggml-xdna.cpp \
//       ggml/src/ggml-xdna/xdna-q8-scale.cpp \
//       ggml/src/ggml-xdna/xdna-q8-c4.cpp \
//       ggml/src/ggml-xdna/xdna-q8-c4-device-scale.cpp \
//       ggml/src/ggml-xdna/xdna-q8-scale-c4.cpp \
//       ggml/src/ggml-xdna/xdna-kernel-registry.cpp \
//       ggml/src/ggml-xdna/xdna-buffer.cpp \
//       ggml/src/ggml-xdna/xdna-whole-array.cpp \
//       ggml/src/ggml-xdna/xdna-persistent.cpp \
//       -lxrt_coreutil -o bench_q8_sweep -lm
//
// Run:
//   XDNA_Q8_ENGINE=c4_ds ./bench_q8_sweep
//   XDNA_Q8_ENGINE=scale ./bench_q8_sweep
//   XDNA_Q8_ENGINE=auto ./bench_q8_sweep

#include "cpu_q8_reference.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include "ggml.h"
#include "ggml-backend.h"

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Build a random Q8_0 tensor [rows, K]
void build_random_q8(std::vector<uint8_t>& q8_data, int rows, int K, float scale_min = 0.1f, float scale_max = 5.0f) {
    int nb = K / QK8_0;
    q8_data.resize((size_t)rows * nb * 34);
    std::mt19937 rng(42 + rows);
    std::uniform_real_distribution<float> scale_dist(scale_min, scale_max);
    std::uniform_int_distribution<int> quant_dist(-127, 127);
    
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nb; b++) {
            float scale = scale_dist(rng);
            uint8_t* ptr = q8_data.data() + ((size_t)r * nb + b) * 34;
            uint16_t h = fp32_to_fp16(scale);
            memcpy(ptr, &h, 2);
            for (int kk = 0; kk < 32; kk++) {
                ptr[2 + kk] = (int8_t)quant_dist(rng);
            }
        }
    }
}

// Run single Q8_0 GEMM via ggml backend API
struct BenchResult {
    double total_ms;
    double host_prep_ms;
    double bo_upload_ms;
    double device_ms;
    double sync_ms;
    double copyback_ms;
    float max_error;
    float rmse;
};

BenchResult run_q8_gemm(ggml_backend_t backend, int M, int K, int N, const float* C_ref, int ref_len) {
    BenchResult result = {};
    
    struct ggml_init_params params = { 0 };
    params.mem_size = 512 * 1024 * 1024;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    
    std::vector<uint8_t> weight_q8;
    std::vector<float> input_f32(K * M);
    
    double t0 = now_ms();
    build_random_q8(weight_q8, N, K);
    for (size_t i = 0; i < input_f32.size(); i++) {
        input_f32[i] = (float)((i * 7 + 3) % 256) / 256.0f;
    }
    double t1 = now_ms();
    result.host_prep_ms = t1 - t0;
    
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    ggml_backend_tensor_set(a, weight_q8.data(), 0, weight_q8.size());
    ggml_backend_tensor_set(b, input_f32.data(), 0, input_f32.size() * sizeof(float));
    
    double t2 = now_ms();
    result.bo_upload_ms = t2 - t1;
    
    double t_dev0 = now_ms();
    ggml_backend_graph_compute(backend, gf);
    double t_dev1 = now_ms();
    result.device_ms = t_dev1 - t_dev0;
    
    double t3 = now_ms();
    std::vector<float> xdna_out(N * M);
    ggml_backend_tensor_get(c, xdna_out.data(), 0, xdna_out.size() * sizeof(float));
    double t4 = now_ms();
    result.copyback_ms = t4 - t3;
    result.total_ms = t4 - t0;
    
    // Compute error against reference
    if (C_ref && ref_len == M * N) {
        double sum_sq = 0.0;
        float max_err = 0.0f;
        for (int i = 0; i < M * N; i++) {
            float err = fabsf(C_ref[i] - xdna_out[i]);
            if (err > max_err) max_err = err;
            sum_sq += (double)err * (double)err;
        }
        result.max_error = max_err;
        result.rmse = (float)sqrt(sum_sq / (M * N));
    }
    
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    
    return result;
}

int main() {
    const int K = 1024;
    const int N = 3584;
    const int M_values[] = {1, 16, 128, 512, 1024};
    const int num_M = sizeof(M_values) / sizeof(M_values[0]);
    
    ggml_backend_load_all();
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, "");
    
    if (!backend_xdna || !backend_cpu) {
        fprintf(stderr, "Backends not available\n");
        return 1;
    }
    
    const char* engine_env = getenv("XDNA_Q8_ENGINE");
    fprintf(stderr, "[SWEEP] XDNA_Q8_ENGINE=%s\n", engine_env ? engine_env : "auto");
    
    fprintf(stderr, "\n=== Q8_0 Shape Sweep (K=%d, N=%d) ===\n", K, N);
    fprintf(stderr, "%-6s %12s %12s %12s %12s %12s %12s %12s %12s\n",
            "M", "Total(ms)", "Host(ms)", "Upload(ms)", "Device(ms)", "Sync(ms)", "Copy(ms)", "MaxErr", "RMSE");
    fprintf(stderr, "%-6s %12s %12s %12s %12s %12s %12s %12s %12s\n",
            "---", "------------", "------------", "------------", "------------", "------------", "------------", "------------", "------------");
    
    for (int idx = 0; idx < num_M; idx++) {
        int M = M_values[idx];
        
        // Build reference on CPU
        std::vector<uint8_t> A_q8, B_q8;
        build_random_q8(A_q8, M, K);
        build_random_q8(B_q8, N, K);
        
        std::vector<float> C_ref(M * N);
        cpu_q8_gemm_ref(A_q8.data(), B_q8.data(), C_ref.data(), M, K, N);
        
        // Run on XDNA
        BenchResult r = run_q8_gemm(backend_xdna, M, K, N, C_ref.data(), M * N);
        
        fprintf(stderr, "%-6d %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4e %12.4e %s\n",
                M, r.total_ms, r.host_prep_ms, r.bo_upload_ms, r.device_ms,
                r.sync_ms, r.copyback_ms, r.max_error, r.rmse,
                r.max_error < 1e-2 ? "PASS" : "FAIL");
        fflush(stderr);
    }
    
    ggml_backend_free(backend_xdna);
    ggml_backend_free(backend_cpu);
    
    fprintf(stderr, "\n[Sweep complete]\n");
    return 0;
}
