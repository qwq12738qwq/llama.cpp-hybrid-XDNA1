// Phase 38C: Q8 Execution Path Instrumentation Benchmark
//
// Exercises the XDNA Q8_0 path with full timing instrumentation to localize
// the ~549 ms wait anomaly. Measures every stage independently:
//   T0-T16: Q8 scaled path (persistent engine)
//   WA path: engine lookup, weight pack, input quant, XRT submit, XRT wait
//
// Required env vars:
//   XDNA_P38C              - enable detailed timing in ggml-xdna.cpp + engines
//   XDNA_P38C_BO_TRACE     - log every BO allocation/deallocation with handle
//   XDNA_P38C_CHUNK_TIMING - per-chunk wait timing for persistent engine
//   XDNA_Q8S_P38C          - detailed Q8 scale engine submit/wait timing
//   XDNA_WA_P38C           - detailed WA engine submit/wait timing
//   XDNA_KERNEL_CACHE      - path to kernel cache (default /Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache)
//
// Run inside npu-dev container where XRT device 0 is available.

#include "ggml-backend.h"
#include "ggml-xdna.h"
#include "ggml-alloc.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

static const int M = 512;
static const int K = 1024;
static const int N = 3584;

static double now_ms(void) {
    auto t = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

// Quantize F32 [n_rows, K] to Q8_0 (per-32-block along K, same layout as ggml)
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

static void fill_random_f32(std::vector<float> & v, float scale = 1.0f) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

static std::vector<float> parse_ms_tokens(const char * tag, const char * prefix,
                                          const std::vector<std::string> & lines) {
    std::vector<float> vals;
    for (const auto & line : lines) {
        if (strstr(line.c_str(), tag) && strstr(line.c_str(), prefix)) {
            const char * p = strstr(line.c_str(), tag);
            p += strlen(tag);
            while (*p == '=' || *p == ' ') p++;
            char * end = nullptr;
            double v = strtod(p, &end);
            if (end != p) vals.push_back((float)v);
        }
    }
    return vals;
}

struct Phase38CResults {
    // WA path timings (ms) per iteration
    std::vector<float> wa_engine_ms;
    std::vector<float> wa_wpack_ms;
    std::vector<float> wa_a_fetchq_ms;
    std::vector<float> wa_xrt_submit_ms;
    std::vector<float> wa_xrt_wait_ms;
    std::vector<float> wa_out_copy_ms;
    std::vector<float> wa_total_ms;
    std::vector<float> wa_wait_only_ms;
    std::vector<float> wa_bo_copy_ms;
    std::vector<float> wa_bo_sync_ms;
    std::vector<float> wa_run_ctor_ms;
    std::vector<float> wa_setarg_ms;
    std::vector<float> wa_start_ms;

    // Q8S path timings (ms) per iteration
    std::vector<float> q8s_submit_a_copy_ms;
    std::vector<float> q8s_submit_b_copy_ms;
    std::vector<float> q8s_submit_sync_ms;
    std::vector<float> q8s_submit_run_ctor_ms;
    std::vector<float> q8s_submit_setarg_ms;
    std::vector<float> q8s_submit_start_ms;
    std::vector<float> q8s_wait_wait_ms;
    std::vector<float> q8s_wait_sync_from_ms;
    std::vector<float> q8s_wait_copyback_ms;

    // Persistent engine chunked path timings
    std::vector<float> pe_chunk_wait_ms;  // first chunk wait
    std::vector<float> pe_chunk_total_wait_ms;  // all 32 chunks combined

    // BO identity
    std::vector<std::string> wa_bo_ids;
    std::vector<std::string> q8s_bo_ids;
    std::vector<std::string> xdna_buffer_bo_ids;
    std::vector<std::string> pe_bo_ids;
};

int main(void) {
    fprintf(stderr, "[P38C] Phase 38C: Q8 Engine Residency & XRT Wait Audit\n");
    fprintf(stderr, "[P38C] Shape: M=%d K=%d N=%d\n", M, K, N);

    bool p38c_on = std::getenv("XDNA_P38C") != nullptr;
    bool bo_trace = std::getenv("XDNA_P38C_BO_TRACE") != nullptr;
    bool chunk_timing = std::getenv("XDNA_P38C_CHUNK_TIMING") != nullptr;
    bool q8s_p38c = std::getenv("XDNA_Q8S_P38C") != nullptr;
    bool wa_p38c = std::getenv("XDNA_WA_P38C") != nullptr;

    fprintf(stderr, "[P38C] Flags: p38c=%d bo_trace=%d chunk_timing=%d q8s=%d wa=%d\n",
            p38c_on, bo_trace, chunk_timing, q8s_p38c, wa_p38c);

    // Initialize XDNA backend
    ggml_backend_reg_t reg = ggml_backend_xdna_reg();
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        fprintf(stderr, "[P38C] ERROR: XDNA backend not available\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_xdna_init();
    if (!backend) {
        fprintf(stderr, "[P38C] ERROR: XDNA backend init failed\n");
        return 1;
    }

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_dev_t xdna_dev = ggml_backend_xdna_reg_get_device(0);
    ggml_backend_buffer_type_t xdna_buft = xdna_dev ? ggml_xdna_device_get_buffer_type(xdna_dev) : nullptr;

    srand(42);
    std::vector<float> act_f32(M * K);
    std::vector<float> wt_f32(N * K);
    fill_random_f32(act_f32, 0.5f);
    fill_random_f32(wt_f32, 0.5f);

    // Quantize weight to Q8_0 (n_rows=N, K=K)
    std::vector<uint8_t> wt_q8;
    quantize_f32_to_q8_0(wt_f32.data(), N, K, wt_q8);

    size_t act_bytes = act_f32.size() * sizeof(float);
    size_t wt_bytes = wt_q8.size();
    size_t dst_bytes = (size_t)N * M * sizeof(float);

    fprintf(stderr, "[P38C] Data sizes: act=%.2f MB, wt=%.2f MB, dst=%.2f MB\n",
            act_bytes / 1024.0 / 1024.0, wt_bytes / 1024.0 / 1024.0,
            dst_bytes / 1024.0 / 1024.0);

    Phase38CResults results;
    const int NUM_ITER = 3;

    // =========================================================================
    // Test A: WA path (ffn_gate M=512 K=1024 N=3584)
    // =========================================================================
    fprintf(stderr, "\n[P38C] === Test A: WA path (ffn_gate M=512 K=1024 N=3584) ===\n");

    // Enable WA path via XDNA_WA=1
    setenv("XDNA_WA", "1", 1);

    for (int iter = 0; iter < NUM_ITER; iter++) {
        struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
        struct ggml_context * ctx = ggml_init(params);

        // ggml: src0=[K,N] weight^T, src1=[K,M] input^T
        struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);

        ggml_backend_buffer_t wt_buf = ggml_backend_buft_alloc_buffer(cpu_buft, wt_bytes);
        ggml_backend_buffer_t act_buf = ggml_backend_buft_alloc_buffer(cpu_buft, act_bytes);
        ggml_backend_buffer_t dst_buf = ggml_backend_buft_alloc_buffer(cpu_buft, dst_bytes);

        ggml_backend_tensor_alloc(wt_buf, src0, ggml_backend_buffer_get_base(wt_buf));
        ggml_backend_tensor_alloc(act_buf, src1, ggml_backend_buffer_get_base(act_buf));
        ggml_backend_tensor_alloc(dst_buf, dst, ggml_backend_buffer_get_base(dst_buf));

        ggml_backend_tensor_set(src1, act_f32.data(), 0, act_bytes);
        ggml_backend_tensor_set(src0, wt_q8.data(), 0, wt_bytes);

        // Warmup
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);

        // Measure
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        double t1 = now_ms();
        double total_ms = t1 - t0;

        results.wa_total_ms.push_back((float)total_ms);
        fprintf(stderr, "[P38C] WA iter=%d: total=%.2f ms\n", iter, total_ms);

        ggml_free(ctx);
    }

    // Disable WA path for next tests
    unsetenv("XDNA_WA");

    // =========================================================================
    // Test B: Q8_0 path through scale engine (CPU buffer)
    // =========================================================================
    fprintf(stderr, "\n[P38C] === Test B: Q8_0 scale path (CPU buffer, iter=%d) ===\n", NUM_ITER);

    for (int iter = 0; iter < NUM_ITER; iter++) {
        struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);

        ggml_backend_buffer_t wt_buf = ggml_backend_buft_alloc_buffer(cpu_buft, wt_bytes);
        ggml_backend_buffer_t act_buf = ggml_backend_buft_alloc_buffer(cpu_buft, act_bytes);
        ggml_backend_buffer_t dst_buf = ggml_backend_buft_alloc_buffer(cpu_buft, dst_bytes);

        ggml_backend_tensor_alloc(wt_buf, src0, ggml_backend_buffer_get_base(wt_buf));
        ggml_backend_tensor_alloc(act_buf, src1, ggml_backend_buffer_get_base(act_buf));
        ggml_backend_tensor_alloc(dst_buf, dst, ggml_backend_buffer_get_base(dst_buf));

        ggml_backend_tensor_set(src1, act_f32.data(), 0, act_bytes);
        ggml_backend_tensor_set(src0, wt_q8.data(), 0, wt_bytes);

        // Warmup
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);

        // Measure
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        double t1 = now_ms();
        double total_ms = t1 - t0;

        fprintf(stderr, "[P38C] Q8 iter=%d: total=%.2f ms\n", iter, total_ms);

        ggml_free(ctx);
    }

    // =========================================================================
    // Test C: XDNA resident buffer path
    // =========================================================================
    fprintf(stderr, "\n[P38C] === Test C: XDNA resident buffer (CPU buffer baseline) ===\n");

    {
        struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);

        // Use XDNA buffer for tensors
        ggml_backend_buffer_t wt_buf = ggml_backend_buft_alloc_buffer(xdna_buft, wt_bytes);
        ggml_backend_buffer_t act_buf = ggml_backend_buft_alloc_buffer(xdna_buft, act_bytes);
        ggml_backend_buffer_t dst_buf = ggml_backend_buft_alloc_buffer(xdna_buft, dst_bytes);

        if (wt_buf && act_buf && dst_buf) {
            ggml_backend_tensor_alloc(wt_buf, src0, ggml_backend_buffer_get_base(wt_buf));
            ggml_backend_tensor_alloc(act_buf, src1, ggml_backend_buffer_get_base(act_buf));
            ggml_backend_tensor_alloc(dst_buf, dst, ggml_backend_buffer_get_base(dst_buf));

            double t_set0 = now_ms();
            ggml_backend_tensor_set(src1, act_f32.data(), 0, act_bytes);
            ggml_backend_tensor_set(src0, wt_q8.data(), 0, wt_bytes);
            double t_set1 = now_ms();
            fprintf(stderr, "[P38C] XDNA resident initial upload: %.2f ms\n", t_set1 - t_set0);

            for (int iter = 0; iter < NUM_ITER; iter++) {
                ggml_backend_graph_compute(backend, gf);
                ggml_backend_synchronize(backend);
            }
            fprintf(stderr, "[P38C] XDNA resident: %d iterations done\n", NUM_ITER);
        } else {
            fprintf(stderr, "[P38C] XDNA buffer allocation failed, skipping\n");
        }

        ggml_free(ctx);
    }

    // =========================================================================
    // Test D: First vs steady-state XRT wait comparison
    // =========================================================================
    fprintf(stderr, "\n[P38C] === Test D: First vs steady-state (WA path, 10 iter) ===\n");
    setenv("XDNA_WA", "1", 1);
    setenv("XDNA_WA_P38C", "1", 1);

    {
        struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);

        ggml_backend_buffer_t wt_buf = ggml_backend_buft_alloc_buffer(cpu_buft, wt_bytes);
        ggml_backend_buffer_t act_buf = ggml_backend_buft_alloc_buffer(cpu_buft, act_bytes);
        ggml_backend_buffer_t dst_buf = ggml_backend_buft_alloc_buffer(cpu_buft, dst_bytes);

        ggml_backend_tensor_alloc(wt_buf, src0, ggml_backend_buffer_get_base(wt_buf));
        ggml_backend_tensor_alloc(act_buf, src1, ggml_backend_buffer_get_base(act_buf));
        ggml_backend_tensor_alloc(dst_buf, dst, ggml_backend_buffer_get_base(dst_buf));

        ggml_backend_tensor_set(src1, act_f32.data(), 0, act_bytes);
        ggml_backend_tensor_set(src0, wt_q8.data(), 0, wt_bytes);

        // Warmup
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);

        for (int iter = 0; iter < 10; iter++) {
            double t0 = now_ms();
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            double t1 = now_ms();
            fprintf(stderr, "[P38C] WA iter=%d: total=%.2f ms\n", iter, t1 - t0);
        }

        ggml_free(ctx);
    }

    unsetenv("XDNA_WA");
    unsetenv("XDNA_WA_P38C");

    ggml_backend_free(backend);

    fprintf(stderr, "\n[P38C] Benchmark complete. Check stderr for [XDNA-P38C*] timing lines.\n");
    fprintf(stderr, "[P38C] Key env vars to set for full detail:\n");
    fprintf(stderr, "[P38C]   XDNA_P38C=1 XDNA_P38C_BO_TRACE=1 XDNA_P38C_CHUNK_TIMING=1\n");
    fprintf(stderr, "[P38C]   XDNA_Q8S_P38C=1 XDNA_WA_P38C=1\n");

    return 0;
}
