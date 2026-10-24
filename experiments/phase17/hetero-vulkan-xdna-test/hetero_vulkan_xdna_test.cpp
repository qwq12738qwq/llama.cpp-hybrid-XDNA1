/**
 * hetero_vulkan_xdna_test.cpp
 *
 * Stage 15: True Vulkan + XDNA N-dimension heterogeneous MUL_MAT for Qwen PP.
 *
 * For one logical MUL_MAT  C = A x B  (A: [K,M] F32 activation, B: [K,N] Q8_0 weight):
 *   - split the N (output) dimension into two slices
 *   - Vulkan computes  C_vk  = A x B[:, 0:Nv)
 *   - XDNA   computes  C_xd  = A x B[:, Nv:N)
 *   - merge:          C = concat(C_vk, C_xd, axis=0)
 *
 * Usage:
 *   hetero-vulkan-xdna-test [--M 512] [--K 1024] [--N 4096] [--nv 1024] [--iters 3]
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-profile.h"
#include "ggml-xdna.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <future>
#include <atomic>
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

static void quantize_weight_q8(const std::vector<float>& w_f32, int K, int N, std::vector<uint8_t>& w_q8) {
    size_t nbytes = (size_t)(K * N / 32) * 34;
    w_q8.resize(nbytes);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w_f32.data(), w_q8.data(), 0, N, K, nullptr);
}

static void copy_q8_cols(const uint8_t* full, int K, int N, int n0, int n1, std::vector<uint8_t>& slice) {
    int nb = K / 32;
    int cols = n1 - n0;
    slice.resize((size_t)cols * nb * 34);
    for (int n = n0; n < n1; n++) {
        for (int b = 0; b < nb; b++) {
            size_t src = (size_t)n * nb + b;
            size_t dst = (size_t)(n - n0) * nb + b;
            memcpy(&slice[dst * 34], &full[src * 34], 34);
        }
    }
}

static std::vector<float> run_full(ggml_backend_t backend, const std::vector<uint8_t>& w_q8,
                                    const std::vector<float>& input_f32, int M, int K, int N,
                                    double* ms) {
    struct ggml_init_params params{0};
    params.mem_size = (size_t)(K * N * 34 / 32) + (size_t)(K * M * 4) + 64 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init failed");

    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) die("alloc_ctx_tensors failed");

    ggml_backend_tensor_set(w, w_q8.data(), 0, w_q8.size());
    ggml_backend_tensor_set(b, input_f32.data(), 0, input_f32.size() * sizeof(float));

    double t0 = now_ms();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) die("graph_compute failed");
    double t1 = now_ms();
    if (ms) *ms = t1 - t0;

    std::vector<float> out((size_t)N * M);
    ggml_backend_tensor_get(c, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static std::vector<float> run_slice(ggml_backend_t backend, const std::vector<uint8_t>& w_q8_slice,
                                     const std::vector<float>& input_f32, int M, int K, int Nslice,
                                     double* ms) {
    struct ggml_init_params params{0};
    params.mem_size = (size_t)(K * Nslice * 34 / 32) + (size_t)(K * M * 4) + 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init failed");

    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, Nslice);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) die("alloc_ctx_tensors failed");

    ggml_backend_tensor_set(w, w_q8_slice.data(), 0, w_q8_slice.size());
    ggml_backend_tensor_set(b, input_f32.data(), 0, input_f32.size() * sizeof(float));

    double t0 = now_ms();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) die("graph_compute(slice) failed");
    double t1 = now_ms();
    if (ms) *ms = t1 - t0;

    std::vector<float> out((size_t)Nslice * M);
    ggml_backend_tensor_get(c, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static void stats(const std::vector<float>& a, const std::vector<float>& b, double& max_abs, double& max_rel) {
    max_abs = 0; max_rel = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > max_abs) max_abs = d;
        double denom = std::fabs((double)a[i]) + 1e-6;
        double rel = d / denom;
        if (rel > max_rel) max_rel = rel;
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    int M = 512, K = 1024, N = 4096, nv = 1024;
    fprintf(stderr, "[MAIN] entered M=%d K=%d N=%d nv=%d\n", M, K, N, nv);
    int iters = 3;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M" && i + 1 < argc) M = std::stoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc) K = std::stoi(argv[++i]);
        else if (a == "--N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (a == "--nv" && i + 1 < argc) nv = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
    }
    int nx = N - nv;

    printf("[HETERO] M=%d K=%d N=%d  Vulkan N=[0,%d)  XDNA N=[%d,%d)\n", M, K, N, nv, nv, N);
    fflush(stdout);

    ggml_backend_load_all();

    ggml_backend_t backend_vk = ggml_backend_init_by_name("Vulkan0", "");
    if (!backend_vk) backend_vk = ggml_backend_init_by_name("Vulkan", "");
    if (!backend_vk) die("Vulkan backend not available");
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    if (!backend_xdna) backend_xdna = ggml_backend_init_by_name("AMD Phoenix XDNA1 NPU", "");
    if (!backend_xdna) die("XDNA backend not available");
    ggml_backend_t backend_cpu = ggml_backend_init_by_name("CPU", "");
    if (!backend_cpu) die("CPU backend not available");

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> w_f32((size_t)K * N);
    std::vector<float> input_f32((size_t)K * M);
    for (auto& v : w_f32) v = dist(rng);
    for (auto& v : input_f32) v = dist(rng);

    std::vector<uint8_t> w_q8;
    quantize_weight_q8(w_f32, K, N, w_q8);

    std::vector<uint8_t> w_q8_vk, w_q8_xd;
    copy_q8_cols(w_q8.data(), K, N, 0, nv, w_q8_vk);
    copy_q8_cols(w_q8.data(), K, N, nv, N, w_q8_xd);

    double t_cpu = 0;
    auto cpu_ref = run_full(backend_cpu, w_q8, input_f32, M, K, N, &t_cpu);

    double t_vk_full = 0;
    auto vk_full = run_full(backend_vk, w_q8, input_f32, M, K, N, &t_vk_full);

    double t_xd_full = 0;
    auto xd_full = run_full(backend_xdna, w_q8, input_f32, M, K, N, &t_xd_full);

    double e_vk, r_vk, e_xd, r_xd;
    stats(vk_full, cpu_ref, e_vk, r_vk);
    stats(xd_full, cpu_ref, e_xd, r_xd);

    printf("\n=== CORRECTNESS (vs CPU Q8_0 reference) ===\n");
    printf("Vulkan-only full : max_abs=%.5f max_rel=%.5f  (%.2f ms)\n", e_vk, r_vk, t_vk_full);
    printf("XDNA-only  full : max_abs=%.5f max_rel=%.5f  (%.2f ms)\n", e_xd, r_xd, t_xd_full);

    double t_vk_slice = 0, t_xd_slice = 0;
    std::vector<float> hetero((size_t)N * M);

    double best_vk = 1e30, best_xd = 1e30, best_wall = 1e30, best_overlap = 0;
    for (int it = 0; it < iters; it++) {
        double t0 = now_ms();
        // Run Vulkan and XDNA concurrently using std::async
        auto fut_vk = std::async(std::launch::async, [&]() {
            double t_vk = 0;
            auto result = run_slice(backend_vk, w_q8_vk, input_f32, M, K, nv, &t_vk);
            return std::make_pair(result, t_vk);
        });
        auto fut_xd = std::async(std::launch::async, [&]() {
            double t_xd = 0;
            auto result = run_slice(backend_xdna, w_q8_xd, input_f32, M, K, nx, &t_xd);
            return std::make_pair(result, t_xd);
        });
        
        auto [c_vk, t_vk_slice] = fut_vk.get();
        auto [c_xd, t_xd_slice] = fut_xd.get();
        double t1 = now_ms();
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < nv; n++) hetero[(size_t)m * N + n] = c_vk[(size_t)m * nv + n];
            for (int n = 0; n < nx; n++) hetero[(size_t)m * N + (nv + n)] = c_xd[(size_t)m * nx + n];
        }
        double wall = t1 - t0;
        double overlap = t_vk_slice + t_xd_slice - wall;
        if (it == iters - 1) { best_vk = t_vk_slice; best_xd = t_xd_slice; best_wall = wall; best_overlap = overlap; }
    }

    double e_h, r_h;
    stats(hetero, cpu_ref, e_h, r_h);
    std::vector<float> ref_vk((size_t)nv * M), ref_xd((size_t)nx * M);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < nv; n++) ref_vk[(size_t)m * nv + n] = cpu_ref[(size_t)m * N + n];
        for (int n = 0; n < nx; n++) ref_xd[(size_t)m * nx + n] = cpu_ref[(size_t)m * N + (nv + n)];
    }
    auto c_vk2 = run_slice(backend_vk, w_q8_vk, input_f32, M, K, nv, nullptr);
    auto c_xd2 = run_slice(backend_xdna, w_q8_xd, input_f32, M, K, nx, nullptr);
    double e_vk_region, r_vk_region, e_xd_region, r_xd_region;
    stats(c_vk2, ref_vk, e_vk_region, r_vk_region);
    stats(c_xd2, ref_xd, e_xd_region, r_xd_region);

    float max_vk = 0, max_xd = 0;
    for (auto v : c_vk2) max_vk = std::max(max_vk, std::fabs(v));
    for (auto v : c_xd2) max_xd = std::max(max_xd, std::fabs(v));

    printf("\n=== HETEROGENEOUS N-PARTITION ===\n");
    printf("[HETERO] op=MUL_MAT M=%d K=%d N=%d\n", M, K, N);
    printf("Vulkan: N=[0,%d) elapsed=%.2f ms  max_out=%.3f\n", nv, best_vk, max_vk);
    printf("XDNA  : N=[%d,%d) elapsed=%.2f ms  max_out=%.3f\n", nv, N, best_xd, max_xd);
    printf("merge : (host concat) copied %zu floats\n", hetero.size());
    printf("sequential wall (vk+xd) = %.2f ms\n", best_vk + best_xd);
    printf("concurrent wall        = %.2f ms  overlap=%.2f ms\n", best_wall, best_overlap);

    printf("\n=== RESULTS ===\n");
    printf("Vulkan region vs CPU  : max_abs=%.5f max_rel=%.5f  (Vulkan work %s 0)\n", e_vk_region, r_vk_region, max_vk > 0 ? "!=" : "==");
    printf("XDNA  region vs CPU   : max_abs=%.5f max_rel=%.5f  (XDNA work %s 0)\n", e_xd_region, r_xd_region, max_xd > 0 ? "!=" : "==");
    printf("Merged vs CPU         : max_abs=%.5f max_rel=%.5f\n", e_h, r_h);
    {
        double e_hv, r_hv; stats(hetero, vk_full, e_hv, r_hv);
        printf("Merged vs Vulkan-full: max_abs=%.5f max_rel=%.5f\n", e_hv, r_hv);
    }

    bool pass = (max_vk > 0) && (max_xd > 0) && (e_h < 1.0) && (r_h < 0.05) && (e_xd_region < 1.0) && (e_vk_region < 1e-3);
    printf("\n[HETERO] CORRECTNESS GATE: %s\n", pass ? "PASS" : "FAIL");
    printf("[HETERO] both backends executed real work: %s\n", (max_vk > 0 && max_xd > 0) ? "YES" : "NO");
    printf("[HETERO] XDNA real XRT submits: see [XDNA] counters above\n");

    ggml_backend_free(backend_vk);
    ggml_backend_free(backend_xdna);
    ggml_backend_free(backend_cpu);
    return pass ? 0 : 2;
}
