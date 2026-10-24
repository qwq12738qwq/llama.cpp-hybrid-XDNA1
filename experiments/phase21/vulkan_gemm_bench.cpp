// vulkan_gemm_bench.cpp -- Phase 21: ggml-vulkan Q8_0 GEMM operator baseline.
//
// Measures ggml_mul_mat(W_Q8_0[N,K], X_F32[M,K]) on the Vulkan (780M) backend,
// the same quantized GEMM Qwen3.5-0.8B prefill runs. Reports compute GTOPS and
// ms so it can be compared against the XDNA whole-array int8 mmul roofline.
//
// Build (from repo root, after build-vk exists):
//   g++ -O2 -std=c++17 -I ggml/include experiments/phase21/vulkan_gemm_bench.cpp \
//       -L build-vk -lggml -lggml-base -lggml-vulkan -Wl,-rpath,$PWD/build-vk \
//       -o experiments/phase21/vulkan_gemm_bench
//
// Run:
//   ./vulkan_gemm_bench <M> <K> <N> [iters]
//
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

int main(int argc, char ** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int K = (argc > 2) ? atoi(argv[2]) : 4096;
    int N = (argc > 3) ? atoi(argv[3]) : 4096;
    int iters = (argc > 4) ? atoi(argv[4]) : 10;

    ggml_backend_load("/workspace/llama.cpp/build-vk/bin/libggml-vulkan.so");
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (!reg) { fprintf(stderr, "no Vulkan reg\n"); return 1; }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "no GPU backend\n"); return 1; }
    fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));

    struct ggml_init_params ip = { 1024ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);   // ne = (K, M)
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);  // ne = (K, N)
    ggml_tensor * c = ggml_mul_mat(ctx, b, a);                        // ne = (N, M)

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A((size_t)M * K);
    for (auto & v : A) v = dist(rng);
    ggml_backend_tensor_set(a, A.data(), 0, ggml_nbytes(a));

    std::vector<float> Bf((size_t)K * N);
    for (auto & v : Bf) v = dist(rng);
    std::vector<uint8_t> Bq(ggml_nbytes(b));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, Bf.data(), Bq.data(), 0, (int64_t)N, K, nullptr);
    ggml_backend_tensor_set(b, Bq.data(), 0, Bq.size());

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    for (int i = 0; i < 2; i++) ggml_backend_graph_compute(backend, gf);

    std::vector<double> times;
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    double ms = times[times.size() / 2];
    double gops = 2.0 * (double)M * K * N / (ms * 1e-3) / 1e9;
    printf("VKGEMM M=%d K=%d N=%d ms=%.4f gops=%.2f type=Q8_0\n", M, K, N, ms, gops);
    printf("CSV vk_q8_0,%d,%d,%d,%.4f,%.2f\n", M, K, N, ms, gops);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
