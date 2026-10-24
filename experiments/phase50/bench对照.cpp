// Phase 50 Stage 3: CPU vs Vulkan same-shape Q8_0 GEMM对照.
// Same op as FFN prefill: C[M,N] = A[M,K](F32 or Q8_0) x W[K,N](Q8_0).
// Reports median ms + GFLOP/s over iters. Build against current build/bin.
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int K = (argc > 2) ? atoi(argv[2]) : 1024;
    int N = (argc > 3) ? atoi(argv[3]) : 3584;
    int iters = (argc > 4) ? atoi(argv[4]) : 9;
    const char * devsel = (argc > 5) ? argv[5] : "Vulkan";

    ggml_backend_t backend = nullptr;
    ggml_backend_load_all();
    if (!strcmp(devsel, "CPU")) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } else {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, devsel);
        if (!backend) backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend) { fprintf(stderr, "no backend %s\n", devsel); return 1; }
    fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));

    struct ggml_init_params ip;
    memset(&ip, 0, sizeof(ip));
    ip.mem_size = 1024ull * 1024 * 1024;
    ip.no_alloc = true;
    struct ggml_context * ctx = ggml_init(ip);

    // FFN GEMM: activation row-major [K,M]? ggml mul_mat(b,a): b=[K,N] Q8_0, a=[K,M] F32, c=[N,M]
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor * c = ggml_mul_mat(ctx, b, a);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    srand(1234);
    std::vector<float> A((size_t)M * K);
    for (auto & v : A) v = (float)(rand() % 2000 - 1000) / 1000.0f;
    ggml_backend_tensor_set(a, A.data(), 0, ggml_nbytes(a));
    std::vector<float> Bf((size_t)K * N);
    for (auto & v : Bf) v = (float)(rand() % 2000 - 1000) / 1000.0f;
    std::vector<uint8_t> Bq(ggml_nbytes(b));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, Bf.data(), Bq.data(), 0, N, K, nullptr);
    ggml_backend_tensor_set(b, Bq.data(), 0, Bq.size());

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    std::vector<double> times;
    for (int i = 0; i < iters; i++) {
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        double t1 = now_ms();
        times.push_back(t1 - t0);
    }
    std::sort(times.begin(), times.end());
    // checksum to defeat DCE + sanity
    std::vector<float> out((size_t)M * N);
    ggml_backend_tensor_get(c, out.data(), 0, out.size() * sizeof(float));
    double sum = 0;
    for (size_t i = 0; i < out.size(); i += 997) sum += out[i];
    double med = times[times.size() / 2];
    double mn = times.front();
    double gflops = 2.0 * (double)M * K * N / (med * 1e-3) / 1e9;
    printf("CMP dev=%s M=%d K=%d N=%d median_ms=%.4f min_ms=%.4f gflops=%.2f cksum=%.6f\n",
           ggml_backend_name(backend), M, K, N, med, mn, gflops, sum);
    printf("all_ms=");
    for (double t : times) printf("%.3f ", t);
    printf("\n");
    return 0;
}
