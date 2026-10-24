// Phase 60: isolated numerical check of the XDNA Q8_0 int8 GEMM path.
//
// Builds one MUL_MAT (Q8_0 weights x F32 activations) on the XDNA backend and
// on the CPU backend and compares the results. Used to separate "the device
// kernel is wrong" from "the host side of the XDNA path is wrong".
//
// build:
//   g++ -O2 -o /tmp/test_npu_gemm experiments/phase60/test_npu_gemm.cpp \
//       -Iggml/include -Iinclude -Lbuild/bin -lggml -lggml-base \
//       -Wl,-rpath,$PWD/build/bin
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float frand(int * s) {
    *s = *s * 1103515245 + 12345;
    return (float)((*s >> 16) & 0x7fff) / 32767.0f - 0.5f;
}

int main(int argc, char ** argv) {
    const int K = argc > 1 ? atoi(argv[1]) : 1024;
    const int N = argc > 2 ? atoi(argv[2]) : 3584;
    const int M = argc > 3 ? atoi(argv[3]) : 512;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t npu = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (strstr(ggml_backend_reg_name(reg), "XDNA") != nullptr) {
            npu = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
            break;
        }
    }
    if (!npu) { fprintf(stderr, "no XDNA backend\n"); return 1; }
    fprintf(stderr, "[TEST] backends: cpu=%s npu=%s K=%d N=%d M=%d\n",
            ggml_backend_name(cpu), ggml_backend_name(npu), K, N, M);

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead() + 4*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    (void) y;

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, npu);

    // Fill the weights with Q8_0 blocks directly.
    {
        std::vector<uint8_t> q8(ggml_nbytes(w));
        int s = 12345;
        const int nb = K / 32;
        for (int n = 0; n < N; n++) {
            for (int b = 0; b < nb; b++) {
                uint8_t * blk = &q8[((size_t) n * nb + b) * 34];
                float amax = 0.0f;
                float tmp[32];
                for (int k = 0; k < 32; k++) { tmp[k] = frand(&s); amax = std::max(amax, std::fabs(tmp[k])); }
                float d = amax / 127.0f;
                ggml_fp16_t h = ggml_fp32_to_fp16(d);
                memcpy(blk, &h, sizeof(h));
                for (int k = 0; k < 32; k++) {
                    int q = (int) llroundf(tmp[k] / d);
                    blk[2 + k] = (uint8_t) (int8_t) std::max(-127, std::min(127, q));
                }
            }
        }
        ggml_backend_tensor_set(w, q8.data(), 0, q8.size());
    }
    {
        std::vector<float> f((size_t) K * M);
        int s = 777;
        for (size_t i = 0; i < f.size(); i++) f[i] = frand(&s);
        ggml_backend_tensor_set(x, f.data(), 0, f.size() * sizeof(float));
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(gf, out);

    ggml_backend_graph_compute(npu, gf);
    std::vector<float> got((size_t) M * N);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

    // Reference on the CPU.
    ggml_context * ctx2 = ggml_init(ip);
    ggml_tensor * w2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_Q8_0, K, N);
    ggml_tensor * x2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, K, M);
    ggml_tensor * y2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, M, N);
    (void) y2;
    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx2, cpu);
    std::vector<uint8_t> q8(ggml_nbytes(w));
    ggml_backend_tensor_get(w, q8.data(), 0, q8.size());
    ggml_backend_tensor_set(w2, q8.data(), 0, q8.size());
    std::vector<float> f((size_t) K * M);
    ggml_backend_tensor_get(x, f.data(), 0, f.size() * sizeof(float));
    ggml_backend_tensor_set(x2, f.data(), 0, f.size() * sizeof(float));

    ggml_cgraph * gf2 = ggml_new_graph(ctx2);
    ggml_tensor * out2 = ggml_mul_mat(ctx2, w2, x2);
    ggml_build_forward_expand(gf2, out2);
    ggml_backend_graph_compute(cpu, gf2);
    std::vector<float> ref((size_t) M * N);
    ggml_backend_tensor_get(out2, ref.data(), 0, ref.size() * sizeof(float));

    double max_abs = 0.0, sum_abs = 0.0, max_ref = 0.0;
    size_t nbad = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = std::fabs((double) got[i] - (double) ref[i]);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        max_ref = std::max(max_ref, (double) std::fabs(ref[i]));
        if (d > 1e-2 * (1.0 + std::fabs((double) ref[i]))) nbad++;
    }
    printf("[TEST] max_abs_err=%.6g  mean_abs_err=%.6g  max|ref|=%.6g  bad=%zu/%zu  rel=%.4g\n",
           max_abs, sum_abs / ref.size(), max_ref, nbad, ref.size(),
           max_abs / (max_ref > 0 ? max_ref : 1.0));
    printf(nbad == 0 ? "[TEST] PASS\n" : "[TEST] FAIL\n");
    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf2);
    ggml_free(ctx);
    ggml_free(ctx2);
    ggml_backend_free(cpu);
    ggml_backend_free(npu);
    return nbad == 0 ? 0 : 1;
}
