// attention_partition_bench.cpp -- Phase 31 attention partition microbenchmark.
//
// Measures the KV-range partition cost structure of the real Vulkan flash
// attention op (GGML_OP_FLASH_ATTN_EXT) against the Qwen3.5-0.8B attention
// geometry (HSK=HSV=256, n_head=8, n_head_kv=2, gqa=4), and checks that the
// Vulkan native split_k partition + online-softmax merge reproduces the unsplit
// result.
//
// Baselines (goal section 13):
//   B0 = Vulkan full attention (KV=C, split_k=1)
//   B1 = Vulkan partition A + Vulkan partition B, serial (two half-KV dispatches)
//   B3 = Vulkan full attention (repeat of B0)
// plus correctness: split_k=2 (native KV-range partition + merge) vs split_k=1,
// and both against a CPU FP32 reference.
//
// Build: see build.sh.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint32_t rng_state = 0x12345678u;
static float rnd(float lo, float hi) {
    rng_state = rng_state * 1664525u + 1013904223u;
    float u = (float)((rng_state >> 8) & 0xFFFFFF) / (float)0x1000000;
    return lo + u * (hi - lo);
}

// Run one FLASH_ATTN_EXT graph on `backend`. Returns median compute+sync ms over
// `reps` runs. If `out` is non-null it is filled with the op result.
static double run_fa(ggml_backend_t backend,
                     int hsk, int hsv, int nh, int nkv, int N, int KV,
                     ggml_type tk, ggml_type tv,
                     std::vector<float> * out, int reps) {
    rng_state = 0x12345678u; // identical inputs across backends / repeats
    ggml_init_params ip = {
        /*mem_size=*/ ggml_tensor_overhead() * 64 + ggml_graph_overhead() + (16u << 20),
        /*mem_buffer=*/ nullptr,
        /*no_alloc=*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, N, nh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, tk, hsk, KV, nkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, tv, hsv, KV, nkv, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KV, N, 1, 1);

    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) hsk), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "alloc failed (KV=%d)\n", KV);
        ggml_free(ctx);
        return -1.0;
    }

    std::vector<float> qh((size_t) hsk * N * nh);
    for (auto & x : qh) x = rnd(-1.0f, 1.0f);
    ggml_backend_tensor_set(q, qh.data(), 0, qh.size() * sizeof(float));

    std::vector<ggml_fp16_t> kh((size_t) hsk * KV * nkv);
    for (auto & x : kh) x = ggml_fp32_to_fp16(rnd(-1.0f, 1.0f));
    ggml_backend_tensor_set(k, kh.data(), 0, kh.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> vh((size_t) hsv * KV * nkv);
    for (auto & x : vh) x = ggml_fp32_to_fp16(rnd(-1.0f, 1.0f));
    ggml_backend_tensor_set(v, vh.data(), 0, vh.size() * sizeof(ggml_fp16_t));

    // Additive attention mask: small per-(kv,query) bias so softmax weights differ.
    std::vector<ggml_fp16_t> mh((size_t) KV * N);
    for (size_t i = 0; i < mh.size(); i++) mh[i] = ggml_fp32_to_fp16(rnd(-2.0f, 0.0f));
    ggml_backend_tensor_set(m, mh.data(), 0, mh.size() * sizeof(ggml_fp16_t));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);

    for (int i = 0; i < 2; i++) {
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }

    std::vector<double> ts;
    for (int i = 0; i < reps; i++) {
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        ts.push_back(now_ms() - t0);
    }
    std::sort(ts.begin(), ts.end());
    double med = ts[ts.size() / 2];

    if (out) {
        out->resize((size_t) ggml_nelements(o));
        ggml_backend_tensor_get(o, out->data(), 0, out->size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return med;
}

static void compare(const char * tag, const std::vector<float> & a, const std::vector<float> & b) {
    double max_abs = 0.0, max_rel = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        double d = fabs((double) a[i] - (double) b[i]);
        max_abs = std::max(max_abs, d);
        double denom = std::max(fabs((double) b[i]), 1e-6);
        max_rel = std::max(max_rel, d / denom);
        dot += (double) a[i] * (double) b[i];
        na += (double) a[i] * (double) a[i];
        nb += (double) b[i] * (double) b[i];
    }
    double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    printf("CORR,%s,max_abs=%.6e,max_rel=%.6e,cosine=%.9f\n", tag, max_abs, max_rel, cos);
}

int main(int argc, char ** argv) {
    const int hsk = 256, hsv = 256, nh = 8, nkv = 2, N = 512;
    const ggml_type tk = GGML_TYPE_F16, tv = GGML_TYPE_F16;
    const int reps = (argc > 1) ? atoi(argv[1]) : 20;

    std::vector<int> kvs = {512, 4096, 16384, 32768};
    if (argc > 2) {
        kvs.clear();
        char * s = strdup(argv[2]);
        for (char * p = strtok(s, ","); p; p = strtok(nullptr, ",")) kvs.push_back(atoi(p));
        free(s);
    }

    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (!vk) { fprintf(stderr, "no vulkan backend\n"); return 1; }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    printf("hsk=%d hsv=%d n_head=%d n_head_kv=%d N=%d reps=%d\n", hsk, hsv, nh, nkv, N, reps);

    for (int KV : kvs) {
        // B0 / B3: full attention, split_k=1
        setenv("GGML_VK_FORCE_SPLIT_K", "1", 1);
        std::vector<float> o_full;
        double t_full = run_fa(vk, hsk, hsv, nh, nkv, N, KV, tk, tv, &o_full, reps);
        printf("TIME,B0,vulkan,full,KV=%d,split_k=1,median_ms=%.4f\n", KV, t_full);
        double t_full2 = run_fa(vk, hsk, hsv, nh, nkv, N, KV, tk, tv, nullptr, reps);
        printf("TIME,B3,vulkan,full,KV=%d,split_k=1,median_ms=%.4f\n", KV, t_full2);

        // B1: partition A (first half) + partition B (second half) serial.
        // Represented as two independent half-KV dispatches (each self-normalizing),
        // measured as one serial pair. This gives T_vk_A + T_vk_B.
        int half = KV / 2;
        double t_a = run_fa(vk, hsk, hsv, nh, nkv, N, half, tk, tv, nullptr, reps);
        double t_b = run_fa(vk, hsk, hsv, nh, nkv, N, half, tk, tv, nullptr, reps);
        printf("TIME,B1,vulkan,split,KV=%d,half=%d,T_A_ms=%.4f,T_B_ms=%.4f,T_serial_ms=%.4f\n",
               KV, half, t_a, t_b, t_a + t_b);

        // Correctness: native Vulkan split_k=2 KV-range partition + merge vs split_k=1.
        setenv("GGML_VK_FORCE_SPLIT_K", "2", 1);
        std::vector<float> o_split;
        double t_split = run_fa(vk, hsk, hsv, nh, nkv, N, KV, tk, tv, &o_split, reps);
        printf("TIME,splitk2,vulkan,full,KV=%d,split_k=2,median_ms=%.4f\n", KV, t_split);
        compare("vk_splitk2_vs_full", o_split, o_full);

        // CPU FP32 reference (only for the smaller KV to keep it quick).
        if (KV <= 4096) {
            std::vector<float> o_cpu;
            run_fa(cpu, hsk, hsv, nh, nkv, N, KV, tk, tv, &o_cpu, 1);
            compare("vk_full_vs_cpu", o_full, o_cpu);
        }
        fflush(stdout);
    }

    ggml_backend_free(cpu);
    ggml_backend_free(vk);
    return 0;
}
