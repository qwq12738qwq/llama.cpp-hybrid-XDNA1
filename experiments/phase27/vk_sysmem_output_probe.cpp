// Phase 27 probe: can ggml-vulkan compute write its output into host/imported
// system memory (VK_EXT_external_memory_host) instead of DEVICE_LOCAL VRAM?
//
// It builds a small mini-graph identical to the XDNA_WA Vulkan slice:
//   wt[K=1024, N=1792] Q8_0  x  bt[K=1024, M=512] F32 -> ct[N, M]
// and runs it three ways, measuring set/compute/get latency and comparing output:
//   (1) CPU reference
//   (2) Vulkan device buffer type (baseline, DEVICE_LOCAL VRAM)
//   (3) Vulkan host-output buffer type (system memory via host pointer import)
//
// Build:
//   g++ -O2 -std=c++17 vk_sysmem_output_probe.cpp -I../../../ggml/include \
//       -L../../../build-vk/bin -lggml-vulkan -lggml-cpu -lggml -lggml-base \
//       -Wl,-rpath,$PWD/../../../build-vk/bin -o vk_sysmem_output_probe

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>

static const int K = 1024;
static const int N = 1792;
static const int M = 512;

// Q8_0: per 32 weights -> fp16 scale (2 bytes) + 32 int8.
static std::vector<uint8_t> make_q8_0(int k, int n, uint64_t & seed) {
    int nb = k / 32;
    std::vector<uint8_t> w((size_t) n * nb * 34);
    for (int col = 0; col < n; col++) {
        for (int b = 0; b < nb; b++) {
            uint8_t * blk = &w[((size_t) col * nb + b) * 34];
            // fp16 1.0
            blk[0] = 0x00; blk[1] = 0x3C;
            for (int i = 0; i < 32; i++) {
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                int8_t v = (int8_t)((seed >> 40) % 255 - 127);
                blk[2 + i] = (uint8_t) v;
            }
        }
    }
    return w;
}

struct Result {
    double alloc_ms = 0, set_ms = 0, compute_ms = 0, get_ms = 0;
    std::vector<float> out;
};

static Result run_graph(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                        const std::vector<uint8_t> & w, const std::vector<float> & a) {
    Result r;

    ggml_init_params ip{};
    ip.mem_size = 16 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * wt = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    ggml_tensor * bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * ct = ggml_mul_mat(ctx, wt, bt);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ct);

    uint64_t t0 = ggml_time_us();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    uint64_t t1 = ggml_time_us();
    if (!buf) {
        fprintf(stderr, "alloc_ctx_tensors_from_buft failed (buf null)\n");
        ggml_free(ctx);
        return r;
    }

    ggml_backend_tensor_set(wt, w.data(), 0, w.size());
    ggml_backend_tensor_set(bt, a.data(), 0, a.size() * sizeof(float));
    uint64_t t2 = ggml_time_us();

    ggml_backend_graph_compute(backend, gf);
    uint64_t t3 = ggml_time_us();

    r.out.resize((size_t) N * M);
    ggml_backend_tensor_get(ct, r.out.data(), 0, r.out.size() * sizeof(float));
    uint64_t t4 = ggml_time_us();

    r.alloc_ms = (t1 - t0) / 1000.0;
    r.set_ms = (t2 - t1) / 1000.0;
    r.compute_ms = (t3 - t2) / 1000.0;
    r.get_ms = (t4 - t3) / 1000.0;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

int main() {
    uint64_t seed = 12345;
    std::vector<uint8_t> w = make_q8_0(K, N, seed);
    std::vector<float> a((size_t) K * M);
    for (size_t i = 0; i < a.size(); i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        a[i] = (float)((int)((seed >> 40) % 2000) - 1000) / 250.0f;
    }

    printf("device count = %d\n", ggml_backend_vk_get_device_count());
    char desc[256] = {0};
    ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
    printf("device 0 = %s\n", desc);
    printf("shape: wt[%d,%d] Q8_0 x bt[%d,%d] F32 -> ct[%d,%d]  (%.2f MB)\n",
           K, N, K, M, N, M, (double) N * M * 4 / 1e6);

    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (!vk) { fprintf(stderr, "Vulkan backend init failed\n"); return 1; }
    ggml_backend_t cpu = ggml_backend_cpu_init();

    ggml_backend_buffer_type_t bt_dev = ggml_backend_vk_buffer_type(0);
    ggml_backend_buffer_type_t bt_host = ggml_backend_vk_host_output_buffer_type(0);
    printf("device buft name = %s\n", ggml_backend_buft_name(bt_dev));
    printf("host-output buft name = %s\n", ggml_backend_buft_name(bt_host));

    // CPU reference
    Result rc = run_graph(cpu, ggml_backend_cpu_buffer_type(), w, a);
    printf("[CPU ] alloc=%.2f set=%.2f compute=%.2f get=%.2f ms\n",
           rc.alloc_ms, rc.set_ms, rc.compute_ms, rc.get_ms);

    // Vulkan baseline (device / VRAM)
    Result rv = run_graph(vk, bt_dev, w, a);
    printf("[VK-device] alloc=%.2f set=%.2f compute=%.2f get=%.2f ms\n",
           rv.alloc_ms, rv.set_ms, rv.compute_ms, rv.get_ms);

    // Vulkan host-output (system memory)
    Result rh = run_graph(vk, bt_host, w, a);
    printf("[VK-hostout] alloc=%.2f set=%.2f compute=%.2f get=%.2f ms\n",
           rh.alloc_ms, rh.set_ms, rh.compute_ms, rh.get_ms);

    // correctness
    size_t bad_cpu = 0, bad_dev = 0;
    float maxrel_cpu = 0, maxrel_dev = 0;
    for (size_t i = 0; i < rc.out.size(); i++) {
        float ref = rc.out[i];
        float v = rv.out[i];
        float h = rh.out[i];
        float dvc = fabsf(v - ref) / (fabsf(ref) + 1e-6f);
        float dhc = fabsf(h - ref) / (fabsf(ref) + 1e-6f);
        if (dvc > 1e-3f) { bad_cpu++; maxrel_cpu = std::max(maxrel_cpu, dvc); }
        if (dhc > 1e-3f) { bad_dev++; maxrel_dev = std::max(maxrel_dev, dhc); }
    }
    printf("[correctness] vk-device vs cpu: bad=%zu maxrel=%.3e\n", bad_cpu, maxrel_cpu);
    printf("[correctness] vk-hostout vs cpu: bad=%zu maxrel=%.3e\n", bad_dev, maxrel_dev);

    // host-output vs device bitwise-ish (same shader -> should match closely)
    size_t diff_dh = 0; float maxrel_dh = 0;
    for (size_t i = 0; i < rc.out.size(); i++) {
        float d = fabsf(rh.out[i] - rv.out[i]) / (fabsf(rv.out[i]) + 1e-6f);
        if (d > 1e-4f) { diff_dh++; maxrel_dh = std::max(maxrel_dh, d); }
    }
    printf("[correctness] vk-hostout vs vk-device: diff=%zu maxrel=%.3e\n", diff_dh, maxrel_dh);
    printf("sample hostout[0..3] = %.5f %.5f %.5f %.5f (cpu %.5f %.5f %.5f %.5f)\n",
           rh.out[0], rh.out[1], rh.out[2], rh.out[3], rc.out[0], rc.out[1], rc.out[2], rc.out[3]);

    ggml_backend_free(cpu);
    ggml_backend_free(vk);
    printf("DONE\n");
    return 0;
}
