// Phase 44B: Minimal Vulkan -> XDNA -> Vulkan scheduler repro.
// Prints backend ownership for every node/src immediately before split_graph.
//
// Graph:
//   Vulkan input (leaf, preallocated)
//     -> XDNA MUL_MAT (Q8_0 weight, Qwen FFN shape)
//     -> Vulkan consumer (ADD)
//
// Usage: phase44_hetero_sched [--M 512] [--K 1024] [--N 3584] [--iters 3]

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-xdna.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

static void die(const char* msg) {
    fprintf(stderr, "[FATAL] %s\n", msg);
    exit(1);
}

static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void quantize_q8(const std::vector<float>& w, int K, int N, std::vector<uint8_t>& out) {
    size_t nbytes = (size_t)(K * N / 32) * 34;
    out.resize(nbytes);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), out.data(), 0, N, K, nullptr);
}

static const char* buft_name(ggml_backend_buffer_t buf) {
    if (!buf) return "(null)";
    return buf->buft && buf->buft->iface.get_name ? buf->buft->iface.get_name(buf->buft) : "(?)";
}

static int backend_id_of(ggml_backend_sched_t sched, ggml_tensor* t, int n_backends) {
    // Use public API: get tensor backend then map back to index
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, t);
    if (!b) return -1;
    // We need the raw id: expose via debug print only.
    // Fall back: compare against known backends is not available here.
    // The instrumented split_graph will print ids.
    (void)n_backends;
    return b ? 0 : -1;
}

int main(int argc, char** argv) {
    int M = 512, K = 1024, N = 3584, iters = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--M") && i + 1 < argc) M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    }

    fprintf(stderr, "[P44] Minimal hetero test M=%d K=%d N=%d iters=%d\n", M, K, N, iters);

    const bool use_view = getenv("P44_USE_VIEW") != nullptr;
    ggml_backend_load_all();

    // Enumerate devices
    ggml_backend_dev_t vk_dev = nullptr, xdna_dev = nullptr, cpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const char* name = ggml_backend_dev_name(d);
        auto type = ggml_backend_dev_type(d);
        fprintf(stderr, "[P44] device[%zu]: %s type=%d\n", i, name, (int)type);
        // this APU reports Vulkan as IGPU; also match by name
        if ((type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
             strstr(name, "Vulkan")) && !vk_dev) {
            vk_dev = d;
        }
        if (strstr(name, "XDNA")) xdna_dev = d;
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU && !cpu_dev) cpu_dev = d;
    }
    if (!vk_dev) die("no Vulkan device");
    if (!xdna_dev) die("no XDNA device");
    if (!cpu_dev) die("no CPU device");

    ggml_backend_t vk = ggml_backend_dev_init(vk_dev, nullptr);
    ggml_backend_t xdna = ggml_backend_dev_init(xdna_dev, nullptr);
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!vk || !xdna || !cpu) die("backend init failed");

    // Explicit test of supports_buft / supports_op
    {
        ggml_backend_buffer_type_t vk_buft = ggml_backend_get_default_buffer_type(vk);
        ggml_backend_buffer_type_t xdna_buft = ggml_backend_get_default_buffer_type(xdna);
        fprintf(stderr, "[P44] vk_buft=%s xdna_buft=%s\n",
                vk_buft && vk_buft->iface.get_name ? vk_buft->iface.get_name(vk_buft) : "?",
                xdna_buft && xdna_buft->iface.get_name ? xdna_buft->iface.get_name(xdna_buft) : "?");
        fprintf(stderr, "[P44] xdna supports_xdna_buft=%d supports_vk_buft=%d\n",
                (int)ggml_backend_supports_buft(xdna, xdna_buft),
                (int)ggml_backend_supports_buft(xdna, vk_buft));
        fprintf(stderr, "[P44] vk supports_vk_buft=%d supports_xdna_buft=%d\n",
                (int)ggml_backend_supports_buft(vk, vk_buft),
                (int)ggml_backend_supports_buft(vk, xdna_buft));
    }

    // Build graph: Vulkan input -> XDNA mul_mat -> Vulkan add
    struct ggml_init_params params = { 64 * 1024 * 1024, nullptr, true }; // no_alloc
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) die("ggml_init");

    // Inputs as tensors (not preallocated yet)
    struct ggml_tensor* act = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_input(act);
    ggml_format_name(act, "act");

    struct ggml_tensor* act_input = act;
    if (use_view) {
        act_input = ggml_view_2d(ctx, act, K, M, act->nb[1], 0);
        ggml_format_name(act_input, "act_view");
    }

    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    ggml_set_input(w);
    ggml_format_name(w, "w");

    struct ggml_tensor* mm = ggml_mul_mat(ctx, w, act_input);
    ggml_format_name(mm, "mm");

    struct ggml_tensor* bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    ggml_set_input(bias);
    ggml_format_name(bias, "bias");

    struct ggml_tensor* out = ggml_add(ctx, mm, bias);
    ggml_set_output(out);
    ggml_format_name(out, "out");

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    fprintf(stderr, "[P44] graph n_nodes=%d n_leafs=%d\n", gf->n_nodes, gf->n_leafs);
    for (int i = 0; i < gf->n_nodes; i++) {
        struct ggml_tensor* n = gf->nodes[i];
        fprintf(stderr, "[P44] node[%d] name=%s op=%s\n", i, n->name, ggml_op_name(n->op));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (!n->src[j]) continue;
            fprintf(stderr, "  src[%d]=%s op=%s view_src=%p buf=%s\n",
                    j, n->src[j]->name, ggml_op_name(n->src[j]->op),
                    (void*)n->src[j]->view_src, buft_name(n->src[j]->buffer));
        }
    }
    for (int i = 0; i < gf->n_leafs; i++) {
        struct ggml_tensor* l = gf->leafs[i];
        fprintf(stderr, "[P44] leaf[%d] name=%s op=%s view_src=%p buf=%s\n",
                i, l->name, ggml_op_name(l->op), (void*)l->view_src, buft_name(l->buffer));
    }

    // Scheduler: Vulkan (0) > XDNA (1) > CPU (2)
    ggml_backend_t backends[3] = { vk, xdna, cpu };
    ggml_backend_buffer_type_t bufts[3] = {
        ggml_backend_get_default_buffer_type(vk),
        ggml_backend_get_default_buffer_type(xdna),
        ggml_backend_get_default_buffer_type(cpu),
    };
    const int n_backends = 3;
    fprintf(stderr, "[P44] sched backends:\n");
    for (int i = 0; i < n_backends; i++) {
        fprintf(stderr, "  backend[%d]=%s buft=%s\n", i,
                ggml_backend_name(backends[i]),
                bufts[i] && bufts[i]->iface.get_name ? bufts[i]->iface.get_name(bufts[i]) : "?");
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n_backends, 4096, false, true);
    if (!sched) die("sched_new");

    // Force true Vulkan -> XDNA -> Vulkan: leaves on Vulkan, MUL_MAT on XDNA, ADD on Vulkan.
    const char* force = getenv("P44_FORCE_XDNA");
    const bool xdna_weight = getenv("P44_XDNA_WEIGHT") != nullptr;
    if (force && *force) {
        for (int i = 0; i < gf->n_leafs; i++) {
            ggml_backend_t leaf_backend = xdna_weight && strcmp(gf->leafs[i]->name, "w") == 0 ? xdna : vk;
            ggml_backend_sched_set_tensor_backend(sched, gf->leafs[i], leaf_backend);
            fprintf(stderr, "[P44] forced %s for leaf %s\n", ggml_backend_name(leaf_backend), gf->leafs[i]->name);
        }
        for (int i = 0; i < gf->n_nodes; i++) {
            if (gf->nodes[i]->op == GGML_OP_MUL_MAT) {
                ggml_backend_sched_set_tensor_backend(sched, gf->nodes[i], xdna);
                fprintf(stderr, "[P44] forced XDNA for node %s\n", gf->nodes[i]->name);
            } else if (gf->nodes[i]->op == GGML_OP_ADD) {
                ggml_backend_sched_set_tensor_backend(sched, gf->nodes[i], vk);
                fprintf(stderr, "[P44] forced Vulkan for node %s\n", gf->nodes[i]->name);
            }
        }
    }

    // Instrumented: dump assignments by calling split via reserve
    fprintf(stderr, "[P44] allocating graph...\n");
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        die("alloc_graph failed (likely SCHED-ASSERT or allocation error)");
    }
    fprintf(stderr, "[P44] n_splits=%d\n", ggml_backend_sched_get_n_splits(sched));

    // Fill inputs
    std::vector<float> act_data((size_t)M * K);
    std::vector<float> bias_data((size_t)M * N);
    std::vector<float> w_data((size_t)N * K);
    for (int i = 0; i < M * K; i++) act_data[i] = 0.01f * ((i * 7) % 100) - 0.5f;
    for (int i = 0; i < M * N; i++) bias_data[i] = 0.001f * ((i * 3) % 50);
    for (int i = 0; i < N * K; i++) w_data[i] = 0.02f * ((i * 11) % 80) - 0.8f;

    std::vector<uint8_t> w_q8;
    quantize_q8(w_data, K, N, w_q8);

    auto set_in = [&](ggml_tensor* t, const void* data, size_t nbytes) {
        if (!t->buffer) { fprintf(stderr, "[P44] no buffer for %s\n", t->name); die("tensor buffer null"); }
        ggml_backend_tensor_set(t, data, 0, nbytes);
    };

    set_in(act, act_data.data(), act_data.size() * sizeof(float));
    set_in(bias, bias_data.data(), bias_data.size() * sizeof(float));
    set_in(w, w_q8.data(), w_q8.size());

    // CPU reference: out[m,n] = sum_k act[m,k]*deq(w[k,n]) + bias[m,n]
    // Q8_0: per K-block of 32: half d, int8 qs[32]; value = d * qs
    std::vector<float> ref((size_t)M * N, 0.0f);
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc = bias_data[(size_t)m * N + n];
            for (int kb = 0; kb < K / 32; kb++) {
                const uint8_t* p = w_q8.data() + ((size_t)n * (K / 32) + kb) * 34;
                uint16_t sc;
                memcpy(&sc, p, 2);
                float d = ggml_fp16_to_fp32(sc);
                const int8_t* qs = (const int8_t*)(p + 2);
                const float* arow = act_data.data() + (size_t)m * K + kb * 32;
                for (int i = 0; i < 32; i++) {
                    acc += arow[i] * (d * (float)qs[i]);
                }
            }
            ref[(size_t)m * N + n] = acc;
        }
    }

    // Run iterations
    for (int it = 0; it < iters; it++) {
        set_in(act, act_data.data(), act_data.size() * sizeof(float));
        set_in(bias, bias_data.data(), bias_data.size() * sizeof(float));
        set_in(w, w_q8.data(), w_q8.size());
        double t0 = now_ms();
        enum ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
        double t1 = now_ms();
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[P44] iter %d compute FAILED status=%d\n", it, (int)st);
            break;
        }
        fprintf(stderr, "[P44] iter %d compute OK %.2f ms n_splits=%d\n",
                it, t1 - t0, ggml_backend_sched_get_n_splits(sched));

        // Read mm (XDNA output) and out (after Vulkan ADD) separately
        std::vector<float> got_mm((size_t)M * N);
        int mm_nonfinite = -1;
        if (mm->buffer) {
            ggml_backend_tensor_get(mm, got_mm.data(), 0, got_mm.size() * sizeof(float));
            mm_nonfinite = 0;
            for (size_t i = 0; i < got_mm.size(); i++) {
                if (!std::isfinite(got_mm[i])) mm_nonfinite++;
            }
        }
        std::vector<float> got((size_t)M * N);
        if (!out->buffer) { fprintf(stderr, "[P44] out has no buffer\n"); die("out buffer"); }
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

        int nonfinite = 0;
        int mismatch = 0;
        double max_err = 0.0, se = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            if (!std::isfinite(got[i])) { nonfinite++; continue; }
            double e = (double)got[i] - (double)ref[i];
            double ae = std::fabs(e);
            if (ae > max_err) max_err = ae;
            se += e * e;
            if (ae > 1e-2f) mismatch++;
        }
        double rmse = std::sqrt(se / (double)got.size());
        fprintf(stderr, "[P44] iter %d mm_nonfinite=%d nonfinite=%d max_err=%.6g rmse=%.6g mismatches=%d first=%g mid=%g\n",
                it, mm_nonfinite, nonfinite, max_err, rmse, mismatch, got[0], got[M * N / 2]);
    }

    fprintf(stderr, "[P44] DONE\n");
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(vk);
    ggml_backend_free(xdna);
    ggml_backend_free(cpu);
    return 0;
}
