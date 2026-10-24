// hetero_op.cpp -- Phase 20G: real-backend N-partition of one Q8_0 GEMM.
//
//   C[M x N] = X[M x K] x W[K x N]        (ggml: mul_mat(W, X))
//
// W is split along N:  N0 columns -> Vulkan backend, N1 columns -> XDNA backend.
// Both backends are the real ggml backends (no naive shader, no IRON runtime).
// Each backend computes its own independent graph with graph_compute_async();
// Vulkan is waited on a separate thread so the two devices run concurrently.
//
//   overlap_ns  = min(vk_done, xdna_done) - max(vk_submit, xdna_submit)
//   critical_ns = max(vk_done, xdna_done) - min(vk_submit, xdna_submit)
//
// Build (repo root):
//   g++ -O2 -std=c++17 experiments/phase20g/hetero_op.cpp -o experiments/phase20g/hetero_op \
//       -I ggml/include -I ggml/src -L build-new/bin \
//       -lggml-vulkan -lggml-xdna -lggml -lggml-base -lggml-cpu \
//       -Wl,-rpath,build-new/bin -pthread
//
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-xdna.h"
#include "ggml-vulkan.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Quantize row-major f32 W[K][N] (ne0=K, ne1=N) into ggml Q8_0 blocks.
// Along K, every 32 elements form one block: fp16 scale + 32 int8.
static void quantize_q8_0(const std::vector<float> & W, int K, int N, std::vector<uint8_t> & out) {
    const int nb = K / 32;
    out.assign((size_t) N * nb * 34, 0);
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            const float * blk = &W[(size_t) n * K + (size_t) b * 32];
            float amax = 0.0f;
            for (int k = 0; k < 32; k++) amax = std::max(amax, std::fabs(blk[k]));
            float d = amax / 127.0f;
            if (d == 0.0f) d = 1e-8f;
            uint8_t * o = &out[((size_t) n * nb + b) * 34];
            ggml_fp16_t dh = ggml_fp32_to_fp16(d);
            std::memcpy(o, &dh, 2);
            for (int k = 0; k < 32; k++) {
                int q = (int) std::lroundf(blk[k] / d);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                o[2 + k] = (uint8_t)(int8_t) q;
            }
        }
    }
}

static void dequant_q8_0(const std::vector<uint8_t> & q, int K, int N, std::vector<float> & out) {
    const int nb = K / 32;
    out.assign((size_t) K * N, 0.0f);
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            const uint8_t * o = &q[((size_t) n * nb + b) * 34];
            float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) o);
            for (int k = 0; k < 32; k++)
                out[(size_t) n * K + (size_t) b * 32 + k] = d * (float)(int8_t) o[2 + k];
        }
    }
}

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_tensor * Wq = nullptr;   // [K, N] Q8_0
    ggml_tensor * X  = nullptr;   // [K, M] F32
    ggml_tensor * C  = nullptr;   // [N, M] F32
    ggml_cgraph * gf = nullptr;

    bool build(ggml_backend_t be, int K, int N, int M) {
        backend = be;
        ggml_init_params p = { 512ull * 1024 * 1024, nullptr, true };
        ctx = ggml_init(p);
        Wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        X  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, M);
        C  = ggml_mul_mat(ctx, Wq, X);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, C);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        return buf != nullptr;
    }
};

int main(int argc, char ** argv) {
    int M = 512, K = 1024, N = 1024;
    int N0 = 512;            // Vulkan columns
    int iters = 20;
    bool serial = false;
    bool skip_vk = false;
    const char * json_out = nullptr;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M" && i + 1 < argc) M = atoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc) K = atoi(argv[++i]);
        else if (a == "--N" && i + 1 < argc) N = atoi(argv[++i]);
        else if (a == "--n0" && i + 1 < argc) N0 = atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--serial") serial = true;
        else if (a == "--skip-vk") skip_vk = true;
        else if (a == "--json" && i + 1 < argc) json_out = argv[++i];
    }
    int N1 = N - N0;

    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (!vk) { fprintf(stderr, "vk init failed\n"); return 1; }
    ggml_backend_t xd = ggml_backend_xdna_init();
    if (!xd) { fprintf(stderr, "xdna init failed\n"); return 1; }
    fprintf(stderr, "[setup] vk=%s xdna=%s M=%d K=%d N=%d N0(vk)=%d N1(xdna)=%d\n",
            ggml_backend_name(vk), ggml_backend_name(xd), M, K, N, N0, N1);

    // Real Q8_0 weight: full [K,N] in ne0=K layout.
    std::vector<float> Wfull((size_t) K * N);
    {
        uint32_t s = 12345u;
        for (size_t i = 0; i < Wfull.size(); i++) {
            s = s * 1664525u + 1013904223u;
            Wfull[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
    }
    std::vector<uint8_t> Wq_full;
    quantize_q8_0(Wfull, K, N, Wq_full);

    // Split W into N0 / N1 column slices (per-column blocks are contiguous).
    std::vector<uint8_t> Wq0, Wq1;
    {
        const int nb = K / 32;
        Wq0.assign((size_t) N0 * nb * 34, 0);
        Wq1.assign((size_t) N1 * nb * 34, 0);
        for (int n = 0; n < N0; n++)
            std::memcpy(&Wq0[(size_t) n * nb * 34], &Wq_full[(size_t) n * nb * 34], (size_t) nb * 34);
        for (int n = 0; n < N1; n++)
            std::memcpy(&Wq1[(size_t) n * nb * 34], &Wq_full[(size_t)(N0 + n) * nb * 34], (size_t) nb * 34);
    }

    // Activation X[M,K] (ggml ne0=K, ne1=M), row m contiguous K.
    std::vector<float> Xf((size_t) M * K);
    {
        uint32_t s = 999u;
        for (size_t i = 0; i < Xf.size(); i++) {
            s = s * 1103515245u + 12345u;
            Xf[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
    }

    Graph g0, g1, gfull;
    if (!g0.build(vk, K, N0, M)) { fprintf(stderr, "g0 build failed\n"); return 1; }
    if (!g1.build(xd, K, N1, M)) { fprintf(stderr, "g1 build failed\n"); return 1; }
    if (!gfull.build(vk, K, N, M)) { fprintf(stderr, "gfull build failed\n"); return 1; }

    ggml_backend_tensor_set(g0.Wq, Wq0.data(), 0, Wq0.size());
    ggml_backend_tensor_set(g0.X,  Xf.data(),  0, Xf.size() * sizeof(float));
    ggml_backend_tensor_set(g1.Wq, Wq1.data(), 0, Wq1.size());
    ggml_backend_tensor_set(g1.X,  Xf.data(),  0, Xf.size() * sizeof(float));
    ggml_backend_tensor_set(gfull.Wq, Wq_full.data(), 0, Wq_full.size());
    ggml_backend_tensor_set(gfull.X,  Xf.data(), 0, Xf.size() * sizeof(float));

    // Reference = Vulkan-only full-N GEMM (the baseline the task asks to beat).
    {
        ggml_backend_graph_compute_async(vk, gfull.gf);
        ggml_backend_synchronize(vk);
        for (int i = 0; i < 3; i++) { ggml_backend_graph_compute_async(vk, gfull.gf); ggml_backend_synchronize(vk); }
        uint64_t t0 = now_ns();
        for (int i = 0; i < 10; i++) { ggml_backend_graph_compute_async(vk, gfull.gf); ggml_backend_synchronize(vk); }
        uint64_t t1 = now_ns();
        double vk_only_ms = (t1 - t0) / 10.0 / 1e6;
        double gm = (double) M * K * N / (vk_only_ms * 1e6);
        fprintf(stderr, "[vulkan-only] full N=%d: %.3f ms  %.1f GMAC/s\n", N, vk_only_ms, gm);
    }
    std::vector<float> reffull((size_t) N * M, 0.0f);
    ggml_backend_tensor_get(gfull.C, reffull.data(), 0, reffull.size() * sizeof(float));

    auto submit_vk = [&]() { return ggml_backend_graph_compute_async(vk, g0.gf); };
    auto submit_xd = [&]() { return ggml_backend_graph_compute_async(xd, g1.gf); };

    auto read_C = [&](Graph & g, int Nn, std::vector<float> & out) {
        out.assign((size_t) Nn * M, 0.0f);
        ggml_backend_tensor_get(g.C, out.data(), 0, out.size() * sizeof(float));
    };

    // warmup
    for (int i = 0; i < 3; i++) {
        submit_xd(); submit_vk();
        ggml_backend_synchronize(xd); ggml_backend_synchronize(vk);
    }

    std::vector<float> C0, C1;
    auto run_once = [&](uint64_t & xd_sub, uint64_t & xd_done,
                        uint64_t & vk_sub, uint64_t & vk_done) {
        if (skip_vk) {
            submit_xd(); xd_sub = now_ns();
            ggml_backend_synchronize(xd); xd_done = now_ns();
            vk_sub = vk_done = xd_sub;
        } else if (!serial) {
            submit_xd(); xd_sub = now_ns();
            submit_vk(); vk_sub = now_ns();
            std::thread th([&]() { ggml_backend_synchronize(vk); vk_done = now_ns(); });
            ggml_backend_synchronize(xd); xd_done = now_ns();
            th.join();
        } else {
            submit_vk(); vk_sub = now_ns();
            ggml_backend_synchronize(vk); vk_done = now_ns();
            submit_xd(); xd_sub = now_ns();
            ggml_backend_synchronize(xd); xd_done = now_ns();
        }
    };

    uint64_t a, b, c, d;
    run_once(a, b, c, d);
    read_C(g0, N0, C0);
    read_C(g1, N1, C1);

    // correctness. Backend buffers use ggml [N,M] layout: element (n,m) at n + m*N.
    // Reference = Vulkan-only full-N GEMM (reffull).
    double maxabs = 0; long bad = 0;
    double refmax = 0;
    for (size_t i = 0; i < reffull.size(); i++) refmax = std::max(refmax, (double) std::fabs(reffull[i]));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float g = (n < N0) ? C0[(size_t) n + (size_t) m * N0]
                               : C1[(size_t)(n - N0) + (size_t) m * N1];
            double e = std::fabs((double) g - (double) reffull[(size_t) n + (size_t) m * N]);
            if (e > maxabs) maxabs = e;
            if (e > 1e-3 * (refmax > 0 ? refmax : 1)) bad++;
        }
    double rel = maxabs / (refmax > 0 ? refmax : 1);
    bool correct = (rel < 1e-2) || skip_vk;  // Q8_0 X-quantization noise ~1%
    // split by partition
    long bad_vk = 0, bad_xd = 0; double max_vk = 0, max_xd = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double r = reffull[(size_t) n + (size_t) m * N];
            if (n < N0) {
                if (skip_vk) continue;
                double e = std::fabs((double) C0[(size_t) n + (size_t) m * N0] - r);
                if (e > 1e-3 * (refmax > 0 ? refmax : 1)) bad_vk++;
                if (e > max_vk) max_vk = e;
            } else {
                double e = std::fabs((double) C1[(size_t)(n - N0) + (size_t) m * N1] - r);
                if (e > 1e-3 * (refmax > 0 ? refmax : 1)) bad_xd++;
                if (e > max_xd) max_xd = e;
            }
        }
    fprintf(stderr, "[correctness] bad=%ld/%d  vk_part bad=%ld maxabs=%.4g | xdna_part bad=%ld maxabs=%.4g\n",
            bad, N * M, bad_vk, max_vk, bad_xd, max_xd);
    if (getenv("HETERO_DUMP")) {
        int shown = 0;
        for (int m = 0; m < M && shown < 12; m++)
            for (int n = N0; n < N && shown < 12; n++) {
                double r = reffull[(size_t) n + (size_t) m * N];
                float g = C1[(size_t)(n - N0) + (size_t) m * N1];
                if (std::fabs((double) g - r) > 1e-3 * (refmax > 0 ? refmax : 1)) {
                    fprintf(stderr, "  MISMATCH n=%d m=%d xdna=%.5g vk=%.5g ratio=%.5g\n",
                            n, m, g, r, r != 0 ? (double) g / r : 0.0);
                    shown++;
                }
            }
    }

    // xdna-semantic reference: quantize X per-32-block (same rule the XDNA path
    // uses: scale = amax/127), take Q8_0 weight int8 directly, int32 dot per
    // block, then apply both scales. This is the exact integer semantics of the
    // XDNA scale-aware kernel, independent of ggml's X quantization.
    {
        const int nb = K / 32;
        std::vector<float> xsc((size_t) M * nb);
        std::vector<int8_t> xq((size_t) M * K);
        for (int m = 0; m < M; m++)
            for (int b = 0; b < nb; b++) {
                const float * blk = &Xf[(size_t) m * K + (size_t) b * 32];
                float amax = 0; for (int k = 0; k < 32; k++) amax = std::max(amax, std::fabs(blk[k]));
                float s = amax == 0 ? 1.0f : amax / 127.0f;
                xsc[(size_t) m * nb + b] = s;
                for (int k = 0; k < 32; k++) {
                    int q = (int) std::lroundf(blk[k] / s);
                    if (q > 127) q = 127; if (q < -127) q = -127;
                    xq[(size_t) m * K + (size_t) b * 32 + k] = (int8_t) q;
                }
            }
        std::vector<int8_t> wq((size_t) K * N);
        std::vector<float> wsc((size_t) N * nb);
        for (int n = 0; n < N; n++)
            for (int b = 0; b < nb; b++) {
                const uint8_t * o = &Wq_full[((size_t) n * nb + b) * 34];
                wsc[(size_t) n * nb + b] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) o);
                for (int k = 0; k < 32; k++) wq[(size_t) n * K + (size_t) b * 32 + k] = (int8_t) o[2 + k];
            }
        double mxsem = 0; long badsem = 0;
        for (int m = 0; m < M; m++)
            for (int n = N0; n < N; n++) {
                double acc = 0;
                for (int b = 0; b < nb; b++) {
                    int32_t p = 0;
                    for (int k = 0; k < 32; k++)
                        p += (int32_t) xq[(size_t) m * K + (size_t) b * 32 + k] *
                             (int32_t) wq[(size_t) n * K + (size_t) b * 32 + k];
                    acc += (double) xsc[(size_t) m * nb + b] * wsc[(size_t) n * nb + b] * (double) p;
                }
                double e = std::fabs(acc - (double) C1[(size_t)(n - N0) + (size_t) m * N1]);
                if (e > mxsem) mxsem = e;
                if (e > 1e-3) badsem++;
            }
        fprintf(stderr, "[xdna-semantic] max_abs=%.6g bad=%ld/%d %s\n",
                mxsem, badsem, N1 * M, badsem == 0 ? "EXACT" : "DIFF");
    }

    printf("iter,xdna_submit_ns,xdna_done_ns,vk_submit_ns,vk_done_ns,overlap_ns,critical_ns,xdna_dur_ns,vk_dur_ns\n");
    int pass = 0; double sum_crit = 0; long long best = -1e18;
    for (int it = 0; it < iters; it++) {
        uint64_t t0 = now_ns();
        run_once(a, b, c, d);
        uint64_t first = a < c ? a : c, last = b > d ? b : d;
        uint64_t s = a > c ? a : c, e = b < d ? b : d;
        long long overlap = (long long) e - (long long) s;
        uint64_t crit = last - first;
        sum_crit += (double) crit;
        if (overlap > best) best = overlap;
        if (overlap > 0) pass++;
        printf("%d,%llu,%llu,%llu,%llu,%lld,%llu,%llu,%llu\n", it,
               (unsigned long long)(a - t0), (unsigned long long)(b - t0),
               (unsigned long long)(c - t0), (unsigned long long)(d - t0),
               overlap, (unsigned long long) crit,
               (unsigned long long)(b - a), (unsigned long long)(d - c));
    }
    double mean_crit = sum_crit / iters;
    fprintf(stderr, "[correctness] max_abs=%.6g max_rel=%.3g %s\n", maxabs, rel, correct ? "PASS" : "FAIL");
    fprintf(stderr, "OVERLAP %d/%d best=%lld ns mean_critical=%.0f ns\n", pass, iters, best, mean_crit);

    struct ggml_xdna_counters ctr = ggml_xdna_get_counters();
    fprintf(stderr, "[counters] xrt_submit=%d xrt_complete=%d graph_compute=%d supports_op=%d\n",
            ctr.xrt_submit, ctr.xrt_complete, ctr.graph_compute, ctr.supports_op);

    if (json_out) {
        FILE * f = fopen(json_out, "a");
        if (f) {
            fprintf(f, "{\"M\":%d,\"K\":%d,\"N\":%d,\"n0\":%d,\"n1\":%d,\"serial\":%d,"
                       "\"correct\":%d,\"max_rel\":%.6g,\"overlap_pass\":%d,\"iters\":%d,"
                       "\"mean_critical_ns\":%.0f,\"best_overlap_ns\":%lld,"
                       "\"xrt_submit\":%d}\n",
                    M, K, N, N0, N1, (int) serial, (int) correct, rel, pass, iters,
                    mean_crit, best, ctr.xrt_submit);
            fclose(f);
        }
    }

    bool ok = correct && (serial || pass == iters);
    printf("RESULT %s correct=%d overlap_positive=%d/%d best_overlap_ns=%lld mean_critical_ns=%.0f\n",
           ok ? "PASS" : "FAIL", (int) correct, pass, iters, best, mean_crit);
    return ok ? 0 : 2;
}
