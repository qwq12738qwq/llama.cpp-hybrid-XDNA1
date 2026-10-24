// Phase 47B Level 1: XDNA FFN execution model.
//
// Drives the real FFN GEMMs (gate K=1024 N=3584, up same, down K=3584 N=1024)
// through the production q8_scale engine and compares execution models:
//
//   seq  : submit gate -> wait -> submit up -> wait -> submit down -> wait
//   pipe : submit gate -> submit up -> wait gate -> wait up -> submit down -> wait
//   pipe3: submit gate -> submit up -> submit down -> wait all
//
// Reports per-submit host time, per-wait time and total wall, so we can tell
// whether two logically independent XRT commands are truly outstanding on the
// device (wait2 ~= 0 while wait1 ~= T) or serialized by the runtime.
//
// Usage: bench_level1 [--iters N] [--engine c1|c4|mmulv2] [--models seq,pipe,pipe3]

#include "xdna-q8-scale.h"
#include "xdna-q8-scale-c4.h"
#include "xdna-q8-c4-mmul-v2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static std::string env_or(const char * name, const char * dflt) {
    const char * v = getenv(name);
    return v && *v ? std::string(v) : std::string(dflt);
}

struct gemm {
    int K, N;
    std::string suffix;
};

struct eng {
    void * p = nullptr;
    std::string kind;
    size_t a_bytes = 0, b_bytes = 0, c_bytes = 0;
};

static eng make_engine(const std::string & kind, int M, int K, int N, const std::string & cache) {
    const char * suffix = (kind == "c1") ? "_C1_Q8SCALEV"
                       : (kind == "c4") ? "_C4_Q8SCALEV" : "_C4_Q8SCALEV_MMUL";
    const int ncols = (kind == "c1") ? 1 : 4;
    const std::string dir = cache + "/M" + std::to_string(M) + "_K" + std::to_string(K) +
                            "_N" + std::to_string(N) + suffix;
    const std::string xclbin = dir + "/design.xclbin";
    const std::string insts  = dir + "/design.insts.bin";
    eng e; e.kind = kind;
    FILE * f = fopen(xclbin.c_str(), "rb");
    if (!f) return e;
    fclose(f);
    if (kind == "mmulv2") e.p = ggml_xdna_q8_c4_mmul_v2_create(M, K, N, 4, 32, 64, 256, 4, xclbin.c_str(), insts.c_str());
    else if (kind == "c4") e.p = ggml_xdna_q8_scale_c4_create(M, K, N, 4, xclbin.c_str(), insts.c_str());
    else e.p = ggml_xdna_q8_scale_create(M, K, N, xclbin.c_str(), insts.c_str());
    if (!e.p) return e;
    const int m = 32, n = 64, kt = 256, nb = kt / 32;
    e.a_bytes = (size_t)(M / m) * (K / kt) * (m * kt + m * nb * 4);
    e.b_bytes = (size_t)(N / ncols / n) * (K / kt) * (kt * n + nb * n * 4);
    e.c_bytes = (size_t) M * N * sizeof(float);
    return e;
}

static void free_engine(eng & e) {
    if (!e.p) return;
    if (e.kind == "mmulv2") ggml_xdna_q8_c4_mmul_v2_free((ggml_xdna_q8_c4_mmul_v2_engine *) e.p);
    else if (e.kind == "c4") ggml_xdna_q8_scale_c4_free((ggml_xdna_q8_scale_c4_engine *) e.p);
    else ggml_xdna_q8_scale_free((ggml_xdna_q8_scale_engine *) e.p);
    e.p = nullptr;
}

static bool do_submit(eng & e, const void * a, const void * b, xrt::run ** run) {
    if (e.kind == "mmulv2") return ggml_xdna_q8_c4_mmul_v2_submit((ggml_xdna_q8_c4_mmul_v2_engine *) e.p, a, b, run);
    if (e.kind == "c4")     return ggml_xdna_q8_scale_c4_submit((ggml_xdna_q8_scale_c4_engine *) e.p, a, b, run);
    return ggml_xdna_q8_scale_submit((ggml_xdna_q8_scale_engine *) e.p, a, b, run);
}

static int current_buf(eng & e) {
    if (e.kind == "mmulv2") return ((ggml_xdna_q8_c4_mmul_v2_engine *) e.p)->current_buffer - 1;
    if (e.kind == "c4")     return ((ggml_xdna_q8_scale_c4_engine *) e.p)->current_buffer - 1;
    return ((ggml_xdna_q8_scale_engine *) e.p)->current_buffer - 1;
}

static bool do_wait(eng & e, xrt::run * run, float * out, int buf) {
    if (e.kind == "mmulv2") return ggml_xdna_q8_c4_mmul_v2_wait((ggml_xdna_q8_c4_mmul_v2_engine *) e.p, run, out, buf, nullptr);
    if (e.kind == "c4")     return ggml_xdna_q8_scale_c4_wait((ggml_xdna_q8_scale_c4_engine *) e.p, run, out, buf);
    return ggml_xdna_q8_scale_wait((ggml_xdna_q8_scale_engine *) e.p, run, out, buf);
}

int main(int argc, char ** argv) {
    int iters = 3;
    std::string kind = env_or("P47B_ENGINE", "c1");
    std::string models = env_or("P47B_MODELS", "seq,pipe");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--engine") && i + 1 < argc) kind = argv[++i];
        else if (!strcmp(argv[i], "--models") && i + 1 < argc) models = argv[++i];
    }
    const std::string cache = env_or("XDNA_KERNEL_CACHE",
        "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache");
    const int M = 512;

    std::vector<std::string> model_list;
    { size_t p = 0; while (p < models.size()) { size_t c = models.find(',', p); if (c == std::string::npos) c = models.size();
      model_list.push_back(models.substr(p, c - p)); p = c + 1; } }

    fprintf(stderr, "[L1] M=%d engine=%s iters=%d models=%s\n", M, kind.c_str(), iters, models.c_str());

    for (const std::string & model : model_list) {
        if (model == "dupA" || model == "dupB") {
            // Same-engine test: two independent GEMMs on the two BO sets, the
            // exact shape of gate+up in the real graph.
            eng e = make_engine(kind, M, 1024, 3584, cache);
            if (!e.p) { fprintf(stderr, "[L1] %s: engine create failed\n", model.c_str()); continue; }
            const int mT = 32, nT = 64, ktT = 256, nbT = ktT / 32;
            const size_t a_chunk = mT * ktT + mT * nbT * 4, b_chunk = ktT * nT + nbT * nT * 4;
            auto pack = [](size_t total, size_t chunk, size_t data_bytes) {
                std::vector<uint8_t> v(total, 1);
                for (size_t c = 0; c < total / chunk; c++) {
                    float * sc = (float *)(v.data() + c * chunk + data_bytes);
                    for (size_t i = 0; i < (chunk - data_bytes) / sizeof(float); i++) sc[i] = 1.0f;
                }
                return v;
            };
            std::vector<uint8_t> a = pack(e.a_bytes, a_chunk, mT * ktT);
            std::vector<uint8_t> b = pack(e.b_bytes, b_chunk, ktT * nT);
            std::vector<float> c(e.c_bytes / 4, 0.0f);
            double tot_wall = 0, w1s = 0, w2s = 0, s1s = 0, s2s = 0;
            int n_ok = 0, bad = 0;
            for (int it = 0; it < iters + 1; it++) {
                const double tb = now_ms();
                xrt::run * r1 = nullptr, * r2 = nullptr;
                double s1, s2, w1, w2, t;
                if (model == "dupA") {
                    t = now_ms();
                    if (!do_submit(e, a.data(), b.data(), &r1)) return 1;
                    s1 = now_ms() - t;
                    const int i1 = current_buf(e);
                    t = now_ms();
                    if (!do_wait(e, r1, c.data(), i1)) return 1;
                    w1 = now_ms() - t;
                    t = now_ms();
                    if (!do_submit(e, a.data(), b.data(), &r2)) return 1;
                    s2 = now_ms() - t;
                    const int i2 = current_buf(e);
                    t = now_ms();
                    if (!do_wait(e, r2, c.data(), i2)) return 1;
                    w2 = now_ms() - t;
                } else {
                    t = now_ms();
                    if (!do_submit(e, a.data(), b.data(), &r1)) return 1;
                    s1 = now_ms() - t;
                    const int i1 = current_buf(e);
                    t = now_ms();
                    if (!do_submit(e, a.data(), b.data(), &r2)) return 1;
                    s2 = now_ms() - t;
                    const int i2 = current_buf(e);
                    t = now_ms();
                    if (!do_wait(e, r1, c.data(), i1)) return 1;
                    w1 = now_ms() - t;
                    t = now_ms();
                    if (!do_wait(e, r2, c.data(), i2)) return 1;
                    w2 = now_ms() - t;
                }
                const double wall = now_ms() - tb;
                if (it > 0) {
                    tot_wall += wall; s1s += s1; s2s += s2; w1s += w1; w2s += w2;
                    double mx = 0; for (float x : c) mx = std::max(mx, (double)std::fabs(x));
                    if (std::fabs(mx - 1024.0) > 1.0) bad++;
                    n_ok++;
                }
            }
            const double gops = 2.0 * 2.0 * M * 1024 * 3584 / 1e9;
            printf("L1 model=%-6s engine=%-7s pair_wall_ms=%.1f submit=%.2f/%.2f wait1=%.1f wait2=%.1f "
                   "pair_GOP/s=%.1f bad_result=%d/%d\n",
                   model.c_str(), kind.c_str(), tot_wall / n_ok, s1s / n_ok, s2s / n_ok,
                   w1s / n_ok, w2s / n_ok, gops / (tot_wall / n_ok / 1000.0), bad, n_ok);
            fflush(stdout);
            free_engine(e);
            continue;
        }
        // one engine per GEMM shape, all alive at once (mirrors the real FFN)
        eng gate = make_engine(kind, M, 1024, 3584, cache);
        eng up   = make_engine(kind, M, 1024, 3584, cache);
        eng down = make_engine(kind, M, 3584, 1024, cache);
        if (!gate.p || !up.p || !down.p) {
            fprintf(stderr, "[L1] %s: engine create failed (gate=%p up=%p down=%p)\n", model.c_str(), gate.p, up.p, down.p);
            free_engine(gate); free_engine(up); free_engine(down);
            continue;
        }
        // int8 payload = 1, per-32-block f32 scales = 1.0 so the expected result is K
        auto pack = [](size_t total, int nchunks, size_t chunk, size_t data_bytes) {
            std::vector<uint8_t> v(total, 1);
            for (size_t c = 0; c < nchunks; c++) {
                float * sc = (float *)(v.data() + c * chunk + data_bytes);
                size_t n = (chunk - data_bytes) / sizeof(float);
                for (size_t i = 0; i < n; i++) sc[i] = 1.0f;
            }
            return v;
        };
        const int mT = 32, nT = 64, ktT = 256, nbT = ktT / 32;
        const size_t a_chunk = mT * ktT + mT * nbT * 4, b_chunk = ktT * nT + nbT * nT * 4;
        std::vector<uint8_t> ag = pack(gate.a_bytes, gate.a_bytes / a_chunk, a_chunk, mT * ktT);
        std::vector<uint8_t> bg = pack(gate.b_bytes, gate.b_bytes / b_chunk, b_chunk, ktT * nT);
        std::vector<uint8_t> au = pack(up.a_bytes,   up.a_bytes   / a_chunk, a_chunk, mT * ktT);
        std::vector<uint8_t> bu = pack(up.b_bytes,   up.b_bytes   / b_chunk, b_chunk, ktT * nT);
        std::vector<uint8_t> ad = pack(down.a_bytes, down.a_bytes / a_chunk, a_chunk, mT * ktT);
        std::vector<uint8_t> bd = pack(down.b_bytes, down.b_bytes / b_chunk, b_chunk, ktT * nT);
        std::vector<float> cg(gate.c_bytes / 4, 0.0f), cu(up.c_bytes / 4, 0.0f), cd(down.c_bytes / 4, 0.0f);

        double tot_wall = 0, tot_submit = 0;
        int    n_ok = 0;
        double last_waits[3] = {0, 0, 0};
        double last_checks[3] = {0, 0, 0};
        for (int it = 0; it < iters + 1; it++) {
            const double t_begin = now_ms();
            double s_gate = 0, s_up = 0, s_down = 0, w_gate = 0, w_up = 0, w_down = 0;
            xrt::run * rg = nullptr, * ru = nullptr, * rd = nullptr;

            if (model == "seq") {
                double t0 = now_ms();
                if (!do_submit(gate, ag.data(), bg.data(), &rg)) { fprintf(stderr, "[L1] submit fail\n"); return 1; }
                s_gate = now_ms() - t0;
                int bg_idx = current_buf(gate);
                double t1 = now_ms();
                if (!do_wait(gate, rg, cg.data(), bg_idx)) { fprintf(stderr, "[L1] wait fail\n"); return 1; }
                w_gate = now_ms() - t1;

                t0 = now_ms();
                if (!do_submit(up, au.data(), bu.data(), &ru)) { fprintf(stderr, "[L1] submit fail\n"); return 1; }
                s_up = now_ms() - t0;
                int bu_idx = current_buf(up);
                t1 = now_ms();
                if (!do_wait(up, ru, cu.data(), bu_idx)) { fprintf(stderr, "[L1] wait fail\n"); return 1; }
                w_up = now_ms() - t1;

                t0 = now_ms();
                if (!do_submit(down, ad.data(), bd.data(), &rd)) { fprintf(stderr, "[L1] submit fail\n"); return 1; }
                s_down = now_ms() - t0;
                int bd_idx = current_buf(down);
                t1 = now_ms();
                if (!do_wait(down, rd, cd.data(), bd_idx)) { fprintf(stderr, "[L1] wait fail\n"); return 1; }
                w_down = now_ms() - t1;
            } else {
                // submit everything first, then wait
                double t0 = now_ms();
                if (!do_submit(gate, ag.data(), bg.data(), &rg)) { fprintf(stderr, "[L1] submit fail\n"); return 1; }
                s_gate = now_ms() - t0;
                t0 = now_ms();
                if (!do_submit(up, au.data(), bu.data(), &ru)) { fprintf(stderr, "[L1] submit fail\n"); return 1; }
                s_up = now_ms() - t0;
                if (model == "pipe3") {
                    t0 = now_ms();
                    if (!do_submit(down, ad.data(), bd.data(), &rd)) { fprintf(stderr, "[L1] submit fail\n"); return 1; }
                    s_down = now_ms() - t0;
                }
                const int bg_idx = current_buf(gate), bu_idx = current_buf(up);
                double t1 = now_ms();
                if (!do_wait(gate, rg, cg.data(), bg_idx)) { fprintf(stderr, "[L1] wait fail\n"); return 1; }
                w_gate = now_ms() - t1;
                t1 = now_ms();
                if (!do_wait(up, ru, cu.data(), bu_idx)) { fprintf(stderr, "[L1] wait fail\n"); return 1; }
                w_up = now_ms() - t1;
                if (model == "pipe3") {
                    const int bd_idx = current_buf(down);
                    t1 = now_ms();
                    if (!do_wait(down, rd, cd.data(), bd_idx)) { fprintf(stderr, "[L1] wait fail\n"); return 1; }
                    w_down = now_ms() - t1;
                }
            }
            const double wall = now_ms() - t_begin;
            if (it > 0) {
                tot_wall += wall;
                tot_submit += s_gate + s_up + s_down;
                n_ok++;
                last_waits[0] = w_gate; last_waits[1] = w_up; last_waits[2] = w_down;
                // constant-operand check: out = K * aval * bval, catches stale/empty output
                auto chk = [](const std::vector<float> & v) {
                    double m = 0; for (float x : v) m = std::max(m, (double)std::fabs(x)); return m; };
                last_checks[0] = chk(cg); last_checks[1] = chk(cu); last_checks[2] = chk(cd);
            }
        }
        const double gops = 3.0 * 2.0 * M * 1024 * 3584 / 1e9;   // gate+up+down
        const double expect_gate = 1024, expect_up = 1024, expect_down = 3584;
        printf("L1 model=%-6s engine=%-7s ffn_wall_ms=%.1f host_submit_ms=%.2f "
               "wait_gate=%.1f wait_up=%.1f wait_down=%.1f FFN_GOP/s=%.1f "
               "c_gate=%.0f/%.0f c_up=%.0f/%.0f c_down=%.0f/%.0f\n",
               model.c_str(), kind.c_str(), tot_wall / n_ok, tot_submit / n_ok,
               last_waits[0], last_waits[1], last_waits[2], gops / (tot_wall / n_ok / 1000.0),
               last_checks[0], expect_gate, last_checks[1], expect_up, last_checks[2], expect_down);
        fflush(stdout);
        free_engine(gate); free_engine(up); free_engine(down);
    }
    return 0;
}
