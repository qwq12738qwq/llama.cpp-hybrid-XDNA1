// Phase 47B Level 0: XDNA kernel-only benchmark.
//
// Drives the XDNA engines directly (no ggml graph, no scheduler, no copies) and
// reports where the time goes for each (M,K,N) shape:
//   host memcpy -> BO, BO sync, xrt::run ctor, set_arg, start, wait, sync_from, copyback.
//
// Usage: bench_level0 [--iters N] [--engines c1,c4,mmulv2] [--shapes 1024x3584,3584x1024]

#include "xdna-q8-scale.h"
#include "xdna-q8-scale-c4.h"
#include "xdna-q8-c4-mmul-v2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static std::string env_or(const char * name, const char * dflt) {
    const char * v = getenv(name);
    return v && *v ? std::string(v) : std::string(dflt);
}

struct acc {
    double pack = 0, bo_sync = 0, run_ctor = 0, set_arg = 0, start = 0, wait = 0, sync_from = 0, copyback = 0;
    int n = 0;
    void add(const acc & o) { pack+=o.pack; bo_sync+=o.bo_sync; run_ctor+=o.run_ctor; set_arg+=o.set_arg;
                              start+=o.start; wait+=o.wait; sync_from+=o.sync_from; copyback+=o.copyback; n+=o.n; }
};

int main(int argc, char ** argv) {
    int iters = 5;
    std::string engines = env_or("P47B_ENGINES", "c1,c4,mmulv2");
    std::string shapes = env_or("P47B_SHAPES", "1024x3584,3584x1024");
    int M = 512;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--engines") && i + 1 < argc) engines = argv[++i];
        else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) shapes = argv[++i];
        else if (!strcmp(argv[i], "--M") && i + 1 < argc) M = atoi(argv[++i]);
    }

    const std::string cache = env_or("XDNA_KERNEL_CACHE",
        "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache");

    std::vector<std::pair<int,int>> shape_list;
    {
        size_t p = 0;
        while (p < shapes.size()) {
            size_t c = shapes.find(',', p);
            if (c == std::string::npos) c = shapes.size();
            std::string s = shapes.substr(p, c - p);
            size_t x = s.find('x');
            if (x != std::string::npos) shape_list.push_back({atoi(s.c_str()), atoi(s.c_str() + x + 1)});
            p = c + 1;
        }
    }
    std::vector<std::string> engine_list;
    {
        size_t p = 0;
        while (p < engines.size()) {
            size_t c = engines.find(',', p);
            if (c == std::string::npos) c = engines.size();
            engine_list.push_back(engines.substr(p, c - p));
            p = c + 1;
        }
    }

    fprintf(stderr, "[L0] M=%d iters=%d cache=%s\n", M, iters, cache.c_str());

    for (const auto & sh : shape_list) {
        const int K = sh.first, N = sh.second;
        const double gops = 2.0 * M * K * N / 1e9;
        for (const std::string & en : engine_list) {
            const int ncols = (en == "c1") ? 1 : 4;
            const std::string suffix = (en == "c1") ? "_C1_Q8SCALEV"
                                : (en == "c4") ? "_C4_Q8SCALEV" : "_C4_Q8SCALEV_MMUL";
            const char * dir_env = getenv("P47B_DIR");
            const std::string dir = dir_env ? std::string(dir_env) :
                cache + "/M" + std::to_string(M) + "_K" + std::to_string(K) +
                "_N" + std::to_string(N) + suffix;
            const std::string xclbin = dir + "/design.xclbin";
            const std::string insts  = dir + "/design.insts.bin";
            FILE * f = fopen(xclbin.c_str(), "rb");
            if (!f) { fprintf(stderr, "[L0] SKIP %s K=%d N=%d (no xclbin)\n", en.c_str(), K, N); continue; }
            fclose(f);

            void * eng = nullptr;
            const int ncols_arg = getenv("P47B_NCOLS") ? atoi(getenv("P47B_NCOLS")) : 4;
            if (en == "mmulv2") eng = ggml_xdna_q8_c4_mmul_v2_create(M, K, N, ncols_arg, 32, 64, 256, 4, xclbin.c_str(), insts.c_str());
            else if (en == "c4") eng = ggml_xdna_q8_scale_c4_create(M, K, N, 4, xclbin.c_str(), insts.c_str());
            else eng = ggml_xdna_q8_scale_create(M, K, N, xclbin.c_str(), insts.c_str());
            if (!eng) { fprintf(stderr, "[L0] SKIP %s K=%d N=%d (engine create failed)\n", en.c_str(), K, N); continue; }

            // Packed operand sizes must match the engine layout.
            const int m = 32, n = 64, kt = 256, nb = kt / 32;
            size_t a_bytes = (size_t)(M / m) * (K / kt) * (m * kt + m * nb * 4);
            size_t b_bytes;
            if (en == "c1") b_bytes = (size_t)(N / n) * (K / kt) * (kt * n + nb * n * 4);
            else b_bytes = (size_t)(N / (getenv("P47B_NCOLS") ? atoi(getenv("P47B_NCOLS")) : 4) / n) * (K / kt) * (kt * n + nb * n * 4);
            const size_t c_bytes = (size_t) M * N * sizeof(float);

            std::vector<uint8_t> a(a_bytes, 1), b(b_bytes, 2);
            std::vector<float> c(c_bytes / sizeof(float), 0.0f);

            acc tot;
            bool failed = false;
            double wall0 = now_ms();
            for (int it = 0; it < iters + 1; it++) {   // +1 warmup
                xrt::run * run = nullptr;
                double t0 = now_ms(), t1, t2, t3;
                acc a1;
                bool ok = false;
                if (en == "mmulv2") {
                    ok = ggml_xdna_q8_c4_mmul_v2_submit((ggml_xdna_q8_c4_mmul_v2_engine *) eng, a.data(), b.data(), &run);
                    t1 = now_ms();
                } else if (en == "c4") {
                    ok = ggml_xdna_q8_scale_c4_submit((ggml_xdna_q8_scale_c4_engine *) eng, a.data(), b.data(), &run);
                    t1 = now_ms();
                } else {
                    ok = ggml_xdna_q8_scale_submit((ggml_xdna_q8_scale_engine *) eng, a.data(), b.data(), &run);
                    t1 = now_ms();
                }
                if (!ok || !run) { fprintf(stderr, "[L0] submit failed\n"); failed = true; break; }
                t2 = now_ms();   // submit returned (async)
                const int buf = ((en == "mmulv2") ? ((ggml_xdna_q8_c4_mmul_v2_engine *) eng)->current_buffer
                            : (en == "c4")     ? ((ggml_xdna_q8_scale_c4_engine *) eng)->current_buffer
                                               : ((ggml_xdna_q8_scale_engine *) eng)->current_buffer) - 1;
                if (en == "mmulv2")      ok = ggml_xdna_q8_c4_mmul_v2_wait((ggml_xdna_q8_c4_mmul_v2_engine *) eng, run, c.data(), buf, nullptr);
                else if (en == "c4")     ok = ggml_xdna_q8_scale_c4_wait((ggml_xdna_q8_scale_c4_engine *) eng, run, c.data(), buf);
                else                     ok = ggml_xdna_q8_scale_wait((ggml_xdna_q8_scale_engine *) eng, run, c.data(), buf);
                t3 = now_ms();
                if (!ok) { fprintf(stderr, "[L0] wait failed\n"); failed = true; break; }
                if (it > 0) {
                    a1.n = 1;
                    a1.start = t1 - t0;
                    a1.wait = t3 - t2;
                    tot.add(a1);
                }
            }
            double wall = now_ms() - wall0;
            if (en == "mmulv2") ggml_xdna_q8_c4_mmul_v2_free((ggml_xdna_q8_c4_mmul_v2_engine *) eng);
            else if (en == "c4") ggml_xdna_q8_scale_c4_free((ggml_xdna_q8_scale_c4_engine *) eng);
            else ggml_xdna_q8_scale_free((ggml_xdna_q8_scale_engine *) eng);
            if (failed || tot.n == 0) { fprintf(stderr, "[L0] %s K=%d N=%d FAILED\n", en.c_str(), K, N); continue; }
            const double it_n = iters;
            const double total = (tot.start + tot.wait) / it_n;
            printf("L0 engine=%-7s M=%d K=%d N=%d | submit_ms=%.2f wait_ms=%.2f total_ms=%.2f | GOP/s=%.1f | wall_ms=%.2f\n",
                   en.c_str(), M, K, N, tot.start / it_n, tot.wait / it_n, total,
                   gops / (total / 1000.0), wall);
            fflush(stdout);
        }
    }
    return 0;
}
