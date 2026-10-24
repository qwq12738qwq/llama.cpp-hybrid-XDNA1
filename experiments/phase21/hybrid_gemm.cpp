// hybrid_gemm.cpp -- Phase 21: heterogeneous N-split GEMM.
//
//   C[M x N] = A[M x K] x B[K x N]
//
// GPU side : real ggml-vulkan backend, Q8_0 W[K,N1] x F32 X[K,M] -> F32 (ne0-major).
// NPU side : native XRT whole-array int8 mmul, A[M,K] i8 x B[K,N0] i8 -> i32.
//
// Both are submitted asynchronously and waited on separate threads, so the
// overlap and critical path are measured, not assumed.
//
//   vk_only_ms  = Vulkan full-N (the baseline to beat)
//   npu_only_ms = NPU whole-array full-N (same shape as vk_only)
//   hybrid      : NPU computes N0 columns, Vulkan computes N1 = N - N0
//                 critical = max(vk_done, npu_done) - min(vk_start, npu_start)
//
// Build (repo root):
//   g++ -O2 -std=c++17 experiments/phase21/hybrid_gemm.cpp -o experiments/phase21/hybrid_gemm \
//       -I ggml/include -I ggml/src -I ggml/src/ggml-xdna/stub -L build-vk/bin \
//       -lggml-vulkan -lggml -lggml-base -lxrt_coreutil -Wl,-rpath,$PWD/build-vk/bin -pthread
//
// Run:
//   ./hybrid_gemm --M 512 --K 4096 --N 4096 --n0 2048 --iters 20 \
//       --npu-xclbin <M512_K4096_N<...>/design.xclbin --npu-insts <...>/design.insts.bin
//
// The NPU xclbin must match (M, K, n0); the NPU-only column below needs a
// separate (M, K, N) xclbin path via --npu-full-xclbin/--npu-full-insts.
//
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>

static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool load_insts(const char * path, std::vector<uint32_t> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(sz / 4);
    size_t rd = fread(out.data(), 1, sz, f);
    fclose(f);
    return rd == (size_t)sz;
}

// Quantize row-major f32 W[K][N] (ne0=K) into ggml Q8_0 blocks.
static void quantize_q8_0(const std::vector<float> & W, int K, int N, std::vector<uint8_t> & out) {
    const int nb = K / 32;
    out.assign((size_t)N * nb * 34, 0);
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            const float * blk = &W[(size_t)n * K + (size_t)b * 32];
            float amax = 0.0f;
            for (int k = 0; k < 32; k++) amax = std::max(amax, std::fabs(blk[k]));
            float d = amax / 127.0f;
            if (d == 0.0f) d = 1e-8f;
            uint8_t * o = &out[((size_t)n * nb + b) * 34];
            ggml_fp16_t dh = ggml_fp32_to_fp16(d);
            memcpy(o, &dh, 2);
            for (int k = 0; k < 32; k++) {
                int q = (int)std::lroundf(blk[k] / d);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                o[2 + k] = (uint8_t)(int8_t)q;
            }
        }
    }
}

struct VkGraph {
    ggml_context * ctx = nullptr;
    ggml_tensor * Wq = nullptr;
    ggml_tensor * X = nullptr;
    ggml_tensor * C = nullptr;
    ggml_cgraph * gf = nullptr;

    bool build(ggml_backend_t be, int K, int N, int M) {
        ggml_init_params p = { 512ull * 1024 * 1024, nullptr, true };
        ctx = ggml_init(p);
        Wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        C = ggml_mul_mat(ctx, Wq, X);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, C);
        return ggml_backend_alloc_ctx_tensors(ctx, be) != nullptr;
    }
};

struct XdnaGemm {
    xrt::device device;
    xrt::kernel kernel;
    xrt::bo inst_bo, in_bo, w_bo, out_bo;
    std::vector<uint32_t> insts;
    size_t inst_bytes = 0;
    int M = 0, K = 0, N = 0;
    bool ok = false;

    XdnaGemm() : device(0), inst_bo(), in_bo(), w_bo(), out_bo() {}

    bool init(int m, int k, int n, const char * xclbin, const char * insts_p) {
        M = m; K = k; N = n;
        try {
            xrt::xclbin xb{std::string(xclbin)};
            device.register_xclbin(xb);
            xrt::uuid uuid = xb.get_uuid();
            xrt::hw_context ctx(device, uuid);
            kernel = xrt::kernel(ctx, "MLIR_AIE");
            if (!load_insts(insts_p, insts)) return false;
            inst_bytes = insts.size() * sizeof(uint32_t);
            inst_bo = xrt::bo(device, inst_bytes, xrt::bo::flags::cacheable, kernel.group_id(1));
            memcpy(inst_bo.map(), insts.data(), inst_bytes);
            inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            in_bo  = xrt::bo(device, (size_t)M * K, xrt::bo::flags::host_only, kernel.group_id(3));
            w_bo   = xrt::bo(device, (size_t)K * N, xrt::bo::flags::host_only, kernel.group_id(3));
            out_bo = xrt::bo(device, (size_t)M * N * 4, xrt::bo::flags::host_only, kernel.group_id(3));
            ok = true;
            return true;
        } catch (const std::exception & e) {
            fprintf(stderr, "[npu] init failed: %s\n", e.what());
            return false;
        }
    }

    void prepare(const std::vector<int8_t> & A, const std::vector<int8_t> & B) {
        memcpy(in_bo.map(), A.data(), (size_t)M * K);
        memcpy(w_bo.map(), B.data(), (size_t)K * N);
        memset(out_bo.map(), 0, (size_t)M * N * 4);
        in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void start(xrt::run & r) {
        r.set_arg(0, (uint64_t)3);
        r.set_arg(1, inst_bo);
        r.set_arg(2, (uint32_t)inst_bytes);
        r.set_arg(3, in_bo);
        r.set_arg(4, w_bo);
        r.set_arg(5, out_bo);
        r.start();
    }

    int wait(xrt::run & r) {
        auto st = r.wait();
        return (int)st;
    }
};

int main(int argc, char ** argv) {
    int M = 512, K = 4096, N = 4096, N0 = 2048, iters = 20;
    bool serial = false;
    const char * npu_xc = nullptr, * npu_in = nullptr;
    const char * npu_full_xc = nullptr, * npu_full_in = nullptr;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M" && i + 1 < argc) M = atoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc) K = atoi(argv[++i]);
        else if (a == "--N" && i + 1 < argc) N = atoi(argv[++i]);
        else if (a == "--n0" && i + 1 < argc) N0 = atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--serial") serial = true;
        else if (a == "--npu-xclbin" && i + 1 < argc) npu_xc = argv[++i];
        else if (a == "--npu-insts" && i + 1 < argc) npu_in = argv[++i];
        else if (a == "--npu-full-xclbin" && i + 1 < argc) npu_full_xc = argv[++i];
        else if (a == "--npu-full-insts" && i + 1 < argc) npu_full_in = argv[++i];
    }
    int N1 = N - N0;
    if (!npu_xc || !npu_in) { fprintf(stderr, "need --npu-xclbin/--npu-insts\n"); return 1; }

    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (!vk) { fprintf(stderr, "vk init failed\n"); return 1; }
    fprintf(stderr, "[setup] %s M=%d K=%d N=%d N0(npu)=%d N1(vk)=%d\n",
            ggml_backend_name(vk), M, K, N, N0, N1);

    XdnaGemm npu;
    if (!npu.init(M, K, N0, npu_xc, npu_in)) return 1;

    // ---- data ----
    std::vector<int8_t> A((size_t)M * K), B0((size_t)K * N0);
    for (size_t i = 0; i < A.size(); i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < B0.size(); i++) B0[i] = (int8_t)((i * 11 + 5) % 255 - 127);
    npu.prepare(A, B0);

    std::vector<float> W1((size_t)K * N1);
    for (size_t i = 0; i < W1.size(); i++) W1[i] = (float)((int)((i * 13 + 1) % 255) - 127) / 127.0f;
    std::vector<uint8_t> Wq1;
    quantize_q8_0(W1, K, N1, Wq1);
    std::vector<float> X((size_t)K * M);
    for (size_t i = 0; i < X.size(); i++) X[i] = (float)((int)((i * 5 + 2) % 255) - 127) / 127.0f;

    VkGraph g1;
    if (!g1.build(vk, K, N1, M)) { fprintf(stderr, "g1 build failed\n"); return 1; }
    ggml_backend_tensor_set(g1.Wq, Wq1.data(), 0, Wq1.size());
    ggml_backend_tensor_set(g1.X, X.data(), 0, X.size() * sizeof(float));

    VkGraph gfull;
    bool have_full = false;
    if (npu_full_xc) {
        std::vector<float> Wf((size_t)K * N);
        for (size_t i = 0; i < Wf.size(); i++) Wf[i] = (float)((int)((i * 13 + 1) % 255) - 127) / 127.0f;
        std::vector<uint8_t> Wqf;
        quantize_q8_0(Wf, K, N, Wqf);
        if (!gfull.build(vk, K, N, M)) { fprintf(stderr, "gfull build failed\n"); return 1; }
        ggml_backend_tensor_set(gfull.Wq, Wqf.data(), 0, Wqf.size());
        ggml_backend_tensor_set(gfull.X, X.data(), 0, X.size() * sizeof(float));
        have_full = true;
    }

    // ---- warmup ----
    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute_async(vk, g1.gf);
        ggml_backend_synchronize(vk);
        xrt::run r(npu.kernel); npu.start(r); npu.wait(r);
    }
    if (have_full) for (int i = 0; i < 3; i++) { ggml_backend_graph_compute_async(vk, gfull.gf); ggml_backend_synchronize(vk); }

    // ---- NPU-only (full N) ----
    double npu_only_ms = 0.0;
    if (npu_full_xc && npu_full_in) {
        XdnaGemm npuf;
        if (npuf.init(M, K, N, npu_full_xc, npu_full_in)) {
            std::vector<int8_t> Bf((size_t)K * N);
            for (size_t i = 0; i < Bf.size(); i++) Bf[i] = (int8_t)((i * 11 + 5) % 255 - 127);
            npuf.prepare(A, Bf);
            std::vector<double> ts;
            for (int i = 0; i < iters; i++) {
                xrt::run r(npuf.kernel);
                uint64_t t0 = now_ns(); npuf.start(r); npuf.wait(r); uint64_t t1 = now_ns();
                ts.push_back((t1 - t0) / 1e6);
            }
            std::sort(ts.begin(), ts.end());
            npu_only_ms = ts[ts.size() / 2];
        }
    }

    // ---- Vulkan-only (full N) ----
    double vk_only_ms = 0.0;
    if (have_full) {
        std::vector<double> ts;
        for (int i = 0; i < iters; i++) {
            uint64_t t0 = now_ns();
            ggml_backend_graph_compute_async(vk, gfull.gf);
            ggml_backend_synchronize(vk);
            uint64_t t1 = now_ns();
            ts.push_back((t1 - t0) / 1e6);
        }
        std::sort(ts.begin(), ts.end());
        vk_only_ms = ts[ts.size() / 2];
    }

    // ---- hybrid: NPU N0 || Vulkan N1 ----
    std::vector<double> crit, ovl, vk_h, npu_h;
    for (int i = 0; i < iters; i++) {
        uint64_t t_submit_min = UINT64_MAX, t_done_max = 0;
        uint64_t vk_s = 0, vk_d = 0, npu_s = 0, npu_d = 0;
        std::thread tv([&]() {
            uint64_t a = now_ns();
            ggml_backend_graph_compute_async(vk, g1.gf);
            ggml_backend_synchronize(vk);
            uint64_t b = now_ns();
            vk_s = a; vk_d = b;
        });
        std::thread tn([&]() {
            uint64_t a = now_ns();
            xrt::run r(npu.kernel);
            npu.start(r);
            npu.wait(r);
            uint64_t b = now_ns();
            npu_s = a; npu_d = b;
        });
        tv.join(); tn.join();
        t_submit_min = std::min(vk_s, npu_s);
        t_done_max = std::max(vk_d, npu_d);
        crit.push_back((t_done_max - t_submit_min) / 1e6);
        vk_h.push_back((vk_d - vk_s) / 1e6);
        npu_h.push_back((npu_d - npu_s) / 1e6);
        uint64_t ov = (std::min(vk_d, npu_d) > std::max(vk_s, npu_s))
                          ? (std::min(vk_d, npu_d) - std::max(vk_s, npu_s)) : 0;
        ovl.push_back(ov / 1e6);
    }
    std::sort(crit.begin(), crit.end());
    std::sort(ovl.begin(), ovl.end());
    std::sort(vk_h.begin(), vk_h.end());
    std::sort(npu_h.begin(), npu_h.end());
    double crit_ms = crit[crit.size() / 2];
    double ovl_ms = ovl[ovl.size() / 2];
    double vk_h_ms = vk_h[vk_h.size() / 2];
    double npu_h_ms = npu_h[npu_h.size() / 2];

    double npu_only_gops = 2.0 * (double)M * K * N / (npu_only_ms * 1e-3) / 1e9;
    double vk_only_gops = vk_only_ms > 0 ? 2.0 * (double)M * K * N / (vk_only_ms * 1e-3) / 1e9 : 0.0;
    double speedup = vk_only_ms > 0 ? vk_only_ms / crit_ms : 0.0;

    printf("HYBRID M=%d K=%d N=%d N0=%d N1=%d vk_only_ms=%.4f npu_only_ms=%.4f "
           "hyb_vk_ms=%.4f hyb_npu_ms=%.4f overlap_ms=%.4f critical_ms=%.4f speedup=%.3f\n",
           M, K, N, N0, N1, vk_only_ms, npu_only_ms, vk_h_ms, npu_h_ms, ovl_ms, crit_ms, speedup);
    printf("CSV hybrid,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n",
           M, K, N, N0, vk_only_ms, npu_only_ms, vk_h_ms, npu_h_ms, ovl_ms, crit_ms, speedup);
    printf("TOPS vk_only=%.3f npu_only=%.3f (N=%d)\n", vk_only_gops / 1000.0, npu_only_gops / 1000.0, N);
    (void)serial;
    return 0;
}
