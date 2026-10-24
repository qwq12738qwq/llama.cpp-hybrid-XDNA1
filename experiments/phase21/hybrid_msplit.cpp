// hybrid_msplit.cpp -- Phase 21: heterogeneous M-split GEMM (NPU rows || Vulkan rows).
//
//   C[M x N] = A[M x K] x B[K x N]
//
//   NPU : rows [0, M0)   -> native XRT whole-array int8 mmul (M0 x K x N)
//   VK  : rows [M0, M)   -> ggml-vulkan Q8_0 W[K,N] x F32 X[K,M1] -> F32
//
// vk_only_ms = Vulkan full-M baseline (the time to beat).
// critical   = max(npu_done, vk_done) - min(npu_start, vk_start).
//
// Build (repo root):
//   g++ -O2 -std=c++17 experiments/phase21/hybrid_msplit.cpp -o experiments/phase21/hybrid_msplit \
//       -I ggml/include -I ggml/src -I ggml/src/ggml-xdna/stub -L build-vk/bin \
//       -lggml-vulkan -lggml -lggml-base -lxrt_coreutil -Wl,-rpath,$PWD/build-vk/bin -pthread
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
    ggml_tensor * Wq = nullptr, * X = nullptr, * C = nullptr;
    ggml_cgraph * gf = nullptr;
    bool build(ggml_backend_t be, int K, int N, int M) {
        ggml_init_params p = { 512ull * 1024 * 1024, nullptr, true };
        ctx = ggml_init(p);
        Wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        X  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        C  = ggml_mul_mat(ctx, Wq, X);
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
    int wait(xrt::run & r) { return (int)r.wait(); }
};

int main(int argc, char ** argv) {
    int M = 2048, K = 4096, N = 4096, M0 = 512, iters = 20;
    const char * npu_xc = nullptr, * npu_in = nullptr;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M" && i + 1 < argc) M = atoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc) K = atoi(argv[++i]);
        else if (a == "--N" && i + 1 < argc) N = atoi(argv[++i]);
        else if (a == "--m0" && i + 1 < argc) M0 = atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--npu-xclbin" && i + 1 < argc) npu_xc = argv[++i];
        else if (a == "--npu-insts" && i + 1 < argc) npu_in = argv[++i];
    }
    int M1 = M - M0;

    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (!vk) { fprintf(stderr, "vk init failed\n"); return 1; }
    fprintf(stderr, "[setup] %s M=%d K=%d N=%d M0(npu)=%d M1(vk)=%d\n",
            ggml_backend_name(vk), M, K, N, M0, M1);

    XdnaGemm npu;
    if (!npu.init(M0, K, N, npu_xc, npu_in)) return 1;
    std::vector<int8_t> A0((size_t)M0 * K), B0((size_t)K * N);
    for (size_t i = 0; i < A0.size(); i++) A0[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < B0.size(); i++) B0[i] = (int8_t)((i * 11 + 5) % 255 - 127);
    npu.prepare(A0, B0);

    // Vulkan half: W[K,N] Q8_0, X[K,M1]
    std::vector<float> W((size_t)K * N);
    for (size_t i = 0; i < W.size(); i++) W[i] = (float)((int)((i * 13 + 1) % 255) - 127) / 127.0f;
    std::vector<uint8_t> Wq; quantize_q8_0(W, K, N, Wq);
    std::vector<float> X1((size_t)K * M1);
    for (size_t i = 0; i < X1.size(); i++) X1[i] = (float)((int)((i * 5 + 2) % 255) - 127) / 127.0f;

    VkGraph g1, gfull;
    if (!g1.build(vk, K, N, M1)) { fprintf(stderr, "g1 build failed\n"); return 1; }
    ggml_backend_tensor_set(g1.Wq, Wq.data(), 0, Wq.size());
    ggml_backend_tensor_set(g1.X, X1.data(), 0, X1.size() * sizeof(float));

    std::vector<float> Xf((size_t)K * M);
    for (size_t i = 0; i < Xf.size(); i++) Xf[i] = (float)((int)((i * 5 + 2) % 255) - 127) / 127.0f;
    if (!gfull.build(vk, K, N, M)) { fprintf(stderr, "gfull build failed\n"); return 1; }
    ggml_backend_tensor_set(gfull.Wq, Wq.data(), 0, Wq.size());
    ggml_backend_tensor_set(gfull.X, Xf.data(), 0, Xf.size() * sizeof(float));

    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute_async(vk, g1.gf); ggml_backend_synchronize(vk);
        ggml_backend_graph_compute_async(vk, gfull.gf); ggml_backend_synchronize(vk);
        xrt::run r(npu.kernel); npu.start(r); npu.wait(r);
    }

    // Vulkan-only full M
    std::vector<double> tv;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = now_ns();
        ggml_backend_graph_compute_async(vk, gfull.gf);
        ggml_backend_synchronize(vk);
        uint64_t t1 = now_ns();
        tv.push_back((t1 - t0) / 1e6);
    }
    std::sort(tv.begin(), tv.end());
    double vk_only_ms = tv[tv.size() / 2];

    // NPU-only (M0 rows, full N)
    std::vector<double> tn;
    for (int i = 0; i < iters; i++) {
        xrt::run r(npu.kernel);
        uint64_t t0 = now_ns(); npu.start(r); npu.wait(r); uint64_t t1 = now_ns();
        tn.push_back((t1 - t0) / 1e6);
    }
    std::sort(tn.begin(), tn.end());
    double npu_only_ms = tn[tn.size() / 2];

    // hybrid
    std::vector<double> crit, ovl, vh, nh;
    for (int i = 0; i < iters; i++) {
        uint64_t vk_s = 0, vk_d = 0, npu_s = 0, npu_d = 0;
        std::thread tvk([&]() {
            uint64_t a = now_ns();
            ggml_backend_graph_compute_async(vk, g1.gf);
            ggml_backend_synchronize(vk);
            uint64_t b = now_ns(); vk_s = a; vk_d = b;
        });
        std::thread tnp([&]() {
            uint64_t a = now_ns();
            xrt::run r(npu.kernel); npu.start(r); npu.wait(r);
            uint64_t b = now_ns(); npu_s = a; npu_d = b;
        });
        tvk.join(); tnp.join();
        crit.push_back((std::max(vk_d, npu_d) - std::min(vk_s, npu_s)) / 1e6);
        vh.push_back((vk_d - vk_s) / 1e6);
        nh.push_back((npu_d - npu_s) / 1e6);
        uint64_t o = (std::min(vk_d, npu_d) > std::max(vk_s, npu_s)) ? (std::min(vk_d, npu_d) - std::max(vk_s, npu_s)) : 0;
        ovl.push_back(o / 1e6);
    }
    std::sort(crit.begin(), crit.end()); std::sort(ovl.begin(), ovl.end());
    std::sort(vh.begin(), vh.end()); std::sort(nh.begin(), nh.end());
    double crit_ms = crit[crit.size() / 2], ovl_ms = ovl[ovl.size() / 2];
    double vh_ms = vh[vh.size() / 2], nh_ms = nh[nh.size() / 2];
    double speedup = vk_only_ms / crit_ms;

    printf("HYBRID_M M=%d K=%d N=%d M0=%d M1=%d vk_only_ms=%.4f npu_only_ms=%.4f "
           "hyb_vk_ms=%.4f hyb_npu_ms=%.4f overlap_ms=%.4f critical_ms=%.4f speedup=%.3f\n",
           M, K, N, M0, M1, vk_only_ms, npu_only_ms, vh_ms, nh_ms, ovl_ms, crit_ms, speedup);
    printf("CSV hybrid_m,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n",
           M, K, N, M0, vk_only_ms, npu_only_ms, vh_ms, nh_ms, ovl_ms, crit_ms, speedup);
    return 0;
}
