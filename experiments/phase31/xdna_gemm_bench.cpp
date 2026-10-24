// xdna_gemm_bench.cpp -- Phase 31: real XDNA int8 GEMM timing at attention shapes.
//
// Anchors the compute cost the NPU would pay for an attention KV-range partition.
// QK^T for one query tile against a KV range is a GEMM A[M,K] i8 x B[K,N] i8 with
// M = query rows, K = head_dim (256), N = KV-range length. Cached XDNA kernels
// exist for exactly K=256, so these shapes run natively (no recompilation).
//
// Reports the end-to-end time (host copy-in + submit + wait + copy-out) and the
// kernel-only span, so the DMA and the on-device compute can be separated.
//
// Build: see build.sh (target xdna_gemm_bench).
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median_ms(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
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
    return rd == (size_t) sz;
}

struct XdnaI8Gemm {
    xrt::device device;
    xrt::kernel kernel;
    xrt::bo inst_bo, in_bo, w_bo, out_bo;
    std::vector<uint32_t> insts;
    size_t inst_bytes = 0;
    int M = 0, K = 0, N = 0;

    XdnaI8Gemm() : device(0), kernel(), inst_bo(), in_bo(), w_bo(), out_bo() {}

    bool init(int m, int k, int n, const char * xclbin, const char * insts_p) {
        M = m; K = k; N = n;
        try {
            xrt::xclbin xb{std::string(xclbin)};
            device.register_xclbin(xb);
            xrt::uuid uuid = xb.get_uuid();
            xrt::hw_context ctx(device, uuid);
            kernel = xrt::kernel(ctx, "MLIR_AIE");
            if (!load_insts(insts_p, insts)) { fprintf(stderr, "insts load failed\n"); return false; }
            inst_bytes = insts.size() * sizeof(uint32_t);
            inst_bo = xrt::bo(device, inst_bytes, xrt::bo::flags::cacheable, kernel.group_id(1));
            std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
            inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            in_bo  = xrt::bo(device, (size_t)M * K, xrt::bo::flags::host_only, kernel.group_id(3));
            w_bo   = xrt::bo(device, (size_t)K * N, xrt::bo::flags::host_only, kernel.group_id(3));
            out_bo = xrt::bo(device, (size_t)M * N * sizeof(int32_t), xrt::bo::flags::host_only, kernel.group_id(3));
            return true;
        } catch (const std::exception & e) {
            fprintf(stderr, "[xdna] init failed: %s\n", e.what());
            return false;
        }
    }

    void prepare(const int8_t * A, const int8_t * B) {
        std::memcpy(in_bo.map(), A, (size_t)M * K);
        std::memcpy(w_bo.map(), B, (size_t)K * N);
        std::memset(out_bo.map(), 0, (size_t)M * N * sizeof(int32_t));
        in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    double run_once() {
        xrt::run run(kernel);
        run.set_arg(0, (uint64_t) 3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t) inst_bytes);
        run.set_arg(3, in_bo);
        run.set_arg(4, w_bo);
        run.set_arg(5, out_bo);
        double t0 = now_ms();
        run.start();
        run.wait();
        out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return now_ms() - t0;
    }
};

int main(int argc, char ** argv) {
    const int reps = (argc > 1) ? atoi(argv[1]) : 20;
    const char * cache = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache";

    struct Shape { int M, K, N; };
    std::vector<Shape> shapes = {
        {512, 256, 3072},
        {512, 256, 4096},
    };

    for (auto sh : shapes) {
        char dir[512];
        snprintf(dir, sizeof(dir), "%s/M%d_K%d_N%d_C1_Tm64_Tk64_Tn64", cache, sh.M, sh.K, sh.N);
        std::string xclbin = std::string(dir) + "/design.xclbin";
        std::string insts  = std::string(dir) + "/design.insts.bin";

        XdnaI8Gemm xd;
        if (!xd.init(sh.M, sh.K, sh.N, xclbin.c_str(), insts.c_str())) {
            printf("GEMM,skip,M=%d,K=%d,N=%d,reason=init_failed\n", sh.M, sh.K, sh.N);
            continue;
        }

        std::vector<int8_t> A((size_t) sh.M * sh.K), B((size_t) sh.K * sh.N);
        for (size_t i = 0; i < A.size(); i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
        for (size_t i = 0; i < B.size(); i++) B[i] = (int8_t)((i * 11 + 5) % 255 - 127);

        xd.prepare(A.data(), B.data());
        xd.run_once(); // warmup

        std::vector<double> ts;
        for (int i = 0; i < reps; i++) ts.push_back(xd.run_once());
        double med = median_ms(ts);

        double flops = 2.0 * sh.M * sh.K * sh.N;
        double gops = flops / (med / 1e3) / 1e9;
        printf("GEMM,M=%d,K=%d,N=%d,median_ms=%.4f,GFLOPs=%.2f\n", sh.M, sh.K, sh.N, med, gops);
        fflush(stdout);
    }
    return 0;
}
