// resident_probe.cpp -- Phase 23: resident / batched multi-GEMM vs per-op staging.
//
// Experiments B (two GEMMs one staging), D/E (persistent context, batched
// submission), 11 (N=2/3/4), and zero-copy activation (xrt::ext::bo userptr).
//
// The kernel is the cached whole-array C4_WA design: A[M,K] i8 * B[K,N] i8 ->
// C[M,N] i32, one submit per GEMM here (no multi-op xclbin available).
//
// Build:
//   g++ -O2 -std=c++17 -I/usr/include/xrt -I<repo>/ggml/src/ggml-xdna/stub \
//       resident_probe.cpp -o resident_probe -lxrt_coreutil
//
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_ext.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <memory>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double med(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static bool load_insts(const char * p, std::vector<uint32_t> & o) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) return false;
    ifs.seekg(0, std::ios::end); size_t s = (size_t) ifs.tellg(); ifs.seekg(0);
    o.resize(s / 4); ifs.read(reinterpret_cast<char *>(o.data()), s);
    return true;
}

int main(int argc, char ** argv) {
    const char * xclbin = (argc > 1) ? argv[1]
        : "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C4_WA/design.xclbin";
    const char * insts_path = (argc > 2) ? argv[2]
        : "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C4_WA/design.insts.bin";
    int M = 512, K = 1024, N = 3584;
    int reps = (argc > 3) ? atoi(argv[3]) : 5;

    xrt::device dev(0);
    xrt::xclbin xb{std::string(xclbin)};
    dev.register_xclbin(xb);
    xrt::hw_context hw(dev, xb.get_uuid());
    xrt::kernel kernel(hw, "MLIR_AIE");
    uint32_t gid1 = kernel.group_id(1), gid3 = kernel.group_id(3);

    std::vector<uint32_t> insts;
    if (!load_insts(insts_path, insts)) return 1;
    size_t inst_bytes = insts.size() * 4;
    xrt::bo inst_bo(dev, inst_bytes, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    size_t a_bytes = (size_t) M * K, b_bytes = (size_t) K * N, c_bytes = (size_t) M * N * 4;

    // shared activation buffer (zero-copy ingress candidate)
    void * act = nullptr;
    posix_memalign(&act, 4096, a_bytes);
    xrt::ext::bo a_shared(dev, act, a_bytes,
                          xrt::ext::bo::access_mode::read_write | xrt::ext::bo::access_mode::local);

    xrt::bo a_bo(dev, a_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo b_bo(dev, b_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo c_bo(dev, c_bytes, xrt::bo::flags::host_only, gid3);

    std::vector<int8_t> A(a_bytes), B(b_bytes);
    for (size_t i = 0; i < a_bytes; i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < b_bytes; i++) B[i] = (int8_t)((i * 11 + 5) % 255 - 127);
    std::memcpy(act, A.data(), a_bytes);
    std::memcpy(a_bo.map(), A.data(), a_bytes);
    std::memcpy(b_bo.map(), B.data(), b_bytes);
    std::memset(c_bo.map(), 0, c_bytes);

    auto launch = [&](xrt::bo & a) {
        xrt::run run(kernel);
        run.set_arg(0, (uint64_t) 3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t) inst_bytes);
        run.set_arg(3, a);
        run.set_arg(4, b_bo);
        run.set_arg(5, c_bo);
        run.start();
        run.wait();
    };
    auto sync_in = [&](xrt::bo & a) {
        a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    };

    // warmup
    launch((xrt::bo &) a_shared);
    launch(a_bo);

    printf("== per-op staging vs resident, M=%d K=%d N=%d ==\n", M, K, N);
    printf("mode,K,per_gemm_ms,total_ms\n");
    for (int K_gemm : {1, 2, 3, 4}) {
        // (A) per-op: restage a/b + sync + launch for every GEMM
        std::vector<double> per;
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            for (int g = 0; g < K_gemm; g++) {
                std::memcpy(a_bo.map(), A.data(), a_bytes);
                std::memcpy(b_bo.map(), B.data(), b_bytes);
                sync_in(a_bo);
                launch(a_bo);
            }
            double t1 = now_ms();
            per.push_back((t1 - t0) / K_gemm);
        }
        printf("perop,%d,%.4f,%.4f\n", K_gemm, med(per), med(per) * K_gemm);

        // (B) resident: stage once, launch K GEMMs, single sync at end
        std::vector<double> res;
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            std::memcpy(a_bo.map(), A.data(), a_bytes);
            std::memcpy(b_bo.map(), B.data(), b_bytes);
            sync_in(a_bo);
            for (int g = 0; g < K_gemm; g++) launch(a_bo);
            c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            double t1 = now_ms();
            res.push_back((t1 - t0) / K_gemm);
        }
        printf("resident,%d,%.4f,%.4f\n", K_gemm, med(res), med(res) * K_gemm);

        // (C) shared activation (ext::bo userptr), resident
        std::vector<double> sh;
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);   // no activation memcpy: act is shared
            for (int g = 0; g < K_gemm; g++) launch((xrt::bo &) a_shared);
            c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            double t1 = now_ms();
            sh.push_back((t1 - t0) / K_gemm);
        }
        printf("shared_resident,%d,%.4f,%.4f\n", K_gemm, med(sh), med(sh) * K_gemm);
    }

    // correctness of shared path
    int32_t * Cp = (int32_t *) c_bo.map();
    long bad = 0;
    for (int mm = 0; mm < 32; mm++)
        for (int nn = 0; nn < 32; nn++) {
            long acc = 0;
            for (int kk = 0; kk < K; kk++) acc += (long) A[mm * K + kk] * (long) B[kk * N + nn];
            if (Cp[mm * N + nn] != (int32_t) acc) bad++;
        }
    printf("CORRECTNESS,shared_activation,mismatches=%ld,%s\n", bad, bad == 0 ? "PASS" : "FAIL");
    printf("DONE\n");
    (void) a_bo;
    return 0;
}
