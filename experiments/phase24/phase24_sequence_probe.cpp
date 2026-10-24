// phase24_sequence_probe.cpp -- Phase 24 minimal feasibility test.
//
// Question: can N XDNA GEMMs be submitted as ONE host submission with a single
// completion (xrt::runlist), instead of submit+wait per GEMM, and does that
// remove the per-op host-side submission/dispatch tax?
//
// Modes (per-GEMM ms, N = 1..4):
//   perop_new   : each GEMM -> fresh xrt::run, start(), wait()
//   perop_reuse : one xrt::run reused, start(), wait() per GEMM
//   runlist     : N runs added to xrt::runlist, execute() once, wait() once
//
// Kernel: cached whole-array C4_WA design (A[M,K] i8 * B[K,N] i8 -> C[M,N] i32,
// kernel name MLIR_AIE).  xrt::runlist executes the queued runs atomically in
// order on one host submission.
//
// Build:
//   g++ -O2 -std=c++17 -I/usr/include/xrt -I<repo>/ggml/src/ggml-xdna/stub \
//       phase24_sequence_probe.cpp -o phase24_sequence_probe -lxrt_coreutil
//
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_kernel.h>   // xrt::runlist

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
    int reps = (argc > 3) ? atoi(argv[3]) : 5;
    int M = 512, K = 1024, N = 3584;

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
    xrt::bo a_bo(dev, a_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo b_bo(dev, b_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo c_bo(dev, c_bytes, xrt::bo::flags::host_only, gid3);

    std::vector<int8_t> A(a_bytes), B(b_bytes);
    for (size_t i = 0; i < a_bytes; i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < b_bytes; i++) B[i] = (int8_t)((i * 11 + 5) % 255 - 127);
    std::memcpy(a_bo.map(), A.data(), a_bytes);
    std::memcpy(b_bo.map(), B.data(), b_bytes);
    std::memset(c_bo.map(), 0, c_bytes);
    a_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto set_args = [&](xrt::run & r) {
        r.set_arg(0, (uint64_t) 3);
        r.set_arg(1, inst_bo);
        r.set_arg(2, (uint32_t) inst_bytes);
        r.set_arg(3, a_bo);
        r.set_arg(4, b_bo);
        r.set_arg(5, c_bo);
    };

    // warmup
    { xrt::run r(kernel); set_args(r); r.start(); r.wait(); }

    printf("== Phase 24 sequence probe, M=%d K=%d N=%d ==\n", M, K, N);
    printf("mode,N,per_gemm_ms,total_ms\n");
    for (int Ng : {1, 2, 3, 4}) {
        // perop_new
        { std::vector<double> v;
          for (int rep = 0; rep < reps; rep++) {
            double t0 = now_ms();
            for (int g = 0; g < Ng; g++) { xrt::run r(kernel); set_args(r); r.start(); r.wait(); }
            v.push_back((now_ms() - t0) / Ng);
          }
          printf("perop_new,%d,%.4f,%.4f\n", Ng, med(v), med(v) * Ng); }

        // perop_reuse
        { std::vector<double> v;
          for (int rep = 0; rep < reps; rep++) {
            xrt::run r(kernel); set_args(r);
            double t0 = now_ms();
            for (int g = 0; g < Ng; g++) { r.start(); r.wait(); }
            v.push_back((now_ms() - t0) / Ng);
          }
          printf("perop_reuse,%d,%.4f,%.4f\n", Ng, med(v), med(v) * Ng); }

        // runlist: N runs, one execute, one wait
        { std::vector<double> v;
          std::exception_ptr err;
          for (int rep = 0; rep < reps; rep++) {
            try {
              xrt::runlist rl(hw);
              std::vector<xrt::run> runs;
              runs.reserve(Ng);
              for (int g = 0; g < Ng; g++) {
                runs.emplace_back(kernel);
                set_args(runs.back());
                rl.add(runs.back());
              }
              double t0 = now_ms();
              rl.execute();
              rl.wait();
              v.push_back((now_ms() - t0) / Ng);
            } catch (const std::exception & e) {
              err = std::current_exception();
              printf("runlist,%d,ERR,%s\n", Ng, e.what());
              break;
            }
          }
          if (!err) printf("runlist,%d,%.4f,%.4f\n", Ng, med(v), med(v) * Ng); }
    }

    // correctness after a runlist of 2
    c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const int32_t * Cp = (const int32_t *) c_bo.map();
    long bad = 0;
    for (int mm = 0; mm < 32; mm++)
        for (int nn = 0; nn < 32; nn++) {
            long acc = 0;
            for (int kk = 0; kk < K; kk++) acc += (long) A[mm * K + kk] * (long) B[kk * N + nn];
            if (Cp[mm * N + nn] != (int32_t) acc) bad++;
        }
    printf("CORRECTNESS,runlist,mismatches=%ld,%s\n", bad, bad == 0 ? "PASS" : "FAIL");
    printf("DONE\n");
    return 0;
}
