// gemm_smoke.cpp -- Phase 21 whole-array native GEMM correctness smoke test.
//
// Loads a cached xclbin + insts.bin produced by gen_gemm.py, runs a single
// XRT submit, and checks C = A @ B (plain row-major) against a CPU reference.
// Supports int8/int16 inputs and int16/int32 outputs.
//
// Build:
//   g++ -O2 -std=c++17 gemm_smoke.cpp -o gemm_smoke -lxrt_coreutil
//
// Run:
//   ./gemm_smoke <xclbin> <insts> <M> <K> <N> <in_elem_bytes> <out_elem_bytes>
//
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>

static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool load_insts(const char * path, std::vector<uint32_t> & out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { fprintf(stderr, "cannot open %s\n", path); return false; }
    ifs.seekg(0, std::ios::end);
    size_t sz = (size_t)ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out.resize(sz / 4);
    ifs.read(reinterpret_cast<char *>(out.data()), sz);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s <xclbin> <insts> <M> <K> <N> <in_bytes> <out_bytes>\n", argv[0]);
        return 1;
    }
    const char * xclbin = argv[1];
    const char * insts_path = argv[2];
    int M = atoi(argv[3]);
    int K = atoi(argv[4]);
    int N = atoi(argv[5]);
    int in_b = atoi(argv[6]);   // 1 or 2
    int out_b = atoi(argv[7]);  // 2 or 4

    fprintf(stderr, "[smoke] M=%d K=%d N=%d in=%d out=%d\n", M, K, N, in_b, out_b);

    xrt::device device(0);
    xrt::xclbin xb{std::string(xclbin)};
    device.register_xclbin(xb);
    xrt::uuid uuid = xb.get_uuid();
    xrt::hw_context hw_ctx(device, uuid);
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");
    uint32_t gid1 = kernel.group_id(1);
    uint32_t gid3 = kernel.group_id(3);

    std::vector<uint32_t> insts;
    if (!load_insts(insts_path, insts)) return 1;
    size_t inst_bytes = insts.size() * sizeof(uint32_t);

    xrt::bo inst_bo(device, inst_bytes, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    size_t in_bytes  = (size_t)M * K * in_b;
    size_t w_bytes   = (size_t)K * N * in_b;
    size_t out_bytes = (size_t)M * N * out_b;

    xrt::bo in_bo(device, in_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo w_bo(device, w_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo out_bo(device, out_bytes, xrt::bo::flags::host_only, gid3);

    std::vector<int32_t> A(M * K), B(K * N);
    for (int i = 0; i < M * K; i++) A[i] = (i * 7 + 3) % 255 - 127;
    for (int i = 0; i < K * N; i++) B[i] = (i * 11 + 5) % 255 - 127;

    if (in_b == 1) {
        std::vector<int8_t> a8(A.size()), b8(B.size());
        for (size_t i = 0; i < A.size(); i++) a8[i] = (int8_t)A[i];
        for (size_t i = 0; i < B.size(); i++) b8[i] = (int8_t)B[i];
        std::memcpy(in_bo.map(), a8.data(), in_bytes);
        std::memcpy(w_bo.map(), b8.data(), w_bytes);
    } else {
        std::vector<int16_t> a16(A.size()), b16(B.size());
        for (size_t i = 0; i < A.size(); i++) a16[i] = (int16_t)A[i];
        for (size_t i = 0; i < B.size(); i++) b16[i] = (int16_t)B[i];
        std::memcpy(in_bo.map(), a16.data(), in_bytes);
        std::memcpy(w_bo.map(), b16.data(), w_bytes);
    }
    std::memset(out_bo.map(), 0, out_bytes);
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run(kernel);
    run.set_arg(0, (uint64_t)3);
    run.set_arg(1, inst_bo);
    run.set_arg(2, (uint32_t)inst_bytes);
    run.set_arg(3, in_bo);
    run.set_arg(4, w_bo);
    run.set_arg(5, out_bo);

    uint64_t t0 = now_ns();
    run.start();
    auto state = run.wait();
    uint64_t t1 = now_ns();
    double ms = (t1 - t0) / 1e6;

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    long bad = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (int mm = 0; mm < M; mm++) {
        for (int nn = 0; nn < N; nn++) {
            double acc = 0.0;
            for (int kk = 0; kk < K; kk++) acc += (double)A[mm * K + kk] * (double)B[kk * N + nn];
            double got;
            if (out_b == 4) got = (double)((const int32_t *)out_bo.map())[mm * N + nn];
            else            got = (double)((const int16_t *)out_bo.map())[mm * N + nn];
            double ad = got - acc < 0 ? acc - got : got - acc;
            if (ad > 0.0) {
                double denom = acc < 0 ? -acc : acc;
                double rd = denom > 0 ? ad / denom : ad;
                if (rd > max_rel) max_rel = rd;
                bad++;
            }
            if (ad > max_abs) max_abs = ad;
        }
    }
    printf("SMOKE %s M=%d K=%d N=%d mismatches=%ld max_abs=%.0f max_rel=%.3e ms=%.3f state=%d\n",
           bad == 0 ? "PASS" : "FAIL", M, K, N, bad, max_abs, max_rel, ms, (int)state);
    return bad == 0 ? 0 : 2;
}
