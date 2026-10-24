// xdna_smoke.cpp -- native XRT async smoke test (no IRON runtime)
//
// Confirms the low-level xrt::run start/wait path works against a cached
// xclbin + insts.bin, and checks the int8 GEMM result against a CPU reference.
//
// Build:
//   g++ -O2 -std=c++17 xdna_smoke.cpp -o xdna_smoke -lxrt_coreutil
//
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <cstdio>
#include <cstdint>
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
    const char * xclbin = (argc > 1) ? argv[1]
        : "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M64_K1024_N64/design.xclbin";
    const char * insts_path = (argc > 2) ? argv[2]
        : "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M64_K1024_N64/design.insts.bin";
    int M = (argc > 3) ? atoi(argv[3]) : 64;
    int K = (argc > 4) ? atoi(argv[4]) : 1024;
    int N = (argc > 5) ? atoi(argv[5]) : 64;

    fprintf(stderr, "[smoke] M=%d K=%d N=%d\n", M, K, N);

    xrt::device device(0);
    fprintf(stderr, "[smoke] device name: %s\n", device.get_info<xrt::info::device::name>().c_str());

    xrt::xclbin xb{std::string(xclbin)};
    device.register_xclbin(xb);
    xrt::uuid uuid = xb.get_uuid();
    xrt::hw_context hw_ctx(device, uuid);
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");
    uint32_t gid1 = kernel.group_id(1);
    uint32_t gid3 = kernel.group_id(3);
    fprintf(stderr, "[smoke] group_id(1)=%u group_id(3)=%u\n", gid1, gid3);

    std::vector<uint32_t> insts;
    if (!load_insts(insts_path, insts)) return 1;
    size_t inst_bytes = insts.size() * sizeof(uint32_t);

    xrt::bo inst_bo(device, inst_bytes, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    size_t in_bytes  = (size_t)M * K;
    size_t w_bytes   = (size_t)K * N;
    size_t out_bytes = (size_t)M * N * sizeof(int32_t);

    xrt::bo in_bo(device, in_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo w_bo(device, w_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo out_bo(device, out_bytes, xrt::bo::flags::host_only, gid3);

    std::vector<int8_t> A(in_bytes), B(w_bytes);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)((i * 11 + 5) % 255 - 127);

    std::memcpy(in_bo.map(), A.data(), in_bytes);
    std::memcpy(w_bo.map(), B.data(), w_bytes);
    std::memset(out_bo.map(), 0, out_bytes);
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // submit (async)
    xrt::run run(kernel);
    run.set_arg(0, (uint64_t)3);
    run.set_arg(1, inst_bo);
    run.set_arg(2, (uint32_t)inst_bytes);
    run.set_arg(3, in_bo);
    run.set_arg(4, w_bo);
    run.set_arg(5, out_bo);

    uint64_t t0 = now_ns();
    run.start();
    uint64_t t1 = now_ns();
    auto state = run.wait();
    uint64_t t2 = now_ns();
    fprintf(stderr, "[smoke] start_ns=%llu wait_ns=%llu state=%d\n",
            (unsigned long long)(t1 - t0), (unsigned long long)(t2 - t1), (int)state);

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const int32_t * C = (const int32_t *)out_bo.map();

    // CPU float64 reference
    long bad = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += (double)A[m * K + k] * (double)B[k * N + n];
            double got = (double)C[m * N + n];
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
    fprintf(stderr, "[smoke] M=%d K=%d N=%d mismatches %ld / %d max_abs=%.0f max_rel=%.3e\n",
            M, K, N, bad, M * N, max_abs, max_rel);
    printf("SMOKE %s M=%d K=%d N=%d mismatches=%ld max_abs=%.0f max_rel=%.3e state=%d\n",
           bad == 0 ? "PASS" : "FAIL", M, K, N, bad, max_abs, max_rel, (int)state);
    return bad == 0 ? 0 : 2;
}
