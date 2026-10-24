// conv_silu_smoke.cpp -- verify + time the XDNA depthwise conv1d kernel.
//
// Matches ggml_ssm_conv layout:
//   sx : [C, Tin] f32  (Tin = T + K - 1, time contiguous)
//   w  : [C, K]   f32
//   out: [T, C]   f32  (channel contiguous)
//   out[c, t] = sum_k sx[c, t + k] * w[c, k]
//
// Build:
//   g++ -O2 -std=c++17 conv_silu_smoke.cpp -o conv_silu_smoke -lxrt_coreutil
//
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>

static bool load_insts(const char * path, std::vector<uint32_t> & out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { fprintf(stderr, "cannot open %s\n", path); return false; }
    ifs.seekg(0, std::ios::end);
    size_t sz = (size_t) ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out.resize(sz / 4);
    ifs.read(reinterpret_cast<char *>(out.data()), sz);
    return true;
}

int main(int argc, char ** argv) {
    const char * xclbin = (argc > 1) ? argv[1] : "/tmp/conv/design.xclbin";
    const char * insts_path = (argc > 2) ? argv[2] : "/tmp/conv/design.insts.bin";
    int C = (argc > 3) ? atoi(argv[3]) : 6144;
    int T = (argc > 4) ? atoi(argv[4]) : 512;
    int K = (argc > 5) ? atoi(argv[5]) : 4;
    int iters = (argc > 6) ? atoi(argv[6]) : 20;
    int Tin = T + K - 1;

    // sx: [C, Tin], w: [C, K]
    std::vector<float> A((size_t) C * Tin), W((size_t) C * K);
    for (size_t i = 0; i < A.size(); i++) A[i] = 0.001f * (float)((int)((i * 13 + 7) % 97) - 48);
    for (size_t i = 0; i < W.size(); i++) W[i] = 0.01f  * (float)((int)((i * 5 + 1) % 17) - 8);

    // CPU reference, out: [T, C]
    std::vector<double> ref((size_t) T * C, 0.0);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += (double) A[(size_t) c * Tin + (t + k)] * W[(size_t) c * K + k];
            ref[(size_t) t * C + c] = acc;
        }

    xrt::device device(0);
    xrt::xclbin xb{std::string(xclbin)};
    device.register_xclbin(xb);
    xrt::hw_context hw_ctx(device, xb.get_uuid());
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");

    std::vector<uint32_t> insts;
    if (!load_insts(insts_path, insts)) return 1;
    size_t inst_bytes = insts.size() * 4;

    xrt::bo inst_bo(device, inst_bytes, xrt::bo::flags::cacheable, kernel.group_id(1));
    std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo a_bo(device, A.size() * 4, xrt::bo::flags::host_only, kernel.group_id(3));
    xrt::bo w_bo(device, W.size() * 4, xrt::bo::flags::host_only, kernel.group_id(3));
    xrt::bo o_bo(device, (size_t) T * C * 4, xrt::bo::flags::host_only, kernel.group_id(3));
    std::memcpy(a_bo.map(), A.data(), A.size() * 4);
    std::memcpy(w_bo.map(), W.data(), W.size() * 4);
    std::memset(o_bo.map(), 0, (size_t) T * C * 4);
    a_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    o_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run(kernel);
    run.set_arg(0, (uint64_t) 3);
    run.set_arg(1, inst_bo);
    run.set_arg(2, (uint32_t) inst_bytes);
    run.set_arg(3, a_bo);
    run.set_arg(4, w_bo);
    run.set_arg(5, o_bo);

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) { run.start(); run.wait(); }
    auto t1 = std::chrono::steady_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    o_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float * O = (const float *) o_bo.map();

    double max_abs = 0.0, max_rel = 0.0;
    long bad = 0; long first_bad = -1;
    long bad_by_q[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < (size_t) T * C; i++) {
        double d = (double) O[i] - ref[i];
        double ad = d < 0 ? -d : d;
        if (ad > max_abs) max_abs = ad;
        double denom = ref[i] < 0 ? -ref[i] : ref[i];
        if (denom > 1e-6) { double r = ad / denom; if (r > max_rel) max_rel = r; }
        if (ad > 1e-3 * (denom > 1 ? denom : 1)) {
            if (first_bad < 0) first_bad = (long) i;
            bad++;
            bad_by_q[(i / ((size_t) T * C / 4)) < 3 ? (i / ((size_t) T * C / 4)) : 3]++;
        }
    }
    fprintf(stderr, "[conv] first_bad=%ld (t=%ld c=%ld) bad_by_quarter=%ld %ld %ld %ld\n",
            first_bad, first_bad / (size_t) C, first_bad % (size_t) C,
            bad_by_q[0], bad_by_q[1], bad_by_q[2], bad_by_q[3]);
    double macs = (double) T * C * K;
    fprintf(stderr, "[conv] C=%d T=%d K=%d iters=%d avg=%.1f us max_abs=%.4g max_rel=%.3e bad=%ld/%d gmac_s=%.3f\n",
            C, T, K, iters, avg_us, max_abs, max_rel, bad, T * C, macs / (avg_us * 1e-6) / 1e9);
    printf("CONV %s C=%d T=%d K=%d avg_us=%.1f max_rel=%.3e bad=%ld/%d gmac_s=%.3f\n",
           bad == 0 ? "PASS" : "FAIL", C, T, K, avg_us, max_rel, bad, T * C, macs / (avg_us * 1e-6) / 1e9);
    return bad == 0 ? 0 : 2;
}
