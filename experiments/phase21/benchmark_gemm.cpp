// benchmark_gemm.cpp -- Phase 21 whole-array native GEMM roofline benchmark.
//
// Runs a cached gen_gemm.py design `repeat` times, records submit->wait latency
// (median + min), verifies once against a CPU reference, and prints / appends a
// CSV row with achieved GTOPS and array utilization.
//
// Build:
//   g++ -O2 -std=c++17 -I<repo>/ggml/src/ggml-xdna/stub benchmark_gemm.cpp \
//       -o benchmark_gemm -lxrt_coreutil
//
// Run:
//   ./benchmark_gemm <xclbin> <insts> <M> <K> <N> <in_b> <out_b> <cols> \
//       <peak_gops> <repeat> <tag> [csv_out]
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
#include <algorithm>

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
    if (argc < 12) {
        fprintf(stderr, "usage: %s <xclbin> <insts> M K N in_b out_b cols peak_gops repeat tag [csv_out]\n", argv[0]);
        return 1;
    }
    const char * xclbin = argv[1];
    const char * insts_path = argv[2];
    int M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]);
    int in_b = atoi(argv[6]), out_b = atoi(argv[7]);
    int cols = atoi(argv[8]);
    double peak_gops = atof(argv[9]);
    int repeat = atoi(argv[10]);
    const char * tag = argv[11];
    const char * csv_out = (argc > 12) ? argv[12] : nullptr;

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

    size_t in_bytes = (size_t)M * K * in_b;
    size_t w_bytes = (size_t)K * N * in_b;
    size_t out_bytes = (size_t)M * N * out_b;

    xrt::bo in_bo(device, in_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo w_bo(device, w_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo out_bo(device, out_bytes, xrt::bo::flags::host_only, gid3);

    std::vector<int32_t> A(M * K), B(K * N);
    for (int i = 0; i < M * K; i++) A[i] = (i * 7 + 3) % 255 - 127;
    for (int i = 0; i < K * N; i++) B[i] = (i * 11 + 5) % 255 - 127;

    std::vector<int8_t> a8(A.size()), b8(B.size());
    std::vector<int16_t> a16(A.size()), b16(B.size());
    if (in_b == 1) {
        for (size_t i = 0; i < A.size(); i++) a8[i] = (int8_t)A[i];
        for (size_t i = 0; i < B.size(); i++) b8[i] = (int8_t)B[i];
        std::memcpy(in_bo.map(), a8.data(), in_bytes);
        std::memcpy(w_bo.map(), b8.data(), w_bytes);
    } else {
        for (size_t i = 0; i < A.size(); i++) a16[i] = (int16_t)A[i];
        for (size_t i = 0; i < B.size(); i++) b16[i] = (int16_t)B[i];
        std::memcpy(in_bo.map(), a16.data(), in_bytes);
        std::memcpy(w_bo.map(), b16.data(), w_bytes);
    }
    std::memset(out_bo.map(), 0, out_bytes);
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<double> times;
    for (int r = 0; r < repeat; r++) {
        xrt::run run(kernel);
        run.set_arg(0, (uint64_t)3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t)inst_bytes);
        run.set_arg(3, in_bo);
        run.set_arg(4, w_bo);
        run.set_arg(5, out_bo);
        uint64_t t0 = now_ns();
        run.start();
        run.wait();
        uint64_t t1 = now_ns();
        times.push_back((t1 - t0) / 1e6);
    }
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::sort(times.begin(), times.end());
    double med = times[times.size() / 2];
    double mn = times.front();

    // verify once (sampled: full reference is O(M*N*K), too slow for large M/N)
    long bad = 0;
    double max_rel = 0.0;
    int vm = M < 128 ? M : 128;
    int vn = N < 128 ? N : 128;
    for (int mm = 0; mm < vm; mm++) {
        for (int nn = 0; nn < vn; nn++) {
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
                if (bad == 0 || rd > 1e-6) bad++;
            }
        }
    }
    double macs = (double)M * K * N;
    double gops = 2.0 * macs / (med * 1e-3) / 1e9;
    double util = peak_gops > 0 ? gops / peak_gops : 0.0;

    char row[512];
    snprintf(row, sizeof(row),
             "%s,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.2f,%.1f,%.1f,%.4f,%s\n",
             tag, M, K, N, in_b, out_b, cols, med, mn, gops, peak_gops, util * 100.0, max_rel,
             bad == 0 ? "PASS" : "FAIL");
    if (bad != 0) printf("BENCH_NOTE %s mismatches=%ld\n", tag, bad);
    printf("BENCH %s M=%d K=%d N=%d cols=%d med_ms=%.4f min_ms=%.4f gops=%.2f util=%.1f%% %s\n",
           tag, M, K, N, cols, med, mn, gops, util * 100.0, bad == 0 ? "PASS" : "FAIL");
    if (csv_out) {
        std::ofstream ofs(csv_out, std::ios::app);
        ofs << row;
    }
    printf("CSV %s", row);
    return bad == 0 ? 0 : 2;
}
