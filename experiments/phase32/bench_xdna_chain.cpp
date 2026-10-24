// bench_xdna_chain.cpp -- Phase 32: persistent XDNA resident-chain benchmark.
//
// Measures the amortization of fixed XDNA submission/DMA cost when the same
// weight tensor is kept resident across multiple sequential GEMM calls.
// Two modes:
//   B1: N independent GEMMs, each re-uploads weight (cold chain)
//   B2: N sequential GEMMs, weight stays resident (persistent chain)
//
// Also measures per-GEMM latency breakdown:
//   - weight upload (once for B1, amortized for B2)
//   - input upload per call
//   - NPU execute per call
//   - output readback per call
//
// Build:
//   g++ -O2 -std=c++17 -I<repo>/ggml/src/ggml-xdna/stub \
//       bench_xdna_chain.cpp -o bench_xdna_chain -lxrt_coreutil
//
// Run:
//   ./bench_xdna_chain <M> <K> <N> <num_gemms> [reps] [kernel_cache_dir]
//
#include <unistd.h>
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
#include <fstream>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median_ms(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
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

struct ChainResult {
    double weight_upload_ms = 0.0;
    double per_gemm_input_ms = 0.0;
    double per_gemm_exec_ms = 0.0;
    double per_gemm_output_ms = 0.0;
    double total_ms = 0.0;
    double per_gemm_median_ms = 0.0;
    int num_gemms = 0;
};

static ChainResult run_chain_cold(int M, int K, int N, int num_gemms,
                                  const std::vector<int8_t> & A,
                                  const std::vector<int8_t> & W,
                                  const char * xclbin, const char * insts_path) {
    ChainResult r;
    r.num_gemms = num_gemms;

    xrt::device device(0);
    xrt::xclbin xb{std::string(xclbin)};
    device.register_xclbin(xb);
    xrt::uuid uuid = xb.get_uuid();
    xrt::hw_context hw_ctx(device, uuid);
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");
    uint32_t gid1 = kernel.group_id(1);
    uint32_t gid3 = kernel.group_id(3);

    std::vector<uint32_t> insts;
    if (!load_insts(insts_path, insts)) { fprintf(stderr, "load insts failed\n"); return r; }
    size_t inst_bytes = insts.size() * sizeof(uint32_t);

    xrt::bo inst_bo(device, inst_bytes, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    size_t in_bytes = (size_t)M * K;
    size_t w_bytes = (size_t)K * N;
    size_t out_bytes = (size_t)M * N * sizeof(int32_t);

    xrt::bo in_bo(device, in_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo w_bo(device, w_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo out_bo(device, out_bytes, xrt::bo::flags::host_only, gid3);

    auto t_total_0 = now_ms();

    // Cold: upload weight each time
    std::vector<double> gemm_times;
    for (int g = 0; g < num_gemms; g++) {
        auto t0 = now_ms();
        // Upload weight (simulating fresh weight each call)
        std::memcpy(w_bo.map(), W.data(), w_bytes);
        w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t1 = now_ms();

        // Upload input
        std::memcpy(in_bo.map(), A.data(), in_bytes);
        in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t2 = now_ms();

        // Execute
        xrt::run run(kernel);
        run.set_arg(0, (uint64_t)3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t)inst_bytes);
        run.set_arg(3, in_bo);
        run.set_arg(4, w_bo);
        run.set_arg(5, out_bo);
        uint64_t t_exec_0 = now_ns();
        run.start();
        run.wait();
        uint64_t t_exec_1 = now_ns();
        auto t3 = now_ms();

        // Readback
        out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto t4 = now_ms();

        gemm_times.push_back((t_exec_1 - t_exec_0) / 1e6);
        if (g == 0) {
            r.weight_upload_ms = t1 - t0;
            r.per_gemm_input_ms = t2 - t1;
            r.per_gemm_exec_ms = (t_exec_1 - t_exec_0) / 1e6;
            r.per_gemm_output_ms = t4 - t3;
        }
    }
    auto t_total_1 = now_ms();
    r.total_ms = t_total_1 - t_total_0;
    r.per_gemm_median_ms = median_ms(gemm_times);
    return r;
}

static ChainResult run_chain_persistent(int M, int K, int N, int num_gemms,
                                        const std::vector<int8_t> & A,
                                        const std::vector<int8_t> & W,
                                        const char * xclbin, const char * insts_path) {
    ChainResult r;
    r.num_gemms = num_gemms;

    xrt::device device(0);
    xrt::xclbin xb{std::string(xclbin)};
    device.register_xclbin(xb);
    xrt::uuid uuid = xb.get_uuid();
    xrt::hw_context hw_ctx(device, uuid);
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");
    uint32_t gid1 = kernel.group_id(1);
    uint32_t gid3 = kernel.group_id(3);

    std::vector<uint32_t> insts;
    if (!load_insts(insts_path, insts)) { fprintf(stderr, "load insts failed\n"); return r; }
    size_t inst_bytes = insts.size() * sizeof(uint32_t);

    xrt::bo inst_bo(device, inst_bytes, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    size_t in_bytes = (size_t)M * K;
    size_t w_bytes = (size_t)K * N;
    size_t out_bytes = (size_t)M * N * sizeof(int32_t);

    xrt::bo in_bo(device, in_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo w_bo(device, w_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo out_bo(device, out_bytes, xrt::bo::flags::host_only, gid3);

    auto t_total_0 = now_ms();

    // Persistent: upload weight once
    std::memcpy(w_bo.map(), W.data(), w_bytes);
    w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t_weight_end = now_ms();
    r.weight_upload_ms = t_weight_end - t_total_0;

    std::vector<double> gemm_times;
    for (int g = 0; g < num_gemms; g++) {
        auto t0 = now_ms();
        // Upload input
        std::memcpy(in_bo.map(), A.data(), in_bytes);
        in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t1 = now_ms();

        // Execute
        xrt::run run(kernel);
        run.set_arg(0, (uint64_t)3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t)inst_bytes);
        run.set_arg(3, in_bo);
        run.set_arg(4, w_bo);
        run.set_arg(5, out_bo);
        uint64_t t_exec_0 = now_ns();
        run.start();
        run.wait();
        uint64_t t_exec_1 = now_ns();
        auto t2 = now_ms();

        // Readback
        out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto t3 = now_ms();

        gemm_times.push_back((t_exec_1 - t_exec_0) / 1e6);
        if (g == 0) {
            r.per_gemm_input_ms = t1 - t0;
            r.per_gemm_exec_ms = (t_exec_1 - t_exec_0) / 1e6;
            r.per_gemm_output_ms = t3 - t2;
        }
    }
    auto t_total_1 = now_ms();
    r.total_ms = t_total_1 - t_total_0;
    r.per_gemm_median_ms = median_ms(gemm_times);
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <M> <K> <N> <num_gemms> [reps] [cache_dir]\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    int num_gemms = atoi(argv[4]);
    int reps = (argc > 5) ? atoi(argv[5]) : 3;
    const char * cache = (argc > 6) ? argv[6]
        : "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache";

    char dir[512];
    snprintf(dir, sizeof(dir), "%s/M%d_K%d_N%d_C1_Tm64_Tk64_Tn64", cache, M, K, N);
    std::string xclbin = std::string(dir) + "/design.xclbin";
    std::string insts  = std::string(dir) + "/design.insts.bin";

    if (access(xclbin.c_str(), R_OK) != 0 || access(insts.c_str(), R_OK) != 0) {
        printf("CHAIN,skip,M=%d,K=%d,N=%d,reason=no_kernel\n", M, K, N);
        return 0;
    }

    std::vector<int8_t> A((size_t)M * K), W((size_t)K * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < W.size(); i++) W[i] = (int8_t)((i * 11 + 5) % 255 - 127);

    // Warmup
    run_chain_cold(M, K, N, 1, A, W, xclbin.c_str(), insts.c_str());
    run_chain_persistent(M, K, N, 1, A, W, xclbin.c_str(), insts.c_str());

    ChainResult cold_med, persist_med;
    std::vector<double> cold_totals, persist_totals;
    for (int r = 0; r < reps; r++) {
        cold_totals.push_back(run_chain_cold(M, K, N, num_gemms, A, W, xclbin.c_str(), insts.c_str()).total_ms);
        persist_totals.push_back(run_chain_persistent(M, K, N, num_gemms, A, W, xclbin.c_str(), insts.c_str()).total_ms);
    }
    cold_med.total_ms = median_ms(cold_totals);
    persist_med.total_ms = median_ms(persist_totals);

    // Re-run once for per-stage breakdown (median run)
    cold_med = run_chain_cold(M, K, N, num_gemms, A, W, xclbin.c_str(), insts.c_str());
    persist_med = run_chain_persistent(M, K, N, num_gemms, A, W, xclbin.c_str(), insts.c_str());

    printf("CHAIN,M=%d,K=%d,N=%d,G=%d,"
           "cold_total_ms=%.4f,cold_w_ms=%.4f,cold_in_ms=%.4f,cold_exec_ms=%.4f,cold_out_ms=%.4f,"
           "persist_total_ms=%.4f,persist_w_ms=%.4f,persist_in_ms=%.4f,persist_exec_ms=%.4f,persist_out_ms=%.4f,"
           "speedup=%.3f\n",
           M, K, N, num_gemms,
           cold_med.total_ms, cold_med.weight_upload_ms, cold_med.per_gemm_input_ms,
           cold_med.per_gemm_exec_ms, cold_med.per_gemm_output_ms,
           persist_med.total_ms, persist_med.weight_upload_ms, persist_med.per_gemm_input_ms,
           persist_med.per_gemm_exec_ms, persist_med.per_gemm_output_ms,
           cold_med.total_ms / (persist_med.total_ms > 0 ? persist_med.total_ms : 1e-9));
    printf("CSV CHAIN,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n",
           M, K, N, num_gemms,
           cold_med.total_ms, cold_med.weight_upload_ms, cold_med.per_gemm_input_ms,
           cold_med.per_gemm_exec_ms, cold_med.per_gemm_output_ms,
           persist_med.total_ms, persist_med.weight_upload_ms, persist_med.per_gemm_input_ms,
           persist_med.per_gemm_exec_ms, persist_med.per_gemm_output_ms,
           cold_med.total_ms / (persist_med.total_ms > 0 ? persist_med.total_ms : 1e-9));
    return 0;
}
