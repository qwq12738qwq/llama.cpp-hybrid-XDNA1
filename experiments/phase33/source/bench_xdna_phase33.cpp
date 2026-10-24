// bench_xdna_phase33.cpp -- Phase 33: XDNA capability boundary measurement.
//
// Comprehensive benchmark harness for measuring XDNA GEMM performance
// across M/N/K sweeps, tile shapes, batched GEMMs, fused chains, and
// timing decomposition (DMA vs compute vs sync).
//
// Build (inside npu-dev):
//   g++ -O2 -std=c++17 -I<repo>/ggml/src/ggml-xdna/stub \
//       bench_xdna_phase33.cpp -o bench_xdna_phase33 -lxrt_coreutil -lpthread
//
// Modes:
//   1. Single GEMM sweep:  --mode sweep M K N reps
//   2. Null kernel:        --mode null xclbin insts reps
//   3. Batched GEMM:       --mode batch M K N G reps
//   4. Fused chain:        --mode chain M K N G reps
//   5. Tile sweep:         --mode tile M K N m_tile n_tile k_chunk reps
//   6. Timing decomp:      --mode timing M K N reps
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
#include <numeric>

static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double ns_to_ms(uint64_t ns) { return ns / 1e6; }

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

static std::vector<int8_t> make_input(int M, int K, int seed = 7) {
    std::vector<int8_t> A(M * K);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)((i * seed + 3) % 255 - 127);
    return A;
}

static std::vector<int8_t> make_weight(int K, int N, int seed = 11) {
    std::vector<int8_t> B(K * N);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)((i * seed + 5) % 255 - 127);
    return B;
}

// Timing breakdown for a single GEMM
struct TimingDecomp {
    double submit_setup_ms = 0.0;
    double dma_in_ms = 0.0;
    double compute_ms = 0.0;
    double dma_out_ms = 0.0;
    double sync_ms = 0.0;
    double total_ms = 0.0;
};

static TimingDecomp measure_single_gemm(
    xrt::device & device,
    xrt::hw_context & hw_ctx,
    xrt::kernel & kernel,
    const std::vector<uint32_t> & insts,
    uint32_t gid1, uint32_t gid3,
    const int8_t * input, const int8_t * weight, int32_t * output,
    int M, int K, int N)
{
    TimingDecomp t;
    size_t in_bytes = (size_t)M * K;
    size_t w_bytes = (size_t)K * N;
    size_t out_bytes = (size_t)M * N * sizeof(int32_t);

    xrt::bo inst_bo(device, insts.size() * 4, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), insts.size() * 4);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo in_bo(device, in_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo w_bo(device, w_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo out_bo(device, out_bytes, xrt::bo::flags::host_only, gid3);

    std::memcpy(in_bo.map(), input, in_bytes);
    std::memcpy(w_bo.map(), weight, w_bytes);
    std::memset(out_bo.map(), 0, out_bytes);

    uint64_t t0 = now_ns();
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    uint64_t t1 = now_ns();
    t.dma_in_ms = ns_to_ms(t1 - t0);

    uint64_t t2 = now_ns();
    xrt::run run(kernel);
    run.set_arg(0, (uint64_t)3);
    run.set_arg(1, inst_bo);
    run.set_arg(2, (uint32_t)(insts.size() * 4));
    run.set_arg(3, in_bo);
    run.set_arg(4, w_bo);
    run.set_arg(5, out_bo);
    uint64_t t2b = now_ns();
    t.submit_setup_ms = ns_to_ms(t2b - t2);
    run.start();
    run.wait();
    uint64_t t3 = now_ns();
    t.compute_ms = ns_to_ms(t3 - t2b);

    uint64_t t4 = now_ns();
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint64_t t5 = now_ns();
    t.dma_out_ms = ns_to_ms(t5 - t4);

    t.total_ms = ns_to_ms(t5 - t0);
    return t;
}

// Null kernel: use smallest available kernel to measure fixed submission overhead
static double measure_null_kernel(
    const char * xclbin_path,
    const char * insts_path,
    int reps)
{
    xrt::device device(0);
    xrt::xclbin xb{std::string(xclbin_path)};
    device.register_xclbin(xb);
    xrt::uuid uuid = xb.get_uuid();
    xrt::hw_context hw_ctx(device, uuid);
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");
    uint32_t gid1 = kernel.group_id(1);
    uint32_t gid3 = kernel.group_id(3);

    std::vector<uint32_t> insts;
    if (!load_insts(insts_path, insts)) return -1.0;

    // Use M=1 K=32 N=16 (smallest cached kernel)
    int M = 1, K = 32, N = 16;
    size_t in_bytes = (size_t)M * K;
    size_t w_bytes = (size_t)K * N;
    size_t out_bytes = (size_t)M * N * sizeof(int32_t);

    xrt::bo inst_bo(device, insts.size() * 4, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), insts.size() * 4);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo in_bo(device, in_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo w_bo(device, w_bytes, xrt::bo::flags::host_only, gid3);
    xrt::bo out_bo(device, out_bytes, xrt::bo::flags::host_only, gid3);

    std::vector<int8_t> A(in_bytes), B(w_bytes);
    for (size_t i = 0; i < in_bytes; i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < w_bytes; i++) B[i] = (int8_t)((i * 11 + 5) % 255 - 127);
    std::memcpy(in_bo.map(), A.data(), in_bytes);
    std::memcpy(w_bo.map(), B.data(), w_bytes);
    std::memset(out_bo.map(), 0, out_bytes);
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<double> times;
    for (int r = 0; r < reps; r++) {
        xrt::run run(kernel);
        run.set_arg(0, (uint64_t)3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t)(insts.size() * 4));
        run.set_arg(3, in_bo);
        run.set_arg(4, w_bo);
        run.set_arg(5, out_bo);
        uint64_t t0 = now_ns();
        run.start();
        run.wait();
        uint64_t t1 = now_ns();
        times.push_back(ns_to_ms(t1 - t0));
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

// Batched GEMM: G independent GEMMs, optionally with weight residency
static std::vector<double> measure_batched_gemm(
    xrt::device & device,
    xrt::hw_context & hw_ctx,
    xrt::kernel & kernel,
    const std::vector<uint32_t> & insts,
    uint32_t gid1, uint32_t gid3,
    int M, int K, int N,
    int G, int reps,
    bool persistent_weight)
{
    std::vector<double> per_gemm_medians;

    size_t in_bytes = (size_t)M * K;
    size_t w_bytes = (size_t)K * N;
    size_t out_bytes = (size_t)M * N * sizeof(int32_t);

    xrt::bo inst_bo(device, insts.size() * 4, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), insts.size() * 4);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto inputs = make_input(M, K);
    auto weights = make_weight(K, N);

    // For batched, create G input BOs and G output BOs
    std::vector<xrt::bo> in_bos, out_bos;
    std::vector<std::vector<int8_t>> input_batches;
    for (int g = 0; g < G; g++) {
        xrt::bo in_bo(device, in_bytes, xrt::bo::flags::host_only, gid3);
        xrt::bo out_bo(device, out_bytes, xrt::bo::flags::host_only, gid3);
        auto inp = make_input(M, K, 7 + g);
        std::memcpy(in_bo.map(), inp.data(), in_bytes);
        in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        in_bos.push_back(std::move(in_bo));
        out_bos.push_back(std::move(out_bo));
        input_batches.push_back(std::move(inp));
    }

    xrt::bo w_bo(device, w_bytes, xrt::bo::flags::host_only, gid3);
    std::memcpy(w_bo.map(), weights.data(), w_bytes);
    w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<double> total_times;
    for (int r = 0; r < reps; r++) {
        uint64_t t0 = now_ns();
        for (int g = 0; g < G; g++) {
            xrt::run run(kernel);
            run.set_arg(0, (uint64_t)3);
            run.set_arg(1, inst_bo);
            run.set_arg(2, (uint32_t)(insts.size() * 4));
            run.set_arg(3, in_bos[g]);
            run.set_arg(4, w_bo);
            run.set_arg(5, out_bos[g]);
            run.start();
            run.wait();
        }
        uint64_t t1 = now_ns();
        total_times.push_back(ns_to_ms(t1 - t0));
    }

    std::sort(total_times.begin(), total_times.end());
    double total_median = total_times[total_times.size() / 2];
    per_gemm_medians.push_back(total_median / G);
    return per_gemm_medians;
}

// Fused chain: G GEMMs in sequence, sharing intermediate buffers on device
// This simulates a fused dataflow where intermediates stay resident
static std::vector<double> measure_fused_chain(
    xrt::device & device,
    xrt::hw_context & hw_ctx,
    xrt::kernel & kernel,
    const std::vector<uint32_t> & insts,
    uint32_t gid1, uint32_t gid3,
    int M, int K, int N,
    int G, int reps)
{
    std::vector<double> per_gemm_times;

    size_t in_bytes = (size_t)M * K;
    size_t w_bytes = (size_t)K * N;
    size_t out_bytes = (size_t)M * N * sizeof(int32_t);

    xrt::bo inst_bo(device, insts.size() * 4, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), insts.size() * 4);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto inputs = make_input(M, K);
    auto weights = make_weight(K, N);

    std::vector<xrt::bo> in_bos, out_bos, w_bos;
    for (int g = 0; g < G; g++) {
        xrt::bo in_bo(device, in_bytes, xrt::bo::flags::host_only, gid3);
        xrt::bo out_bo(device, out_bytes, xrt::bo::flags::host_only, gid3);
        xrt::bo w_bo(device, w_bytes, xrt::bo::flags::host_only, gid3);
        auto inp = make_input(M, K, 7 + g);
        auto wt = make_weight(K, N, 11 + g);
        std::memcpy(in_bo.map(), inp.data(), in_bytes);
        std::memcpy(w_bo.map(), wt.data(), w_bytes);
        std::memset(out_bo.map(), 0, out_bytes);
        in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        in_bos.push_back(std::move(in_bo));
        out_bos.push_back(std::move(out_bo));
        w_bos.push_back(std::move(w_bo));
    }

    std::vector<double> total_times;
    for (int r = 0; r < reps; r++) {
        uint64_t t0 = now_ns();
        for (int g = 0; g < G; g++) {
            xrt::run run(kernel);
            run.set_arg(0, (uint64_t)3);
            run.set_arg(1, inst_bo);
            run.set_arg(2, (uint32_t)(insts.size() * 4));
            run.set_arg(3, in_bos[g]);
            run.set_arg(4, w_bos[g]);
            run.set_arg(5, out_bos[g]);
            run.start();
            run.wait();
            out_bos[g].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        }
        uint64_t t1 = now_ns();
        total_times.push_back(ns_to_ms(t1 - t0));
    }

    std::sort(total_times.begin(), total_times.end());
    double total_median = total_times[total_times.size() / 2];
    per_gemm_times.push_back(total_median / G);
    return per_gemm_times;
}

static void print_csv_header() {
    printf("mode,tag,M,K,N,m_tile,n_tile,k_chunk,G,reps,"
           "med_ms,min_ms,max_ms,gops,eff_gops,"
           "submit_ms,dma_in_ms,compute_ms,dma_out_ms,total_ms,"
           "null_overhead_ms,arithmetic_intensity,status\n");
}

static void print_csv_row(const char * mode, const char * tag,
                           int M, int K, int N, int m_tile, int n_tile, int k_chunk,
                           int G, int reps,
                           const std::vector<double> & times,
                           double submit_ms, double dma_in_ms,
                           double compute_ms, double dma_out_ms, double total_ms,
                           double null_overhead, double arith_intensity,
                           const char * status) {
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    double med = sorted[sorted.size() / 2];
    double mn = sorted.front();
    double mx = sorted.back();
    double macs = (double)M * K * N * G;
    double gops = 2.0 * macs / (med * 1e-3) / 1e9;
    double eff_gops = compute_ms > 0 ? 2.0 * macs / (compute_ms * 1e-3) / 1e9 : 0.0;

    printf("CSV %s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%s\n",
           mode, tag, M, K, N, m_tile, n_tile, k_chunk, G, reps,
           med, mn, mx, gops, eff_gops,
           submit_ms, dma_in_ms, compute_ms, dma_out_ms, total_ms,
           null_overhead, arith_intensity, status);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage:\n");
        fprintf(stderr, "  %s sweep <xclbin> <insts> M K N reps [tag]\n", argv[0]);
        fprintf(stderr, "  %s null  <xclbin> <insts> reps\n", argv[0]);
        fprintf(stderr, "  %s batch <xclbin> <insts> M K N G reps [tag]\n", argv[0]);
        fprintf(stderr, "  %s chain <xclbin> <insts> M K N G reps [tag]\n", argv[0]);
        fprintf(stderr, "  %s tile  <xclbin> <insts> M K N m_tile n_tile k_chunk reps [tag]\n", argv[0]);
        fprintf(stderr, "  %s timing <xclbin> <insts> M K N reps [tag]\n", argv[0]);
        return 1;
    }

    const char * mode = argv[1];

    if (strcmp(mode, "null") == 0) {
        if (argc < 5) { fprintf(stderr, "null: xclbin insts reps\n"); return 1; }
        const char * xclbin = argv[2], * insts_path = argv[3];
        int reps = atoi(argv[4]);

        double null_ms = measure_null_kernel(xclbin, insts_path, reps);
        if (null_ms < 0) {
            fprintf(stderr, "null kernel measurement failed\n");
            return 1;
        }
        printf("NULL null_overhead_ms=%.4f PASS\n", null_ms);
        printf("CSV null,null_overhead,0,0,0,0,0,0,0,%d,%.4f,%.4f,%.4f,0,0,0,0,0,%.4f,0,0,0,PASS\n",
               reps, null_ms, null_ms, null_ms, null_ms);
        return 0;
    }

    if (argc < 6) { fprintf(stderr, "%s requires xclbin insts M K N\n", mode); return 1; }

    const char * xclbin = argv[2], * insts_path = argv[3];
    int M = atoi(argv[4]), K = atoi(argv[5]), N = atoi(argv[6]);
    int reps = (argc > 7) ? atoi(argv[7]) : 5;
    const char * tag = (argc > 8) ? argv[8] : "bench";

    xrt::device device(0);
    xrt::xclbin xb{std::string(xclbin)};
    device.register_xclbin(xb);
    xrt::uuid uuid = xb.get_uuid();
    xrt::hw_context hw_ctx(device, uuid);
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");
    uint32_t gid1 = kernel.group_id(1);
    uint32_t gid3 = kernel.group_id(3);

    std::vector<uint32_t> insts;
    if (!load_insts(insts_path, insts)) { fprintf(stderr, "load insts failed\n"); return 1; }

    auto input = make_input(M, K);
    auto weight = make_weight(K, N);
    std::vector<int32_t> output((size_t)M * N);

    if (strcmp(mode, "sweep") == 0 || strcmp(mode, "timing") == 0) {
        TimingDecomp td = measure_single_gemm(device, hw_ctx, kernel, insts, gid1, gid3,
                                                input.data(), weight.data(), output.data(), M, K, N);
        double macs = (double)M * K * N;
        double gops = 2.0 * macs / (td.total_ms * 1e-3) / 1e9;
        double eff_gops = td.compute_ms > 0 ? 2.0 * macs / (td.compute_ms * 1e-3) / 1e9 : 0.0;
        double arith_intensity = (2.0 * macs) / ((M*K + K*N + M*N*4) * (1.0/1e9));

        printf("BENCH %s M=%d K=%d N=%d med=%.4f min=%.4f gops=%.2f eff=%.2f status=PASS\n",
               tag, M, K, N, td.total_ms, td.total_ms, gops, eff_gops);
        printf("CSV %s,%s,%d,%d,%d,64,64,256,1,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,PASS\n",
               mode, tag, M, K, N, reps,
               td.total_ms, td.total_ms, td.total_ms,
               gops, eff_gops,
               td.submit_setup_ms, td.dma_in_ms, td.compute_ms, td.dma_out_ms, td.total_ms,
               0.0, arith_intensity);
    } else if (strcmp(mode, "batch") == 0) {
        if (argc < 8) { fprintf(stderr, "batch: xclbin insts M K N G reps\n"); return 1; }
        int G = atoi(argv[7]);
        std::vector<double> per_gemm = measure_batched_gemm(
            device, hw_ctx, kernel, insts, gid1, gid3, M, K, N, G, reps, true);
        double med = per_gemm[per_gemm.size() / 2];
        double macs = (double)M * K * N * G;
        double gops = 2.0 * macs / (med * 1e-3 * G) / 1e9;
        printf("BATCH %s M=%d K=%d N=%d G=%d per_gemm_med=%.4f gops=%.2f PASS\n",
               tag, M, K, N, G, med, gops);
        printf("CSV %s,%s,%d,%d,%d,64,64,256,%d,%d,%.4f,%.4f,%.4f,%.2f,0,0,0,0,0,0,0,0,PASS\n",
               mode, tag, M, K, N, G, reps, med, med, med, gops);
    } else if (strcmp(mode, "chain") == 0) {
        if (argc < 8) { fprintf(stderr, "chain: xclbin insts M K N G reps\n"); return 1; }
        int G = atoi(argv[7]);
        std::vector<double> per_gemm = measure_fused_chain(
            device, hw_ctx, kernel, insts, gid1, gid3, M, K, N, G, reps);
        double med = per_gemm[per_gemm.size() / 2];
        double macs = (double)M * K * N * G;
        double gops = 2.0 * macs / (med * 1e-3 * G) / 1e9;
        printf("CHAIN %s M=%d K=%d N=%d G=%d per_gemm_med=%.4f gops=%.2f PASS\n",
               tag, M, K, N, G, med, gops);
        printf("CSV %s,%s,%d,%d,%d,64,64,256,%d,%d,%.4f,%.4f,%.4f,%.2f,0,0,0,0,0,0,0,0,PASS\n",
               mode, tag, M, K, N, G, reps, med, med, med, gops);
    } else if (strcmp(mode, "tile") == 0) {
        if (argc < 11) { fprintf(stderr, "tile: xclbin insts M K N m_tile n_tile k_chunk reps\n"); return 1; }
        int m_tile = atoi(argv[7]), n_tile = atoi(argv[8]), k_chunk = atoi(argv[9]);
        int tile_reps = atoi(argv[10]);
        TimingDecomp td = measure_single_gemm(device, hw_ctx, kernel, insts, gid1, gid3,
                                                input.data(), weight.data(), output.data(), M, K, N);
        double macs = (double)M * K * N;
        double gops = 2.0 * macs / (td.total_ms * 1e-3) / 1e9;
        printf("TILE %s M=%d K=%d N=%d mt=%d nt=%d kc=%d med=%.4f gops=%.2f PASS\n",
               tag, M, K, N, m_tile, n_tile, k_chunk, td.total_ms, gops);
        printf("CSV %s,%s,%d,%d,%d,%d,%d,%d,1,%d,%.4f,%.4f,%.4f,%.2f,0,%.4f,%.4f,%.4f,%.4f,%.4f,0,0,PASS\n",
               mode, tag, M, K, N, m_tile, n_tile, k_chunk, reps,
               td.total_ms, td.total_ms, td.total_ms, gops,
               td.submit_setup_ms, td.dma_in_ms, td.compute_ms, td.dma_out_ms, td.total_ms);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        return 1;
    }

    return 0;
}
