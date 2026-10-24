/**
 * phase19_m_sweep.cpp
 *
 * Direct XDNA XRT kernel execution for M-sweep + K-sweep experiments.
 * Uses xrt::run + set_arg matching proven xdna-persistent.cpp pattern.
 * Host-side M-tiling. XRT contexts properly destroyed between kernels.
 *
 * Build (inside container):
 *   g++ -O2 -std=c++17 -o phase19_m_sweep phase19_m_sweep.cpp \
 *       -I/workspace/llama.cpp/ggml/src/ggml-xdna/stub \
 *       -L/usr/lib/x86_64-linux-gnu -l:libxrt_coreutil.so -lpthread
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <sys/time.h>

#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static bool file_exists(const char* p) {
    struct stat st;
    return stat(p, &st) == 0;
}

// ============================================================================
// XDNA kernel wrapper with persistent BOs
// ============================================================================
struct XdnaKernel {
    std::string xclbin_path, inst_path;
    int M_tile, K_tile, N_tile;
    size_t inst_bytes = 0;
    xrt::device* dev = nullptr;
    xrt::xclbin* xcl = nullptr;
    xrt::hw_context* hw_ctx = nullptr;
    xrt::kernel* krnl = nullptr;
    xrt::bo* inst_bo = nullptr;
    xrt::bo* input_bo = nullptr;
    xrt::bo* weight_bo = nullptr;
    xrt::bo* output_bo = nullptr;
    xrt::run* pre_run = nullptr;
};

static bool xdna_init(XdnaKernel& k) {
    try {
        k.dev = new xrt::device(0);
        k.xcl = new xrt::xclbin(k.xclbin_path);
        k.dev->register_xclbin(*k.xcl);
        k.hw_ctx = new xrt::hw_context(*k.dev, k.xcl->get_uuid());
        k.krnl = new xrt::kernel(*k.hw_ctx, "MLIR_AIE");

        std::ifstream ifs(k.inst_path, std::ios::binary);
        if (!ifs) { fprintf(stderr, "[XDNA] Cannot open %s\n", k.inst_path.c_str()); return false; }
        ifs.seekg(0, std::ios::end);
        k.inst_bytes = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::vector<uint32_t> insts(k.inst_bytes / 4);
        ifs.read(reinterpret_cast<char*>(insts.data()), k.inst_bytes);
        ifs.close();

        uint32_t gid1 = k.krnl->group_id(1);
        uint32_t gid3 = k.krnl->group_id(3);

        k.inst_bo = new xrt::bo(*k.dev, k.inst_bytes, xrt::bo::flags::cacheable, gid1);
        std::memcpy(k.inst_bo->map(), insts.data(), k.inst_bytes);
        k.inst_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Persistent data BOs
        k.input_bo  = new xrt::bo(*k.dev, (size_t)k.M_tile * k.K_tile, xrt::bo::flags::host_only, gid3);
        k.weight_bo = new xrt::bo(*k.dev, (size_t)k.K_tile * k.N_tile, xrt::bo::flags::host_only, gid3);
        k.output_bo = new xrt::bo(*k.dev, (size_t)k.M_tile * k.N_tile * 4, xrt::bo::flags::host_only, gid3);

        // Pre-create run object with all args
        k.pre_run = new xrt::run(*k.krnl);
        k.pre_run->set_arg(0, (uint64_t)3);
        k.pre_run->set_arg(1, *k.inst_bo);
        k.pre_run->set_arg(2, (uint32_t)k.inst_bytes);
        k.pre_run->set_arg(3, *k.input_bo);
        k.pre_run->set_arg(4, *k.weight_bo);
        k.pre_run->set_arg(5, *k.output_bo);

        fprintf(stderr, "[XDNA] Init OK: M=%d K=%d N=%d gid1=%u gid3=%u\n",
                k.M_tile, k.K_tile, k.N_tile, gid1, gid3);
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[XDNA] Init FAILED: %s\n", e.what());
        return false;
    }
}

static void xdna_fini(XdnaKernel& k) {
    delete k.pre_run;   k.pre_run = nullptr;
    delete k.output_bo; k.output_bo = nullptr;
    delete k.weight_bo; k.weight_bo = nullptr;
    delete k.input_bo;  k.input_bo = nullptr;
    delete k.inst_bo;   k.inst_bo = nullptr;
    delete k.krnl;      k.krnl = nullptr;
    delete k.hw_ctx;    k.hw_ctx = nullptr;
    delete k.xcl;       k.xcl = nullptr;
    delete k.dev;       k.dev = nullptr;
}

// Execute one kernel call matching xdna-persistent.cpp pattern.
// Kernel args: 0=uint64_t(3), 1=inst_bo, 2=inst_bytes, 3=input_bo, 4=weight_bo, 5=output_bo
static void xdna_exec_one(XdnaKernel& k,
                           const int8_t* input_padded, int K,
                           const int8_t* weight_row, int N,
                           int32_t* output_row)
{
    size_t in_b  = (size_t)k.M_tile * K;
    size_t wt_b  = (size_t)K * N;
    size_t out_b = (size_t)k.M_tile * N * sizeof(int32_t);

    std::memcpy(k.input_bo->map(), input_padded, in_b);
    k.input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::memcpy(k.weight_bo->map(), weight_row, wt_b);
    k.weight_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::memset(k.output_bo->map(), 0, out_b);
    k.output_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    k.pre_run->start();
    k.pre_run->wait();

    k.output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(output_row, k.output_bo->map(), out_b);
}

// ============================================================================
// CPU reference: raw int8 GEMM  C = A * W
// A: [M, K] row-major, W: [K, N] row-major, C: [M, N] row-major
// ============================================================================
static void cpu_gemm_i8(const int8_t* A, const int8_t* W, int32_t* C,
                         int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int64_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int64_t)A[(size_t)m * K + k] * (int64_t)W[(size_t)k * N + n];
            C[(size_t)m * N + n] = (int32_t)acc;
        }
    }
}

// ============================================================================
struct TestData {
    std::vector<int8_t> input;   // [M, K] row-major
    std::vector<int8_t> weight;  // [K, N] row-major (matches XDNA kernel layout)
    std::vector<int32_t> ref;    // [M, N] row-major
};

static TestData gen_data(int M, int K, int N, unsigned seed = 42) {
    TestData td;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-127, 127);
    td.input.resize((size_t)M * K);
    td.weight.resize((size_t)K * N);
    for (auto& v : td.input) v = dist(rng);
    for (auto& v : td.weight) v = dist(rng);
    td.ref.resize((size_t)M * N);
    cpu_gemm_i8(td.input.data(), td.weight.data(), td.ref.data(), M, K, N);
    return td;
}

struct BenchResult {
    double time_ms, gops, dma_gbps;
    bool correct;
    double max_err;
};

static BenchResult bench_xdna(XdnaKernel& k,
                               const int8_t* A, const int8_t* B,
                               const int32_t* ref, int M, int K, int N,
                               int warmup, int iters)
{
    BenchResult br = {};
    int Mt = k.M_tile;
    int tiles = (M + Mt - 1) / Mt;
    std::vector<int8_t> padded((size_t)Mt * K);
    std::vector<int32_t> kd_out((size_t)tiles * Mt * N);

    for (int w = 0; w < warmup; w++) {
        for (int t = 0; t < tiles; t++) {
            int m0 = t * Mt;
            int mr = std::min(Mt, M - m0);
            std::memset(padded.data(), 0, padded.size());
            std::memcpy(padded.data(), A + (size_t)m0 * K, (size_t)mr * K);
            xdna_exec_one(k, padded.data(), K, B, N,
                          kd_out.data() + (size_t)t * Mt * N);
        }
    }

    double total = 0;
    for (int it = 0; it < iters; it++) {
        double t0 = now_ms();
        for (int t = 0; t < tiles; t++) {
            int m0 = t * Mt;
            int mr = std::min(Mt, M - m0);
            std::memset(padded.data(), 0, padded.size());
            std::memcpy(padded.data(), A + (size_t)m0 * K, (size_t)mr * K);
            xdna_exec_one(k, padded.data(), K, B, N,
                          kd_out.data() + (size_t)t * Mt * N);
        }
        total += now_ms() - t0;
    }
    br.time_ms = total / iters;

    double flops = 2.0 * M * K * N;
    br.gops = flops / (br.time_ms * 1e6);
    size_t dma = (size_t)M * K + (size_t)K * N + (size_t)M * N * 4;
    br.dma_gbps = (dma / (1024.0 * 1024.0)) / (br.time_ms / 1000.0);

    br.correct = true;
    br.max_err = 0;
    for (int m = 0; m < M; m++) {
        int t = m / Mt;
        int r = m % Mt;
        for (int n = 0; n < N; n++) {
            int32_t got = kd_out[(size_t)(t * Mt + r) * N + n];
            int32_t expect = ref[(size_t)m * N + n];
            double err = std::fabs((double)got - (double)expect);
            if (err > br.max_err) br.max_err = err;
            if (err > 0.5) br.correct = false;
        }
    }
    return br;
}

// ============================================================================
// Find kernel in cache
// ============================================================================
static std::string find_kernel(int Mt, int K, int N) {
    const char* fmts[] = {
        "M%d_K%d_N%d_C1_Tm64_Tk64_Tn64",
        "M%d_K%d_N%d_C1_Tm64_Tk32_Tn64",
        "M%d_K%d_N%d_C1",
        "M%d_K%d_N%d",
    };
    for (auto fmt : fmts) {
        char dir[512];
        snprintf(dir, sizeof(dir),
            "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/%s", fmt);
        char full[512];
        snprintf(full, sizeof(full), dir, Mt, K, N);
        std::string xcl = std::string(full) + "/design.xclbin";
        std::string inst = std::string(full) + "/design.insts.bin";
        if (file_exists(xcl.c_str()) && file_exists(inst.c_str()))
            return std::string(full);
    }
    return "";
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("=== Phase 19: XDNA M-sweep + K-sweep ===\n\n");

    int m_tiles[] = {64, 128, 256, 512, 768, 1024};
    int n_vals[] = {512, 1024, 2048, 3072, 4096, 6144, 8192};
    int k_vals[] = {1024, 2048, 4096};

    int nfound = 0;
    for (int kv : k_vals)
        for (int mt : m_tiles)
            for (int n : n_vals)
                if (!find_kernel(mt, kv, n).empty()) nfound++;
    printf("Available kernels: %d\n\n", nfound);

    std::ofstream csv("/workspace/llama.cpp/phase19_m_sweep.csv");
    csv << "M,K,N,M_tile,XDNA_ms,GOPS,GB_s,Correct,MaxErr,Tiles\n";

    // ---- M-sweep ----
    printf("=== M-sweep: K=1024, N=3072/4096/6144 ===\n");
    printf("%-8s %-6s %-6s %-12s %-8s %-8s %-8s %-8s %-6s\n",
           "M", "K", "N", "XDNA_ms", "GOPS", "GB/s", "Correct", "MaxErr", "Tiles");
    printf("-------- ------ ------ ------------ -------- -------- -------- -------- ------\n");

    int m_values[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int n_test[] = {3072, 4096, 6144};

    for (int N : n_test) {
        int best_mt = -1;
        for (int mt : {512, 256, 128, 64}) {
            if (!find_kernel(mt, 1024, N).empty()) { best_mt = mt; break; }
        }
        if (best_mt < 0) { fprintf(stderr, "[SKIP] No kernel K=1024 N=%d\n", N); continue; }

        XdnaKernel k;
        std::string kdir = find_kernel(best_mt, 1024, N);
        k.xclbin_path = kdir + "/design.xclbin";
        k.inst_path = kdir + "/design.insts.bin";
        k.M_tile = best_mt;
        k.K_tile = 1024;
        k.N_tile = N;
        if (!xdna_init(k)) continue;

        for (int M : m_values) {
            size_t mem = (size_t)M * 1024 + (size_t)1024 * N + (size_t)M * N * 4;
            if (mem > 1500ULL * 1024 * 1024) {
                printf("%-8d %-6d %-6d SKIPPED (need %zuMB)\n", M, 1024, N, mem/(1024*1024));
                continue;
            }
            auto td = gen_data(M, 1024, N, 42);
            int tiles = (M + best_mt - 1) / best_mt;
            auto br = bench_xdna(k, td.input.data(), td.weight.data(), td.ref.data(),
                                 M, 1024, N, 1, 1);
            printf("%-8d %-6d %-6d %-12.2f %-8.2f %-8.2f %-8s %-8.1f %-6d\n",
                   M, 1024, N, br.time_ms, br.gops, br.dma_gbps,
                   br.correct ? "YES" : "NO", br.max_err, tiles);
            csv << M << ",1024," << N << "," << best_mt << ","
                << br.time_ms << "," << br.gops << "," << br.dma_gbps << ","
                << (br.correct ? "1" : "0") << "," << br.max_err << "," << tiles << "\n";
            csv.flush();
        }
        xdna_fini(k);
        printf("\n");
    }

    // ---- K-sweep ----
    printf("=== K-sweep: M=4096, N=3072 ===\n");
    printf("%-8s %-6s %-6s %-12s %-8s %-8s %-8s %-8s\n",
           "M", "K", "N", "XDNA_ms", "GOPS", "GB/s", "Correct", "MaxErr");
    printf("-------- ------ ------ ------------ -------- -------- -------- --------\n");

    for (int K : k_vals) {
        int best_mt = -1;
        for (int mt : {512, 256, 128, 64}) {
            if (!find_kernel(mt, K, 3072).empty()) { best_mt = mt; break; }
        }
        if (best_mt < 0) { fprintf(stderr, "[SKIP] No kernel K=%d N=3072\n", K); continue; }

        XdnaKernel k;
        std::string kdir = find_kernel(best_mt, K, 3072);
        k.xclbin_path = kdir + "/design.xclbin";
        k.inst_path = kdir + "/design.insts.bin";
        k.M_tile = best_mt;
        k.K_tile = K;
        k.N_tile = 3072;
        if (!xdna_init(k)) continue;

        int M = 4096;
        auto td = gen_data(M, K, 3072, 42);
        auto br = bench_xdna(k, td.input.data(), td.weight.data(), td.ref.data(),
                             M, K, 3072, 1, 1);
        printf("%-8d %-6d %-6d %-12.2f %-8.2f %-8.2f %-8s %-8.1f\n",
               M, K, 3072, br.time_ms, br.gops, br.dma_gbps,
               br.correct ? "YES" : "NO", br.max_err);
        csv << M << "," << K << ",3072," << best_mt << ","
            << br.time_ms << "," << br.gops << "," << br.dma_gbps << ","
            << (br.correct ? "1" : "0") << "," << br.max_err << ","
            << ((M+best_mt-1)/best_mt) << "\n";
        csv.flush();
        xdna_fini(k);
    }

    csv.close();
    printf("\n=== Phase 19 complete. CSV: phase19_m_sweep.csv ===\n");
    return 0;
}
