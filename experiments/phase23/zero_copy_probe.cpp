// zero_copy_probe.cpp -- Phase 23: classify the host<->NPU staging cost.
//
// Questions:
//   1. What is raw host_only BO sync(TO_DEVICE)/sync(FROM_DEVICE) bandwidth?
//   2. Is sync(host_only) a real data move or just cache/IOMMU bookkeeping?
//      (compare with-sync vs no-sync, and whether the kernel still sees data)
//   3. Do other BO flags (normal / device_only / cacheable) help?
//   4. Does xrt::ext::bo(userptr) (wrap an existing page-aligned host buffer,
//      the "shared allocation" escape hatch) change the cost?
//
// Build:
//   g++ -O2 -std=c++17 -I/usr/include/xrt zero_copy_probe.cpp \
//       -o zero_copy_probe -lxrt_coreutil
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
#include <optional>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

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

// Median of a vector after sorting a copy.
static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static const char * flag_name(int f) {
    switch (f) {
        case 0: return "host_only";
        case 1: return "normal";
        case 2: return "device_only";
        case 3: return "cacheable";
        case 4: return "userptr";
        default: return "?";
    }
}

static xrt::bo::flags flag_of(int f) {
    switch (f) {
        case 0: return xrt::bo::flags::host_only;
        case 1: return xrt::bo::flags::normal;
        case 2: return xrt::bo::flags::device_only;
        case 3: return xrt::bo::flags::cacheable;
        default: return xrt::bo::flags::host_only;
    }
}

// --- raw BO sync/copy bandwidth ------------------------------------------
static void bench_sync(const xrt::device & dev, size_t bytes, int flag,
                       int repeat, bool do_sync, int group) {
    std::vector<char> src(bytes, 0x11), dst(bytes, 0);
    void * user = nullptr;
    bool is_user = (flag == 4);
    std::optional<xrt::bo> bo;
    if (is_user) {
        // page-aligned user allocation, wrapped host_only shared.
        if (posix_memalign(&user, 4096, bytes) != 0) { fprintf(stderr, "alloc fail\n"); return; }
        xrt::ext::bo e(dev, user, bytes, xrt::ext::bo::access_mode::read_write |
                                            xrt::ext::bo::access_mode::local);
        bo = static_cast<xrt::bo &>(e);
    } else {
        bo = xrt::bo(dev, bytes, flag_of(flag), group);
    }
    void * map = bo->map();

    std::vector<double> tc_in, t_sync_to, t_sync_from, tc_out;
    for (int r = 0; r < repeat; r++) {
        double a = now_ms();
        std::memcpy(map, src.data(), bytes);
        double b = now_ms();
        if (do_sync) bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        double c = now_ms();
        if (do_sync) bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        double d = now_ms();
        std::memcpy(dst.data(), map, bytes);
        double e = now_ms();
        tc_in.push_back(b - a);
        t_sync_to.push_back(c - b);
        t_sync_from.push_back(d - c);
        tc_out.push_back(e - d);
    }
    auto bw = [&](double ms) { return ms > 0 ? (bytes / (ms * 1e-3)) / 1e9 : 0.0; };
    double m_in = median(tc_in), m_to = median(t_sync_to),
           m_fr = median(t_sync_from), m_out = median(tc_out);
    printf("SYNC,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,sync=%d\n",
           flag_name(flag), (int)(bytes >> 20), m_in, m_to, m_fr, m_out,
           bw(m_in), bw(m_to), bw(m_fr), bw(m_out), do_sync ? 1 : 0);
    bo.reset();
    if (is_user) free(user);
}

// --- kernel sees data without sync? --------------------------------------
// Runs the real WA GEMM kernel: A[M,K] i8 * B[K,N] i8 -> C[M,N] i32.
// Fills A/B via map(), optionally syncs, runs, reads C, checks a few MACs.
static void bench_kernel_sync(const xrt::device & dev, xrt::hw_context & hw_ctx,
                              xrt::kernel & kernel, xrt::bo & inst_bo, size_t inst_bytes,
                              int M, int K, int N, int group, bool do_sync, int repeat) {
    size_t a_bytes = (size_t) M * K;
    size_t b_bytes = (size_t) K * N;
    size_t c_bytes = (size_t) M * N * sizeof(int32_t);

    xrt::bo a_bo(dev, a_bytes, xrt::bo::flags::host_only, group);
    xrt::bo b_bo(dev, b_bytes, xrt::bo::flags::host_only, group);
    xrt::bo c_bo(dev, c_bytes, xrt::bo::flags::host_only, group);

    std::vector<int8_t> A(a_bytes), B(b_bytes);
    for (size_t i = 0; i < a_bytes; i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < b_bytes; i++) B[i] = (int8_t)((i * 11 + 5) % 255 - 127);

    std::memcpy(a_bo.map(), A.data(), a_bytes);
    std::memcpy(b_bo.map(), B.data(), b_bytes);
    std::memset(c_bo.map(), 0, c_bytes);
    if (do_sync) {
        a_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        c_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    std::vector<double> tv, tcopy;
    for (int r = 0; r < repeat; r++) {
        xrt::run run(kernel);
        run.set_arg(0, (uint64_t) 3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t) inst_bytes);
        run.set_arg(3, a_bo);
        run.set_arg(4, b_bo);
        run.set_arg(5, c_bo);
        double t0 = now_ms();
        run.start();
        run.wait();
        double t1 = now_ms();
        tv.push_back(t1 - t0);
    }
    double t2 = now_ms();
    c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(c_bo.map(), c_bo.map(), 0);  // touch
    double t3 = now_ms();
    (void) tcopy;

    // verify a small corner against reference
    long bad = 0;
    const int32_t * Cp = (const int32_t *) c_bo.map();
    int vm = M < 64 ? M : 64, vn = N < 64 ? N : 64;
    for (int mm = 0; mm < vm; mm++) {
        for (int nn = 0; nn < vn; nn++) {
            long acc = 0;
            for (int kk = 0; kk < K; kk++) acc += (long) A[mm * K + kk] * (long) B[kk * N + nn];
            if (Cp[mm * N + nn] != (int32_t) acc) bad++;
        }
    }
    printf("KERNEL,M=%d,K=%d,N=%d,sync=%d,med_ms=%.4f,min_ms=%.4f,sync_from_ms=%.3f,mismatches=%ld,%s\n",
           M, K, N, do_sync ? 1 : 0, median(tv),
           *std::min_element(tv.begin(), tv.end()), t3 - t2, bad,
           bad == 0 ? "PASS" : "FAIL");
}

int main(int argc, char ** argv) {
    const char * xclbin = (argc > 1) ? argv[1]
        : "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C4_WA/design.xclbin";
    const char * insts_path = (argc > 2) ? argv[2]
        : "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C4_WA/design.insts.bin";
    int repeat = (argc > 3) ? atoi(argv[3]) : 20;

    xrt::device dev(0);
    auto nm = dev.get_info<xrt::info::device::name>();
    printf("INFO,device=%s\n", nm.c_str());

    // --- part 1: raw sync/copy bandwidth across flags and sizes -----------
    const size_t sizes[] = {1u<<20, 4u<<20, 16u<<20, 32u<<20, 64u<<20, 128u<<20};
    printf("== RAW BO: flag,MB,memcpy_in_ms,sync_to_ms,sync_from_ms,memcpy_out_ms,"
           "in_GBs,to_GBs,from_GBs,out_GBs,sync\n");
    for (int flag : {0, 1, 2, 3}) {
        for (size_t sz : sizes) {
            try { bench_sync(dev, sz, flag, repeat, true, 0); }
            catch (const std::exception & ex) {
                printf("SYNC,%s,%d,ERR,%s\n", flag_name(flag), (int)(sz >> 20), ex.what());
                break;
            }
        }
    }
    // no-sync variant (host_only + userptr) to see if sync is bookkeeping
    for (int flag : {0, 4}) {
        for (size_t sz : sizes) {
            try { bench_sync(dev, sz, flag, repeat, false, 0); }
            catch (const std::exception & ex) {
                printf("SYNC,%s,%d,ERR,%s\n", flag_name(flag), (int)(sz >> 20), ex.what());
                break;
            }
        }
    }
    // userptr with sync (shared allocation wrapping host buffer)
    for (size_t sz : sizes) {
        try { bench_sync(dev, sz, 4, repeat, true, 0); }
        catch (const std::exception & ex) {
            printf("SYNC,userptr,%d,ERR,%s\n", (int)(sz >> 20), ex.what());
            break;
        }
    }

    // --- part 2: real kernel, with vs without sync ------------------------
    xrt::xclbin xb{std::string(xclbin)};
    dev.register_xclbin(xb);
    xrt::uuid uuid = xb.get_uuid();
    xrt::hw_context hw_ctx(dev, uuid);
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");
    uint32_t gid1 = kernel.group_id(1);
    uint32_t gid3 = kernel.group_id(3);

    std::vector<uint32_t> insts;
    if (!load_insts(insts_path, insts)) return 1;
    size_t inst_bytes = insts.size() * sizeof(uint32_t);
    xrt::bo inst_bo(dev, inst_bytes, xrt::bo::flags::cacheable, gid1);
    std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    printf("== KERNEL M=512 K=1024 N=3584 ==\n");
    bench_kernel_sync(dev, hw_ctx, kernel, inst_bo, inst_bytes, 512, 1024, 3584, gid3, true,  repeat);
    bench_kernel_sync(dev, hw_ctx, kernel, inst_bo, inst_bytes, 512, 1024, 3584, gid3, false, repeat);

    printf("DONE\n");
    return 0;
}
