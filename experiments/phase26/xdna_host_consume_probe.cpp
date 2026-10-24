// Phase 26: does an XRT userptr BO over plain host memory actually feed the XDNA
// kernel in place (no hidden device copy), and what does the sync cost?
//
// A[M,K] i8 x B[K,N] i8 -> C[M,N] i32 on host memory shared with Vulkan.
// Verifies the kernel result against a CPU reference and times submit/wait/sync.
//
// Build:
//   g++ -O2 -std=c++17 xdna_host_consume_probe.cpp -o xdna_host_consume_probe \
//       $(pkg-config --cflags --libs xrt) -I/usr/include/xrt \
//       -I/workspace/llama.cpp/ggml/src/ggml-xdna/stub
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>

#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<uint32_t> load_insts(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> v(sz / 4);
    if (fread(v.data(), 1, sz, f) != (size_t) sz) { perror("fread"); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    const int M = 512, K = 1024, N = 1792;
    const char * cache = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N1792_C4_WA";
    std::string xclbin = std::string(cache) + "/design.xclbin";
    std::string insts_path = std::string(cache) + "/design.insts.bin";

    size_t a_bytes = (size_t) M * K;
    size_t b_bytes = (size_t) K * N;
    size_t c_bytes = (size_t) M * N * sizeof(int32_t);

    // Host memory shared with (in the real path) Vulkan. 64K aligned.
    void * a = nullptr, * b = nullptr, * c = nullptr;
    posix_memalign(&a, 64 * 1024, a_bytes);
    posix_memalign(&b, 64 * 1024, b_bytes);
    posix_memalign(&c, 64 * 1024, c_bytes);
    printf("host a=%p b=%p c=%p  (a=%zu b=%zu c=%zu bytes)\n", a, b, c, a_bytes, b_bytes, c_bytes);

    srand(1234);
    int8_t * A = (int8_t*) a; int8_t * B = (int8_t*) b;
    for (size_t i = 0; i < a_bytes; i++) A[i] = (int8_t)(rand() % 255 - 127);
    for (size_t i = 0; i < b_bytes; i++) B[i] = (int8_t)(rand() % 255 - 127);
    memset(c, 0, c_bytes);

    try {
        xrt::device dev{0};
        xrt::xclbin xcl(xclbin);
        dev.register_xclbin(xcl);
        xrt::hw_context hw(dev, xcl.get_uuid());
        xrt::kernel k(hw, "MLIR_AIE");

        std::vector<uint32_t> insts = load_insts(insts_path.c_str());
        size_t inst_bytes = insts.size() * 4;

        uint32_t gid1 = k.group_id(1), gid3 = k.group_id(3);

        xrt::bo inst_bo(dev, inst_bytes, xrt::bo::flags::cacheable, gid1);
        memcpy(inst_bo.map(), insts.data(), inst_bytes);
        inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // userptr BOs over the host memory -> the "shared allocation" hypothesis.
        double t0 = now_ms();
        xrt::bo a_bo(hw, a, a_bytes, xrt::bo::flags::host_only, gid3);
        xrt::bo b_bo(hw, b, b_bytes, xrt::bo::flags::host_only, gid3);
        xrt::bo c_bo(hw, c, c_bytes, xrt::bo::flags::host_only, gid3);
        double t1 = now_ms();
        printf("userptr bo create: %.2fms  a.map=%p(=%p) c.map=%p(=%p)\n",
               t1 - t0, a_bo.map(), a, c_bo.map(), c);

        // run
        for (int it = 0; it < 3; it++) {
            double s0 = now_ms();
            a_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            double s1 = now_ms();
            xrt::run run(k);
            run.set_arg(0, (uint64_t) 3);
            run.set_arg(1, inst_bo);
            run.set_arg(2, (uint32_t) inst_bytes);
            run.set_arg(3, a_bo);
            run.set_arg(4, b_bo);
            run.set_arg(5, c_bo);
            run.start();
            double s2 = now_ms();
            run.wait();
            double s3 = now_ms();
            c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            double s4 = now_ms();
            printf("it%d sync_in=%.2f start=%.2f wait=%.2f sync_out=%.2f total=%.2fms\n",
                   it, s1 - s0, s2 - s1, s3 - s2, s4 - s3, s4 - s0);
        }

        // verify a few outputs against CPU reference
        int32_t * C = (int32_t*) c;
        int bad = 0, checked = 0;
        for (int m = 0; m < 2; m++) {
            for (int n = 0; n < N; n += 211) {
                int64_t acc = 0;
                for (int kk = 0; kk < K; kk++) acc += (int)A[(size_t) m * K + kk] * (int)B[(size_t) kk * N + n];
                int32_t got = C[(size_t) m * N + n];
                checked++;
                if (got != (int32_t) acc) {
                    if (bad < 5) printf("  mismatch m=%d n=%d got=%d exp=%lld\n", m, n, got, (long long) acc);
                    bad++;
                }
            }
        }
        printf("verify: %d/%d checked, %d mismatches\n", checked - bad, checked, bad);
        printf("RESULT: XRT userptr host BO feeds XDNA kernel %s\n", bad == 0 ? "IN PLACE (correct)" : "INCORRECTLY");
    } catch (const std::exception & ex) {
        printf("[FAIL] %s\n", ex.what());
        return 1;
    }
    return 0;
}
