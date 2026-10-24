// q8_scale_smoke.cpp -- verify the scale-aware Q8_0 full-K kernel (step 5).
//
// Native XRT only (no IRON runtime). Packs A/B with per-32-block f32 scales into
// the tile-major layout the kernel expects, runs one submit over the full K, and
// compares against a CPU float64 Q8_0 reference.
//
// Build:
//   g++ -O2 -std=c++17 q8_scale_smoke.cpp -o q8_scale_smoke \
//       -I/workspace/llama.cpp/ggml/src/ggml-xdna/stub -lxrt_coreutil
//
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

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
    const char * xclbin = (argc > 1) ? argv[1] : "/tmp/q8s/design.xclbin";
    const char * insts_path = (argc > 2) ? argv[2] : "/tmp/q8s/design.insts.bin";
    int M = (argc > 3) ? atoi(argv[3]) : 64;
    int K = (argc > 4) ? atoi(argv[4]) : 1024;
    int N = (argc > 5) ? atoi(argv[5]) : 64;
    const int m = 64, n = 64;               // tile sizes
    const int KT = 32;                      // K-block
    const int nb = K / KT;
    if (M % m || N % n || K % KT) { fprintf(stderr, "bad shape\n"); return 1; }

    int a_tile = m * KT + m * 4;
    int b_tile = KT * n + n * 4;
    int a_tiles = (M / m) * nb;
    int b_tiles = (N / n) * nb;

    std::vector<int8_t> A((size_t)M * K), B((size_t)K * N);
    std::vector<float> As((size_t)M * nb), Bs((size_t)nb * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = (int8_t)((int)(i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < B.size(); i++) B[i] = (int8_t)((int)(i * 11 + 5) % 255 - 127);
    int uniform = (argc > 6) ? atoi(argv[6]) : 0;
    for (size_t i = 0; i < As.size(); i++) As[i] = uniform ? 1.0f : 0.001f * (float)((i % 17) + 1);
    for (size_t i = 0; i < Bs.size(); i++) Bs[i] = uniform ? 1.0f : 0.002f * (float)((i % 13) + 1);

    // pack A: tile (mt,blk) = [m*KT int8][m f32]
    std::vector<int8_t> Ap((size_t)a_tiles * a_tile, 0);
    for (int mt = 0; mt < M / m; mt++)
        for (int blk = 0; blk < nb; blk++) {
            int8_t * t = &Ap[(size_t)(mt * nb + blk) * a_tile];
            for (int i = 0; i < m; i++)
                for (int l = 0; l < KT; l++)
                    t[i * KT + l] = A[(size_t)(mt * m + i) * K + blk * KT + l];
            float * ts = (float *)(t + m * KT);
            for (int i = 0; i < m; i++) ts[i] = As[(size_t)(mt * m + i) * nb + blk];
        }

    // pack B: tile (nt,blk) = [KT*n int8][n f32]
    std::vector<int8_t> Bp((size_t)b_tiles * b_tile, 0);
    for (int nt = 0; nt < N / n; nt++)
        for (int blk = 0; blk < nb; blk++) {
            int8_t * t = &Bp[(size_t)(nt * nb + blk) * b_tile];
            for (int l = 0; l < KT; l++)
                for (int j = 0; j < n; j++)
                    t[l * n + j] = B[(size_t)(blk * KT + l) * N + nt * n + j];
            float * ts = (float *)(t + KT * n);
            for (int j = 0; j < n; j++) ts[j] = Bs[(size_t)blk * N + nt * n + j];
        }

    // CPU float64 reference
    std::vector<double> ref((size_t)M * N, 0.0);
    for (int mm = 0; mm < M; mm++)
        for (int nn = 0; nn < N; nn++) {
            double acc = 0.0;
            for (int blk = 0; blk < nb; blk++) {
                int64_t psum = 0;
                for (int l = 0; l < KT; l++)
                    psum += (int32_t)A[(size_t)mm * K + blk * KT + l] * (int32_t)B[(size_t)(blk * KT + l) * N + nn];
                acc += (double)psum * As[(size_t)mm * nb + blk] * Bs[(size_t)blk * N + nn];
            }
            ref[(size_t)mm * N + nn] = acc;
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

    xrt::bo a_bo(device, Ap.size(), xrt::bo::flags::host_only, kernel.group_id(3));
    xrt::bo b_bo(device, Bp.size(), xrt::bo::flags::host_only, kernel.group_id(3));
    xrt::bo c_bo(device, (size_t)M * N * 4, xrt::bo::flags::host_only, kernel.group_id(3));

    std::memcpy(a_bo.map(), Ap.data(), Ap.size());
    std::memcpy(b_bo.map(), Bp.data(), Bp.size());
    std::memset(c_bo.map(), 0, (size_t)M * N * 4);
    a_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run(kernel);
    run.set_arg(0, (uint64_t)3);
    run.set_arg(1, inst_bo);
    run.set_arg(2, (uint32_t)inst_bytes);
    run.set_arg(3, a_bo);
    run.set_arg(4, b_bo);
    run.set_arg(5, c_bo);
    run.start();
    auto state = run.wait();

    c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float * C = (const float *)c_bo.map();

    double max_abs = 0.0, max_rel = 0.0;
    long bad = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        double d = (double)C[i] - ref[i];
        double ad = d < 0 ? -d : d;
        if (ad > max_abs) max_abs = ad;
        double denom = ref[i] < 0 ? -ref[i] : ref[i];
        if (denom > 1e-6) { double r = ad / denom; if (r > max_rel) max_rel = r; }
        if (ad > 1e-3 * (denom > 1 ? denom : 1)) bad++;
    }
    fprintf(stderr, "[q8s] M=%d K=%d N=%d state=%d max_abs=%.6g max_rel=%.3e bad=%ld/%d\n",
            M, K, N, (int)state, max_abs, max_rel, bad, M * N);
    printf("Q8_SCALE %s M=%d K=%d N=%d state=%d max_rel=%.3e bad=%ld/%d\n",
           bad == 0 ? "PASS" : "FAIL", M, K, N, (int)state, max_rel, bad, M * N);
    return bad == 0 ? 0 : 2;
}
