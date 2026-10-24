// q8_mmul_smoke.cpp -- verify the matrix-unit scale-aware Q8_0 kernel.
//
// Layout (matches gen_q8_scale_mmul.py):
//   A chunk (mt,blk): (16*4) tiles of 4x8 int8, then 64 f32 scales  (2304 B)
//   B chunk (nt,blk): (4*8) tiles of 8x8 int8, then 64 f32 scales   (2304 B)
//   C: [M, N] f32
//
// Build:
//   g++ -O2 -std=c++17 q8_mmul_smoke.cpp -o q8_mmul_smoke \
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
    const char * xclbin = (argc > 1) ? argv[1] : "/tmp/q8m/design.xclbin";
    const char * insts_path = (argc > 2) ? argv[2] : "/tmp/q8m/design.insts.bin";
    int M = (argc > 3) ? atoi(argv[3]) : 64;
    int K = (argc > 4) ? atoi(argv[4]) : 1024;
    int N = (argc > 5) ? atoi(argv[5]) : 64;
    int uniform = (argc > 6) ? atoi(argv[6]) : 0;
    int iters = (argc > 7) ? atoi(argv[7]) : 1;
    const int MT = 64, NT = 64, KC = 256;
    const int NBLK = KC / 32, KA = KC / 8, KB = KC / 8;
    if (M % MT || N % NT || K % KC) { fprintf(stderr, "bad shape\n"); return 1; }
    int n_mt = M / MT, n_nt = N / NT, n_chunk = K / KC;
    int nb = K / 32;

    int a_bytes = MT * KC + MT * NBLK * 4;   // 18432
    int b_bytes = KC * NT + NBLK * NT * 4;   // 18432

    std::vector<int8_t> A((size_t) M * K), B((size_t) K * N);
    std::vector<float> As((size_t) M * nb), Bs((size_t) nb * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = (int8_t)((int)(i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < B.size(); i++) B[i] = (int8_t)((int)(i * 11 + 5) % 255 - 127);
    for (size_t i = 0; i < As.size(); i++) As[i] = uniform ? 1.0f : 0.001f * (float)((i % 17) + 1);
    for (size_t i = 0; i < Bs.size(); i++) Bs[i] = uniform ? 1.0f : 0.002f * (float)((i % 13) + 1);

    // pack A: (mt,chunk) = (16 * KA) tiles of 4x8 + 64*NBLK f32 scales
    std::vector<int8_t> Ap((size_t) n_mt * n_chunk * a_bytes, 0);
    for (int mt = 0; mt < n_mt; mt++)
        for (int ch = 0; ch < n_chunk; ch++) {
            int8_t * t = &Ap[(size_t)(mt * n_chunk + ch) * a_bytes];
            for (int i = 0; i < 16; i++)
                for (int kt = 0; kt < KA; kt++)
                    for (int rr = 0; rr < 4; rr++)
                        for (int ss = 0; ss < 8; ss++)
                            t[(i * KA + kt) * 32 + rr * 8 + ss] =
                                A[(size_t)(mt * MT + i * 4 + rr) * K + ch * KC + kt * 8 + ss];
            float * ts = (float *)(t + MT * KC);
            for (int row = 0; row < MT; row++)
                for (int bb = 0; bb < NBLK; bb++)
                    ts[row * NBLK + bb] = As[(size_t)(mt * MT + row) * nb + ch * NBLK + bb];
        }

    // pack B: (nt,chunk) = (8 * KB) tiles of 8x8 + NBLK*64 f32 scales
    std::vector<int8_t> Bp((size_t) n_nt * n_chunk * b_bytes, 0);
    for (int nt = 0; nt < n_nt; nt++)
        for (int ch = 0; ch < n_chunk; ch++) {
            int8_t * t = &Bp[(size_t)(nt * n_chunk + ch) * b_bytes];
            for (int j = 0; j < 8; j++)
                for (int kt = 0; kt < KB; kt++)
                    for (int ss = 0; ss < 8; ss++)
                        for (int tt = 0; tt < 8; tt++)
                            t[(j * KB + kt) * 64 + ss * 8 + tt] =
                                B[(size_t)(ch * KC + kt * 8 + ss) * N + nt * NT + j * 8 + tt];
            float * ts = (float *)(t + KC * NT);
            for (int bb = 0; bb < NBLK; bb++)
                for (int col = 0; col < NT; col++)
                    ts[bb * NT + col] = Bs[(size_t)(ch * NBLK + bb) * N + nt * NT + col];
        }

    std::vector<double> ref((size_t) M * N, 0.0);
    for (int mm = 0; mm < M; mm++)
        for (int nn = 0; nn < N; nn++) {
            double acc = 0.0;
            for (int blk = 0; blk < nb; blk++) {
                int64_t psum = 0;
                for (int l = 0; l < 32; l++)
                    psum += (int32_t) A[(size_t) mm * K + blk * 32 + l] * (int32_t) B[(size_t)(blk * 32 + l) * N + nn];
                acc += (double) psum * As[(size_t) mm * nb + blk] * Bs[(size_t) blk * N + nn];
            }
            ref[(size_t) mm * N + nn] = acc;
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
    xrt::bo c_bo(device, (size_t) M * N * 4, xrt::bo::flags::host_only, kernel.group_id(3));
    std::memcpy(a_bo.map(), Ap.data(), Ap.size());
    std::memcpy(b_bo.map(), Bp.data(), Bp.size());
    std::memset(c_bo.map(), 0, (size_t) M * N * 4);
    a_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run(kernel);
    run.set_arg(0, (uint64_t) 3);
    run.set_arg(1, inst_bo);
    run.set_arg(2, (uint32_t) inst_bytes);
    run.set_arg(3, a_bo);
    run.set_arg(4, b_bo);
    run.set_arg(5, c_bo);

    if (iters < 1) iters = 1;
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) { run.start(); run.wait(); }
    auto t1 = std::chrono::steady_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    fprintf(stderr, "[q8m] avg submit+wait = %.1f us over %d iters\n", avg_us, iters);

    c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float * C = (const float *) c_bo.map();
    fprintf(stderr, "[q8m] C[0..3]=%.4g %.4g %.4g %.4g ref[0..3]=%.4g %.4g %.4g %.4g\n",
            C[0], C[1], C[2], C[3], ref[0], ref[1], ref[2], ref[3]);

    double max_abs = 0.0, max_rel = 0.0;
    long bad = 0;
    for (size_t i = 0; i < (size_t) M * N; i++) {
        double d = (double) C[i] - ref[i];
        double ad = d < 0 ? -d : d;
        if (ad > max_abs) max_abs = ad;
        double denom = ref[i] < 0 ? -ref[i] : ref[i];
        if (denom > 1e-6) { double r = ad / denom; if (r > max_rel) max_rel = r; }
        if (ad > 1e-3 * (denom > 1 ? denom : 1)) bad++;
    }
    fprintf(stderr, "[q8m] M=%d K=%d N=%d max_abs=%.6g max_rel=%.3e bad=%ld/%d\n",
            M, K, N, max_abs, max_rel, bad, M * N);
    printf("Q8MMUL %s M=%d K=%d N=%d max_rel=%.3e bad=%ld/%d\n",
           bad == 0 ? "PASS" : "FAIL", M, K, N, max_rel, bad, M * N);
    return bad == 0 ? 0 : 2;
}
