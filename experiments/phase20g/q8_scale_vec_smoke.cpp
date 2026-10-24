// q8_scale_vec_smoke.cpp -- verify the vectorized scale-aware Q8_0 kernel.
//
// Layout (matches gen_q8_scale_vec.py):
//   A: n_mt*n_kc chunks, each = [m_tile, k_chunk] int8 + [m_tile, nblk] f32
//   B: n_kc*n_nt chunks, each = [k_chunk, n_tile] int8 + [nblk, n_tile] f32
//   C: [M, N] f32
//
// Build:
//   g++ -O2 -std=c++17 q8_scale_vec_smoke.cpp -o q8_scale_vec_smoke \
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
    const char * xclbin = (argc > 1) ? argv[1] : "/tmp/q8v/design.xclbin";
    const char * insts_path = (argc > 2) ? argv[2] : "/tmp/q8v/design.insts.bin";
    int M = (argc > 3) ? atoi(argv[3]) : 64;
    int K = (argc > 4) ? atoi(argv[4]) : 1024;
    int N = (argc > 5) ? atoi(argv[5]) : 64;
    int uniform = (argc > 6) ? atoi(argv[6]) : 0;
    const int m_tile = 32, n_tile = 64, k_chunk = 256;
    const int nblk = k_chunk / 32;
    if (M % m_tile || N % n_tile || K % k_chunk) { fprintf(stderr, "bad shape\n"); return 1; }
    int n_mt = M / m_tile, n_nt = N / n_tile, n_kc = K / k_chunk;
    int nb = K / 32;

    int a_bytes = m_tile * k_chunk + m_tile * nblk * 4;
    int b_bytes = k_chunk * n_tile + nblk * n_tile * 4;

    std::vector<int8_t> A((size_t) M * K), B((size_t) K * N);
    std::vector<float> As((size_t) M * nb), Bs((size_t) nb * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = (int8_t)((int)(i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < B.size(); i++) B[i] = (int8_t)((int)(i * 11 + 5) % 255 - 127);
    for (size_t i = 0; i < As.size(); i++) As[i] = uniform ? 1.0f : 0.001f * (float)((i % 17) + 1);
    for (size_t i = 0; i < Bs.size(); i++) Bs[i] = uniform ? 1.0f : 0.002f * (float)((i % 13) + 1);

    std::vector<int8_t> Ap((size_t) n_mt * n_kc * a_bytes, 0);
    for (int mt = 0; mt < n_mt; mt++)
        for (int kc = 0; kc < n_kc; kc++) {
            int8_t * t = &Ap[(size_t)(mt * n_kc + kc) * a_bytes];
            for (int i = 0; i < m_tile; i++)
                for (int l = 0; l < k_chunk; l++)
                    t[i * k_chunk + l] = A[(size_t)(mt * m_tile + i) * K + kc * k_chunk + l];
            float * ts = (float *)(t + m_tile * k_chunk);
            for (int i = 0; i < m_tile; i++)
                for (int bb = 0; bb < nblk; bb++)
                    ts[i * nblk + bb] = As[(size_t)(mt * m_tile + i) * nb + kc * nblk + bb];
        }

    std::vector<int8_t> Bp((size_t) n_kc * n_nt * b_bytes, 0);
    for (int nt = 0; nt < n_nt; nt++)
        for (int kc = 0; kc < n_kc; kc++) {
            int8_t * t = &Bp[(size_t)(nt * n_kc + kc) * b_bytes];
            for (int l = 0; l < k_chunk; l++)
                for (int j = 0; j < n_tile; j++)
                    t[l * n_tile + j] = B[(size_t)(kc * k_chunk + l) * N + nt * n_tile + j];
            float * ts = (float *)(t + k_chunk * n_tile);
            for (int bb = 0; bb < nblk; bb++)
                for (int j = 0; j < n_tile; j++)
                    ts[bb * n_tile + j] = Bs[(size_t)(kc * nblk + bb) * N + nt * n_tile + j];
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
    int iters = (argc > 7) ? atoi(argv[7]) : 1;
    if (iters < 1) iters = 1;
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) {
        run.start();
        run.wait();
    }
    auto t1 = std::chrono::steady_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    fprintf(stderr, "[q8v] avg submit+wait = %.1f us over %d iters\n", avg_us, iters);
    int state = 4;

    c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float * C = (const float *) c_bo.map();

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
    fprintf(stderr, "[q8v] M=%d K=%d N=%d state=%d max_abs=%.6g max_rel=%.3e bad=%ld/%d\n",
            M, K, N, (int) state, max_abs, max_rel, bad, M * N);
    printf("Q8VEC %s M=%d K=%d N=%d state=%d max_rel=%.3e bad=%ld/%d\n",
           bad == 0 ? "PASS" : "FAIL", M, K, N, (int) state, max_rel, bad, M * N);
    return bad == 0 ? 0 : 2;
}
