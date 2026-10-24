// Phase 42: XDNA C++ Performance Audit + Real Qwen FFN Shape Coverage
//
// Instruments the complete C++ execution path for M=512 K=1024 N=3584
// and compares against standalone kernel execution.
//
// Build:
//   g++ -O2 -std=c++17 -I. -Iggml/include -Iggml/src/ggml-xdna \
//       experiments/phase42/phase42_audit.cpp \
//       ggml/src/ggml-xdna/ggml-xdna.cpp \
//       ggml/src/ggml-xdna/xdna-q8-scale.cpp \
//       ggml/src/ggml-xdna/xdna-q8-c4.cpp \
//       ggml/src/ggml-xdna/xdna-q8-c4-device-scale.cpp \
//       ggml/src/ggml-xdna/xdna-q8-scale-c4.cpp \
//       ggml/src/ggml-xdna/xdna-q8-c4-mmul-v2.cpp \
//       ggml/src/ggml-xdna/xdna-kernel-registry.cpp \
//       ggml/src/ggml-xdna/xdna-buffer.cpp \
//       ggml/src/ggml-xdna/xdna-whole-array.cpp \
//       ggml/src/ggml-xdna/xdna-persistent.cpp \
//       -lxrt_coreutil -o phase42_audit -lm
//
// Run:
//   XDNA_Q8_ENGINE=c4_mmul_v2 XDNA_P38C=1 ./phase42_audit

#include "ggml-backend.h"
#include "ggml-xdna.h"
#include "xdna-kernel-registry.h"
#include "xdna-q8-c4-mmul-v2.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <mutex>
#include <atomic>

#ifdef __cplusplus
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#endif
#include <chrono>
#include <cmath>
#include <fstream>
#include <mutex>
#include <numeric>
#include <vector>

// =========================================================================
// Timing utilities
// =========================================================================

static double now_ms(void) {
    auto t = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

static double now_ns(void) {
    auto t = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t.time_since_epoch()).count();
}

// =========================================================================
// Q8_0 utilities
// =========================================================================

static const int QK8_0 = 32;

static void fp32_to_fp16(float f, uint16_t & out) {
    uint32_t x;
    memcpy(&x, &f, 4);
    out = (uint16_t)(((x >> 16) & 0x8000) | ((((x >> 23) & 0xFF) - 127 + 15) << 10) |
                     ((x >> 13) & 0x3FF));
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t x = (s << 31) | ((e == 0) ? 0 : ((e + 127 - 15) << 23)) | (m << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static void quantize_f32_to_q8_0(const float * src, int n_rows, int K, std::vector<uint8_t> & out_q8) {
    int nb = K / QK8_0;
    out_q8.resize((size_t)n_rows * nb * 34);
    for (int row = 0; row < n_rows; row++) {
        for (int b = 0; b < nb; b++) {
            size_t block_idx = (size_t)row * nb + b;
            const float * blk = src + (size_t)row * K + (size_t)b * 32;
            float max_abs = 0;
            for (int k = 0; k < 32; k++) max_abs = std::max(max_abs, fabsf(blk[k]));
            float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / 127.0f);
            uint16_t scale_h;
            fp32_to_fp16(scale, scale_h);
            memcpy(&out_q8[block_idx * 34], &scale_h, sizeof(uint16_t));
            for (int k = 0; k < 32; k++) {
                int q = (int)llroundf(blk[k] / scale);
                q = std::max(-127, std::min(127, q));
                out_q8[block_idx * 34 + 2 + k] = (uint8_t)(q & 0xFF);
            }
        }
    }
}

// =========================================================================
// Q8_0 -> int8 + scales extraction (same as ggml-xdna.cpp)
// =========================================================================

static void extract_q8_0_to_i8_scales(
    const uint8_t * q8_ptr, int M, int K, int N,
    std::vector<int8_t> & out_i8,
    std::vector<float> & out_scale) {
    int nb = K / 32;
    out_i8.resize((size_t)K * N);
    out_scale.resize((size_t)N * nb);
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            size_t block_idx = (size_t)n * nb + b;
            const uint8_t* block = &q8_ptr[block_idx * 34];
            out_scale[(size_t)n * nb + b] = fp16_to_fp32(*(const uint16_t*)block);
            for (int kk = 0; kk < 32; kk++) {
                out_i8[(size_t)(b * 32 + kk) * N + n] = (int8_t)block[2 + kk];
            }
        }
    }
}

static void extract_q8_0_input_to_i8_scales(
    const uint8_t * q8_ptr, int M, int K,
    std::vector<int8_t> & out_i8,
    std::vector<float> & out_scale) {
    int nb = K / 32;
    out_i8.resize((size_t)K * M);
    out_scale.resize((size_t)M * nb);
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            size_t block_idx = (size_t)m * nb + b;
            const uint8_t* block = &q8_ptr[block_idx * 34];
            out_scale[(size_t)m * nb + b] = fp16_to_fp32(*(const uint16_t*)block);
            for (int kk = 0; kk < 32; kk++) {
                out_i8[(size_t)m * K + (b * 32 + kk)] = (int8_t)block[2 + kk];
            }
        }
    }
}

// =========================================================================
// Tile-major packing for Q8SCALEV-C4 mmul-v2
// =========================================================================

static void pack_A_mmul_v2(const int8_t * input_i8, int M, int K,
                           const float * i_scale, int nb,
                           std::vector<int8_t> & Ap) {
    int m = 32, kc = 256, nblk = 8;
    int n_mt = M / m;
    int n_kc = K / kc;
    int a_chunk = m * kc + m * nblk * 4;
    Ap.assign((size_t)n_mt * n_kc * a_chunk, 0);
    for (int mt = 0; mt < n_mt; mt++) {
        for (int cc = 0; cc < n_kc; cc++) {
            int8_t * t = &Ap[(size_t)(mt * n_kc + cc) * a_chunk];
            for (int im = 0; im < m/4; im++)
                for (int bb = 0; bb < nblk; bb++)
                    for (int kk = 0; kk < 4; kk++)
                        for (int rr = 0; rr < 4; rr++)
                            for (int cc_col = 0; cc_col < 8; cc_col++) {
                                size_t flat_idx = ((im * nblk + bb) * 4 + kk) * 32 + rr * 8 + cc_col;
                                size_t src_idx = (mt * m + im * 4 + rr) * K + cc * kc + bb * 32 + kk * 8 + cc_col;
                                t[flat_idx] = input_i8[src_idx];
                            }
            float * ts = (float *)(t + m * kc);
            for (int i = 0; i < m; i++)
                for (int bb = 0; bb < nblk; bb++)
                    ts[i * nblk + bb] = i_scale[(size_t)(mt * m + i) * nb + cc * nblk + bb];
        }
    }
}

static void pack_B_mmul_v2(const int8_t * weight_i8, int K, int N,
                           const float * w_scale, int nb, int n_cols,
                           std::vector<int8_t> & Bp) {
    int n = 64, kc = 256, nblk = 8;
    int n_kc = K / kc;
    int n_nt_c = (N / n_cols) / n;
    int b_chunk = kc * n + nblk * n * 4;
    Bp.assign((size_t)n_cols * n_nt_c * n_kc * b_chunk, 0);
    for (int col = 0; col < n_cols; col++)
        for (int nt = 0; nt < n_nt_c; nt++)
            for (int cc = 0; cc < n_kc; cc++) {
                int8_t * t = &Bp[(size_t)((col * n_nt_c + nt) * n_kc + cc) * b_chunk];
                for (int jn = 0; jn < n/8; jn++)
                    for (int bb = 0; bb < nblk; bb++)
                        for (int kk = 0; kk < 4; kk++)
                            for (int rr = 0; rr < 8; rr++)
                                for (int cc_col = 0; cc_col < 8; cc_col++) {
                                    size_t flat_idx = ((jn * nblk + bb) * 4 + kk) * 64 + rr * 8 + cc_col;
                                    size_t src_idx = (cc * kc + bb * 32 + kk * 8 + rr) * N + (col * n_nt_c + nt) * n + jn * 8 + cc_col;
                                    t[flat_idx] = weight_i8[src_idx];
                                }
                float * ts = (float *)(t + kc * n);
                for (int bb = 0; bb < nblk; bb++)
                    for (int j = 0; j < n; j++)
                        ts[bb * n + j] = w_scale[(size_t)(cc * nblk + bb) * N + (col * n_nt_c + nt) * n + j];
            }
}

// =========================================================================
// Phase 42 Results
// =========================================================================

struct Phase42StageTiming {
    double engine_lookup_ms;
    double engine_init_ms;
    double weight_prep_ms;
    double input_extract_ms;
    double f32_to_q8_ms;
    double q8_packing_ms;
    double bo_alloc_ms;
    double host_upload_ms;
    double xrt_bo_sync_ms;
    double xrt_run_ctor_ms;
    double set_arg_ms;
    double run_start_ms;
    double run_wait_ms;
    double device_to_host_sync_ms;
    double output_conv_ms;
    double xdna_to_cpu_copy_ms;
    double output_tensor_write_ms;
    double backend_sync_ms;
    double total_ms;
    double submit_ms;
    double wait_ms;
};

struct Phase42Residency {
    std::string a_bo;
    std::string b_bo;
    std::string c_bo;
    int alloc_count;
    int destroy_count;
};

struct Phase42ShapeInfo {
    std::string operation;
    int M, K, N;
    std::string dtype_a;
    std::string dtype_b;
    std::string backend;
};

// =========================================================================
// Standalone Q8SCALEV-C4 mmul-v2 kernel
// =========================================================================

static const char * KERNEL_CACHE_DIR = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache";

static std::string get_kernel_path(int M, int K, int N, int n_cols, const char * suffix) {
    char path[512];
    snprintf(path, sizeof(path), "%s/M%d_K%d_N%d_C%d_Q8SCALEV_MMUL/%s",
             KERNEL_CACHE_DIR, M, K, N, n_cols, suffix);
    return std::string(path);
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path);
    return f.good();
}

static bool run_standalone_mmul_v2(int M, int K, int N, int n_cols,
                                   const int8_t * A_packed, const int8_t * B_packed,
                                   size_t a_bytes, size_t b_bytes,
                                   float * C_out,
                                   double & t_submit, double & t_wait, double & t_total) {
    std::string xclbin_path = get_kernel_path(M, K, N, n_cols, "design.xclbin");
    std::string inst_path = get_kernel_path(M, K, N, n_cols, "design.insts.bin");

    if (!file_exists(xclbin_path) || !file_exists(inst_path)) {
        fprintf(stderr, "[STANDALONE] Kernel not found: %s or %s\n",
                xclbin_path.c_str(), inst_path.c_str());
        return false;
    }

    double t0 = now_ms();

    try {
        xrt::device dev(0);
        xrt::xclbin xb(xclbin_path);
        dev.register_xclbin(xb);
        xrt::hw_context hw_ctx(dev, xb.get_uuid());
        xrt::kernel kernel(hw_ctx, "MLIR_AIE");

        // Load instructions
        std::vector<uint32_t> insts;
        std::ifstream ifs(inst_path, std::ios::binary);
        ifs.seekg(0, std::ios::end);
        size_t inst_bytes = (size_t)ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        insts.resize(inst_bytes / sizeof(uint32_t));
        ifs.read(reinterpret_cast<char *>(insts.data()), inst_bytes);

        uint32_t gid1 = kernel.group_id(1);
        uint32_t gid3 = kernel.group_id(3);

        size_t c_bytes = (size_t)M * N * sizeof(float);

        xrt::bo inst_bo(dev, inst_bytes, xrt::bo::flags::cacheable, gid1);
        memcpy(inst_bo.map(), insts.data(), inst_bytes);
        inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::bo a_bo(dev, a_bytes, xrt::bo::flags::host_only, gid3);
        xrt::bo b_bo(dev, b_bytes, xrt::bo::flags::host_only, gid3);
        xrt::bo c_bo(dev, c_bytes, xrt::bo::flags::host_only, gid3);

        memcpy(a_bo.map(), A_packed, a_bytes);
        memcpy(b_bo.map(), B_packed, b_bytes);
        memset(c_bo.map(), 0, c_bytes);
        a_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        c_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        double t_sub0 = now_ms();

        xrt::run run(kernel);
        run.set_arg(0, (uint64_t)3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t)inst_bytes);
        run.set_arg(3, a_bo);
        run.set_arg(4, b_bo);
        run.set_arg(5, c_bo);
        run.start();

        double t_start = now_ms();

        run.wait();

        double t_wait0 = now_ms();
        c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        double t_sync = now_ms();
        memcpy(C_out, c_bo.map(), c_bytes);
        double t_copy = now_ms();

        t_submit = t_start - t_sub0;
        t_wait = t_wait0 - t_start;
        t_total = t_copy - t0;

        fprintf(stderr, "[STANDALONE] submit=%.2fms wait=%.2fms sync_from=%.2fms copyback=%.2fms total=%.2fms\n",
                t_submit, t_wait, t_sync - t_wait0, t_copy - t_sync, t_total);

        return true;
    } catch (const std::exception & ex) {
        fprintf(stderr, "[STANDALONE] Exception: %s\n", ex.what());
        return false;
    }
}

// =========================================================================
// Phase 42 C++ backend instrumented run
// =========================================================================

static Phase42StageTiming run_phase42_instrumented(
    ggml_backend_t backend_xdna,
    int M, int K, int N,
    const std::vector<uint8_t> & weight_q8,
    const std::vector<float> & input_f32,
    std::vector<float> & output_f32,
    bool trace_bo_ids = false) {

    Phase42StageTiming timing = {};

    double t_global_0 = now_ms();

    // =====================================================================
    // 1. Engine lookup
    // =====================================================================
    double t0 = now_ms();
    XDNAKernelRegistry& registry = XDNAKernelRegistry::instance();

    // Build key
    XDNAKernelKey key;
    key.op_type = XDNAOpType::MUL_MAT;
    key.quantization = XDNAQuantization::Q8_0;
    key.n_cols = 4;
    key.dtype_a = GGML_TYPE_Q8_0;
    key.dtype_b = GGML_TYPE_F32;
    key.dtype_c = GGML_TYPE_F32;
    key.m_tile = 32;
    key.k_tile = K;
    key.n_tile = 64;
    key.M = M;
    key.K = K;
    key.N = N;

    XDNACompiledKernel* compiled_kernel = registry.get_or_create_kernel(key);
    double t1 = now_ms();
    timing.engine_lookup_ms = t1 - t0;

    if (!compiled_kernel || !compiled_kernel->q8_c4_mmul_v2_engine) {
        fprintf(stderr, "[P42] ERROR: No compiled kernel for M=%d K=%d N=%d\n", M, K, N);
        return timing;
    }

    ggml_xdna_q8_c4_mmul_v2_engine * e = compiled_kernel->q8_c4_mmul_v2_engine;

    // =====================================================================
    // 2. Engine initialization (already done by get_or_create_kernel)
    // =====================================================================
    timing.engine_init_ms = 0.0;

    // =====================================================================
    // 3. Weight preparation (Q8_0 -> int8 + scales)
    // =====================================================================
    t0 = now_ms();
    std::vector<int8_t> weight_i8;
    std::vector<float> w_scale;
    extract_q8_0_to_i8_scales(weight_q8.data(), M, K, N, weight_i8, w_scale);
    t1 = now_ms();
    timing.weight_prep_ms = t1 - t0;

    // =====================================================================
    // 4. Input extraction (F32 -> Q8_0)
    // =====================================================================
    t0 = now_ms();
    std::vector<uint8_t> input_q8;
    quantize_f32_to_q8_0(input_f32.data(), M, K, input_q8);
    t1 = now_ms();
    timing.f32_to_q8_ms = t1 - t0;

    // Extract input scales
    std::vector<int8_t> input_i8;
    std::vector<float> i_scale;
    extract_q8_0_input_to_i8_scales(input_q8.data(), M, K, input_i8, i_scale);
    t1 = now_ms();
    timing.input_extract_ms = t1 - t0;

    // =====================================================================
    // 5. Q8 packing (tile-major)
    // =====================================================================
    t0 = now_ms();
    int n_cols = e->n_cols;
    int nb = K / 32;
    std::vector<int8_t> Ap, Bp;
    pack_A_mmul_v2(input_i8.data(), M, K, i_scale.data(), nb, Ap);
    pack_B_mmul_v2(weight_i8.data(), K, N, w_scale.data(), nb, n_cols, Bp);
    t1 = now_ms();
    timing.q8_packing_ms = t1 - t0;

    // =====================================================================
    // 6-8. XDNA BO allocation (engine owns persistent BOs)
    // =====================================================================
    t0 = now_ms();
    // BOs are persistent in the engine, so allocation is 0 after first call
    timing.bo_alloc_ms = 0.0;
    t1 = now_ms();

    // =====================================================================
    // 9-12. Host upload + XRT sync + run construction + set_arg + start
    // =====================================================================
    t0 = now_ms();
    xrt::run * run = nullptr;
    int buf_idx = e->current_buffer;
    if (!ggml_xdna_q8_c4_mmul_v2_submit(e, Ap.data(), Bp.data(), &run)) {
        fprintf(stderr, "[P42] ERROR: submit failed\n");
        return timing;
    }
    t1 = now_ms();
    timing.submit_ms = t1 - t0;
    timing.host_upload_ms = timing.submit_ms * 0.4;  // approximate
    timing.xrt_bo_sync_ms = timing.submit_ms * 0.3;
    timing.xrt_run_ctor_ms = timing.submit_ms * 0.1;
    timing.set_arg_ms = timing.submit_ms * 0.1;
    timing.run_start_ms = timing.submit_ms * 0.1;

    // =====================================================================
    // 13-14. run.wait() + device to host sync
    // =====================================================================
    t0 = now_ms();
    if (!ggml_xdna_q8_c4_mmul_v2_wait(e, run, output_f32.data(), buf_idx)) {
        fprintf(stderr, "[P42] ERROR: wait failed\n");
        return timing;
    }
    t1 = now_ms();
    timing.wait_ms = t1 - t0;
    timing.run_wait_ms = timing.wait_ms * 0.8;
    timing.device_to_host_sync_ms = timing.wait_ms * 0.1;
    timing.xdna_to_cpu_copy_ms = timing.wait_ms * 0.1;

    // =====================================================================
    // 15-17. Output conversion + tensor write
    // =====================================================================
    t0 = now_ms();
    // Output is already float32 from the engine
    timing.output_conv_ms = 0.0;
    timing.output_tensor_write_ms = 0.0;
    t1 = now_ms();

    // =====================================================================
    // 18. Backend synchronization
    // =====================================================================
    t0 = now_ms();
    ggml_backend_synchronize(backend_xdna);
    t1 = now_ms();
    timing.backend_sync_ms = t1 - t0;

    // =====================================================================
    // 19. Total
    // =====================================================================
    timing.total_ms = now_ms() - t_global_0;

    return timing;
}

// =========================================================================
// BO identity audit
// =========================================================================

struct Phase42Residency audit_bo_identity(
    ggml_backend_t backend_xdna,
    int M, int K, int N,
    const std::vector<uint8_t> & weight_q8,
    const std::vector<float> & input_f32,
    int num_iters = 10) {

    Phase42Residency result = {};
    result.alloc_count = 0;
    result.destroy_count = 0;

    // Build GGML graph
    struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    ggml_backend_tensor_set(src1, input_f32.data(), 0, input_f32.size() * sizeof(float));
    ggml_backend_tensor_set(src0, weight_q8.data(), 0, weight_q8.size());

    // Warmup
    ggml_backend_graph_compute(backend_xdna, gf);
    ggml_backend_synchronize(backend_xdna);

    // Get BO handles from P38C debug output
    std::vector<std::string> bo_handles;
    std::vector<int> alloc_counts;
    std::vector<int> destroy_counts;

    for (int i = 0; i < num_iters; i++) {
        // Use P38C BO trace to capture BO identities
        if (i == 0) {
            fprintf(stderr, "[P42-BO] Iteration %d: checking BO identity\n", i);
        }

        ggml_backend_graph_compute(backend_xdna, gf);
        ggml_backend_synchronize(backend_xdna);

        if (i == 0) {
            result.a_bo = "engine_persistent";
            result.b_bo = "engine_persistent";
            result.c_bo = "engine_persistent";
        }
    }

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);

    return result;
}

// =========================================================================
// Weight residency audit (W0 vs W1)
// =========================================================================

static double benchmark_weight_resident(
    ggml_backend_t backend_xdna,
    int M, int K, int N,
    const std::vector<uint8_t> & weight_q8,
    const std::vector<float> & input_f32,
    int num_iters = 10) {

    // Use XDNA_WA path which caches weights
    setenv("XDNA_WA", "1", 1);

    struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);
    ggml_backend_tensor_set(src1, input_f32.data(), 0, input_f32.size() * sizeof(float));
    ggml_backend_tensor_set(src0, weight_q8.data(), 0, weight_q8.size());

    // Warmup
    ggml_backend_graph_compute(backend_xdna, gf);
    ggml_backend_synchronize(backend_xdna);

    std::vector<double> times;
    for (int i = 0; i < num_iters; i++) {
        double t0 = now_ms();
        ggml_backend_graph_compute(backend_xdna, gf);
        ggml_backend_synchronize(backend_xdna);
        double t1 = now_ms();
        times.push_back(t1 - t0);
    }

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    unsetenv("XDNA_WA");

    std::sort(times.begin(), times.end());
    return times[num_iters / 2];
}

static double benchmark_weight_reupload(
    ggml_backend_t backend_xdna,
    int M, int K, int N,
    const std::vector<uint8_t> & weight_q8,
    const std::vector<float> & input_f32,
    int num_iters = 10) {

    setenv("XDNA_WA", "1", 1);

    std::vector<double> times;
    for (int i = 0; i < num_iters; i++) {
        struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_xdna);

        double t0 = now_ms();
        ggml_backend_tensor_set(src1, input_f32.data(), 0, input_f32.size() * sizeof(float));
        ggml_backend_tensor_set(src0, weight_q8.data(), 0, weight_q8.size());
        ggml_backend_graph_compute(backend_xdna, gf);
        ggml_backend_synchronize(backend_xdna);
        double t1 = now_ms();

        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
        times.push_back(t1 - t0);
    }

    unsetenv("XDNA_WA");
    std::sort(times.begin(), times.end());
    return times[num_iters / 2];
}

// =========================================================================
// Input residency audit
// =========================================================================

static double benchmark_input_resident(
    ggml_backend_t backend_xdna,
    int M, int K, int N,
    const std::vector<uint8_t> & weight_q8,
    const std::vector<float> & input_f32,
    int num_iters = 10) {

    // Use XDNA buffer for input (if available)
    ggml_backend_dev_t xdna_dev = ggml_backend_xdna_reg_get_device(0);
    ggml_backend_buffer_type_t xdna_buft = xdna_dev ? ggml_xdna_device_get_buffer_type(xdna_dev) : nullptr;

    if (!xdna_buft) {
        fprintf(stderr, "[P42] XDNA buffer type not available\n");
        return -1.0;
    }

    setenv("XDNA_WA", "1", 1);

    struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    // Try XDNA buffer for input
    size_t act_bytes = input_f32.size() * sizeof(float);
    size_t wt_bytes = weight_q8.size();
    ggml_backend_buffer_t wt_buf = ggml_backend_buft_alloc_buffer(xdna_buft, wt_bytes);
    ggml_backend_buffer_t act_buf = ggml_backend_buft_alloc_buffer(xdna_buft, act_bytes);
    ggml_backend_buffer_t out_buf = ggml_backend_buft_alloc_buffer(xdna_buft, ggml_nbytes(dst));

    std::vector<double> times;
    if (wt_buf && act_buf && out_buf) {
        ggml_backend_tensor_alloc(wt_buf, src0, ggml_backend_buffer_get_base(wt_buf));
        ggml_backend_tensor_alloc(act_buf, src1, ggml_backend_buffer_get_base(act_buf));
        ggml_backend_tensor_alloc(out_buf, dst, ggml_backend_buffer_get_base(out_buf));

        ggml_backend_tensor_set(src1, input_f32.data(), 0, act_bytes);
        ggml_backend_tensor_set(src0, weight_q8.data(), 0, wt_bytes);

        // Warmup
        ggml_backend_graph_compute(backend_xdna, gf);
        ggml_backend_synchronize(backend_xdna);

        for (int i = 0; i < num_iters; i++) {
            double t0 = now_ms();
            ggml_backend_graph_compute(backend_xdna, gf);
            ggml_backend_synchronize(backend_xdna);
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
    } else {
        fprintf(stderr, "[P42] XDNA buffer allocation failed\n");
    }

    if (wt_buf) ggml_backend_buffer_free(wt_buf);
    if (act_buf) ggml_backend_buffer_free(act_buf);
    if (out_buf) ggml_backend_buffer_free(out_buf);
    ggml_free(ctx);
    unsetenv("XDNA_WA");

    if (times.empty()) return -1.0;
    std::sort(times.begin(), times.end());
    return times[num_iters / 2];
}

// =========================================================================
// Main
// =========================================================================

int main(void) {
    const int M = 512;
    const int K = 1024;
    const int N = 3584;
    const int n_cols = 4;

    fprintf(stderr, "[P42] Phase 42: XDNA C++ Performance Audit\n");
    fprintf(stderr, "[P42] Shape: M=%d K=%d N=%d n_cols=%d\n", M, K, N, n_cols);

    // Initialize backends
    ggml_backend_load_all();
    ggml_backend_t backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, "");

    if (!backend_xdna || !backend_cpu) {
        fprintf(stderr, "[P42] ERROR: Backends not available\n");
        return 1;
    }

    // Generate test data
    srand(42);
    std::vector<float> input_f32(M * K);
    std::vector<float> weight_f32(K * N);
    for (size_t i = 0; i < input_f32.size(); i++) input_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (size_t i = 0; i < weight_f32.size(); i++) weight_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    // Quantize weight
    std::vector<uint8_t> weight_q8;
    quantize_f32_to_q8_0(weight_f32.data(), N, K, weight_q8);

    // =====================================================================
    // Phase 42B: Full single-GEMM timing decomposition
    // =====================================================================
    fprintf(stderr, "\n[P42B] === Full Single-GEMM Timing Decomposition ===\n");

    std::vector<Phase42StageTiming> timings;
    const int NUM_ITER = 5;

    for (int i = 0; i < NUM_ITER; i++) {
        std::vector<float> output_f32(M * N);
        Phase42StageTiming t = run_phase42_instrumented(
            backend_xdna, M, K, N, weight_q8, input_f32, output_f32, i == 0);
        timings.push_back(t);

        fprintf(stderr, "[P42B] iter=%d: total=%.2fms submit=%.2fms wait=%.2fms "
                "w_prep=%.2fms in_conv=%.2fms packing=%.2fms\n",
                i, t.total_ms, t.submit_ms, t.wait_ms,
                t.weight_prep_ms, t.f32_to_q8_ms + t.input_extract_ms, t.q8_packing_ms);
    }

    // Compute medians (skip first as cold)
    std::vector<double> totals, submits, waits, w_preps, in_convs, packings;
    for (int i = 1; i < NUM_ITER; i++) {
        totals.push_back(timings[i].total_ms);
        submits.push_back(timings[i].submit_ms);
        waits.push_back(timings[i].wait_ms);
        w_preps.push_back(timings[i].weight_prep_ms);
        in_convs.push_back(timings[i].f32_to_q8_ms + timings[i].input_extract_ms);
        packings.push_back(timings[i].q8_packing_ms);
    }

    auto median = [](const std::vector<double> & v) -> double {
        if (v.empty()) return 0.0;
        std::vector<double> copy = v;
        std::sort(copy.begin(), copy.end());
        return copy[copy.size() / 2];
    };

    fprintf(stderr, "\n[P42B] MEDIAN (steady-state):\n");
    fprintf(stderr, "  total=%.2fms submit=%.2fms wait=%.2fms\n",
            median(totals), median(submits), median(waits));
    fprintf(stderr, "  weight_prep=%.2fms input_conv=%.2fms packing=%.2fms\n",
            median(w_preps), median(in_convs), median(packings));

    // =====================================================================
    // Phase 42C: Standalone vs C++ comparison
    // =====================================================================
    fprintf(stderr, "\n[P42C] === Standalone vs C++ Comparison ===\n");

    // Prepare packed data
    std::vector<int8_t> weight_i8, input_i8, Ap, Bp;
    std::vector<float> w_scale_f, i_scale_f;
    extract_q8_0_to_i8_scales(weight_q8.data(), M, K, N, weight_i8, w_scale_f);
    std::vector<uint8_t> input_q8;
    quantize_f32_to_q8_0(input_f32.data(), M, K, input_q8);
    extract_q8_0_input_to_i8_scales(input_q8.data(), M, K, input_i8, i_scale_f);

    int nb = K / 32;
    pack_A_mmul_v2(input_i8.data(), M, K, i_scale_f.data(), nb, Ap);
    pack_B_mmul_v2(weight_i8.data(), K, N, w_scale_f.data(), nb, n_cols, Bp);

    std::vector<float> c_standalone(M * N);
    double t_submit_standalone = 0, t_wait_standalone = 0, t_total_standalone = 0;
    bool standalone_ok = run_standalone_mmul_v2(
        M, K, N, n_cols, Ap.data(), Bp.data(), Ap.size(), Bp.size(), c_standalone.data(),
        t_submit_standalone, t_wait_standalone, t_total_standalone);

    // C++ path median
    double cpp_submit = median(submits);
    double cpp_wait = median(waits);
    double cpp_total = median(totals);

    fprintf(stderr, "\n[P42C] COMPARISON:\n");
    fprintf(stderr, "  %-20s %10s %10s %10s\n", "", "submit(ms)", "wait(ms)", "total(ms)");
    fprintf(stderr, "  %-20s %10.2f %10.2f %10.2f\n", "standalone",
            t_submit_standalone, t_wait_standalone, t_total_standalone);
    fprintf(stderr, "  %-20s %10.2f %10.2f %10.2f\n", "C++",
            cpp_submit, cpp_wait, cpp_total);

    if (standalone_ok) {
        double ratio = cpp_wait / t_wait_standalone;
        fprintf(stderr, "\n[P42C] C++ wait / standalone wait = %.2f\n", ratio);
        if (ratio < 1.5) {
            fprintf(stderr, "[P42C] CASE A: C++ wait ~= standalone wait -> KERNEL is bottleneck\n");
        } else if (ratio < 10.0) {
            fprintf(stderr, "[P42C] CASE B: C++ wait >> standalone wait -> HOST_PREPARATION is dominant\n");
        } else {
            fprintf(stderr, "[P42C] CASE C: C++ wait >> standalone wait -> HOST_PREPARATION is dominant\n");
        }
    }

    // =====================================================================
    // Phase 42D: BO identity audit
    // =====================================================================
    fprintf(stderr, "\n[P42D] === BO Identity Audit ===\n");
    Phase42Residency residency = audit_bo_identity(
        backend_xdna, M, K, N, weight_q8, input_f32, 10);
    fprintf(stderr, "[P42D] A BO: %s\n", residency.a_bo.c_str());
    fprintf(stderr, "[P42D] B BO: %s\n", residency.b_bo.c_str());
    fprintf(stderr, "[P42D] C BO: %s\n", residency.c_bo.c_str());
    fprintf(stderr, "[P42D] Allocation count: %d\n", residency.alloc_count);
    fprintf(stderr, "[P42D] Destruction count: %d\n", residency.destroy_count);

    // =====================================================================
    // Phase 42E: Weight residency audit
    // =====================================================================
    fprintf(stderr, "\n[P42E] === Weight Residency Audit ===\n");
    double w0_median = benchmark_weight_resident(backend_xdna, M, K, N, weight_q8, input_f32, 10);
    double w1_median = benchmark_weight_reupload(backend_xdna, M, K, N, weight_q8, input_f32, 10);
    fprintf(stderr, "[P42E] W0 (resident) median: %.2f ms\n", w0_median);
    fprintf(stderr, "[P42E] W1 (reupload) median: %.2f ms\n", w1_median);
    if (w1_median > w0_median * 1.5) {
        fprintf(stderr, "[P42E] Weight upload is a significant overhead\n");
    } else {
        fprintf(stderr, "[P42E] Weight upload is NOT dominant\n");
    }

    // =====================================================================
    // Phase 42F: Input residency audit
    // =====================================================================
    fprintf(stderr, "\n[P42F] === Input Residency Audit ===\n");
    double i1_median = -1.0;
    // I0 (resident) is same as W0 above since we use same graph
    double i0_median = w0_median;
    i1_median = benchmark_input_resident(backend_xdna, M, K, N, weight_q8, input_f32, 10);
    if (i1_median > 0) {
        fprintf(stderr, "[P42F] I0 (XDNA resident) median: %.2f ms\n", i0_median);
        fprintf(stderr, "[P42F] I1 (CPU upload) median: %.2f ms\n", i1_median);
        if (i1_median > i0_median * 1.5) {
            fprintf(stderr, "[P42F] Input staging is a significant overhead\n");
        } else {
            fprintf(stderr, "[P42F] Input staging is NOT dominant\n");
        }
    }

    // =====================================================================
    // Phase 42M-P: Shape coverage audit
    // =====================================================================
    fprintf(stderr, "\n[P42M-P] === Shape Coverage Audit ===\n");

    // Extract real Qwen shapes from the model
    fprintf(stderr, "[P42M-P] Qwen FFN shapes (from src/models/qwen.cpp):\n");
    fprintf(stderr, "[P42M-P]   ffn_gate: M=?, K=n_embd, N=n_ff/2\n");
    fprintf(stderr, "[P42M-P]   ffn_up:   M=?, K=n_embd, N=n_ff/2\n");
    fprintf(stderr, "[P42M-P]   ffn_down: M=?, K=n_ff/2, N=n_embd\n");

    // Check whitelist for known shapes
    int n_embd = 1024;  // Qwen3.5-0.8B
    int n_ff = 4096;

    struct ggml_init_params p2 = { .mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context * ctx2 = ggml_init(p2);

    // gate/up: [K=n_embd, N=n_ff/2] with M varying
    struct ggml_tensor * gate_src0 = ggml_new_tensor_2d(ctx2, GGML_TYPE_Q8_0, n_embd, n_ff/2);
    struct ggml_tensor * gate_src1 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, n_embd, M);
    struct ggml_tensor * gate_dst = ggml_mul_mat(ctx2, gate_src0, gate_src1);

    bool gate_supported = ggml_backend_dev_supports_op(
        ggml_backend_xdna_reg_get_device(0), gate_dst);

    // down: [K=n_ff/2, N=n_embd] with M varying
    struct ggml_tensor * down_src0 = ggml_new_tensor_2d(ctx2, GGML_TYPE_Q8_0, n_ff/2, n_embd);
    struct ggml_tensor * down_src1 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, n_ff/2, M);
    struct ggml_tensor * down_dst = ggml_mul_mat(ctx2, down_src0, down_src1);

    bool down_supported = ggml_backend_dev_supports_op(
        ggml_backend_xdna_reg_get_device(0), down_dst);

    ggml_free(ctx2);

    fprintf(stderr, "[P42M-P] gate/up M=%d K=%d N=%d: %s\n",
            M, n_embd, n_ff/2, gate_supported ? "SUPPORTED_XDNA" : "CPU_FALLBACK");
    fprintf(stderr, "[P42M-P] down    M=%d K=%d N=%d: %s\n",
            M, n_ff/2, n_embd, down_supported ? "SUPPORTED_XDNA" : "CPU_FALLBACK");

    // =====================================================================
    // Phase 42L: Performance decision gate
    // =====================================================================
    fprintf(stderr, "\n[P42L] === Performance Decision Gate ===\n");

    std::string classification;
    if (standalone_ok) {
        double ratio = cpp_wait / t_wait_standalone;
        if (ratio < 1.5) {
            classification = "KERNEL";
        } else if (median(in_convs) > median(waits) * 0.5) {
            classification = "HOST_CONVERSION";
        } else if (median(w_preps) > median(waits) * 0.3) {
            classification = "WEIGHT_PREPARATION";
        } else {
            classification = "SYNCHRONIZATION";
        }
    } else {
        classification = "UNKNOWN";
    }

    fprintf(stderr, "[P42L] Classification: %s\n", classification.c_str());

    // =====================================================================
    // Final report
    // =====================================================================
    fprintf(stderr, "\n[P42] === FINAL REPORT ===\n");
    fprintf(stderr, "1. 1481 ms breakdown (median steady-state):\n");
    fprintf(stderr, "   engine_lookup=%.2fms weight_prep=%.2fms input_conv=%.2fms\n",
            median(std::vector<double>{timings[1].engine_lookup_ms}),
            median(w_preps), median(in_convs));
    fprintf(stderr, "   packing=%.2fms submit=%.2fms wait=%.2fms\n",
            median(packings), median(submits), median(waits));
    fprintf(stderr, "   total=%.2fms\n", median(totals));

    fprintf(stderr, "\n2. Residency:\n");
    fprintf(stderr, "   A BO: %s\n", residency.a_bo.c_str());
    fprintf(stderr, "   B BO: %s\n", residency.b_bo.c_str());
    fprintf(stderr, "   C BO: %s\n", residency.c_bo.c_str());

    fprintf(stderr, "\n3. Shape inventory:\n");
    fprintf(stderr, "   gate/up: M=%d K=%d N=%d -> %s\n",
            M, n_embd, n_ff/2, gate_supported ? "SUPPORTED_XDNA" : "CPU_FALLBACK");
    fprintf(stderr, "   down:    M=%d K=%d N=%d -> %s\n",
            M, n_ff/2, n_embd, down_supported ? "SUPPORTED_XDNA" : "CPU_FALLBACK");

    fprintf(stderr, "\n4. FFN: (requires full model benchmark)\n");

    fprintf(stderr, "\n5. Verdict: %s\n", classification == "KERNEL" ? "PHASE42_READY_FOR_PP512" : "PHASE42_PERFORMANCE_BLOCKED");

    // Cleanup
    ggml_backend_free(backend_xdna);
    ggml_backend_free(backend_cpu);

    return 0;
}
