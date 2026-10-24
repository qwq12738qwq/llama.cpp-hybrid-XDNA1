#pragma once

// Phase 40M: Q8_0 full-K GEMM using aie::mmul<4,8,8> with device-side per-block scales.
//
// Uses the q8_c4_mmul_v2 kernel (4 AIE columns, mmul<4,8,8> + device-side scales).
// Q8_0 per-32-block scales are preserved on device; no host-side scale folding.
// One xrt::run covers the whole K.
//
// Cache namespace: M{K}_N{N}_C{n_cols}_Q8SCALEV_MMUL

#include <cstddef>
#include <cstdint>
#include <mutex>

#ifdef __cplusplus
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <vector>

struct ggml_xdna_xrt_trace {
    uint64_t host_prep_start_ns;
    uint64_t host_prep_end_ns;
    uint64_t submit_start_ns;
    uint64_t submit_end_ns;
    uint64_t run_start_ns;
    uint64_t wait_start_ns;
    uint64_t wait_end_ns;
    uint64_t sync_end_ns;
    uint64_t copy_end_ns;
    int buf_idx;
    int run_objects;
    int submissions;
    int starts;
    int waits;
    int k_block_size;
    int expected_k_blocks;
    int actual_k_blocks;
    int q8_scale_blocks;
    int k_chunks;
    int m_tiles;
    int n_tiles;
    int n_cols;
    double host_prep_ms;
    double submit_ms;
    double a_copy_ms;
    double b_copy_ms;
    double bo_sync_ms;
    double run_ctor_ms;
    double set_arg_ms;
    double run_start_ms;
    double run_wait_ms;
    double queue_wait_ms;
    double sync_from_ms;
    double copyback_ms;
    double total_ms;
    int wait_state;
    uint32_t return_code;
    const void * inst_bo;
    const void * input_bo;
    const void * weight_bo;
    const void * output_bo;
};

struct ggml_xdna_q8_c4_mmul_v2_engine {
    int M, K, N;
    int n_cols;
    int m, n, kt, nb;
    int mmul;  // rows per A block: 4 = mmul<4,8,8>, 8 = mmul<8,8,8>
    int s12;  // Phase 50: 1 = pre-broadcast scales + [i][j][4][8] C + linear drain
    int c256; // Phase 62: 1 = per-K-chunk (256 element) scales, int32 accumulation
    size_t a_bytes, b_bytes, c_bytes;
    size_t inst_bytes;
    xrt::device * device;
    xrt::xclbin * xclbin;
    xrt::hw_context * hw_ctx;
    xrt::kernel * kernel;
    xrt::bo * inst_bo;
    xrt::bo * a_bo[2];
    xrt::bo * b_bo[2];
    xrt::bo * c_bo[2];
    int current_buffer;
    std::mutex lifecycle_mutex;
    bool freeing = false;
    std::vector<xrt::run *> active_runs;
};

struct ggml_xdna_q8_c4_mmul_v2_engine * ggml_xdna_q8_c4_mmul_v2_create(
    int M, int K, int N, int n_cols,
    int m_tile, int n_tile, int k_tile, int mmul,
    const char * xclbin_path, const char * inst_path, int s12 = 0, int pure_int8 = 0,
    int c256 = 0);

bool ggml_xdna_q8_c4_mmul_v2_submit(
    struct ggml_xdna_q8_c4_mmul_v2_engine * e,
    const void * a_packed,
    const void * b_packed,
    xrt::run ** out_run,
    ggml_xdna_xrt_trace * trace = nullptr);

bool ggml_xdna_q8_c4_mmul_v2_wait(
    struct ggml_xdna_q8_c4_mmul_v2_engine * e,
    xrt::run * run,
    float * out,
    int buf_idx,
    ggml_xdna_xrt_trace * trace = nullptr);

void ggml_xdna_q8_c4_mmul_v2_free(struct ggml_xdna_q8_c4_mmul_v2_engine * e);
#endif
