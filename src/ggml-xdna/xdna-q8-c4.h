#pragma once

// Phase 39C: Q8_0 full-K GEMM using C4_WA mmul kernel with host-side scale folding.
// 
// Uses the existing C4_WA int8 mmul kernel (4 AIE columns, mmul-based).
// Q8_0 per-32-block scales are folded into the int8 data on host via re-quantization.
// One XRT submit per K-block (32 elements). Results accumulated on host as float32.

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <vector>

struct ggml_xdna_q8_c4_engine {
    int M, K, N;
    int nb;              // number of K-blocks (K/32)
    size_t a_bytes;      // int8 A tile: M * 32
    size_t b_bytes;      // int8 B tile: 32 * N
    size_t c_bytes;      // int32 C tile: M * N * 4
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
};

struct ggml_xdna_q8_c4_engine * ggml_xdna_q8_c4_create(
    int M, int K, int N, const char * xclbin_path, const char * inst_path);

// Submit one K-block (32 elements) to the C4_WA kernel.
// a_block: int8[M, 32], row-major
// b_block: int8[32, N], row-major
// c_out: int32[M, N], row-major (output from kernel)
bool ggml_xdna_q8_c4_submit(
    struct ggml_xdna_q8_c4_engine * e,
    const int8_t * a_block,
    const int8_t * b_block,
    xrt::run ** out_run);

bool ggml_xdna_q8_c4_wait(
    struct ggml_xdna_q8_c4_engine * e,
    xrt::run * run,
    int32_t * c_out,
    int buf_idx);

void ggml_xdna_q8_c4_free(struct ggml_xdna_q8_c4_engine * e);
#endif