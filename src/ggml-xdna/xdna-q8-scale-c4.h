#pragma once

// Phase 40H: Scale-aware Q8_0 full-K GEMM, N split across AIE columns (Q8SCALEV-C4).
//
// Uses the q8_chunk_vec kernel (same as Q8SCALEV-C1) with N dimension split
// across n_cols parallel compute columns. A is broadcast, B and C are N-sliced.
// One xrt::run covers the whole K and all columns.
//
// Cache namespace: M{K}_N{N}_C{n_cols}_Q8SCALEV

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <vector>

struct ggml_xdna_q8_scale_c4_engine {
    int M, K, N;
    int n_cols;
    int m, n, kt, nb;
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
};

struct ggml_xdna_q8_scale_c4_engine * ggml_xdna_q8_scale_c4_create(
    int M, int K, int N, int n_cols,
    const char * xclbin_path, const char * inst_path);

// a_packed / b_packed are host buffers in tile-major layout (N-sliced for B).
bool ggml_xdna_q8_scale_c4_submit(
    struct ggml_xdna_q8_scale_c4_engine * e,
    const void * a_packed,
    const void * b_packed,
    xrt::run ** out_run);

bool ggml_xdna_q8_scale_c4_wait(
    struct ggml_xdna_q8_scale_c4_engine * e,
    xrt::run * run,
    float * out,
    int buf_idx);

void ggml_xdna_q8_scale_c4_free(struct ggml_xdna_q8_scale_c4_engine * e);
#endif
