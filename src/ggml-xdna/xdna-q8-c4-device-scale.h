#pragma once

// Phase 40: Q8_0 full-K GEMM using C4 mmul kernel with device-side per-block scales.
//
// Uses gen_q8_c4_mmul.py kernel (4 AIE columns, mmul<4,8,8> + device-side scales).
// Q8_0 per-32-block scales are preserved on device; no host-side scale folding.
// One xrt::run covers the whole K.

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
#include "xdna-q8-c4-mmul-v2.h"
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <vector>

struct ggml_xdna_q8_c4_ds_engine {
    int M, K, N;
    int n_cols;
    size_t a_bytes;
    size_t b_bytes;
    size_t c_bytes;
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

struct ggml_xdna_q8_c4_ds_engine * ggml_xdna_q8_c4_ds_create(
    int M, int K, int N, int n_cols,
    const char * xclbin_path, const char * inst_path);

bool ggml_xdna_q8_c4_ds_submit(
    struct ggml_xdna_q8_c4_ds_engine * e,
    const void * a_packed,
    const void * b_packed,
    xrt::run ** out_run,
    ggml_xdna_xrt_trace * trace = nullptr);

bool ggml_xdna_q8_c4_ds_wait(
    struct ggml_xdna_q8_c4_ds_engine * e,
    xrt::run * run,
    float * out,
    int buf_idx,
    ggml_xdna_xrt_trace * trace = nullptr);

void ggml_xdna_q8_c4_ds_free(struct ggml_xdna_q8_c4_ds_engine * e);
#endif
