#pragma once

// Phase 20G step 5: scale-aware Q8_0 full-K XDNA execution.
//
// One xrt::run covers the whole K. The device kernel loops over 32-element
// K-blocks and applies per-block scales itself, so there is no host K chunking.
// Host operands are tile-major packed (see gen_q8_scale_gemm.py).

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#endif

#ifdef __cplusplus
struct ggml_xdna_q8_scale_engine {
    int M, K, N;
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

struct ggml_xdna_q8_scale_engine * ggml_xdna_q8_scale_create(
    int M, int K, int N, const char * xclbin_path, const char * inst_path);

// a_packed / b_packed are host buffers already in the tile-major layout.
bool ggml_xdna_q8_scale_submit(
    struct ggml_xdna_q8_scale_engine * e,
    const void * a_packed,
    const void * b_packed,
    xrt::run ** out_run);

bool ggml_xdna_q8_scale_wait(
    struct ggml_xdna_q8_scale_engine * e,
    xrt::run * run,
    float * out,
    int buf_idx);

void ggml_xdna_q8_scale_free(struct ggml_xdna_q8_scale_engine * e);
#endif
