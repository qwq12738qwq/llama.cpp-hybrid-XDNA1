#pragma once

// Phase 22: whole-array int8 GEMM engine for the XDNA backend.
//
// Loads the 4-column whole-array kernel produced by experiments/phase21/gen_gemm.py
// (A[M,K] i8 x B[K,N] i8 -> C[M,N] i32, all row-major) and runs it with a single
// XRT submit. This is the fast path used by the Phase 21 hybrid; the backend's
// previous scale-aware C1 kernel was ~150x slower for the same shape.

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

struct ggml_xdna_wa_engine {
    int M, K, N;
    size_t inst_bytes, a_bytes, b_bytes, c_bytes;
    xrt::device * device;
    xrt::xclbin * xclbin;
    xrt::hw_context * hw_ctx;
    xrt::kernel * kernel;
    xrt::bo * inst_bo;
    xrt::bo * a_bo;
    xrt::bo * b_bo;
    xrt::bo * c_bo;
};

ggml_xdna_wa_engine * ggml_xdna_wa_create(int M, int K, int N,
                                          const char * xclbin_path,
                                          const char * inst_path);
bool ggml_xdna_wa_submit(ggml_xdna_wa_engine * e, const int8_t * a, const int8_t * b,
                         xrt::run ** out_run);
bool ggml_xdna_wa_wait(ggml_xdna_wa_engine * e, xrt::run * run, int32_t * c_out);
void ggml_xdna_wa_free(ggml_xdna_wa_engine * e);
#endif
