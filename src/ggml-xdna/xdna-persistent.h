#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>

#ifdef __cplusplus
#include <string>
#include <xrt/deprecated/xrt.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#endif

#ifdef __cplusplus
struct xrt_context {
    xrt::device device;
    xrt::xclbin xclbin;
    xrt::uuid uuid;
    xrt::hw_context hw_ctx;
    xrt::kernel kernel;
    size_t inst_bytes;
    xrt::bo* inst_bo;  // Shared instruction BO
    std::string xclbin_path;
    std::string inst_path;
};

struct ggml_xdna_persistent_engine {
    void * ctx;
    int    m;
    int    k;
    int    n;
    bool   initialized;
    bool   two_layer;

    // XRT persistent objects
    struct xrt_context * xrt_ctx;

    // Double-buffered BOs for async pipeline
    xrt::bo * inst_bo;
    xrt::bo * weight_bo[2];
    xrt::bo * input_bo[2];
    xrt::bo * output_bo[2];
    int current_buffer;

    // For async execution tracking
    xrt::run * pending_run;
    int pending_buffer;
};
#else
// Opaque struct for C compatibility
struct xrt_context;
struct ggml_xdna_persistent_engine;
#endif

// External execution counters
extern std::atomic<int> g_xdna_persistent_execute_count;
extern std::atomic<int> g_xdna_xrt_submit_count;
extern std::atomic<int> g_xdna_xrt_complete_count;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_xdna_persistent_engine * ggml_xdna_persistent_engine_create(int m, int k, int n, bool two_layer);
void ggml_xdna_persistent_engine_free(struct ggml_xdna_persistent_engine * engine);
bool ggml_xdna_persistent_engine_execute(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    const int8_t * weight1,
    const int8_t * weight2,
    int32_t * output
);

// K-chunked execute: processes full K dimension by chunking into engine->k sized chunks
// input:  [m, K_total]  (row-major)
// weight: [K_total, n]  (row-major)
bool ggml_xdna_persistent_engine_execute_chunked(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    int K_total,
    const int8_t * weight,
    int32_t * output
);

// N-parallel execute: runs kernel once with full problem dimensions (kernel handles internal tiling)
// input:  [m, K]  (row-major)
// weight: [K, n]  (row-major)
bool ggml_xdna_persistent_engine_execute_nparallel(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    const int8_t * weight,
    int32_t * output
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Phase 9: Async execution interface (C++ only)
bool ggml_xdna_persistent_engine_init_xrt(
    struct ggml_xdna_persistent_engine * engine,
    const char * xclbin_path,
    const char * inst_path
);
bool ggml_xdna_persistent_engine_submit(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    const int8_t * weight1,
    int32_t * output,
    xrt::run ** out_run
);
bool ggml_xdna_persistent_engine_wait(
    struct ggml_xdna_persistent_engine * engine,
    xrt::run * run,
    int32_t * output
);
#endif

// C wrappers for async functions (for Python/ctypes access)
#ifdef __cplusplus
extern "C" {
#endif

bool ggml_xdna_persistent_engine_init_xrt_c(
    struct ggml_xdna_persistent_engine * engine,
    const char * xclbin_path,
    const char * inst_path
);
bool ggml_xdna_persistent_engine_submit_c(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    const int8_t * weight1,
    int32_t * output,
    void ** out_run
);
bool ggml_xdna_persistent_engine_wait_c(
    struct ggml_xdna_persistent_engine * engine,
    void * run,
    int32_t * output
);

#ifdef __cplusplus
}
#endif