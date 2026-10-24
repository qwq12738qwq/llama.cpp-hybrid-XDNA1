#pragma once

#include "ggml-backend-impl.h"
#include <vector>
#include <mutex>

#ifdef __cplusplus
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_xdna_buffer_type(void);
ggml_backend_buffer_type_t ggml_xdna_host_buffer_type(void);

// Buffer interface for XDNA-owned XRT BO-backed buffers
GGML_BACKEND_API const struct ggml_backend_buffer_type_i * ggml_xdna_get_buffer_type_i(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
bool ggml_xdna_session_is_fatal();
void ggml_xdna_session_mark_fatal(const char * reason);
void ggml_xdna_session_mark_fatal_if_xrt_error(const char * reason);
void ggml_xdna_synchronize_all();
bool ggml_xdna_wait_run(xrt::run * run, const char * owner,
                        int * wait_state = nullptr, uint32_t * return_code = nullptr);
void ggml_xdna_cleanup_run(xrt::run * run);

struct xdna_buffer_context {
    xrt::device * device;
    std::vector<xrt::bo*> bos;
    std::vector<void*> mapped_ptrs;
    std::vector<size_t> bo_sizes;
    size_t total_size;
    std::mutex sync_mutex;
};

// Process-wide XRT device shared by buffer allocs and Q8 engines.
// Do not delete the returned pointer.
xrt::device * ggml_xdna_shared_xrt_device(void);
#endif
