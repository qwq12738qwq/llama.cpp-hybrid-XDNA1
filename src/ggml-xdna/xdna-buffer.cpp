#include "xdna-buffer.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <exception>
#include <string>
#include <cctype>
#include <xrt/detail/ert.h>

#ifdef __cplusplus
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#endif

static const char * GGML_XDNA_BUFFER_NAME = "XDNA";

static std::atomic<bool> g_xdna_session_fatal{false};
static std::mutex g_xdna_session_fatal_mutex;

bool ggml_xdna_session_is_fatal() {
    return g_xdna_session_fatal.load(std::memory_order_acquire);
}

void ggml_xdna_session_mark_fatal(const char * reason) {
    std::lock_guard<std::mutex> lock(g_xdna_session_fatal_mutex);
    if (g_xdna_session_fatal.exchange(true, std::memory_order_release)) {
        return;
    }
    fprintf(stderr, "[XDNA] FATAL_XDNA_SESSION_ERROR: %s\n", reason ? reason : "unknown XRT error");
    fflush(stderr);
}

void ggml_xdna_session_mark_fatal_if_xrt_error(const char * reason) {
    if (!reason) {
        return;
    }
    std::string text(reason);
    for (char & c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    const char * fatal_terms[] = {
        "broken pipe", "io_page_fault", "bad message", "bad state",
        "unexpected ret", "mailbox", "device lost", "create context",
        "create_hwctx", "init hwctx", "resource allocation failed",
        "aie resource", "channel"
    };
    for (const char * term : fatal_terms) {
        if (text.find(term) != std::string::npos) {
            ggml_xdna_session_mark_fatal(reason);
            return;
        }
    }
}

bool ggml_xdna_wait_run(xrt::run * run, const char * owner,
                        int * wait_state, uint32_t * return_code) {
    if (!run) {
        return false;
    }
    try {
        const ert_cmd_state state = run->wait();
        const uint32_t code = run->return_code();
        if (wait_state) {
            *wait_state = static_cast<int>(state);
        }
        if (return_code) {
            *return_code = code;
        }
        if (state != ERT_CMD_STATE_COMPLETED || code != 0) {
            char reason[256];
            snprintf(reason, sizeof(reason), "%s wait_state=%d return_code=%u",
                     owner ? owner : "XRT run", static_cast<int>(state), code);
            ggml_xdna_session_mark_fatal(reason);
            return false;
        }
        return true;
    } catch (const std::exception & ex) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s wait exception: %s", owner ? owner : "XRT run", ex.what());
        ggml_xdna_session_mark_fatal(reason);
        return false;
    } catch (...) {
        char reason[256];
        snprintf(reason, sizeof(reason), "%s wait exception: unknown", owner ? owner : "XRT run");
        ggml_xdna_session_mark_fatal(reason);
        return false;
    }
}

void ggml_xdna_cleanup_run(xrt::run * run) {
    if (!run) {
        return;
    }
    try {
        run->wait();
    } catch (...) {
    }
    delete run;
}

// Forward declaration
struct ggml_xdna_backend_context;
extern struct ggml_xdna_backend_context * ggml_xdna_get_backend_context(void);

#ifdef __cplusplus
extern "C" {
#endif

static void ggml_xdna_buffer_free(ggml_backend_buffer_t buffer) {
    if (!buffer) return;
    xdna_buffer_context * ctx = (xdna_buffer_context *) buffer->context;
    if (!ctx) return;
    ggml_xdna_synchronize_all();
    if (std::getenv("XDNA_P38C_BO_TRACE")) {
        for (size_t i = 0; i < ctx->bos.size(); i++) {
            fprintf(stderr, "[XDNA-P38C-BO] BUFFER_FREE bo[%zu]=%p size=%zu\n",
                    i, (void*)ctx->bos[i], ctx->bo_sizes[i]);
        }
    }
    if (std::getenv("XDNA_BO_STATS")) {
        static size_t g_freed = 0;
        g_freed += ctx->total_size;
        fprintf(stderr, "[XDNA-BO] free size=%zu total_freed=%zu\n", ctx->total_size, g_freed);
    }
    for (size_t i = 0; i < ctx->bos.size(); i++) {
        if (ctx->bos[i]) {
            delete ctx->bos[i];
            ctx->bos[i] = nullptr;
        }
    }
    delete ctx;
}

static void * ggml_xdna_buffer_get_base(ggml_backend_buffer_t buffer) {
    xdna_buffer_context * ctx = (xdna_buffer_context *) buffer->context;
    if (!ctx || ctx->mapped_ptrs.empty()) return nullptr;
    return ctx->mapped_ptrs[0];
}

static enum ggml_status ggml_xdna_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    (void) buffer;
    (void) tensor;
    return GGML_STATUS_SUCCESS;
}

static void ggml_xdna_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    xdna_buffer_context * ctx = (xdna_buffer_context *) buffer->context;
    if (!ctx || ctx->mapped_ptrs.empty() || !tensor->data) return;
    char * base = (char *) tensor->data + offset;
    memset(base, value, size);
    size_t abs_off = (size_t)((char *)tensor->data - (char *)ctx->mapped_ptrs[0]) + offset;
    size_t bo_off = 0;
    for (size_t i = 0; i < ctx->bos.size(); i++) {
        if (abs_off >= bo_off && abs_off < bo_off + ctx->bo_sizes[i]) {
            ctx->bos[i]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            break;
        }
        bo_off += ctx->bo_sizes[i];
    }
}

static void ggml_xdna_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    xdna_buffer_context * ctx = (xdna_buffer_context *) buffer->context;
    if (!ctx || ctx->mapped_ptrs.empty() || !tensor->data) return;
    char * base = (char *) tensor->data + offset;
    memcpy(base, data, size);
    size_t abs_off = (size_t)((char *)tensor->data - (char *)ctx->mapped_ptrs[0]) + offset;
    size_t bo_off = 0;
    for (size_t i = 0; i < ctx->bos.size(); i++) {
        if (abs_off >= bo_off && abs_off < bo_off + ctx->bo_sizes[i]) {
            ctx->bos[i]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            break;
        }
        bo_off += ctx->bo_sizes[i];
    }
}

static void ggml_xdna_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    xdna_buffer_context * ctx = (xdna_buffer_context *) buffer->context;
    if (!ctx || ctx->mapped_ptrs.empty() || !tensor->data) return;
    // Phase 60: GEMM results are written into the host-mapped buffer by the
    // drain path, not by the device. Copying out before the drain returns the
    // stale (zero) contents, which is what made the first consumer of a
    // split see zero activations.
    ggml_xdna_synchronize_all();
    size_t abs_off = (size_t)((char *)tensor->data - (char *)ctx->mapped_ptrs[0]) + offset;
    size_t bo_off = 0;
    for (size_t i = 0; i < ctx->bos.size(); i++) {
        if (abs_off >= bo_off && abs_off < bo_off + ctx->bo_sizes[i]) {
            ctx->bos[i]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            break;
        }
        bo_off += ctx->bo_sizes[i];
    }
    const char * base = (const char *) tensor->data + offset;
    memcpy(data, base, size);
}

static void ggml_xdna_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    (void) buffer;
    (void) tensor;
    (void) data;
    (void) offset;
    (void) size;
    (void) n_copies;
    (void) stride_tensor;
    (void) stride_data;
}

static void ggml_xdna_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    (void) buffer;
    (void) tensor;
    (void) data;
    (void) offset;
    (void) size;
    (void) n_copies;
    (void) stride_tensor;
    (void) stride_data;
}

static bool ggml_xdna_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    xdna_buffer_context * ctx = (xdna_buffer_context *) buffer->context;
    if (!ctx) return false;
    
    size_t nbytes = ggml_nbytes(src);
    void * tmp = malloc(nbytes);
    if (!tmp) return false;
    
    ggml_backend_tensor_get(src, tmp, 0, nbytes);
    ggml_backend_tensor_set(dst, tmp, 0, nbytes);
    free(tmp);
    return true;
}

static void ggml_xdna_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    xdna_buffer_context * ctx = (xdna_buffer_context *) buffer->context;
    if (!ctx || ctx->mapped_ptrs.empty()) return;
    memset(ctx->mapped_ptrs[0], value, ctx->total_size);
    for (size_t i = 0; i < ctx->bos.size(); i++) {
        ctx->bos[i]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
}

static void ggml_xdna_buffer_reset(ggml_backend_buffer_t buffer) {
    (void) buffer;
}

static const struct ggml_backend_buffer_i ggml_xdna_buffer_i = {
    /* .free_buffer      = */ ggml_xdna_buffer_free,
    /* .get_base         = */ ggml_xdna_buffer_get_base,
    /* .init_tensor      = */ ggml_xdna_buffer_init_tensor,
    /* .memset_tensor    = */ ggml_xdna_buffer_memset_tensor,
    /* .set_tensor       = */ ggml_xdna_buffer_set_tensor,
    /* .get_tensor       = */ ggml_xdna_buffer_get_tensor,
    /* .set_tensor_2d    = */ ggml_xdna_buffer_set_tensor_2d,
    /* .get_tensor_2d    = */ ggml_xdna_buffer_get_tensor_2d,
    /* .cpy_tensor       = */ ggml_xdna_buffer_cpy_tensor,
    /* .clear            = */ ggml_xdna_buffer_clear,
    /* .reset            = */ ggml_xdna_buffer_reset,
};

static const char * ggml_xdna_buffer_type_name(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return GGML_XDNA_BUFFER_NAME;
}

static size_t ggml_xdna_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return 128;
}

static size_t ggml_xdna_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return SIZE_MAX;
}

static size_t ggml_xdna_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    (void) buft;
    if (!tensor) return 0;
    return ggml_nbytes(tensor);
}

static bool ggml_xdna_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return false;
}

// Shared XRT device singleton
static xrt::device * g_xdna_shared_device = nullptr;
static std::mutex g_xdna_shared_device_mutex;

static xrt::device * ggml_xdna_get_or_create_shared_device(void) {
    std::lock_guard<std::mutex> lock(g_xdna_shared_device_mutex);
    if (!g_xdna_shared_device) {
        try {
            g_xdna_shared_device = new xrt::device(0);
            fprintf(stderr, "[XDNA-BUFFER] Shared XRT device created\n");
        } catch (const std::exception & e) {
            fprintf(stderr, "[XDNA-BUFFER] XRT device creation failed: %s\n", e.what());
            ggml_xdna_session_mark_fatal_if_xrt_error(e.what());
            return nullptr;
        }
    }
    return g_xdna_shared_device;
}



static ggml_backend_buffer_t ggml_xdna_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (ggml_xdna_session_is_fatal()) {
        return nullptr;
    }
    if (size == 0) {
        return ggml_backend_buffer_init(buft, ggml_xdna_buffer_i, nullptr, 0);
    }
    
    xdna_buffer_context * ctx = new xdna_buffer_context();
    ctx->total_size = size;
    
    xrt::device * dev = ggml_xdna_get_or_create_shared_device();
    if (!dev) {
        fprintf(stderr, "[XDNA-BUFFER] Failed to get XRT device, falling back to CPU buffer\n");
        delete ctx;
        return nullptr;
    }
    
    xrt::bo * bo = nullptr;
    try {
        size_t aligned_size = (size + 255) & ~255;
        uint32_t gid3 = 3;
        bo = new xrt::bo(*dev, aligned_size, xrt::bo::flags::host_only, gid3);
        void * mapped = bo->map();
        ctx->bos.push_back(bo);
        bo = nullptr;
        ctx->mapped_ptrs.push_back(mapped);
        ctx->bo_sizes.push_back(aligned_size);
        ctx->device = dev;
        
        if (std::getenv("XDNA_BO_STATS")) {
            static size_t g_live = 0, g_peak = 0, g_count = 0;
            g_live += aligned_size;
            g_count++;
            if (g_live > g_peak) g_peak = g_live;
            fprintf(stderr, "[XDNA-BO] alloc size=%zu live=%zu peak=%zu count=%zu\n",
                    aligned_size, g_live, g_peak, g_count);
        }
        if (std::getenv("XDNA_P38C_BO_TRACE")) {
            fprintf(stderr, "[XDNA-P38C-BO] BUFFER_ALLOC bo=%p size=%zu mapped=%p\n",
                    (void*)ctx->bos.back(), aligned_size, mapped);
        }
        
        return ggml_backend_buffer_init(buft, ggml_xdna_buffer_i, ctx, size);
    } catch (const std::exception & e) {
        fprintf(stderr, "[XDNA-BUFFER] BO creation failed: %s\n", e.what());
        ggml_xdna_session_mark_fatal_if_xrt_error(e.what());
        delete bo;
        for (xrt::bo * bos : ctx->bos) {
            delete bos;
        }
        delete ctx;
        return nullptr;
    }
}

static const struct ggml_backend_buffer_type_i ggml_xdna_buffer_type_i = {
    /* .get_name       = */ ggml_xdna_buffer_type_name,
    /* .alloc_buffer   = */ ggml_xdna_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_xdna_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_xdna_buffer_type_get_max_size,
    /* .get_alloc_size = */ ggml_xdna_buffer_type_get_alloc_size,
    /* .is_host        = */ ggml_xdna_buffer_type_is_host,
};

GGML_BACKEND_API const struct ggml_backend_buffer_type_i * ggml_xdna_get_buffer_type_i(void) {
    return &ggml_xdna_buffer_type_i;
}

ggml_backend_buffer_type_t ggml_xdna_buffer_type(void) {
    extern ggml_backend_reg_t ggml_backend_xdna_reg(void);
    ggml_backend_reg_t reg = ggml_backend_xdna_reg();
    if (!reg) return nullptr;
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (!dev) return nullptr;
    extern ggml_backend_buffer_type_t ggml_xdna_device_get_buffer_type(ggml_backend_dev_t dev);
    return ggml_xdna_device_get_buffer_type(dev);
}

ggml_backend_buffer_type_t ggml_xdna_host_buffer_type(void) {
    return ggml_backend_cpu_buffer_type();
}

#ifdef __cplusplus
}

xrt::device * ggml_xdna_shared_xrt_device(void) {
    return ggml_xdna_get_or_create_shared_device();
}
#endif
