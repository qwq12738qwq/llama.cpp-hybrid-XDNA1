#pragma once

#include "ggml-backend-impl.h"
#include "../../include/ggml-xdna.h"
#include "xdna-persistent.h"
#include "xdna-buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_xdna_device_context;

#ifdef __cplusplus
}
#endif

// Accessor for the XDNA backend context (used by xdna-buffer.cpp)
GGML_BACKEND_API struct ggml_xdna_backend_context * ggml_xdna_get_backend_context(void);

// Phase 38C: exported steady-clock millisecond timer for cross-TU instrumentation
extern "C" double ggml_xdna_now_ms(void);
