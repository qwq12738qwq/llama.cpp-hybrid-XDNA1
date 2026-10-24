#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_XDNA_NAME "XDNA"

GGML_BACKEND_API ggml_backend_t ggml_backend_xdna_init(void);

GGML_BACKEND_API bool ggml_backend_is_xdna(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void);

GGML_BACKEND_API ggml_backend_dev_t ggml_backend_xdna_reg_get_device(size_t index);

GGML_BACKEND_API size_t ggml_backend_xdna_reg_get_device_count(void);

GGML_BACKEND_API enum ggml_backend_dev_type ggml_xdna_device_get_type(ggml_backend_dev_t dev);

GGML_BACKEND_API const char * ggml_xdna_device_get_name(ggml_backend_dev_t dev);

GGML_BACKEND_API const char * ggml_xdna_device_get_description(ggml_backend_dev_t dev);

GGML_BACKEND_API void ggml_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_xdna_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props);

GGML_BACKEND_API ggml_backend_t ggml_xdna_device_init(ggml_backend_dev_t dev, const char * params);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_xdna_device_get_buffer_type(ggml_backend_dev_t dev);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_xdna_device_get_host_buffer_type(ggml_backend_dev_t dev);

GGML_BACKEND_API ggml_backend_buffer_t ggml_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

GGML_BACKEND_API bool ggml_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op);

GGML_BACKEND_API bool ggml_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

GGML_BACKEND_API bool ggml_xdna_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op);

GGML_BACKEND_API ggml_backend_event_t ggml_xdna_device_event_new(ggml_backend_dev_t dev);

GGML_BACKEND_API void ggml_xdna_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event);

GGML_BACKEND_API void ggml_xdna_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event);

struct ggml_xdna_counters {
    int supports_op;
    int scheduler_assigned;
    int graph_compute;
    int persistent_execute;
    int xrt_submit;
    int xrt_complete;
    int decode_token;
};

GGML_BACKEND_API void ggml_xdna_print_counters(void);

GGML_BACKEND_API struct ggml_xdna_counters ggml_xdna_get_counters(void);

GGML_BACKEND_API void ggml_xdna_print_counters_delta(const struct ggml_xdna_counters * prev, const struct ggml_xdna_counters * cur);

#ifdef __cplusplus
}
#endif