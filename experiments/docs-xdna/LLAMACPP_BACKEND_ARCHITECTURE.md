# llama.cpp Backend Architecture

## Overview

llama.cpp uses a pluggable backend system that allows tensor operations to be offloaded
to different hardware accelerators. Each backend implements a common interface defined in
`ggml-backend.h` and `ggml-backend-impl.h`.

## Backend Registration

### Static Registration

Backends are registered at compile time in `ggml/src/ggml-backend-reg.cpp`:

```cpp
#ifdef GGML_USE_VULKAN
    register_backend(ggml_backend_vk_reg());
#endif
```

The `ggml_add_backend()` function in `ggml/src/CMakeLists.txt` handles this:

```cmake
ggml_add_backend(Hexagon)
ggml_add_backend(XDNA)
```

### Dynamic Loading

When `GGML_BACKEND_DL` is enabled, backends are loaded as shared libraries from:
- `GGML_BACKEND_DIR` (if set)
- Executable directory
- Current working directory

The library name format is `libggml-<name>.so` (Linux) or `libggml-<name>.dll` (Windows).

### Backend Registry

Each backend provides a `ggml_backend_reg_t` via `ggml_backend_<name>_reg()`:

```cpp
struct ggml_backend_reg_i {
    const char * (*get_name)(ggml_backend_reg_t reg);
    size_t             (*get_device_count)(ggml_backend_reg_t reg);
    ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);
    void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
};
```

## Device Discovery

Devices are enumerated via the registry:

```cpp
ggml_backend_reg_t reg = ggml_backend_xdna_reg();
size_t n_devices = ggml_backend_reg_dev_count(reg);
for (size_t i = 0; i < n_devices; i++) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
}
```

The registry creates devices in its constructor. For XDNA, this means:
1. Check if `/dev/accel/accel0` exists
2. Initialize XRT
3. Create device context

## Buffer Allocation

### Buffer Type Interface

```cpp
struct ggml_backend_buffer_type_i {
    const char * (*get_name)(ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size);
    size_t (*get_alignment)(ggml_backend_buffer_type_t buft);
    size_t (*get_max_size)(ggml_backend_buffer_type_t buft);
    size_t (*get_alloc_size)(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);
    bool (*is_host)(ggml_backend_buffer_type_t buft);
};
```

### Buffer Interface

```cpp
struct ggml_backend_buffer_i {
    void (*free_buffer)(ggml_backend_buffer_t buffer);
    void * (*get_base)(ggml_backend_buffer_t buffer);
    enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, ggml_tensor * tensor);
    void (*memset_tensor)(...);
    void (*set_tensor)(...);
    void (*get_tensor)(...);
    void (*set_tensor_2d)(...);
    void (*get_tensor_2d)(...);
    bool (*cpy_tensor)(...);
    void (*clear)(...);
    void (*reset)(...);
};
```

## Operation Support

### supports_op()

The scheduler queries each backend to determine which operations it can execute:

```cpp
bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);
```

For XDNA, `supports_op()` returns true only for:
- `GGML_OP_MUL_MAT`
- With exact shape: M=1, K=192, N=128
- With type: Q8_0

All other operations return false, causing fallback to CPU.

### offload_op()

Optional hint for expensive operations that should run on the accelerator even if
weights are in an incompatible buffer.

## Graph Compute

### graph_compute()

The scheduler calls `graph_compute()` on the backend for each subgraph:

```cpp
enum ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);
```

The backend iterates through nodes and executes supported operations. Unsupported
operations are skipped (they will be executed by another backend).

### Async Execution

Backends may implement async operations:
- `set_tensor_async`
- `get_tensor_async`
- `synchronize`

XDNA currently does not support async execution.

## Backend Selection

The scheduler selects backends based on:
1. Operation support (`supports_op`)
2. Tensor location (weights in a specific buffer type)
3. Backend priority (order in the scheduler's backend list)

## Backend Priority

Backends are ordered by preference:
1. GPU backends (CUDA, Vulkan, Metal)
2. Accelerator backends (XDNA, Hexagon, BLAS)
3. CPU backend

The XDNA backend registers as `GGML_BACKEND_DEVICE_TYPE_ACCEL`.

## XDNA Backend Design

### Files

```
ggml/src/ggml-xdna/
  ggml-xdna.cpp          Main backend implementation
  ggml-xdna.h            Internal header
  xdna-persistent.h      Persistent engine interface
  xdna-persistent.cpp    Persistent engine reference implementation
  CMakeLists.txt         Build configuration

ggml/include/
  ggml-xdna.h            Public API header

docs/xdna/
  LLAMACPP_BACKEND_ARCHITECTURE.md  This document
  XDNA_BACKEND_DESIGN.md            Design details

tests/
  test_xdna_backend.cpp   Unit test
```

### Constraints

The XDNA backend initially supports only:
- `GGML_OP_MUL_MAT` with M=1, K=192, N=128
- Q8_0 tensor type

All other operations return false from `supports_op()`.

### Persistent Engine

The persistent engine is initialized once during backend creation and remains resident.
It bridges ggml tensors to the Phase 4 persistent MLP implementation.

### Debug Logging

Enable with `GGML_XDNA_DEBUG=1`:

```
[XDNA] backend initialized
[XDNA] persistent engine ready
[XDNA] op accepted
[XDNA] GEMV dispatch M=1 K=192 N=128
[XDNA] execution_us=1234
[XDNA] output copied
[XDNA] graph_compute end
```
