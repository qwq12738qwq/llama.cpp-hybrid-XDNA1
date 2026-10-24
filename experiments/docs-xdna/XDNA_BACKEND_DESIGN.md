# XDNA Backend Design

## Purpose

Experimental `ggml-xdna` backend for AMD Phoenix XDNA1 NPU.

## Goals

1. Dispatch at least one real llama.cpp operation to XDNA
2. Reuse proven Phase 4 persistent engine
3. Do NOT attempt full Transformer offload
4. Do NOT optimize performance yet

## First Success Criterion

A real GGUF model can execute through llama.cpp while at least one deliberately
selected compatible operation is dispatched to the XDNA backend.

## Architecture

```
ggml tensor
    ↓
XDNA adapter (ggml-xdna.cpp)
    ↓
persistent engine (xdna-persistent.cpp)
    ↓
output tensor
```

## Supported Operation

Initially exactly ONE operation:

- `GGML_OP_MUL_MAT`
- Shape: M=1, K=192, N=128
- Type: Q8_0

All other shapes and types are rejected by `supports_op()`.

## Device Discovery

- Check `/dev/accel/accel0`
- Fall back to `/dev/accel/accel1` if needed
- Initialize XRT 2.21.75
- Keep persistent XDNA engine alive

## Buffer Strategy

- No buffer allocation implemented yet (returns nullptr)
- Device memory reporting is placeholder (2GB)
- Host buffers not supported

## Error Handling

- No silent fallback inside XDNA backend
- Unsupported operations are skipped, not emulated
- Backend init fails gracefully if device unavailable

## Performance

No performance optimization at this stage. The persistent engine provides
a reference implementation for correctness validation.

## Debug Logging

Set `GGML_XDNA_DEBUG=1` to enable verbose logging.
