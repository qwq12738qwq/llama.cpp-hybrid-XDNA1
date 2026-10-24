# Phase 6E Scheduler Audit

**Date:** 2026-09-09
**Status:** IN PROGRESS

---

## Summary

XDNA backend works when directly invoked (Phase 6D proven), but `scheduler_assigned = 0` in normal llama.cpp inference. The model loading pipeline doesn't assign layers to XDNA/ACCEL devices.

---

## 1. Backend Registration & Scheduler Integration

### Current Backend Registration (ggml-backend-reg.cpp)
```cpp
// Lines 214-218
#ifdef GGML_USE_XDNA
    register_backend(ggml_backend_xdna_reg());
#endif
```
✅ XDNA backend IS registered globally.

### Scheduler Backend List Construction (llama-context.cpp:325-358)
```cpp
// add ACCEL backends (such as BLAS)
for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        backends.emplace_back(backend);
    }
}
```
✅ XDNA device IS found (type ACCEL) and backend IS created and added to scheduler backends.

**Debug log confirms:**
```
[GGML_REG] register_device called for XDNA0
[XDNA] persistent engine created m=1 k=192 n=128 two_layer=0
```

### Scheduler Backends Debug (with GGML_SCHED_DEBUG=1)
```
n_splits=1
split #0: backend_id=0 (CPU)
```
Only CPU backend is used for splits.

---

## 2. Model Layer Buffer Assignment (llama-model.cpp)

### get_layer_buft_list() (lines 1479-1489)
```cpp
auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
    const bool is_swa = il < n_layer_all && hparams.is_swa(il);
    if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
        return {cpu_dev, &pimpl->cpu_buft_list};  // CPU only
    }
    const int layer_gpu = std::upper_bound(...);
    auto * dev = devices.at(layer_gpu).dev;  // GPU device
    return {dev, &pimpl->gpu_buft_list.at(dev)};  // GPU buffer type
};
```

**CRITICAL FINDING:** This function ONLY returns CPU or GPU devices. XDNA/ACCEL is NEVER considered for layer assignment.

The `devices` vector only contains GPU devices (line 1477: `for (size_t i = 0; i < devices.size(); ++i)` where devices = GPU list).

---

## 3. Buffer Type Selection

### select_weight_buft() (llama-model-loader.cpp:1054-1071)
```cpp
for (const auto & cur : *buft_list) {
    ggml_backend_dev_t cur_dev = cur.first;
    ggml_backend_buffer_type_t cur_buft = cur.second;
    if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
        return cur_buft;
    }
}
```
The `buft_list` comes from `get_layer_buft_list()` which only contains CPU/GPU entries.

### weight_buft_supported() (line 920-965)
```cpp
bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
```
This DOES call `ggml_backend_dev_supports_op()` which IS the XDNA `supports_op()`.

But it's never called with XDNA because XDNA is never in the `buft_list`.

---

## 4. Scheduler Node Assignment

### ggml_backend_sched_set_if_supported() (ggml-backend.cpp:1058-1074)
```cpp
if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
    bool buf_ok = true;
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        if (!ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
            buf_ok = false; break;
        }
    }
    if (buf_ok) { *node_backend_id = cur_backend_id; SET_CAUSE(node, "2.sup"); }
}
```

### ggml_backend_sched_buffer_supported() (line 1037-1052)
```cpp
ggml_backend_buffer_type_t buft = NULL;
if (buf) { buft = buf->buft; }
else {
    int tensor_backend_id = tensor_backend_id(t);
    if (tensor_backend_id != -1) { buft = sched->bufts[tensor_backend_id]; }
}
if (!buft) { return false; }
return ggml_backend_buft_is_supported(backend, buft);
```

**Issue:** Tensor buffer type must be XDNA-compatible. Since weights are assigned CPU/GPU buffer types, `ggml_backend_sched_buffer_supported()` returns false for XDNA backend.

---

## 5. Tensor Buffer Type Assignment

### create_tensor() (llama-model.cpp:1841-1846)
```cpp
const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
return ml.create_tensor(hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, 
                        pimpl->dev_output.buft_list, buft_list_layer, tn, ne, flags);
```

The `buft_list_layer` comes from `dev_layer[il].buft_list` which is set by `get_layer_buft_list()` — which only returns CPU/GPU.

---

## Root Cause Summary

| Component | Issue |
|-----------|-------|
| **Backend Registration** | ✅ XDNA registered globally |
| **Scheduler Backend List** | ✅ XDNA backend added to scheduler |
| **Layer Device Assignment** | ❌ `get_layer_buft_list()` only returns CPU/GPU |
| **Buffer Type Lists** | ❌ No XDNA buffer types in layer buft lists |
| **Tensor Assignment** | ❌ Weights assigned CPU/GPU bufts, not XDNA |
| **Scheduler Assignment** | ❌ `buffer_supported` fails (wrong buft) |
| **scheduler_assigned** | 0 |

---

## Required Fix

**Minimal change:** Modify `get_layer_buft_list()` in `llama-model.cpp` to optionally include XDNA for specific operations (ssm_beta/ssm_alpha).

**Approach:**
1. Add XDNA device to the available devices list
2. For layers containing `ssm_beta`/`ssm_alpha`, include XDNA in the buft list
3. Or: create a special `xdna_buft_list` for those specific tensors

**Risk:** Must ensure weights for ssm_beta/ssm_alpha can actually be placed in XDNA buffers and that the scheduler can handle mixed backend assignments.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/llama-model.cpp` | Modify `get_layer_buft_list()` or add XDNA to layer buft lists |
| `src/llama-model-loader.cpp` | Ensure `select_weight_buft` can pick XDNA for ssm_beta/alpha |
| `src/llama-context.cpp` | Verify scheduler backend list includes XDNA |

---

## Next Steps

1. Add XDNA device to available devices in `llama-model.cpp`
2. Create `xdna_buft_list` in model impl
3. Modify `get_layer_buft_list` to conditionally include XDNA for ssm layers
4. Test with `GGML_SCHED_DEBUG=1` to verify assignment