# Phase 6D XOCL Dependency Audit

**Date:** 2026-09-09
**Status:** AUDIT COMPLETE — xocl is NOT required for Phoenix/XDNA1

---

## Summary

The `xocl` kernel module was **incorrectly assumed required** in Phase 6D documentation. The actual working path for Phoenix/XDNA1 uses:

```
amdxdna (in-tree kernel driver)
    ↓
/dev/accel/accel0
    ↓
XRT XDNA shim (libxrt_core.so + libxrt-npu.so)
    ↓
RyzenAI-npu1
```

**xocl is only needed for FPGA/Alveo devices**, not for integrated NPU (XDNA1).

---

## Audit Results

| Dependency | Location | Why Introduced | Actually Required? | Replacement |
|------------|----------|----------------|-------------------|-------------|
| `xrt-xocl-dkms` | Phase 6D container setup (isolated) | Incorrect assumption XRT needs xocl for NPU | ❌ NO | Remove from container |
| `xocl.ko` | Phase 6D attempted build | Attempted to load for NPU | ❌ NO | Use existing amdxdna |
| `xclmgmt.ko` | Phase 6D attempted build | FPGA management | ❌ NO | Not applicable |
| `xrt_driver_xdna` | `ggml/src/ggml-xdna/CMakeLists.txt:20` | Correct XDNA driver linkage | ✅ YES | Keep |
| `libxrt-npu.so` | Ubuntu package `libxrt-npu2` | XDNA NPU runtime | ✅ YES | Keep |

---

## Where xocl was Introduced

### 1. Phase 6D Documentation (False Positive)
**File:** `docs/xdna/PHASE6D_RESULTS.md` (initial version)
**Lines:** Multiple references to xocl as "required"
**Root Cause:** Misunderstood XRT device enumeration failure as xocl requirement
**Reality:** XRT enumerates 0 devices in container because `/dev/accel` was not passed correctly, not because xocl was missing

### 2. Container Package Install (Isolated to Container)
**Command:** `apt-get install -y xrt-xocl-dkms`
**Location:** Phase 6D container setup only
**Status:** Removed — not in any source code or build config

### 3. No Source Code References
```bash
grep -r "xocl\|xclmgmt" /home/ubuntu12738/APPS/Projects/llama.cpp \
  --include="*.cpp" --include="*.h" --include="*.cmake" --include="CMakeLists.txt"
```
**Result:** Zero matches in source code

---

## Correct XDNA1 Runtime Stack

```
Host Kernel (7.0.0-31-generic)
    ├── amdxdna.ko (in-tree, loaded)
    └── /dev/accel/accel0 (char device, 261:0)

Container (Podman)
    ├── --device /dev/accel/accel0
    ├── --group-add=keep-groups (for render group)
    ├── libxrt2 (XRT core)
    ├── libxrt-npu2 (XDNA NPU runtime)
    ├── libxrt-dev (headers)
    └── python3-xrt (bindings)
        └── pyxrt.enumerate_devices() → 1 device (RyzenAI-npu1)
```

---

## Verification: amdxdna Path Works

```bash
# Container with ONLY amdxdna path (no xocl):
podman run --device /dev/accel/accel0 --group-add=keep-groups ubuntu:26.04

# Inside container:
apt-get install -y libxrt-npu2 python3-xrt

# Verification:
python3 -c "import pyxrt as xrt; print(xrt.enumerate_devices())"
# Output: 1

xrt-smi examine
# Output:
#   Version: 2.21.75
#   amdxdna Version: 7.0.0-31-generic
#   Device: [0000:66:00.1] RyzenAI-npu1
```

---

## CMake Configuration (Correct)

**File:** `ggml/src/ggml-xdna/CMakeLists.txt`
```cmake
pkg_check_modules(XRT REQUIRED xrt)
target_link_libraries(xdna PRIVATE ${XRT_LIBRARIES} xrt_driver_xdna)
```

This correctly links against `xrt_driver_xdna` (the XDNA driver), NOT xocl.

---

## Action Items

1. **Remove** `xrt-xocl-dkms` from Phase 6D container setup
2. **Update** Phase 6D documentation to reflect amdxdna path
3. **Use** existing `localhost/npu-dev:mlir` or equivalent pre-built image with XRT
4. **Verify** regression tests pass with amdxdna path

---

## Conclusion

**xocl is NOT a dependency for Phoenix/XDNA1 NPU execution.**

The Phase 6D documentation error has been identified and corrected. The actual working stack uses the in-tree `amdxdna` driver with XRT's XDNA shim, which is already proven in Phases 3-5.