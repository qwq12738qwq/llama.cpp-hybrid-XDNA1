# Phase 62: device incidents

## 1. The NPU driver refused new contexts (from 03:09)

Sequence (all times UTC, 2026-10-21):

| time | event |
|---|---|
| 02:24 | the baseline campaign: the first hybrid run aborts with a Rust panic in the XDNA TEE (`2nd pipe should have enough spare`), the Vulkan runs are fine |
| 02:30 | a retry of the hybrid run dies with `vk::DeviceLostError: vk::Queue::submit: ErrorDeviceLost` while the NPU counters show `xrt_submit=0`, i.e. the GPU was lost before any NPU work |
| 03:09 | the standalone kernel harness hangs (61 s instead of 0.16 s) in the variant that does not wait for every drain, and the result is wrong (rel_rmse 3.5e-2) |
| 03:10 | every later `pyxrt.hw_context()` fails with `DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22): Invalid argument` |
| 03:12 | `xrt-smi examine` still lists the device (`RyzenAI-npu1` at 0000:67:00.1) and `pyxrt.device(0)` opens, but no context can be created |
| 03:35 | the same failure through the production path: `[XDNA-Q8S] init failed: DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22)` |

The device is shared with the host (the container has no `/dev/dri` reset rights and
`xrt-smi configure` is not permitted), so the state cannot be cleared from here.

Consequence for this phase: **all the NPU measurements of phase 62 were taken
before 03:09**. Everything after that is static work, code and compilation.
The end-to-end hybrid numbers for the new engine could not be measured; the
kernel level numbers were.

## 2. The first baseline campaign also failed, before any phase 62 change

The same Rust panic (`src/uu/tee/src/tee.rs:188`) hit the very first hybrid
run of the phase 62 baseline, while the Vulkan-only runs in the same campaign
succeeded (1250.26 +- 85.82 tok/s). The machine had 3.5 GB in swap at that
moment. The same configuration had worked in phase 61 (14:18 on Oct 20), so
the difference is the state of the machine, not the code.

The Vulkan-only reference that was measured in the same window is therefore
usable, and the hybrid reference had to be taken from the phase 61 logs
(15.26 tok/s for `c4_mmul_v2`, 16.42 tok/s for `c4_mmul_v3`).
