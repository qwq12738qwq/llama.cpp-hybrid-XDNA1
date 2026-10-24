# Phase 61 post-reboot: incidents and device state log

## Device recovery after the Oct-21 wedge (15:22 UTC, Oct 23)

The host was rebooted between the end of phase 63 (2026-10-21 07:27) and this
session (uptime 1 day 13 h at 15:23). The XDNA device came back clean:

* `/dev/accel/accel0` present (created 15:18), `xrt-smi examine` lists
  `RyzenAI-npu1` at `0000:67:00.1`.
* `pyxrt.hw_context()` succeeds again (it had failed with
  `DRM_IOCTL_AMDXDNA_GET_INFO err=-22` for all of phase 63).
* First device check: the production `c4_mmul_v4` (kbench62 `B_c256x2_c4`,
  M512 K1024 N3584, wcols 4) ran at **114.004 ms**, matching the phase 62
  113.1 ms. So the recovery is real, not a stale cache.

## GPU (Vulkan) instability under sustained hybrid load

The Radeon 780M lost its context twice during the perplexity gate:

* 15:36 - during model load of `llama-perplexity -b 2048`:
  `radv/amdgpu: The CS has been cancelled because the context is lost` /
  `vk::Queue::submit: ErrorDeviceLost`. Host had ~0.8 GB free, 6.2 GB in swap.
* 15:44 - at 57 s into the `c4_mmul_v4` perplexity run (`-b 512`, 252 NPU GEMM
  traces executed fine, then `vk::DeviceLostError`).

Both times a fresh short `llama-bench` run after the process exited recovered
the GPU (1176 tok/s). The device-lost is a host-memory thrashing symptom
(the 7.2 GB machine is ~50-60% swapped during the campaign), not a kernel
defect: the NPU ops completed (missing_kernel=0) and the crash is on the Vulkan
`Queue::submit`. The NPU perplexity gate therefore could not complete a full
6-chunk run in a stable window this session.

## NPU re-wedge during the kbench scaling sweep (15:57, Oct 23)

After ~8 back-to-back `hw_context` create/destroy cycles in the kbench worker
scaling sweeps the driver returned to the phase 62/63 wedge signature:

```
RuntimeError: DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22): Invalid argument
```

`pyxrt.device(0)` still opens, `xrt-smi examine` still lists the device, but
every `hw_context()` fails. This is the same in-driver state that required a
reboot to clear before. It is NOT recoverable from inside the container:

* PCI `reset` is a read-only sysfs file here.
* `xrt-smi validate` needs an archive (`amdxdna/bins/xrt_smi_phx.a`) that is not
  installed in this container image.

Device submissions were stopped immediately and this session continues with
device-independent analysis only, per the campaign rule. All the kbench numbers
below were captured BEFORE the wedge (they are real device measurements).

## Evidence captured before the wedge (kbench62, M512 K1024 N3584, GPU-free)

The single-chain `c256` kernel scales LINEARLY with worker count, refuting the
phase 62 headline that the multi-tile design "never runs its workers
concurrently" - that conclusion held for the `c256x2` two-chain production
kernel but not for the one-chain layout. See RESULTS.md section "multi-tile".
DEVICE OK after hangs

## Oct 24 session: persistent-array device runs

Rebooted device healthy (v4 115.3ms, PPL gate passed 16.3963 x2 deterministic).
phase36 persistent-array measurements (single-pass, per call, bf16->f32 unless noted):
T1=21.25ms/177 GOP/s, T2=10.88/346, T4=5.66/667, int8 T4=6.4/578.
T8 (2x4) and T16 (4x4) persistent designs HANG with ERT_CMD_STATE_TIMEOUT on the
current MLIR-AIE 1.4.3 stack, for both dtypes. Three hangs did NOT wedge the
driver (hw_context still OK after). The original phase36 RESULT.txt already
recorded "Device timeout on repeated runs (requires container restart)" - the
4x4-era design was never stable; the 2120-2368 GOPS figures were obtained in a
different container state. The reported GOPS formula (2*M*K*N/call_time) is a
single-GEMM accounting (verified: T1 time = one GEMM + DMA, not 4 iters).
