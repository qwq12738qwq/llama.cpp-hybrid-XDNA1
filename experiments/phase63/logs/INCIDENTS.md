# Phase 63: device incidents

## The NPU was already dead when phase 63 started

The NPU driver was wedged at the end of phase 62: every
`pyxrt.hw_context()` and every `xrt-smi validate` fails with

```
DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22): Invalid argument
```

while `pyxrt.device(0)` (the plain open) still works and `xrt-smi examine`
still lists `RyzenAI-npu1` at 0000:67:00.1.  The PCI function is present,
the driver is loaded, and there are no AER errors
(`aer_dev_fatal: Undefined 0`), so the state is inside the driver, not the
device.

Recovery was attempted and is not possible from inside the container:

| attempt | result |
|---|---|
| `xrt-smi validate -d 0000:67:00.1` | the same ioctl error |
| `echo 1 > /sys/bus/pci/devices/0000:67:00.1/reset` | read-only file system |
| `xrt-smi configure` (the only configuration command) | no reset option |
| `xrt-smi examine` | works (a different ioctl path) |

**Consequence for phase 63:** no NPU execution was possible.  Per the phase
63 stopping rule ("if the device cannot create a context, stop the hardware
submissions, save the scene and continue with the static analysis, the
implementation and the offline compilation"), the phase 63 work is:

* the **resource audit** of every topology, measured with the real compiler
  (`aiecc`): the compiler's own allocation is the measurement, and it does
  not need the device,
* the **16 worker MemTile relay design** (`experiments/phase63/gen_mtile.py`),
  compiled with the real `aiecc` for the production shapes,
* the **offline verification** of the data flow of that design
  (`experiments/phase63/scripts/check_relay_flow.py`): the host packing, the
  kernel's reading convention and the MemTile gather are transcribed into
  numpy and checked against a float64 reference,
* the **cost driven routing policy** in the backend, and the Vulkan side
  measurements of the cost table.

What is *not* in phase 63: the device time of any phase 63 candidate, the
end-to-end hybrid numbers and the perplexity of the new engine.

## The host memory

The machine is shared and its memory is the binding constraint for the
campaigns: 7.2 GB total, 3.1-5.4 GB used by the host, 2.6-3.9 GB in swap
during the session.  The Vulkan-only runs need about 1.5 GB and the NPU
harness about 0.5 GB, so the two can run one after the other but not
together.  `GGML_XDNA_DISABLE=1` was added in phase 62 so that a wedged
device does not also stop the GPU-only measurements; it is used for every
Vulkan number of phase 63.
