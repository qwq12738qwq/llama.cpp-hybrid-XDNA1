# IRON_LAYOUT_REPORT

Phase 20G step 3 (layout archaeology). Scope: determine the exact
host-logical -> device-physical mapping for the validated IRON kernel, using
`int8_gemm_nparallel.py` (M=64, K=1024, N=64, n_cols=1) as the authoritative
oracle.

Two independent evidence sources were used:

1. Source: `int8_gemm_nparallel.py::_build_nparallel_design`
2. Artifact: the generated `aie.mlir` runtime sequence + `kernels_main.json` ABI

A third source, a runtime cross-check, was started but not completed (see
"Open item" below).

## Conclusion first

**No host-side repacking is required.** For this kernel the host-visible
operands are plain row-major contiguous 2D arrays. `TensorTiler2D` /
`dimensionsToStream` transformations exist only between L2 (MemTile) and L1
(core) and are entirely internal to the device program.

This falsifies the earlier working assumption that the all-zero manual-XRT
result was caused by a `TensorTiler2D` operand layout mismatch.

## A. Input operand A

Logical: `int8[M][K]`, row-major, contiguous.

```
A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
```

DMA descriptor for M=64, K=1024:

```
aie.dma_bd(%arg0 : memref<64x1024xi8> offset = 0 len = 65536
           sizes = [1, 16, 64, 64] strides = [0, 64, 1024, 1])
```

- `physical shape` = `64x1024`, contiguous, 65536 bytes
- last dim: 64 elements, stride 1   -> one k-tile (64 contiguous int8)
- 3rd dim: 64, stride 1024          -> 64 M rows
- 2nd dim: 16, stride 64            -> 16 k-tiles (16 * 64 = 1024 = K)
- `tile shape` (device-internal only) = 64x64
- `offset` = 0, no padding, no transpose

## B. Weight operand B

Logical: `int8[K][N]`, row-major, contiguous.

```
B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
```

```
aie.dma_bd(%arg1 : memref<1024x64xi8> offset = 0 len = 65536
           sizes = [1, 1, 1024, 64] strides = [0, 0, 64, 1])
```

- `physical shape` = `1024x64`, contiguous, 65536 bytes
- single 2D block, row-major, no tiling, no transpose at L3/L2

## C. Output operand C

Logical: `int32[M][N]`, row-major, contiguous.

```
C_ty = np.ndarray[(M, N), np.dtype[np.int32]]
```

```
aie.dma_bd(%arg2 : memref<64x64xi32> offset = 0 len = 4096
           sizes = [1, 1, 64, 64] strides = [0, 0, 64, 1])
```

- `physical shape` = `64x64`, contiguous, 4096 bytes
- row-major, no padding, no transpose

## D. ObjectFifo producer/consumer

From `aie.objectfifo` declarations in `aie.mlir`:

```
A_L3L2_0 (ShimNOC -> MemTile)   memref<4096xi8>   depth 2
A_L2L1_0 (MemTile -> Core)      memref<4096xi8>   depth 2
B_L3L2_0 (ShimNOC -> MemTile)   memref<4096xi8>   depth 2
B_L2L1_0 (MemTile -> Core)      memref<4096xi8>   depth 2
C_L1L2_0 (Core -> MemTile)      memref<4096xi32>  depth 2
C_L2L3_0 (MemTile -> ShimNOC)   memref<4096xi32>  depth 2
```

Links: `A_L3L2_0 -> A_L2L1_0`, `B_L3L2_0 -> B_L2L1_0`, `C_L1L2_0 -> C_L2L3_0`.

## E. Reorder / pad / transpose

- **Reordered at host level: none.**
- **Padded at host level: none.**
- **Transposed at host level: none.**
- Reordering exists **only** at the MemTile -> Core boundary via
  `dimensionsToStream`:

```
A_L2L1_0 [<size=16,stride=256>, <size=8,stride=8>, <size=4,stride=64>, <size=8,stride=1>]
B_L2L1_0 [<size=8, stride=512>, <size=8,stride=8>, <size=8,stride=64>, <size=8,stride=1>]
C_L2L3_0 [<size=16,stride=256>, <size=4,stride=8>, <size=8,stride=32>, <size=8,stride=1>]
```

All three factor to 4096 elements. These are device-internal.

## XRT kernel ABI

From generated `design.prj/kernels_main.json`:

```
arg0 opcode  uint64_t  offset 0x00
arg1 instr   char*     offset 0x08   memory-connection = SRAM
arg2 ninstr  uint32_t  offset 0x10
arg3 bo0     void*     offset 0x14   memory-connection = HOST
arg4 bo1     void*     offset 0x1C   memory-connection = HOST
arg5 bo2     void*     offset 0x24   memory-connection = HOST
arg6 bo3     void*     offset 0x2C
arg7 bo4     void*     offset 0x34
```

Observed ABI usage:

| arg | IRON (`hostruntime.py:311`) | manual (`xdna-persistent.cpp`) |
|-----|------------------------------|-------------------------------|
| 0 opcode | `3` | `3` |
| 1 instr | `bo` flags `cacheable`, `group_id(1)` | same |
| 2 ninstr | `insts.nbytes` = 420 | `inst_bytes` = 420 |
| 3.. | `*buffers` (A, B, C) | input_bo, weight_bo, output_bo |

Observed `group_id`: `group_id(1) = 65537 (0x10001)`, `group_id(3) = 65536 (0x10000)`.
Identical on both paths.

## Runtime-path comparison (IRON vs manual C++)

| step | IRON | manual |
|------|------|--------|
| device | `self._device` | `xrt::device(0)` |
| xclbin | `pyxrt.xclbin(path)` | `xrt::xclbin(path)` |
| register | `register_xclbin` | `register_xclbin` |
| uuid | `xclbin.get_uuid()` | `xclbin.get_uuid()` |
| hw_context | `pyxrt.hw_context(dev, uuid)` | `xrt::hw_context(dev, uuid)` |
| kernel | `pyxrt.kernel(ctx, "MLIR_AIE")` | `xrt::kernel(ctx, "MLIR_AIE")` |
| ninstr | element size in bytes | element size in bytes |
| **wait status** | **checked** (`r != ERT_CMD_STATE_COMPLETED` raises) | **discarded** |

This is the only functional difference found on the call path. It is not yet
confirmed to be the cause; see Open item.

## Artifact cross-check

IRON's per-run cache is at
`/home/ubuntu/.npu/cache/<hash>/` (transient; exists only while the process
runs).

| file | IRON cache | `kernel_cache/M64_K1024_N64/` |
|------|------------|-------------------------------|
| `final.xclbin` / `design.xclbin` | `1c798dfc...` 11097 B | `72c7fa05...` 11417 B |
| `insts.bin` | `cb958182...` 420 B | `cb958182...` 420 B (byte-identical) |
| `main.pdi` | 4640 B | 4960 B |
| core kernel object | `matmul_i8_i32_549bf467.o` | `matmul_i8_i32_91e3a8c6.o` |

`diff aie.mlir` shows differences **only** in MLIR-AIE IR syntax generation,
not in the runtime sequence:

```
// IRON build (works)
%1 = aie.objectfifo.acquire @C_L1L2_0(Produce, 1) : memref<4096xi32>

// kernel_cache build (all-zero)
%1 = aie.objectfifo.acquire @C_L1L2_0(Produce, 1) : !aie.objectfifosubview<memref<4096xi32>>
%2 = aie.objectfifo.subview.access %1[0] : ... -> memref<4096xi32>
```

The `aie.runtime_sequence` (DMA descriptors, i.e. the layout) is identical in
both.

**Repacking the operands would not have changed the outcome.** Feeding the
manual path IRON's newly generated `final.xclbin` + `insts.bin` still produced
4096/4096 mismatches, so neither the xclbin generation nor the operand layout
is the cause.

## Falsified hypotheses

- TensorTiler2D operand repacking needed at host level - **false**
- XRT kernel ABI mismatch (scale args, arg order, ninstr meaning) - **false**
- BO flags / group_id mismatch - **false**
- stale or invalid xclbin in `kernel_cache/` - **false**
- `insts.bin` mismatch - **false** (byte-identical)

## Open item (remaining real blocker)

The manual XRT path does not inspect the return value of `run.wait()`. IRON
does, and treats anything other than `ERT_CMD_STATE_COMPLETED` as an error.
A Python `pyxrt` probe was written to read the state from the same call shape
as the manual path, but it did not finish: `pyxrt.bo` exposes `sync` without a
`to_device` / `from_device` direction constant, so the probe needs the correct
sync-direction invocation before it can run.

Next concrete step: finish the probe, read `run.wait()`'s state on the manual
call shape, and confirm whether the kernel is reporting an error (which would
explain all-zero output as "kernel never ran") or completing successfully
(which would move suspicion back to the operand BO contents).

---

# RESOLVED (supersedes the "Open item" section above)

The all-zero output was **never a library or layout problem**. It was a bug in
the test harness.

## Root cause

`numpy.ndarray.flatten()` always returns a **copy**; `ravel()` returns a view
for a C-contiguous array. The harness did:

```python
output_flat = output_xdna.flatten()          # COPY
libxdna..._wait_c(engine, run_ptr, output_flat.ctypes.data_as(POINTER(c_int32)))
mismatches = np.sum(output_xdna != output_cpu)   # reads the ORIGINAL array
```

The kernel wrote correct results into the copy, the copy was discarded, and the
verification read `output_xdna`, which had never been written. Hence
"4096/4096 mismatches" with all zeros.

Proof:

```
flatten shares memory: False
ravel   shares memory: True
```

After changing `.flatten()` -> `.ravel()` in the harness:
`mismatches: 0 / 4096`.

## ERT_CMD_STATE value correction

`run.wait()` returning `4` is **COMPLETED**, not ERROR. From
`/usr/include/xrt/detail/ert.h`:

```c
ERT_CMD_STATE_NEW = 1, QUEUED = 2, RUNNING = 3,
ERT_CMD_STATE_COMPLETED = 4, ERT_CMD_STATE_ERROR = 5, ...
```

An independent C++ probe (`/tmp/xrt_cxx_probe.cpp`, links the same
`libxrt_coreutil.so.2`) reproduced the library's exact call sequence with
`state=4` and a correct `C[0,0]`. The earlier reading of `state=4` as an error
was wrong.

## Consequence

- No host-side packing adapter is needed (host operands are plain row-major).
- ggml-xdna's full-K single-submit path is correct on both `execute()` and
  `submit()`/`wait()`.
- `test_async.py` and `test_exec.py` PASS for M=64 K=1024 N=64.
- Gate A (async submit / wait separation) is now backed by a correctness result,
  not just a "returned without waiting" observation.
