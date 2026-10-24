# Phase 24 - Persistent runtime sequence feasibility (minimal)

Goal (per user): test whether N XDNA GEMMs can be submitted as ONE host
submission with a single completion, instead of submit+wait per GEMM, using
XRT's runlist / MLIR-AIE dynamic runtime sequence mechanism.

Artifacts:
- phase24_sequence_probe.cpp / phase24_sequence_probe
- results/phase24_sequence.csv

## Mechanism used
`xrt::runlist` (`/usr/include/xrt/experimental/xrt_kernel.h`): "manage a list of
xrt::run objects such that they can be executed atomically in the order they are
added"; `execute()` submits the whole list, `wait()` waits once. pyxrt also
exposes module/runlist. This is the closest available XRT primitive to a
dynamic runtime sequence. NOTE: the cached design.xclbin has no separate
ELF/PDI control program (only design.xclbin + main.pdi + design.insts.bin), so
the MLIR-AIE 1.4 dynamic runtime sequence control flow is not compiled in;
runlist batches the ERT submissions host-side instead.

## Result, M=512 K=1024 N=3584, median per-GEMM ms (reps=5)

  mode          N=1     N=2     N=3     N=4
  perop_new    4.19    4.25    4.31    3.66     (fresh run each GEMM)
  perop_reuse  4.14    4.05    3.50    3.87     (one run reused)
  runlist      4.64    3.85    3.34    3.30     (one execute, one wait)
  correctness runlist: mismatches=0 PASS

Standalone kernel time (phase21 benchmark_gemm) = 3.72 ms, so the per-GEMM time
is dominated by kernel execution, not host submission.

## Interpretation
- One-submission batching (runlist) saves ~0.3-1.0 ms/GEMM at N>=3 (N=4: 3.30 vs
  3.66 perop_new; N=3: 3.34 vs 4.31) and reproduces correct results.
- It does NOT reduce the ~3.3 ms/GEMM kernel execution floor. The host-side
  submit/wait overhead it removes is only ~0.3-0.6 ms/GEMM.
- Combined with Phase 23: the per-op hybrid tax is NOT host submission and NOT
  NPU DMA/sync; it is (a) the ~200 MB/s Vulkan slice readback and (b) CPU
  fp32<->int8 quantize/merge. runlist cannot touch either.

## Verdict
Feasibility: PASS (single host submission for N GEMMs works and is slightly
faster). Impact on the E2E hybrid: small (~0.3-1.0 ms/op of a ~20-26 ms/op
hybrid tax), insufficient to flip E2E to a win.

To go further one would need the actual MLIR-AIE dynamic-runtime-sequence
control program compiled into the design (device-side sequence, no per-GEMM ERT
packet), plus the cross-device activation/weight lifetime work.
