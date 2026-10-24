# Phase 20G step 10 - XDNA layout correctness sweep

Status: **PASS**. All 8 shapes match the CPU float64 reference exactly (`max_rel = 0`).
No layout mismatch, so `STOP_NATIVE_LAYOUT` is not triggered.

## Goal (step 10)

Verify the native XRT int8 GEMM layout is correct across multiple M and N, not just
the single M=64/N=64 shape:

```
M in {64, 128, 256, 512}
K = 1024
N in {1024, 2048}
```

CPU float64 reference vs XDNA; any mismatch requires `STOP_NATIVE_LAYOUT`.

## Method

`xdna_smoke.cpp` is parameterized (`xclbin insts M K N`). `run_sweep.sh` drives it over
the 8 cached kernels. For each shape it generates deterministic int8 `A[M][K]`,
`B[K][N]` (values in [-127,127]), runs one native XRT `xrt::run.start()` / `run.wait()`
(full K=1024, single submit), and compares the int32 output against a CPU float64
reference.

Log: `results/step10_sweep.txt`.

## Result

```
SMOKE PASS M=64  K=1024 N=1024 mismatches=0 max_abs=0 max_rel=0.000e+00 state=4
SMOKE PASS M=64  K=1024 N=2048 mismatches=0 max_abs=0 max_rel=0.000e+00 state=4
SMOKE PASS M=128 K=1024 N=1024 mismatches=0 max_abs=0 max_rel=0.000e+00 state=4
SMOKE PASS M=128 K=1024 N=2048 mismatches=0 max_abs=0 max_rel=0.000e+00 state=4
SMOKE PASS M=256 K=1024 N=1024 mismatches=0 max_abs=0 max_rel=0.000e+00 state=4
SMOKE PASS M=256 K=1024 N=2048 mismatches=0 max_abs=0 max_rel=0.000e+00 state=4
SMOKE PASS M=512 K=1024 N=1024 mismatches=0 max_abs=0 max_rel=0.000e+00 state=4
SMOKE PASS M=512 K=1024 N=2048 mismatches=0 max_abs=0 max_rel=0.000e+00 state=4
SWEEP_DONE fail=0
```

`max_abs = 0` and `max_rel = 0` on every shape: int8 x int8 -> int32 is exact integer
arithmetic, so no tolerance is involved or adjusted. `state = 4`
(`ERT_CMD_STATE_COMPLETED`) everywhere.

## Interpretation

The host-visible operand layout is confirmed plain row-major across the whole M/N range
(kernels `M{M}_K1024_N{N}` / `..._C1[_Tm64_Tk64_Tn64]`, tile 64x64x64). This is
consistent with the earlier `IRON_LAYOUT_REPORT` conclusion: no host-side repacking is
needed and the TensorTiler2D transforms stay device-internal. No missing primitive.

## Constraints honored

- Full K=1024 per submit; no K=32 loop.
- Native `xrt::run.start()` / `run.wait()`; no IRON runtime.

## Not yet done (next)

- Step 8: ggml backend async contract (declare `async`/`events`, real
  `graph_compute_async`, `event_record`/`event_wait`).
- Step 14: async partition-ratio sweep (10/25/50/75%).
- Step 15: E5 PP end-to-end (`PP4096..65536`, CPU fallback = 0, per-MUL_MAT submit = 1).
- Step 16: structured per-MUL_MAT overlap logging.
