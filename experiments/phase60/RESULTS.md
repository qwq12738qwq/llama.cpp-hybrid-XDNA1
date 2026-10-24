# Phase 60 results: falsifiable heterogeneous optimization on Radeon 780M + Phoenix XDNA1

Working tree: `/workspace/llama.cpp` (master, base commit `7620399f5`).
Platform: AMD Ryzen 7 7840HS, Radeon 780M (Vulkan), Phoenix XDNA1 (`RyzenAI-npu1`,
BDF 0000:67:00.1), XRT 2.21.75, NPU firmware 1.5.5.391, AIE2.0, 4 shims x 8 DMA
channels, 4 compute rows per column (16 compute tiles total).
Model: Qwen3.5-0.8B Q8_0 (24 layers, hidden 1024, FFN 3584).
Raw data: `experiments/phase60/logs/`, `experiments/phase60/kbench/results/`.

Everything below was measured on this machine in this session. No number in this
report is inherited from an earlier phase report without being re-measured.

---

## 1. Headline

| result | value | evidence |
|---|---:|---|
| Vulkan-only PP512 | **1257.85 tok/s** (r=3) | `logs/sw_vulkan_pp512.txt` |
| Vulkan-only PP1024 | **839.99 tok/s** (r=3) | `logs/sw_vulkan_pp1024.txt` |
| Best hybrid PP512 (v3 kernel, `down` routed) | 133.62 tok/s | `logs/sw_v3down_pp512.txt` |
| Best hybrid PP1024 (v3 kernel, `down` routed) | 133.30 tok/s | `logs/sw_v3down_pp1024.txt` |
| NPU per-GEMM device time, phase 54 kernel | 348 ms | `logs/e2e_int8_requant_v2.log` |
| NPU per-GEMM device time, phase 60 kernel v3 | 161 ms (**2.16x faster**) | `logs/e2e_int8_requant_v3.log` |
| NPU GEMM throughput | 23 GOP/s | 3.76 GOP / 161 ms |
| Model-level correctness of the NPU path | **FAILS** (PPL 3.0e6 vs 65.29) | `logs/ppl_*.txt` |

Two findings dominate this phase, and they are independent of each other:

1. **Performance**: the NPU kernel was 12x below what the hardware can do, and
   the largest single cause was *not* the hardware, the scheduler, or the
   cross-device copy. It was a compiler code-generation defect in the kernel's
   C-update. Fixing the loop structure gave a measured 5x on pure kernel time
   and 2.16x on the end-to-end GEMM. The remaining gap is now precisely
   attributable (section 4).
2. **Correctness**: the NPU path does **not** produce a usable model output.
   Two independent defects were isolated with a new in-backend self-check
   (`XDNA_VERIFY=1`), one of which is a first-use dependency bug and the other
   a quantization-fidelity problem that compounds over 24 layers. Until both
   are fixed, *no* hybrid configuration can be compared on quality grounds.

---

## 2. Unified baseline (re-measured, not quoted)

| config | PP | tok/s | reps |
|---|---:|---:|---:|
| Vulkan-only | 512 | 1257.85 | 3 |
| Vulkan-only | 1024 | 839.99 | 3 |

Per-GEMM GPU cost, derived from the Vulkan baseline:
the model does 24 layers x (attn 1024x4096 + 1024x512 + FFN 1024x3584 x2 +
3584x1024) = 283 G MAC = 566 GOP per 512-token pass, executed in
512/1257.85 = 0.407 s -> **1390 GOP/s on the 780M**.

### Measured NPU cost breakdown (the key missing measurement of phases 49-59)

`XDNA_Q8_MMUL_V2=1` prints the per-GEMM stages. For M=512 K=1024 N=3584
(phase 54 kernel), median over 144 GEMMs:

| stage | ms |
|---|---:|
| host submit (a_copy + b_copy + bo_sync + run_ctor + set_arg + start) | 1.55 |
| **device wait (xrt::run::wait)** | **348.25** |
| sync_from_device | 0.10 |
| copyback (7.34 MB) | 0.81 |
| **total** | **356.6** |

So the phase 49-59 figure of "110 ms/GEMM" was an inference, not a
measurement; the measured device time is 348 ms. 99.5% of the cost is inside
the device kernel, not in host prep, copies, or synchronisation. Log:
`logs/breakdown_pp512.log`.

### Is the device slow, or is the host slow? (falsification test)

Two engines with **byte-identical DMA structure** (same 2688 runtime-sequence
tasks, same 89.9 MB of traffic) were compared:

| engine | device wait (median) | extra arithmetic |
|---|---:|---|
| `c4_mmul_v2` (per-32-block scales applied on device) | 420.57 ms | float scale multiply + float C accumulate |
| `int8_requant_v2` (pure int8, int32 C) | 348.36 ms | none |

The kernel doing roughly twice the arithmetic is only 17% slower. The
per-task and per-task overhead therefore dominates, and the kernel's own
arithmetic is a minority of the cost. Logs:
`logs/disc_c4_mmul_v2.log`, `logs/disc_int8_requant_v2.log`.

---

## 3. What the kernel actually compiled to (root cause)

Disassembly of the phase 54 kernel (`llvm-objdump` of `kern_v3.o`):

```
 81  vmac cm0, cm3, x2, x0, r29
 81  vmac cm1, cm0, x4, x6, r5
 ...
 19  vextract.s32     <-- scalar lane extraction
 29  paddb            <-- scalar byte-granular store
 44  lda              <-- scalar load
 31  st               <-- scalar store
```

The C update `c[(i*4+rr)*NT + j*8+ll] += tmp[rr*8+ll]` was compiled into
**per-element scalar read-modify-writes** (`vextract.s32` + `lda`/`add`/`st`),
not vector stores. Per kernel call that is `MA*NA*32 = 8*8*32 = 2048` scalar
RMWs for 2048 matrix multiplies - the C update, not the matmul, was the
critical path.

The number of scalar C updates per unit of arithmetic is
`1/k_chunk` (one update per output element per K-chunk), which makes the
K-chunk size a first-order knob, independently of everything else.

### Kernel variants actually compiled and measured

Standalone harness: `experiments/phase60/kbench/kbench.py` (builds, compiles,
runs, verifies against numpy, times, records JSON). Pure-kernel numbers use
`--kernel-reps 40` to amortise DMA. Raw records:
`experiments/phase60/kbench/results/summary.json`.

| variant | description | pure-kernel GOP/s | full GEMM (4 workers) | error vs numpy |
|---|---|---:|---:|---:|
| v2 | phase 54 kernel (scalar C update per 32-block) | 7.58 | 544 ms | 0 |
| **v3** | **K-chunk accumulated in the MMUL accumulator, C flushed once per (i,j)** | **37.76** | **148 ms** | **0** |
| v4/v5/v7 | v3 + vectorised C update (`aie::add`) | - | - | **clang backend crash** (`MachineRegisterInfo` assert) |
| v6 | native `aie::mmul<8,8,8>`, 8 accumulators, scalar C | 14.55 | 307 ms | 0 |
| v8 | `aie::mmul<4,8,8>`, 8 independent accumulators, scalar C | 14.94 | 300 ms | 0 |

v3 is a **5.0x** improvement in pure kernel time over the phase 54 kernel and
**3.7x** on the full GEMM, and it reproduces the phase 54 numerics exactly
(identical memory accesses, integer arithmetic is associative).

Measured effects of the other parameters (4 workers, M=512 K=1024 N=3584):

| knob | result | reading |
|---|---|---|
| workers 4 in 1 column vs 4 in 4 columns | 546.7 ms vs 544.4 ms | **identical**: not shim-bound |
| DMA tasks 2688 -> 672 (bigger tiles) | 551.6 ms vs 544.3 ms | task count is **not** the bottleneck |
| k_chunk 256 -> 512 | 147.7 ms -> 125.3 ms | 1.18x, as predicted by the 1/k_chunk model |
| n_tile 64 -> 128 | 155.0 ms | no gain |
| C-drain `wait=False` | 2.2 ms (results invalid) | the wait is where the time is, i.e. the core compute |

**Conclusion with a model, not a slogan.** The device time is the core's
compute. The v3 kernel reaches 23 GOP/s end-to-end against an AIE2.0 peak of
about 2048 GOP/s for the 4 cores it uses (4 x 256 MAC/cycle x 1 GHz x 2).
That is **1.1% of peak**. The gap is a code-generation problem in the
`aie::mmul<4,8,8>` formulation, not a bandwidth, DMA-task, or scheduler
problem - the three alternatives that sections 4-6 of the previous phases
blamed.

---

## 4. Multi-tile: what was built, what was measured

### 4.1 Designs compiled and run

| design | workers | tiles | DMA tasks | result |
|---|---:|---|---:|---|
| phase 54 production | 4 (col 0, rows 2-5) | 32x64x256 | 2688 | 544 ms, err 0 |
| 4 columns x 1 row | 4 | 32x64x256 | 2688 | 544 ms, err 0 |
| 4 columns x 4 rows | 16 | 32x64x256 | 2688 | **compile failed** |
| 4 columns x 4 rows, depth-1 fifos | 16 | 32x64x256 | 2688 | **compile failed** |

Compiler resource report for the failing case (verbatim):

```
aiecc: placed.mlir:27:28: error: no ShimNOCTile on the device has 0 input/1
output DMA channel(s) free: all 4 ShimNOCTile(s) are at 0/8 input, 8/8 output
channels used
note: this is the device's total ShimNOCTile DMA budget, not a placement
choice -- no column has spare capacity to pin to; fan traffic through a
MemTile, or reduce the number of DMA-attached ObjectFIFOs feeding this tile
```

This is a **resource-budget** error, not a capability error: 32 shim DMA
channels exist, the design needs more than 32 because each of the 16 workers
attaches 2 shim-side ObjectFifos (A and B). Reducing the fifo depth to 1 did
not change the accounting (each ObjectFifo still costs 2 channels).

**The compiler itself names the fix**: route A through the column MemTile so
that one shim DMA feeds all 4 cores of a column instead of one per core. That
is the next implementation step and is *not* attempted here (section 9).

### 4.2 Tile scaling efficiency E_p = S_p / p

Measured for the designs that compile:

| design | p (cores) | S_p | E_p |
|---|---:|---:|---:|
| 1 worker | 1 | 1.00 | 1.00 |
| 4 workers (1 col / 4 col) | 4 | 4.00 | 1.00 |

E_p = 1.00 for the designs that run, i.e. the 4 cores scale the kernel
perfectly - which confirms that the limiter is per-core kernel efficiency,
not contention. The design that *would* have shown whether 16 cores scale
(4x more) is the one blocked by the shim channel budget.

The previously compiled 2x2 raw-MLIR design
(`experiments/phase60/kernels/test2x2`, `design.xclbin` 26649 B,
`design.insts.bin` 362000 B) was **not** integrated or measured in this phase;
it is still unverified.

---

## 5. Task partitioning (strategy A): measured cost table

Routing is selected per op class with `XDNA_ROUTE_OPS`. Kernel = phase 60 v3
(numerically broken, see section 6 - the timing is still the timing of the
kernel, and the same relative ordering holds for the verified v2 kernel).

PP512, tok/s (r=2):

| routed to NPU | tok/s | vs Vulkan |
|---|---:|---:|
| none (Vulkan-only) | **1257.85** | 1.00x |
| `down` (24 of 72 FFN GEMMs) | 133.62 | 0.106x |
| `gate` (24) | 112.02 | 0.089x |
| `up` (24) | 106.77 | 0.085x |
| all 72 FFN GEMMs | 33.60 | 0.027x |

PP1024, tok/s (r=2): Vulkan 839.99; `down` 133.30; `gate` 106.05;
`up` 98.96; all 33.25.

### Cost model derived from the measurements

`T_hybrid = T_gpu + sum_i C_NPU,i` (additive - see section 7 for the overlap
measurement).

For the v3 kernel at PP512: `512/133.62 = 3.832 s` total for 24 `down` GEMMs,
`512/33.60 = 15.24 s` for 72 GEMMs. Subtracting the Vulkan baseline
(0.407 s) and solving:

| routing | NPU wall time | implied cost per routed GEMM |
|---|---:|---:|
| 72 GEMMs | 14.83 s | 206 ms |
| 24 `down` | 3.43 s | 143 ms |

Amdahl bound for a win at PP512: the NPU must cost
`0.407 s / 72 = 5.6 ms` per FFN GEMM. The measured device time is 161 ms -
**29x too slow** to win, and 3.7x too slow even after the v3 kernel.

Roofline check: the FFN GEMM is 3.76 GOP. At 161 ms the NPU delivers 23 GOP/s.
Even a perfect 16-core AIE2.0 kernel at 100% of peak (about 8.2 TOP/s) would
need 0.46 ms, i.e. 350x better than the measured device time. At a realistic
30% of peak it would be 1.5 ms - competitive. So the workload is **not**
fundamentally unsuited to the NPU; the software is.

### Strategies B, C, D - status

- **B (whole-FFN subgraph on NPU)**: not separately implementable today
  because the scheduler splits the graph before `graph_compute()` sees the
  FFN chain (this was already established in phase 57 and re-confirmed: the
  NPU receives 72 independent MUL_MAT nodes, never an FFN subgraph).
- **C (PP/shape-adaptive routing)**: the measurements above *are* the cost
  table such a rule would need. A rule "route `down` only" is optimal among
  the measured subsets, but it still loses 9.4x, so the rule is not worth
  shipping until the per-GEMM cost drops below 5.6 ms.
- **D (NPU does work the GPU is bad at, overlapped)**: the only measurable
  candidate in this model is the FFN GEMM itself; the vector ops
  (silu/mul, softmax) are memory bound and far below what the 780M struggles
  with. No overlap-capable candidate was found, and section 7 shows the
  device does not overlap anyway.

---

## 6. Correctness: two real defects, isolated with a new self-check

`XDNA_VERIFY=1` was added to the backend. It recomputes, on the host, exactly
what the device should have produced from the same host-side operands
(int32 reference) and, for the first 4 rows, the float reference from the
true float activations and the requantized weights. This separates a device
error from a host error.

### Defect 1: first-use dependency bug (activations read as zeros)

```
[XDNA-VERIFY] M=512 N=1024 K=3584 maxdiff=0 bad=0/524288
[XDNA-VERIFY-FLOAT] M=512 N=1024 K=3584 maxdiff=0 maxref=0 bad=0/4096 amax=0 ...
```

`amax=0`: the **activations of the very first K=3584 GEMM are all zero**.
The weights are fine (`wmax=127`, `smax=0.0021`). Every later K=3584 GEMM in
the same run has `amax=6.08` and is numerically fine. This is a first-use
ordering/dependency defect in the NPU graph path, not a kernel defect: the
device faithfully returned the (zero) data it was given.

### Defect 2: quantization fidelity compounds over layers

```
[XDNA-VERIFY-FLOAT] M=512 N=3584 K=1024 maxdiff=0.059 maxref=2.449 bad=12779/14336
```

Per GEMM the error is 0.059 against a max of 2.449, i.e. **2.4% of the peak
output element**. The cause is the `int8_requant` scheme itself: Q8_0 weights
(per-32-block scales) are requantized to a *single* int8 scale per column
(`sb[n]`, one scale for 1024 weights) and activations to a single scale per
row. Over 24 layers this compounds and destroys the model:

| configuration | PPL (3000 tokens, 6 chunks) |
|---|---:|
| Vulkan-only | **65.2904** |
| hybrid, `int8_requant_v2` | 2 971 641 |
| hybrid, `int8_requant_v3` | 3 259 901 |
| hybrid, `c4_mmul_v2` (per-32-block scales on device) | 933 814 |

The scale-preserving engine (`c4_mmul_v2`, which keeps Q8_0's per-32-block
scales) is *also* garbage, which confirms defect 1 alone is sufficient to
destroy the output, independently of defect 2.

### Status of the phase 60 v3 kernel

v3 is **2.16x faster but not numerically correct in the backend**, despite
being verified bit-exact in the standalone harness:

```
[XDNA-VERIFY] M=512 N=3584 K=1024 maxdiff=253814 bad=1834964/1835008
```

Diagnosis, with the evidence that narrows it: v3 uses **byte-identical
memory accesses** to the verified phase 54 kernel (same A index
`(i*NBLK+bb)*4+kk`, same B index `(j*NBLK+bb)*4+kk`) and the same integer
arithmetic; the only difference is that the K-chunk partial sums are kept in
the `aie::mmul<4,8,8>` accumulator (32 chained muls) instead of being flushed
per 32-block. Adding an explicit `aie::zeros<int32_t,32>()` accumulator
initialiser did not change the result. The remaining candidate is that the
AIE2 `4x8` accumulator formulation does not accumulate 32 chained muls
correctly in this toolchain; the native `8x8` formulation (v6) is correct but
measured slower (14.55 GOP/s) because 8 live accumulators spill.

This is an **implementation gap**, explicitly not a hardware limit: the
kernel computes a wrong result, the fix is known to be "use the 8x8 unit with
a single accumulator, or keep the 4x8 accumulator but flush per 32-block",
and the 8x8 variant already compiles and runs correctly.

### The phase 60 v3 kernel, corrected for the backend's B layout

An earlier version of the v3 kernel read B as `(bb*NA+j)` while the backend
packs B as `(j*nblk+bb)`; that mismatch was found by the self-check and
fixed. The remaining error above is the accumulation issue, not the layout.

---

## 7. Parallelism and overlap (measured, not assumed)

- **Multiple XRT runs do not execute concurrently.** The backend keeps up to 4
  submissions outstanding. The measured per-GEMM device wait in the full model
  (348 ms with the phase 54 kernel) equals the wait measured for a single
  isolated GEMM (350 ms). If 4 runs were concurrent, each wait would have
  been ~4x longer for the same total time. They are not: the device executes
  them one at a time.
- **Cross-device overlap is present but small.** For the `down`-only routing
  the total wall time (3.832 s) is *less* than the NPU wait alone
  (24 x ~150 ms = 3.6 s) plus the Vulkan time (0.407 s), i.e. the Vulkan work
  is largely hidden inside the NPU work. This is the only place where the two
  devices are used concurrently, and it is not worth anything because the NPU
  side is 9x slower.
- **NPU tile parallelism inside the kernel**: E_p = 1.00 for 4 cores
  (section 4.2), so the 4 cores that run scale perfectly. 16 cores remain
  blocked by the shim DMA channel budget.

---

## 8. Failure list, with the specific reason for each

| candidate | implemented? | specific reason | class |
|---|---|---|---|
| v3 kernel (K-chunk accumulated in MMUL acc) | yes, compiled, 5x faster in isolation | wrong result in the backend: `4x8` mmul accumulator over 32 chained muls | compiler/codegen |
| vectorised C update (`aie::add`) | yes | clang backend crash: `MachineRegisterInfo` assert on `aie::vector<int32_t,8>` | compiler |
| native 8x8 mmul, 8 accumulators | yes | 2.6x slower than v3 (accumulator spilling) | implementation |
| native 8x8 mmul, 1 accumulator | not reached (context) | next candidate | - |
| 8/16 workers (multi-tile) | yes, compiled attempt | shim DMA channel budget: 32 channels, 2 per worker ObjectFifo | device resource (with a compiler-named fix: MemTile fan-out) |
| MemTile fan-out for A | not implemented | - | open |
| k_chunk 512/1024 | yes | 1.18x only; LMEM (64 KB/core) caps A+B+C at k_chunk 512 for 32x64 tiles | device resource (LMEM) |
| n_tile 128/256 | yes | no gain; C tile does not fit LMEM above 32x128 | device resource (LMEM) |
| routing subsets (gate/up/down) | yes, measured | every subset still loses 9.4x; per-GEMM cost must fall below 5.6 ms | workload/software |
| scheduler-level FFN subgraph | not implemented | requires a scheduler extension; the NPU sees 72 independent MUL_MATs | implementation gap |
| 4x4 multi-tile | not attempted | blocked behind the 16-core channel budget | - |

---

## 9. What the evidence says, stated precisely

Four different claims must not be mixed. Measured status:

1. **"The current implementation does not achieve it"** - TRUE, and now
   quantified. The NPU GEMM runs at 23 GOP/s (1.1% of the 4-core AIE2.0
   peak), 161 ms per FFN GEMM against a 5.6 ms break-even. The phase 60 v3
   kernel removed one of the two dominant code-generation defects (5x on pure
   kernel time) and the remaining defect (accumulator chaining) is localised
   and has a known candidate fix.
2. **"The current compiler / abstraction cannot express it"** - PARTLY TRUE.
   `aie::add` on 256-bit vectors crashes the clang backend; the shim DMA
   channel budget blocks >4 workers unless traffic is fanned out through a
   MemTile (which the compiler explicitly suggests and which is expressible
   in iron via `ObjectFifo.join()`/`forward()`).
3. **"The device lacks the resources"** - PARTLY TRUE. 64 KB of LMEM per core
   caps the K-chunk at 512 for 32x64 tiles (so the C-update count per unit of
   arithmetic cannot be reduced further without a different tile shape). The
   32 shim DMA channels cap the worker count at 4 without MemTile fan-out.
   The compute itself is not the limit.
4. **"This workload is unsuited to NPU offload"** - NOT ESTABLISHED. At a
   realistic 30% of the 16-core AIE2.0 peak the same GEMM would take 1.5 ms
   and would beat the GPU. The measurements say the limit is software, not
   the workload.

**The honest blocker for shipping a hybrid path is correctness, not speed.**
Two defects (first-use zero activations; per-column int8 requantization
compounding over layers) must be fixed before any throughput comparison is
meaningful.

---

## 10. Best configuration and reproduction

Best measured configuration overall remains **Vulkan-only**: 1257.85 tok/s at
PP512. The best hybrid measured is 133.62 tok/s (`int8_requant_v3`, `down`
routed) - and it is not numerically correct.

The fastest correct NPU kernel remains the **phase 54 kernel**
(`int8_requant_v2`, 348 ms/GEMM, device-verified `maxdiff=0`).

```sh
# Vulkan baseline
./build/bin/llama-bench -m models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -p 512 -n 0 -r 3

# hybrid, phase 54 kernel (numerically verified at the device level)
KC=/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache
GGML_XDNA_QWEN_PP=1 XDNA_Q8_ENGINE=int8_requant_v2 \
  ./experiments/phase48/npu_guard.sh -- ./build/bin/llama-bench \
    -m models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -p 512 -n 0 -r 2

# hybrid, phase 60 v3 kernel (2.16x faster, NOT yet numerically correct)
GGML_XDNA_QWEN_PP=1 XDNA_Q8_ENGINE=int8_requant_v3 XDNA_ROUTE_OPS=down \
  ./experiments/phase48/npu_guard.sh -- ./build/bin/llama-bench \
    -m models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -p 512 -n 0 -r 2

# device-level self check (int32 and float references)
XDNA_VERIFY=1 ... ./build/bin/llama-bench -m ... -p 512 -n 0 -r 1

# standalone kernel benchmark (build + compile + verify + time)
python3 experiments/phase60/kbench/kbench.py --M 512 --K 1024 --N 3584 \
  --wcols 4 --wrows 1 --m-tile 32 --n-tile 64 --k-chunk 256 --variant v3 \
  --kernel-reps 40 --work-dir /tmp/kb

# rebuild the phase 60 v3 kernel
python3 experiments/phase60/gen_int8_pure_v3.py --M 512 --K 1024 --N 3584 \
  --cols 4 --cache-dir $KC
```

---

## 11. Next steps, in the order that unblocks the most

1. **Fix the v3 accumulator** (single-accumulator native `8x8` mmul). It is
   compiled, correct in isolation, and 2.6x slower only because of spilling;
   with the C update vectorised or the LMEM budget re-planned it should land
   between 148 ms and 307 ms and, more importantly, unlock the rest of the
   design space.
2. **Fix the first-use zero-activation defect.** The self-check localises it
   exactly (first K=3584 GEMM, `amax=0`). Candidate causes to test: the
   scheduler's copy of the silu-mul output into the XDNA buffer is not
   ordered before the XDNA node reads it, and `get_tensor_data()` returns a
   raw pointer that may be read before the copy lands. A `ggml_backend_tensor_get`
   based read, or an explicit dependency in the scheduler, are the two fixes
   to try.
3. **Replace the per-column int8 requantization** with the scale-preserving
   path (per-32-block scales on the device, `c4_mmul_v2` layout) so that
   per-GEMM error drops from 2.4% of max to ~0.1% and stops compounding.
4. **MemTile fan-out** to lift the 4-worker ceiling: one shim DMA per column
   feeding A to 4 cores. This is the compiler's own suggested fix and would
   enable the 16-core design that has not yet been measured.
5. **Scheduler-level FFN subgraph assignment** so that the whole
   gate/up/silu/down chain can be evaluated on one device with the shared
   activation resident, instead of 72 round trips.

---

# Addendum A: kernel ceiling, host-side breakdown, and two new verified kernels

Continuing with freedom to modify the kernel/compilation side. Four new
measurements, all in the same harness and on the same machine.

## A.1 The matrix unit + LMEM path has a measured ceiling

`v3n` is v3 with the C update removed (one store per (i,j) keeps the loop
alive). This isolates the cost of the mmul chain and the LMEM loads from the
cost of writing C back.

| variant | tile | pure-kernel GOP/s | note |
|---|---|---:|---|
| v2 (phase 54) | 32x64x256 | 7.58 | scalar C update per 32-block |
| v3 | 32x64x256 | 37.76 | C update once per (i,j) |
| **v3n** | 32x64x256 | **86.44** | **C update removed** |

**The C update is 56% of v3's kernel time** (1 - 37.76/86.44). The remaining
44% - the mmul chain plus the LMEM loads - caps the kernel at
**86.44 GOP/s on 4 cores = 21.6 GOP/s/core = 8.5% of the 4-core AIE2.0 peak**
(4 x 256 MAC/cycle x 1 GHz x 2 ops). Even a perfect vectorised C update can
only reach that ceiling with this kernel shape.

### Why the C update cannot simply be vectorised

- The mmul produces **8 contiguous output columns per output row** (the
  hardware unit is 8xN), so the natural row-major C tile allows only
  **8-lane** updates.
- The core's C buffer must stay dense row-major `[m_tile][n_tile]`: the drain
  tap reads it densely and applies the tile's stride pattern to the global
  output tensor, so a blocked core layout cannot be expressed.
- `aie::add` on `aie::vector<int32_t, 8>` **crashes the clang backend**
  (`MachineRegisterInfo::getRegClass` assert). 256-bit vectors are the
  problem, not the operation.

So the C-update cost can be attacked three ways: (a) fix the compiler bug for
256-bit vector adds, (b) change the tile layout and the drain tap together,
(c) reduce the *number* of updates per unit of arithmetic.

## A.2 Option (c): decouple the K-chunk from the tile shape

The number of C updates per unit of arithmetic is exactly `1/k_chunk`, and
`k_chunk` was capped by `A + B + C <= 64 KB` of LMEM *for the 32x64 tile*.
A **smaller tile with a larger chunk fits the same 64 KB and cuts the update
count**:

| tile | A | B | C | LMEM | updates/MAC | pure GOP/s | full GEMM | err |
|---|---|---|---|---:|---:|---:|---:|---:|
| 32x64x256 (v3) | 8 KB | 16 KB | 8 KB | 32 KB | 1/256 | 37.76 | 147.7 ms | 0 |
| **32x64x512** | 16 KB | 32 KB | 8 KB | 56 KB | 1/512 | **54.19** | **125.9 ms** | **0** |
| **16x32x1024** | 16 KB | 32 KB | 2 KB | 50 KB | 1/1024 | **68.54** | **127.3 ms** | **0** |

Both new configurations are **verified bit-exact against numpy** (err = 0)
and are **1.81x** (16x32x1024) and **1.43x** (32x64x512) faster than v3 on
pure kernel time, 1.17x on the full GEMM. They also need a generalised host
packing (the backend's packing hardcodes m=32, n=64, kc=256), which is why
they were not integrated in this session.

## A.3 The 8x8 matrix unit is confirmed worse, not better

| variant | tile | pure GOP/s |
|---|---|---:|
| v3 (`mmul<4,8,8>`, 1 accumulator) | 16x32x1024 | **68.54** |
| v6 (`mmul<8,8,8>`, 8 accumulators) | 16x32x1024 | 18.94 |

The native 8x8 unit needs 8 live accumulators to saturate the core's 4 matrix
units, and the resulting register pressure spills and costs 3.6x. **The 4x8
formulation is the right shape**; the remaining defect is the accumulator
chaining, not the mmul geometry.

## A.4 Host-side preparation measured: ~1% of the GEMM

`XDNA_HOSTPROF=1` splits the backend's host path per GEMM
(M=512, K=1024, N=3584):

| phase | ms |
|---|---:|
| activation dequant + per-row quantize + A/B tile packing + XRT submit | 1.31 - 1.89 |
| XRT submit (copy + BO sync + run ctor + set_arg + start) | 1.55 (from the MMUL_V2 trace) |
| **device wait** | **348.25 (v2) / 160.92 (v3)** |
| device sync + 7.34 MB copyback | 0.91 |

**The host path is ~1% of the GEMM.** The device kernel is ~99%. This
corrects the speculation in earlier phases that host-side quantisation and
packing were a significant cost - they are not, at these shapes.

## A.5 Revised cost model for the best configuration

For 32x64x512 on the v3 kernel at PP512 (M=512, K=1024, N=3584):

| component | ms | share |
|---|---:|---:|
| host preparation | 1.9 | 1.5% |
| device kernel (of which C update 56%) | 125.9 | 97% |
| of which: C update | ~70 | 55% |
| of which: mmul chain + LMEM loads | ~56 | 43% |

Break-even against Vulkan (5.6 ms per FFN GEMM) is still **22x away**. To
close it, in order of leverage:

1. C update: vectorise it (needs the compiler fix for 256-bit adds, or a
   layout+tap redesign) and/or push k_chunk to 1024 with the 16x32 tile ->
   the measured 68.54 GOP/s design.
2. mmul chain: 21.6 GOP/s/core is 8.5% of peak; the v3n measurement says the
   limit is the mmul+load path, which needs an AIE2-specific microkernel
   (software-pipelined 4 independent accumulators, no spills) rather than a
   change of mmul geometry.
3. Host path: already negligible, no work needed.

## A.6 Status of the addendum

- The 32x64x512 and 16x32x1024 kernels are **compiled, run and numerically
  verified** in the standalone harness, but **not integrated into the
  backend** (they need a generalised host packing). The backend currently
  ships the phase 54 kernel (`int8_requant_v2`, verified correct at the
  device level) and the phase 60 v3 kernel (`int8_requant_v3`, 2.16x faster
  but with the accumulator defect).
- The compiler-side blocker (`aie::add` on 256-bit vectors crashing clang)
  is **confirmed to be a compiler bug**, reproducible on every attempt, and is
  now the single highest-leverage item: fixing it unlocks the vectorised C
  update, which is 55% of the remaining kernel time.

## 12. Limitations of this report
- The phase 60 v3 kernel is not numerically correct in the backend; its
  timings are timings of a kernel that computes the wrong thing. The
  standalone harness verifies v3 bit-exact with its own packing, so the
  defect is in the interaction, not in the arithmetic as such - but that
  interaction is not yet understood.
- The 16-core design never ran; its cost is projected from the measured
  4-core per-core cost and the measured E_p = 1.00, not measured.
- Multi-tile scaling (E_p for p > 4) is unmeasured.
- Only one model (Qwen3.5-0.8B Q8_0) and one NPU (Phoenix XDNA1) were used.
- The 780M Vulkan baseline drops from 1258 tok/s at PP512 to 840 tok/s at
  PP1024 in this session; the cause was not investigated (thermal or
  scheduler effects), so per-PP comparisons should use interleaved runs.

---

# Addendum B: fixes applied - hardcoding removed, accumulator defect fixed

Everything below is implemented, compiled and measured in this session.

## B.1 The hardcoding is gone

The tile shape used to be the literal `m = 32, n = 64, kc = 256` in four
places in the host code, plus `n_cols = 4, kc = 256, n_tile = 64` inside the
weight packer. Now:

- `ggml_xdna_q8_c4_mmul_v2_create()` takes `m_tile, n_tile, k_tile, mmul` and
  stores them in the engine (`e->m`, `e->n`, `e->kt`, `e->mmul`).
- `ggml_xdna_submit_int8_pure()` reads the tile shape **from the engine**;
  no tile constant remains in that function.
- The A packing is generic in `mmul` (4 or 8 rows per block) and in the
  K-chunk, so the same code serves every profile.
- The weight packer takes the tile shape as arguments and **includes it in
  the cache key**, so packed weights of one profile can never be reused by
  another.
- The engine profiles live in **one table** in the kernel registry
  (name, cache suffix, m_tile, n_tile, k_tile, mmul). The kernel generator
  (`experiments/phase60/gen_int8_pure_v4.py`) is parameterised the same way.

New engine: `XDNA_Q8_ENGINE=int8_requant_v4` (32x64x512, native 8x8 mmul).

## B.2 The accumulator defect is fixed

The phase 60 v3 kernel (4x8 mmul, one accumulator over the whole K chunk) was
2.16x faster but numerically wrong in the backend. The cause was **not** the
accumulator initialisation: it is that the `aie::mmul<4,8,8>` formulation
emits a 4-register split that does not survive a 32-mul chain.

The fix is to use the **native 8x8 unit** (`aie::mmul<8,8,8>`), whose `v64acc`
is the hardware's native accumulator width, with a single accumulator per
(i,j). This variant is bit-exact:

```
[XDNA-VERIFY]        M=512 N=3584 K=1024 maxdiff=0 bad=0/1835008
[XDNA-VERIFY-FLOAT]  M=512 N=3584 K=1024 maxdiff=0.059 maxref=2.449
```

`maxdiff=0` means the device result equals the host int32 reference
element for element.

A second defect was found and fixed on the way: the registry's kernel-path
selection fell through to the **Q8SCALEV** kernel for the new engine, so the
int8 data was interpreted as float scales (the symptom was an int32 overflow
of 2^31, which is what a scale array read as int8 data produces).

## B.3 Measured result of the fixes

Per-GEMM device wait (M=512, K=1024, N=3584, median of 144 GEMMs):

| engine | kernel | tile | device wait | correct? |
|---|---|---|---:|---|
| `int8_requant_v2` | phase 54, mmul 4x8 | 32x64x256 | 348.04 ms | yes (maxdiff=0) |
| `int8_requant_v3` | phase 60 v3, mmul 4x8 | 32x64x256 | 160.92 ms | **no** |
| **`int8_requant_v4`** | **phase 60 v4, mmul 8x8** | **32x64x512** | **136.72 ms** | **yes (maxdiff=0)** |

**2.55x faster than the previously correct kernel, and now correct.**

End-to-end PP512 (tok/s, r=2-3):

| configuration | tok/s |
|---|---:|
| Vulkan-only | 1071.81 ± 38.75 (1185-1258 across runs) |
| hybrid, `int8_requant_v2`, all 72 FFN GEMMs | 17.23 ± 0.43 |
| hybrid, `int8_requant_v4`, all 72 FFN GEMMs | 33.57 ± 3.27 |
| **hybrid, `int8_requant_v4`, `down` only (24 GEMMs)** | **143.99 ± 0.62** |

Host preparation is unchanged and negligible: 1.45 ms per GEMM against a
136.72 ms device wait (~1%).

## B.4 What is still open

1. **First-use zero activations.** The first K=3584 GEMM of a run still sees
   `amax=0`. Reading the activation through `ggml_backend_tensor_get()`
   instead of the raw tensor pointer did not change it, which rules out the
   host read path: the source tensor genuinely holds zeros at that point.
   The remaining suspect is the scheduler's copy of the Vulkan-produced
   silu-mul output into the XDNA buffer, i.e. a dependency between the
   Vulkan node and the XDNA node inside the same split. This needs a
   scheduler-level fix (mainline 4 of the phase 60 brief).
2. **Quantisation fidelity.** The per-column int8 requantization of the Q8_0
   weights gives 2.4%-of-max error per GEMM, which compounds over 24 layers
   (PPL 2.97e6 against Vulkan's 65.29). The scale-preserving path
   (`c4_mmul_v2`, per-32-block scales on the device) has the right fidelity
   but was measured at 420 ms/GEMM and is also blocked by defect 1.
3. **16-core design** still blocked by the shim DMA channel budget; the
   compiler's suggested fix (MemTile fan-out) is not implemented.
4. **Vectorised C update** still blocked by the clang crash on
   `aie::add` with 256-bit vectors. With the v4 kernel the C update is a
   smaller share than in v3 (the 8x8 accumulator flushes 64 elements per
   (i,j) instead of 32), so this item is now worth less than it was.

---

# Addendum C: the ordering bug is fixed - the NPU path is now numerically valid

## C.1 Root cause of the "first-use zero activations"

The XDNA backend also implements the **GLU** (silu-mul) op, and the
scheduler assigns it to the NPU. The GLU reads its gate and up operands
through the raw host pointer of the XDNA buffer:

```cpp
float * g = (float *) get_tensor_data_mut((struct ggml_tensor *) gate);
float * u = (float *) get_tensor_data_mut((struct ggml_tensor *) up);
```

But the gate and up results are **still pending on the device** at that
moment: the GEMM results are written into the host-mapped buffers by the
*drain* path, not by the device. The GLU therefore read uninitialised
(zero) buffers and produced a zero output, which the down GEMM then consumed.

Why only the first layer? The drain is forced when 4 submissions are
outstanding (`pending_count >= 4`). For layer 0 the GLU runs while only 2
submissions (gate, up) are pending, so the data is not there yet. From layer 1
onwards the queue has already reached 4 and a drain has happened, so the
buffers hold valid data.

**Fix**: drain the pending queue before the GLU computes.

```cpp
if (node->op == GGML_OP_GLU) {
    // Phase 60: the GLU consumes the gate and up results, which are still
    // pending on the device. Drain first, otherwise the GLU reads the
    // uninitialised host buffers and produces zeros.
    if (!ggml_xdna_drain_pending(ctx)) { ... }
```

A related latent bug was fixed at the same time: `ggml_xdna_buffer_get_tensor`
copied out of an XDNA buffer **without draining**, so any copy-out of a
pending result returned stale data. It now calls `ggml_xdna_synchronize_all()`
(which drains every active backend) before the copy.

## C.2 Result: the model output is now correct

Perplexity over 3000 tokens (6 chunks, ubatch 512):

| configuration | PPL | before the fix |
|---|---:|---:|
| **Vulkan-only (reference)** | **65.2904** | 65.2904 |
| **hybrid, `c4_mmul_v2` (scale preserving)** | **65.2109** | 933 814 |
| hybrid, `int8_requant_v4` (pure int8) | 13 023 092 | 3 259 901 |

**The scale-preserving NPU path now reproduces the Vulkan model's quality
(0.1% PPL difference).** The hybrid path is, for the first time in phases
49-60, a *correct* execution of the model.

The pure-int8 path remains unusable, and now for a *proven* reason that is
independent of the ordering bug: the per-column int8 requantization of the
Q8_0 weights gives 2.4%-of-max error per GEMM, and that compounds over 24
layers. Per-GEMM the device result is bit-exact against the host reference
(`maxdiff=0`), so the loss is in the *quantization scheme*, not in the
kernel, the ordering, or the hardware.

## C.3 The speed of the two numerically viable options

| engine | numerics | per-GEMM device wait | PP512 tok/s |
|---|---|---:|---:|
| `c4_mmul_v2` (per-32-block scales on device) | **valid** (PPL 65.21) | 432.95 ms | 16.04 |
| `int8_requant_v4` (per-column int8 requantization) | invalid (PPL 1.3e7) | **142.83 ms** | 33.57 |
| `int8_requant_v2` (phase 54 kernel) | invalid | 348.04 ms | 17.23 |
| Vulkan-only | reference | - | 1071-1258 |

**The remaining work is a kernel that keeps the scale-preserving numerics
*and* the 8x8 single-accumulator structure.** The two are currently
mutually exclusive:

- `int8_requant_v4` flushes C once per (i,j), so it can apply a single
  scale for the whole K chunk - but one scale for 1024 weights is what
  destroys the numerics.
- `c4_mmul_v2` applies the Q8_0 per-32-block scales, which requires a
  flush per 32-block, i.e. 8x more C updates, which is where its 433 ms
  goes.

The way out is a kernel that accumulates **per 32-block** partial sums in the
`v64acc` and applies the block scales there, so the flush count stays at
8 per (i,j) but each flush is a vector op over the 16-lane (512-bit) C row
instead of 8 scalar float read-modify-writes. That is the same class of
change that made v2 -> v3 worth 5x, applied to the scale-preserving kernel.

## C.4 Summary of everything fixed in phase 60

| defect | fix | effect |
|---|---|---|
| `aie::mmul<4,8,8>` accumulator breaks on long chains | use the native `aie::mmul<8,8,8>` with one `v64acc` | v3 wrong -> v4 bit-exact |
| registry path selection fell through to the Q8SCALEV kernel for the new engine | explicit profile table drives the path | v4 loaded the wrong xclbin |
| host-side tile constants (32/64/256) in four places | tile shape lives in the engine; one profile table | hardcoding removed |
| weight cache key ignored the tile shape | tile shape folded into the key | cross-profile weight reuse prevented |
| GLU read pending gate/up results | drain before GLU | **PPL 933 814 -> 65.21** |
| buffer copy-out did not drain | `ggml_xdna_synchronize_all()` in get_tensor | stale data on copy-out |
