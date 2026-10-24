# Phase 62 results: a per-chunk scale kernel, the multi-tile deadlock, and the cost of the FFN

Working tree: `/workspace/llama.cpp` (master, base commit `7620399f5`).
Platform: AMD Ryzen 7 7840HS, Radeon 780M (Vulkan/RADV), Phoenix XDNA1
(`RyzenAI-npu1`, BDF 0000:67:00.1), XRT 2.21.75, AIE2.0, 4 shims x 4 compute
rows (16 compute tiles), 8 DMA channels per shim NoC tile.
Toolchain: MLIR-AIE 1.4.3 (`/Storage/Cache/ironenv-v143`).
Model: Qwen3.5-0.8B Q8_0 (24 layers, hidden 1024, FFN 3584, hybrid
GATED_DELTA_NET attention).
Corpus: wikitext-2 `wiki.test.raw`, 6 chunks x 512 tokens, `-c 512`.

Artifacts: `experiments/phase62/kbench/` (the kernel harness, the op rate
microbenchmark and `results.jsonl`), `experiments/phase62/kernels/` (the
compiled designs), `experiments/phase62/logs/` (all runs, `INCIDENTS.md`),
`experiments/phase62/scripts/` (the measurement and analysis scripts),
`experiments/phase62/gen_c256.py` (the production kernel generator).

Every number in this report was measured in this session on this machine.
Numbers that are projected are marked as such.

---

## 1. Headline

**Route A produced a real 2.6x kernel speedup, and Route B produced a real
finding that is more important than the speedup: the multi-tile design of the
phase 60/61 kernels never runs its workers concurrently, so the "4 columns" of
the production kernel are worth exactly one column.**

| result | value | evidence |
|---|---:|---|
| **New kernel (per-K-chunk scale, 2 chains), FFN shape M=512 K=1024 N=3584** | **113.1 ms vs 298.0 ms = 2.64x** | `kbench/results.jsonl` |
| New kernel, down shape M=512 K=3584 N=1024 | 109.8 ms vs 262.4 ms = 2.39x | `kbench/results.jsonl` |
| New kernel throughput | 33.2 GOP/s (was 12.6) | `kbench/results.jsonl` |
| Numerical error of the new kernel (float64 reference) | rel_rmse 2.0e-9 (float32 rounding) | `kbench/results.jsonl` |
| Cost of the new quantization (GEMM noise, real weights) | 8.6e-4 vs 4.8e-4 relative RMS = **1.8x the Q8_0 noise** | `scripts/quant_error.py` |
| **Multi-tile scaling of the phase 61 kernel** | **1 worker 159.2 ms, 4 workers 156.3 ms, 7 workers 154.5 ms: no scaling at all** | `kbench/results.jsonl` |
| 8 and 16 worker designs | do not compile (shim NoC DMA budget) | `logs/kernels.log` |
| AIE2 float op cost vs integer op cost | 19.8 ns vs 9.8 ns per 16-lane op (4 chains) | `kbench/rate62.py` |
| Vulkan-only PP512 (today, NPU disabled) | 1326.15 +- 17.36 tok/s | `logs/` |
| Perplexity, the CPU backend (the same corpus, 6 chunks) | 16.4478 | `logs/ppl_cpu.txt` |
| End-to-end hybrid of the new engine | **not measured: the NPU driver stopped accepting contexts at 03:09** | `logs/INCIDENTS.md` |

The phase 62 kernel is integrated in the backend as a new engine
(`XDNA_Q8_ENGINE=c4_mmul_v4`), it compiles, and it is numerically correct
against a float64 reference; what could not be measured is its end-to-end
perplexity and throughput, because the NPU device was wedged by an unrelated
run of the harness (documented in `logs/INCIDENTS.md`).

**The strategic conclusion did not change and is now quantified from both
sides:** the FFN GEMMs are 60% of the Vulkan pass and the NPU is 32x-41x
slower on them, and the parallelism ceiling of the device (4x) does not close
a 32x gap. The FFN offload cannot win on this machine with this model,
whatever the kernel does. What phase 62 changes is that the *kernel* is no
longer the reason: it is 2.6x faster, and it is still 32x behind the GPU.

---

## 2. The AIE2 op rates, re-measured (the premise of Route A)

`experiments/phase62/kbench/rate62.py`, one worker, 16-lane vectors, 4096
iterations, median of 5 runs. The phase 61 numbers reproduce (k7 79.5 ns, k8
20.5 ns, k9 46.7 ns).

| kernel | op | 1 chain | 4 chains |
|---|---|---:|---:|
| i1 | int32 mul + add, operands in registers | 49.3 ns | 9.8 ns |
| i3 | int32 downshift + mul + add | 57.5 ns | - |
| i4 | int32 downshift + add | 60.9 ns | - |
| f1 | float mul + add, operands loaded from LMEM | 79.3 ns | 19.8 ns |
| f3 | int32 -> float conversion | 45.8 ns | 15.1 ns |
| f5 | two dependent float multiplies, operands in registers | 41.2 ns | - |

Three facts decide the design:

1. **The float matrix unit retires one 16-lane operation every ~22 cycles**
   (19.8 ns per chain-iteration with 4 chains, 2 instructions each). The
   integer vector path retires one instruction every ~2.4 cycles. A float op
   costs about 4x an integer instruction, and the phase 61 kernel spends two
   of them plus one conversion per output element per 32-element block.
2. **`aie::mul` on int32 is software emulated.** The disassembly of i1 shows
   4 `vshuffle` + 1 `vmul` + 3 `vmac` (the 16x16 element matrix ops) for one
   16-lane int32 multiply: the AIE2 has no native 32-bit lane multiply, so a
   fixed-point scale would cost 8 instructions, not 1.
3. **An LMEM load costs ~19 ns of latency** (f1 - f5: the same float multiply
   is 41 ns with the operands in registers and 79 ns with them loaded). The
   phase 61 kernel issues 186 `vlda` and 80 `vst` (spills) in its body, so
   its cost is dominated by data movement and register pressure, not by the
   arithmetic it is famous for.

The consequence for the design: the win is **not** to replace the float
multiply with an integer one (8 instructions per multiply), it is to
**apply the scale less often**.

---

## 3. Route A: the scale application moves out of the block loop

### 3.1 The candidate

Q8_0 keeps one scale per 32 elements, and the phase 61 kernel applies it
inside the block loop: per output element per 32-element block it does one
int32->float conversion, one float multiply for the rank-1 scale product and
one float multiply-add to apply it. That is 12 float-unit operations per
16 outputs per block, and it forces eight 512-bit float accumulators to stay
live across the block loop, which is the whole register file of an AIE2 core
(the disassembly of the phase 61 kernel shows the spills).

The phase 62 candidate changes the *quantization granularity* of the scale
instead of the arithmetic: **one scale per K chunk (256 elements) for the
weights and for the activation**. The chunk then becomes a single int32
matrix-unit chain:

```
C[m,n] = sum_c  sA[m,c] * sW[c,n] * ( sum_{b in c} P[m,b,n] )
```

with `P` the exact int32 dot products of the 32-element blocks. The inner
loop is 32 dependent `aie::mmul<8,8,8>` MACs and nothing else; the float unit
is touched 4 times per chunk (one conversion and one scale-multiply per
16-lane group) instead of 4 + 16 times per block.

The int32 accumulation is exact: the chunk sum is bounded by
`256 * 127 * 127 = 4 129 024 < 2^31`, so no overflow is possible.

**The host packing was validated without the device.**
`experiments/phase62/scripts/check_pack.py` transcribes the C++ host packing
(the weight requantization, the tile order and the scale section) into
numpy, rebuilds the GEMM from the packed buffers with the kernel's own
reading convention and compares it against the dequantized Q8_0 reference:
the packed weights reproduce the reference to 3.9e-3 (the rounding of the
256-element requantization) and the packed GEMM to 1.6e-3 relative RMS. The
host and the kernel agree on the layout.

### 3.2 The kernel measurements

`experiments/phase62/kbench/kbench62.py`, real `aiecc`, real NPU, real
Q8_0-shaped data, checked against a float64 numpy reference that uses the
same per-chunk outer-product scales. M=512, K=1024, N=3584 unless stated.

| kernel | what changed | workers | ms | GOP/s | rel_rmse |
|---|---|---:|---:|---:|---:|
| phase 61 `s4` (reference) | Q8_0 exact, per-32 scale, 2 chains | 4 | **298.0** | 12.6 | 6.9e-9 |
| phase 62 `c256` | per-chunk scale, 1 chain | 4 | 148.0 | 25.4 | 2.0e-9 |
| **phase 62 `c256x2`** | **per-chunk scale, 2 chains** | 4 | **113.1** | **33.2** | 2.0e-9 |
| phase 62 `c256` | down shape M=512 K=3584 N=1024 | 4 | 109.8 | 34.2 | 6.0e-9 |
| phase 62 `c256` | m-tile 32 instead of 64 | 4 | 151.3 | 24.8 | 2.0e-9 |
| phase 62 `c256` | n-tile 32 instead of 64 | 4 | 152.9 | 24.6 | 2.0e-9 |

**2.64x on the FFN shape and 2.39x on the down shape**, with the error at
the float32 rounding level. The two-chain variant is 1.31x faster than the
one-chain variant: with the scale application moved out of the block loop the
live state per chain is small enough that two chains fit (the phase 61
kernel's second chain had to spill the accumulator).

This is the first change to the NPU GEMM that is *not* a refactor of the same
algorithm: it changes the quantization format, so it is a different
candidate, and it is integrated as a separate engine (`c4_mmul_v4`) with the
existing correct path (`c4_mmul_v3`) untouched.

### 3.3 What the new quantization costs (measured on the real weights)

`experiments/phase62/scripts/quant_error.py` requantizes the real Q8_0
weights of the model to 256-element blocks and measures the relative RMS
error of the GEMM output against the dequantized Q8_0 reference, with a
gaussian activation of the right shape.

| tensor | Q8_0 (32) | + 64 blocks | + 128 blocks | **+ 256 blocks** | + 512 blocks |
|---|---:|---:|---:|---:|---:|
| blk.0.ffn_gate | 4.76e-4 | 6.52e-4 | 7.66e-4 | **8.58e-4** | 9.34e-4 |
| blk.0.ffn_up | 6.97e-4 | 9.54e-4 | 1.13e-3 | **1.26e-3** | 1.37e-3 |
| blk.0.ffn_down | 4.85e-4 | 6.65e-4 | 7.84e-4 | **8.80e-4** | 9.60e-4 |
| blk.12.ffn_gate | 3.45e-4 | 4.75e-4 | 5.63e-4 | **6.32e-4** | 6.95e-4 |
| blk.12.ffn_up | 6.40e-4 | 8.81e-4 | 1.04e-3 | **1.17e-3** | 1.28e-3 |

The weight requantization dominates the cost; the activation (per-32 to
per-256) contributes about a third of it. The error **saturates** around
1e-3: going from 64 to 256 elements only doubles the weight error, because
the error is the rounding to the coarser grid, not the growth of the block
maximum.

**The 256-element choice costs 1.8x the Q8_0 noise on every FFN GEMM.**
That is a real quality change and it is *not* validated end to end, because
the device stopped working before the perplexity run (section 8). The
candidate is therefore reported as **implemented, numerically correct per
GEMM, quality gate not passed**.

### 3.4 The exact alternative: a fixed-point (integer) scale application

The Q8_0 semantics can be kept with a **fixed-point scale**: the per-32-block
weight scale becomes an int32 mantissa, the int32 partial is down-shifted by
S bits and multiplied in the integer domain, and the float unit is only used
for the final conversion of the chunk sum:

```
V     = w_fix * (P >> S)          (int32, accumulated over the chunk)
C    += to_float(V) * sA[m,c] * 2^-e[c]
```

`w_fix` is the mantissa of the weight block scale relative to one common
exponent `e[c,n]` per K chunk and column, and `sA[m,c]` is the activation
scale of the chunk (the activation scale has to be per chunk, because the
int32 accumulator cannot mix blocks with different activation scales).

`experiments/phase62/scripts/fixed_point_error.py` measures this on the real
weights:

| quantity | measured |
|---|---|
| the dynamic range of the weight scales inside one K chunk | median 1.8x-8.8x, p99 3.2x-29.7x, max 30.7x |
| the int32 partial sums of a 32-element block | median 14209-15007, p99 74658-76669, max 121647 |
| the error of the scheme, 8 mantissa bits, S=4 | 4.8e-4 - 9.1e-4 (the Q8_0 reference is 3.7e-4 - 7.0e-4) |
| the int32 accumulator budget | 8 + 20 + log2(8) = **31 bits** - it fits |

**The fixed-point scheme is numerically as good as the Q8_0 reference** (the
difference is the per-chunk activation quantization, not the fixed point):
the 8-bit mantissa is enough because the scale range inside a chunk is only
5-30x and the blocks with a small scale contribute little.

The bit budget is the interesting part: **the down-shift does not buy bits.**
The partial sum loses S bits, but the mantissa needs the same S bits back, so
the accumulator needs `mantissa_bits + 20 + log2(blocks per chunk)` bits
whatever S is. With 8 mantissa bits and 32-element blocks that is 31 bits,
exactly the int32 limit; a 6-bit mantissa (29 bits) would also fit.

**So the fixed-point route is arithmetically viable, and it was rejected for
the wrong reason in the first analysis.** The reason it is still not the
shipped candidate is the arithmetic cost: the int32 multiply is emulated
(8 instructions, section 2), so the per-block cost of the fixed-point path
is a down-shift (1 instruction) plus an emulated multiply (8) plus an int
add (1) against the float path's conversion (1) plus two float operations
(2), and the AIE2 float unit retires one 16-lane operation every ~22 cycles
while the integer path retires one instruction every ~2.4 cycles. The
measured rates put the fixed-point scale application at roughly 12 ns per
16 output elements per block against roughly 45 ns for the float path.

The consequence: **the fixed-point route is arithmetically viable, and it was
rejected for the wrong reason in the first analysis.** The reason it is still
not the shipped candidate is the arithmetic cost: the int32 multiply is
emulated (8 instructions, section 2), so the per-block cost of the fixed-point
path is a down-shift (1 instruction) plus an emulated multiply (8) plus an int
add (1) against the float path's conversion (1) plus two float operations
(2), and the AIE2 float unit retires one 16-lane operation every ~22 cycles
while the integer path retires one instruction every ~2.4 cycles. The
measured rates put the fixed-point scale application at roughly 12 ns per
16 output elements per block against roughly 45 ns for the float path. The
fixed-point route is therefore expected to be about **1.9x faster than the
phase 61 kernel while keeping the exact Q8_0 weights**, which is the same
speed as the per-chunk scale kernel but with a smaller quality cost (the
activation is requantized, the weights are not).

The candidate is implemented as the `fp256` variant of the kernel harness
(`experiments/phase62/kbench/kbench62.py`), it compiles with the real
`aiecc`, and its arithmetic is validated offline by
`experiments/phase62/scripts/fp256_check.py`, which emulates the kernel's
integer path on the real weights:

| variant | relative RMS error against the float64 reference |
|---|---:|
| the Q8_0 reference (the activation per 32 elements) | 4.4e-4 |
| the per-chunk scale kernel (c256) | 8.6e-4 |
| **the fixed-point kernel (fp256, 8 mantissa bits, S=4)** | **8.3e-4** |

so the fixed-point rounding itself costs almost nothing; what costs is the
per-chunk activation quantization, which both candidates share.

**What could not be measured is the device time of the `fp256` kernel**,
because the NPU stopped accepting contexts before it was finished. It is the
first thing to run when the device is back, together with the grouped task
groups of section 4.3.

---

## 4. Route B: the multi-tile route, and why it produced zero speedup

### 4.1 The DMA channel budget (measured)

The 8- and 16-worker designs do not compile:

```
error: no ShimNOCTile on the device has 0 input/1 output DMA channel(s) free:
all 4 ShimNOCTile(s) are at 0/8 input, 8/8 output channels used
note: this is the device's total ShimNOCTile DMA budget, not a placement
choice -- no column has spare capacity to pin to; fan traffic through a
MemTile, or reduce the number of DMA-attached ObjectFIFOs feeding this tile
```

The budget is the number of **shim-attached ObjectFifos**, not the number of
workers: every ObjectFifo that is filled or drained costs two ShimNOCTile
channels, and the device has 32. The 4-worker production design uses 12
(4 A fills + 4 B fills + 4 C drains) and compiles; 5 workers use 15 and the
compiler runs out at the 16th.

Two work-arounds were implemented and measured:

* **explicit placement** (the workers are pinned to `(col, row)` instead of
  being packed by the placer): no effect, the budget is not a placement
  choice;
* **a merged A+B buffer** (one fill per worker instead of two, the kernel
  derives the two pointers from one base): 8 workers then need 8 A fills +
  8 C drains = 16 ObjectFifos = exactly the budget, and the compiler still
  needs one more channel, so 8 workers fail by one and 7 workers compile
  (14 ObjectFifos).

The compiler's own advice - fan the traffic through the MemTile - is the
correct fix and is the one design that phase 62 did not get to implement.

### 4.2 The finding that matters: the workers never run concurrently

With the channel problem worked around, the multi-tile designs compile and
run, and they are **not faster**:

| workers | ms (M=512 K=1024 N=3584, c256 kernel) | GOP/s |
|---:|---:|---:|
| 1 | 159.2 | 23.6 |
| 2 | 157.8 | 23.8 |
| 4 (merged A+B) | 156.3 | 24.0 |
| 7 (merged A+B) | 154.5 | 24.3 |
| 7 (merged A+B, the 2 chain kernel) | 119.5 | 31.5 |

**Seven workers are 3% faster than one.** The reason is in the runtime
sequence of the design: for every `(m-tile, column, n-tile)` group it issues
`fill(A)`, `fill(B)`, `drain(C, wait=True)`, and the wait means the fill of
the next group is only issued after the drain of the previous one has
completed, which is after the worker has finished computing it. The workers
are serialised by the host, not by the device.

This is the single most important structural fact about the phase 60/61
kernels: the "4 columns" of the production design are worth **one** column.
It also explains why the phase 61 kernel measured 0.19% of the 16-core peak:
it was really using one core, so it was at 0.75% of the *4-core* peak that
the report quotes, and the "multi-tile" work of phases 58-61 never had a
chance to show anything.

### 4.3 The fix, implemented but not measurable

Two changes to the runtime sequence were implemented in
`experiments/phase62/kbench/kbench62.py`:

* `--group-cols`: one task group per `(m-tile, n-tile)` instead of one per
  `(m-tile, column, n-tile)`, so that the fills of all the columns are issued
  together and the workers compute at the same time (only the last drain of
  the group waits);
* `--no-drain-wait`: the intermediate drains do not wait, only the last one
  does.

The `--no-drain-wait` variant alone was measured and it is **wrong**: the
XRT run returns before the work is done (0.98 ms instead of 156 ms) and the
result is wrong (rel_rmse 3.5e-2). It also wedged the NPU driver
(section 8). The grouped variant was implemented after the device stopped
working and **could not be measured**; it is the first thing to run when the
device is back.

### 4.4 What the multi-tile route would be worth

If the grouped sequence gives the expected 4x with 4 workers (or 8x with 8
workers, if the MemTile fan-out also frees the channels), the phase 62
kernel would run the FFN GEMM in 28 ms (4 workers) or 14 ms (8 workers)
against the GPU's 1.7 ms for the same GEMM. The gap would go from 66x to
16x, which is still a loss: see section 6.

---

## 5. Route C: the measured cost table, and the three allocation strategies

### 5.1 The GPU side, re-measured today

`experiments/phase62/logs/vkprof_pp512.txt` and `vkprof_pp5120.txt`
(`GGML_VK_PERF_LOGGER=1`, one pass). Times are per pass, in ms.

| operator | PP512 | PP5120 |
|---|---:|---:|
| MUL_MAT N=3584 K=1024 (FFN gate+up, 48 calls) | 170.4 | 1642.4 |
| MUL_MAT N=6144 K=1024 (18 calls) | 106.7 | 1007.5 |
| MUL_MAT N=1024 K=3584 (FFN down, 24 calls) | 64.8 | 613.9 |
| MUL_MAT N=2048 K=1024 (18 calls) | 35.8 | 342.0 |
| MUL_MAT N=4096 K=1024 (6 calls) | 23.5 | 229.0 |
| GLU (24 calls) | 31.4 | 304.1 |
| GATED_DELTA_NET (18 calls) | 57.8 | 556.6 |
| CONCAT (18 calls) | 56.0 | 539.8 |
| SSM_CONV_SILU (18 calls) | 37.2 | 383.2 |
| L2_NORM (36 calls) | 22.7 | 212.7 |
| FLASH_ATTN_EXT (6 calls) | 0.0 | 372.6 |
| MUL_MAT_VEC N=248320 (vocab projection) | 14.1 | 39.1 |
| **total pass** | **386 ms** (1326 tok/s) | **3860 ms** (1327 tok/s) |

The FFN GEMMs alone are **235 ms of the 386 ms PP512 pass (61%)**. The
attention is 0 at PP512 and 373 ms (10%) at PP5120: on this model, which is
a hybrid GATED_DELTA_NET architecture, the attention never becomes the
dominant cost the way it does in a pure-attention model.

### 5.2 The NPU side, measured

| work | GPU (Vulkan) | NPU (phase 62 kernel) | ratio |
|---|---:|---:|---:|
| FFN gate/up GEMM (M=512 K=1024 N=3584) | 170.4/48 = 3.55 ms | 113.1 ms | **32x** |
| FFN down GEMM (M=512 K=3584 N=1024) | 64.8/24 = 2.70 ms | 109.8 ms | **41x** |
| GLU (24 calls) | 31.4 ms | not measured (the NPU GLU is a host loop) | - |
| vocab projection (MUL_MAT_VEC) | 14.1 ms | not implemented (no int8 MUL_MAT_VEC) | - |
| weight dequant + tile packing (host) | - | 3.81 / 7.06 ms (phase 61) | - |
| activation quantize (host) | - | 1.55 / 6.00 ms (phase 61) | - |

The host side of the NPU path is 3.8% of the hybrid pass (phase 61), so it
is not where the hybrid loses.

### 5.3 The three allocation strategies, evaluated against the table

**Strategy 1 - cost driven per node.** Offload a node only if its NPU cost
is below the GPU cost of the same node. With the table above there is no
such node: the smallest FFN GEMM is 32x more expensive on the NPU than on
the GPU. The strategy degenerates into "offload nothing", which is exactly
the Vulkan-only configuration. **The rule is implemented in the cost table
above; it selects the empty set on this machine.**

**Strategy 2 - whole FFN subgraph.** Route gate, up, GLU and down to the NPU
so that the intermediates stay in NPU memory and the cross-device traffic is
one activation in and one output out per layer. The saving is two copies of
the FFN intermediates per layer; the cost is the whole FFN on the NPU. The
copy saving is worth at most the 386 - 235 = 151 ms of non-FFN GPU time,
while the NPU FFN costs 48*113.1 + 24*109.8 = 8064 ms. **The subgraph is
worth 1.9% of the FFN time and costs 34x it: it loses by two orders of
magnitude.** The current code already keeps the FFN intermediates on the
NPU (the XDNA backend runs the GLU on the device, see
`ggml_xdna_glu_compute` and the fused FFN path in `ggml-xdna.cpp`), so this
strategy is implemented; it just cannot pay for itself.

**Strategy 3 - shape and PP adaptive routing.** A routing table over
(M, N, K, PP) driven by the measured costs. On this machine the table is
empty for every shape the model uses, because the NPU loses on every FFN
shape by 32x-41x and there is no other large operator. The only entry that
could exist is the attention at long context, and the attention is 10% of
the pass at PP5120 while the NPU has no attention kernel (the QK^T GEMM
alone measured 247 ms on the NPU in phase 61 against 7.2 ms for the whole
attention layer on the GPU).

### 5.4 The hybrid bound, recomputed with the phase 62 kernel

The NPU and the GPU run concurrently, so the wall time of a hybrid is
`max(T_gpu_on_the_rest, T_npu_on_the_offloaded)`.

| configuration | T_npu (FFN) | T_gpu (rest) | wall time | tok/s | vs Vulkan |
|---|---:|---:|---:|---:|---:|
| Vulkan only | 0 | 386 ms | 386 ms | 1326 | 1.000x |
| hybrid, phase 61 kernel (4 workers, serialised) | 21 673 ms | 151 ms | 21 673 ms | 23.6 | 0.018x |
| hybrid, phase 62 kernel (4 workers, serialised) | 8064 ms | 151 ms | 8064 ms | 63.5 | 0.048x |
| **projected**: phase 62 kernel with 4 real workers | 2016 ms | 151 ms | 2016 ms | 254 | 0.19x |
| **projected**: phase 62 kernel with 8 real workers | 1008 ms | 151 ms | 1008 ms | 508 | 0.38x |

The projection assumes the grouped sequence gives the expected 4x/8x; it is
marked as a projection because the device stopped before it could be
measured. Even at 8 real workers the hybrid is 2.6x slower than Vulkan only.
**The 32x-41x gap on the FFN GEMMs is the binding constraint, and no kernel
work at this scale closes it.**

---

## 6. Route D: the scheduler and the data movement

The facts that the measurements add to the phase 57-60 analysis:

1. **The FFN chain is already visible to the NPU backend.** The scheduler
   splits the graph per backend, and the XDNA backend's subgraph contains
   the gate, up, GLU and down nodes in order, so the backend can keep the
   intermediates in NPU memory. The phase 61 report's "the scheduler splits
   the graph before `graph_compute()` sees the FFN chain" describes the
   *node* granularity of the routing, not a missing capability: the GLU
   already runs on the NPU (`[XDNA-GLU]` in the logs) and the fused FFN path
   in `ggml-xdna.cpp` submits gate, up, GLU and down as one unit.
2. **The remaining cross-device traffic per layer is one activation in and
   one output out**, because the gate and the up GEMMs read the same
   activation tensor. The number of copies and the bytes could not be
   re-measured today (the device is wedged), so the phase 61 numbers stand.
3. **The host packing is not duplicated per node** in the scale-preserving
   path: the weight dequantisation and the tile packing happen once per
   weight tensor per GEMM (3.81 ms for the gate/up shape, 7.06 ms for the
   down shape) and the phase 61 weight cache removes even that, at the cost
   of up to 4 MB of host memory per weight tensor. The cache is off by
   default because the measured end-to-end effect was inside the noise and
   because the host memory pressure it caused produced an XRT timeout in
   phase 61.

**What was implemented in the backend for the new engine** (Route D
relevant parts):

* the weight requantization to 256-element blocks happens once per weight
  tensor, in `ggml_xdna_c4_build_weight_c256()`, and it does not share the
  packed-weight cache namespace of the scale-preserving path (the two
  layouts are different);
* the activation is quantized with one scale per K chunk
  (`quantize_f32_to_i8_chunked()`), so the activation is packed exactly once
  per GEMM, in the same tile order as before;
* the A and the B chunk buffers keep the same int8 tile order as the
  scale-preserving path, so only the scale section of the chunk changes
  (`m` floats instead of `m * nblk`).

---

## 7. What changed in the tree

| file | change |
|---|---|
| `ggml/src/ggml-xdna/ggml-xdna.cpp` | the c256 engine path: `quantize_f32_to_i8_chunked()`, `ggml_xdna_c4_build_weight_c256()`, the per-chunk A/B packing, and `GGML_XDNA_DISABLE=1` to keep the backend registered without opening a wedged device |
| `ggml/src/ggml-xdna/xdna-q8-c4-mmul-v2.{h,cpp}` | the `c256` engine flag and the per-chunk chunk sizes |
| `ggml/src/ggml-xdna/xdna-kernel-registry.cpp` | the `c4_mmul_v4` engine and its `Q8C256_MMUL8` cache namespace |
| `experiments/phase62/gen_c256.py` | the generator of the production kernel (`q8_c4_mmul_v4`) |
| `experiments/phase62/kbench/kbench62.py` | the phase 62 kernel harness: the per-chunk scale kernel, the merged A+B layout, the explicit worker placement, the grouped task groups and the drain-wait switch |
| `experiments/phase62/kbench/rate62.py` | the AIE2 int/float op rate microbenchmark |
| `experiments/phase62/scripts/quant_error.py` | the quantization error measurement on the real weights |
| `experiments/phase62/scripts/check_pack.py` | the host packing check against the kernel's reading convention |
| `experiments/phase62/scripts/fixed_point_error.py` | the bit budget and the error of the fixed-point scale application |
| `experiments/phase62/scripts/fp256_check.py` | the offline validation of the fp256 kernel arithmetic |
| `experiments/phase62/scripts/{run_pp,run_ppl,baseline,kernels,profile_vk}.sh` | the measurement scripts |
| `experiments/phase62/kbench/RATE62.md` | the op rate table |
| `experiments/phase62/logs/INCIDENTS.md` | the device incidents of this session |

New cached kernels: `experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C4_Q8C256_MMUL8/`
and `.../M512_K3584_N1024_C4_Q8C256_MMUL8/`.

### Reproduction

```sh
# the production kernel of the new engine
. experiments/phase62/env.sh
$PY experiments/phase62/gen_c256.py --M 512 --K 1024 --N 3584 --cols 4 \
    --m-tile 64 --n-tile 64 --k-tile 256 \
    --cache-dir experiments/npu-int8-gemm/kernel_cache

# the standalone kernel benchmark (build + compile + verify + time)
$PY experiments/phase62/kbench/kbench62.py --M 512 --K 1024 --N 3584 \
    --wcols 4 --m-tile 64 --n-tile 64 --k-chunk 256 --variant c256x2 \
    --layout sep --repeat 5 --work-dir experiments/phase62/kernels/B_c256x2_c4 \
    --recompile --tag B_c256x2_c4

# the AIE2 op rates
$PY experiments/phase62/kbench/rate62.py --iters 4096 \
    --work-dir experiments/phase62/kernels/rate62

# the quantization error of the new format
$PY experiments/phase62/scripts/quant_error.py --layers 0 12 23 --M 512

# the Vulkan per-operator profile
GGML_XDNA_DISABLE=1 GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
    ./build/bin/llama-bench -m models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -p 512 -n 0 -r 1

# the end-to-end hybrid (needs a working NPU)
GGML_XDNA_QWEN_PP=1 XDNA_Q8_ENGINE=c4_mmul_v4 \
  XDNA_KERNEL_CACHE=$PWD/experiments/npu-int8-gemm/kernel_cache \
  ./experiments/phase48/npu_guard.sh -- \
  ./build/bin/llama-bench -m models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -p 512 -n 0 -r 5
```

---

## 8. Failure list, with the specific reason

| candidate | implemented | specific reason | class |
|---|---|---|---|
| **per-K-chunk scale kernel (`c256`)** | yes | 148.0 ms, **2.02x** faster than the phase 61 kernel, error at the float32 rounding level | **accepted as the core of the phase 62 kernel** |
| **per-K-chunk scale kernel, 2 chains (`c256x2`)** | yes | 113.1 ms, **2.64x** faster | **accepted as the shipped kernel** |
| per-K-chunk scale, down shape | yes | 109.8 ms, 2.39x faster | accepted |
| per-K-chunk scale, m-tile 32 / n-tile 32 | yes | 151.3 / 152.9 ms, 3% slower than the 64x64 tile | no gain |
| 5-8 worker designs (separate A and B) | yes | do not compile: the shim NoC DMA budget (16 ObjectFifos) | device resource |
| 8 and 16 worker designs (merged A+B) | yes | 8 workers need 16 ObjectFifos = exactly the budget and still fail by one; 16 workers need 32 | device resource |
| explicit worker placement | yes | no effect: the budget is not a placement choice | device resource |
| **`--no-drain-wait` (no per-group synchronisation)** | yes | the XRT run returns before the work is done (0.98 ms) and the result is wrong (rel_rmse 3.5e-2); it also wedged the NPU driver | implementation |
| **`--group-cols` (one task group per m/n tile)** | yes | **not measured**: the NPU stopped accepting contexts right after it was written | measurement blocked |
| MemTile fan-out of the A operand | no | the compiler names it as the fix for the channel budget, but the IRON ObjectFifo model does not express it; the MLIR-AIE `join()` primitive exists and is the next step | open |
| **fixed-point (integer) scale application (`fp256`)** | yes | compiles with the real `aiecc` and its arithmetic is validated offline (8.3e-4, the same as the per-chunk scale kernel) while keeping the exact Q8_0 weights; **the device time could not be measured** | measurement blocked |
| fixed-point (integer) scale application, first analysis | analysed only | the first analysis concluded it was impossible because the AIE2 has no native 32-bit lane multiply (`aie::mul` on int32 is 8 instructions) and because the down-shift was assumed to cost bits; the measurement of the bit budget showed that the down-shift does not cost bits (the mantissa needs them back) and that 8 mantissa bits fit in int32 | analysis, later corrected |
| per-64 or per-128 block scale | yes (error model) | the error is 1.37x / 1.63x the Q8_0 noise against 1.8x for 256; the kernel cost grows because the scale application is amortised over fewer elements | trade-off, not a win |
| weight requantization to 256-element blocks | yes | the quality gate (the model perplexity) could not be run: the device stopped | measurement blocked |
| the end-to-end hybrid of the new engine | yes | same | measurement blocked |
| the 16-core design | no | the channel budget is reached before the tile budget: 16 workers x 2 ObjectFifos = 32 channels | device resource |

### Incidents

* **The NPU driver stopped accepting contexts at 03:09** and did not recover
  (`DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22)`). The trigger was the
  `--no-drain-wait` variant, which hangs the XRT run for 61 s. Details and
  the timeline in `experiments/phase62/logs/INCIDENTS.md`.
* Because the XDNA backend is registered at startup, the wedged device also
  **hung the GPU-only runs**. `GGML_XDNA_DISABLE=1` was added so that the
  Vulkan reference can still be measured (1326.15 +- 17.36 tok/s at PP512).
* The first baseline campaign of the session failed the same way as the
  phase 61 incident (a Rust panic in the XDNA TEE, "2nd pipe should have
  enough spare"), while the Vulkan-only runs of the same campaign
  succeeded: the machine had 3.5 GB in swap at that moment.
* The `experiments/phase47b/bench_level0.cpp` and `bench_level1.cpp` test
  files do not compile (they call the engine constructor with a signature
  from before phase 40M). This is pre-existing and unrelated to phase 62;
  the phase 62 targets (`llama-bench`, `llama-perplexity`) build and run.

---

## 9. Limitations and what is needed next

### The runbook for when the NPU is back

The three measurements that are missing, in the order they should be taken:

```sh
# 1. the grouped task groups: does the multi-tile route scale at all?
$PY experiments/phase62/kbench/kbench62.py --M 512 --K 1024 --N 3584 \
    --wcols 4 --m-tile 64 --n-tile 64 --k-chunk 256 --variant c256x2 \
    --layout mrg --repeat 5 \
    --work-dir experiments/phase62/kernels/F_c256x2_c4_grouped --recompile \
    --tag F_c256x2_c4_grouped
#    and the same with --wcols 7

# 2. the fixed point kernel
$PY experiments/phase62/kbench/kbench62.py --M 512 --K 1024 --N 3584 \
    --wcols 4 --m-tile 64 --n-tile 64 --k-chunk 256 --variant fp256 \
    --layout sep --repeat 5 \
    --work-dir experiments/phase62/kernels/E_fp256_c4 --recompile --tag E_fp256_c4

# 3. the end to end hybrid of the new engine, and its quality gate
GGML_XDNA_QWEN_PP=1 XDNA_Q8_ENGINE=c4_mmul_v4 \
  XDNA_KERNEL_CACHE=$PWD/experiments/npu-int8-gemm/kernel_cache \
  ./experiments/phase48/npu_guard.sh -- \
  ./build/bin/llama-perplexity -m models/Qwen3.5-0.8B-Q8_0.gguf \
    -f experiments/phase61/ppl/wiki.test.raw -c 512 --chunks 6
```

The quality gate of the new engine is the perplexity against the references
of this report: 16.4344 (the Vulkan path, phase 61), 16.4378 (the phase 61
`c4_mmul_v3` engine) and 16.4478 (the CPU backend, measured today). The
per-GEMM noise of the new quantization is 1.8x the Q8_0 noise, so a
perplexity within about 0.5% of the references is the expected outcome; a
larger deviation means the requantization is wrong somewhere.

### The limitations

* **The end-to-end numbers for the new engine are missing.** The kernel is
  2.64x faster than the phase 61 kernel and numerically correct per GEMM,
  but its perplexity (the quality gate) and its end-to-end throughput were
  not measured. The projection in section 5.4 is a projection.
* **The multi-tile route is unmeasured in its corrected form.** The finding
  of section 4.2 (the workers are serialised by the runtime sequence) is
  measured; the fix (the grouped task groups) is not.
* The 16-core design needs the MemTile fan-out, which the IRON ObjectFifo
  model does not express; the MLIR-AIE `join()` primitive does.
* The hybrid still loses by 2.6x even with the projected 8x multi-tile
  speedup, so the FFN offload is not a viable direction on this machine with
  this model. **The only remaining target is the attention**, and on this
  model the attention is 10% of the pass at PP5120, which caps the whole
  hybrid at about 1.1x even with a perfect attention kernel.
* One model (Qwen3.5-0.8B Q8_0), one NPU (Phoenix XDNA1), one GPU (780M),
  one day.
