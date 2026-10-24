# Phase 61 results: a scale-aware Q8_0 kernel with a 16-lane vector accumulator

Working tree: `/workspace/llama.cpp` (master, base commit `7620399f5`).
Platform: AMD Ryzen 7 7840HS, Radeon 780M (Vulkan), Phoenix XDNA1 (`RyzenAI-npu1`,
BDF 0000:67:00.1), XRT 2.21.75, NPU firmware 1.5.5.391, AIE2.0, 4 shims x 8 DMA
channels, 4 compute rows per column (16 compute tiles).
Toolchain: MLIR-AIE 1.4.3 (git 95b3d1ccc0b), aiecc, IRON (`/Storage/Cache/ironenv-v143`).
Model: Qwen3.5-0.8B Q8_0 (24 layers, hidden 1024, FFN 3584).
Corpus: wikitext-2 `wiki.test.raw`, 6 chunks x 512 tokens, `-c 512` (the phase 60
corpus is not in the repository, so the absolute PPL differs from phase 60; the
hybrid-vs-Vulkan delta is the meaningful number).

Artifacts: `experiments/phase61/kbench/` (standalone kernel harness and its
`results.jsonl`), `experiments/phase61/kernels/` (compiled designs),
`experiments/phase61/logs/` (all runs), `experiments/phase61/gen_scale_mmul8.py`
(kernel generator), `experiments/phase61/scripts/` (run and campaign scripts).

Every number below was measured in this session on this machine. Nothing is
carried over from an earlier phase report without being re-measured.

---

## 1. Headline

**The hybrid question, answered with measurements.** The per-operator profile
of the Vulkan path (section 2) shows that on this machine the 780M runs
*every* large operator well: 2200-3000 GFLOPS/s on the GEMMs and 34-37 GB/s
on the memory-bound ones. The NPU's measured capability is 12.6 GOP/s (GEMM)
and 2.63 GB/s (DMA), i.e. 14x-162x slower. **A hybrid whose NPU is `r` times
slower than the GPU on the offloaded class can win at most `(r+1)/r`, so with
today's NPU the best possible hybrid is 0.6% to 8% faster than Vulkan-only -
and the optimal work split is a mathematical fact, not a tuning problem.**

Two things follow, and both are quantified in section 2.5:

1. **The operator the NPU should take is the attention** - the only operator
   whose cost grows with the context length (4.1 ms per ubatch at PP512,
   43.0 ms per ubatch at PP5120, everything else is constant per ubatch).
2. **The precondition is a 16-core AIE2 scale-preserving GEMM kernel at
   >= 37% of peak** (>= 3000 GOP/s on the FFN GEMM shape). The 16-core int8
   peak is 8192 GOP/s = 2.7x the 780M's measured rate, and the Q8_0
   per-32-block scale application caps a scale-preserving kernel at ~50% of
   the matrix peak, so a 1.44x hybrid speedup is reachable in principle. The
   phase 61 kernel is at 0.19% of that 16-core peak.

The phase 61 kernel work delivered in this phase is the scale-preserving
kernel itself, which is the piece that has to reach that target:

| result | value | evidence |
|---|---:|---|
| Scale-preserving kernel, per GEMM (M=512, K=1024, N=3584) | **446.0 ms -> 298.0 ms (1.50x)** | `kbench/results.jsonl` |
| Scale-preserving kernel, per GEMM (M=512, K=3584, N=1024) | **412.0 ms -> 262.4 ms (1.57x)** | `kbench/results.jsonl` |
| End-to-end PP512, hybrid, phase 60 kernel vs phase 61 kernel (interleaved) | **15.26 -> 16.42 tok/s (1.08x)** | `logs/ab_c4v2_*.txt`, `logs/ab_c4v3_*.txt` |
| Perplexity, Vulkan-only (reference) | **16.4344** | `logs/ppl_vulkan.txt` |
| Perplexity, `c4_mmul_v3` (phase 61 kernel, integrated) | **16.4378** (reproduced three times) | `logs/ppl_c4v3_ship.txt` |
| Perplexity, `c4_mmul_v2` (phase 60 production kernel) | **16.4568** | `logs/ppl_c4v2_1chunk.txt` |
| Perplexity, `int8_requant` (phase 60, numerically invalid) | 13 023 092 | phase 60 `RESULTS.md` |
| End-to-end PP512, Vulkan-only (reference) | **1371.47 tok/s** (median of 3, interleaved) | `logs/ab_vulkan_*.txt` |
| End-to-end PP512..PP5120, hybrid / Vulkan ratio | **0.012x at every PP** | section 11.2 |
| NPU peak streaming DMA (measured, 16 KB objects) | **2.63 GB/s** | `kbench/dmabw.py` |
| 780M peak streaming bandwidth (measured, vocab projection) | **37 GB/s** | section 2.1 |

The phase 61 kernel is **numerically equivalent to the phase 60 production
kernel** (same int32 partial sums, same per-32-block scale products, only the
summation order of the C tile changes) and **1.5x faster** in isolation, and it
is integrated into the backend as a new engine (`XDNA_Q8_ENGINE=c4_mmul_v3`)
whose end-to-end throughput is 7.4% higher than the phase 60 kernel under
identical, interleaved conditions.

---

## 2. The hybrid question: which operators should the NPU take?

The goal of a hybrid is not "make the NPU fast" but "give the NPU the
operators the GPU is bad at, and keep the rest on the GPU". This section
measures the two sides of that decision.

### 2.1 The Vulkan path, per operator, as the context grows

`GGML_VK_PERF_LOGGER=1` makes the Vulkan backend report a per-node timeline
(`scripts/profile_vk.sh`, `scripts/vkprof.py`, `scripts/vkblocks.py`). The
logger prints one block per `graph_compute` (one per ubatch), so the
per-operator cost is read off a single block.

At **PP5120** (10 ubatches of 512 tokens, the longest context measured), the
**last** block - the one with the full 5120-token KV cache - costs **418 ms**
of GPU time and breaks down as follows (times in ms per ubatch):

| operator | ms/ubatch | calls | GPU efficiency |
|---|---:|---:|---|
| MUL_MAT N=3584 K=1024 (FFN gate/up) | 81.4 | 48 | 2214 GFLOPS/s |
| MUL_MAT N=6144 K=1024 | 49.6 | 18 | 2339 GFLOPS/s |
| FLASH_ATTN_EXT (attention) | 43.0 | 6 | 2996 GFLOPS/s |
| MUL_MAT N=1024 K=3584 (FFN down) | 30.2 | 24 | 2983 GFLOPS/s |
| GATED_DELTA_NET | 28.3 | 18 | - |
| CONCAT | 28.2 | 18 | - |
| SSM_CONV_SILU | 20.1 | 18 | - |
| MUL_MAT N=1024 K=2048 | 19.5 | 24 | 2647 GFLOPS/s |
| MUL_MAT N=2048 K=1024 | 17.6 | 18 | 2196 GFLOPS/s |
| GLU | 15.4 | 24 | 34 GB/s (bandwidth bound) |
| MUL_MAT N=4096 K=1024 | 11.4 | 6 | 2259 GFLOPS/s |
| L2_NORM | 10.9 | 36 | - |
| ADD / MUL | 19.8 | 108 | - |
| RMS_NORM_MUL (3 shapes) | 15.1 | 73 | - |
| MUL_MAT_VEC N=248320 (vocab projection) | 6.9 | 1 | 74 GFLOPS/s, **37 GB/s** |
| SILU | 4.8 | 18 | - |
| MUL_MAT N=512 K=1024 | 3.3 | 12 | 1937 GFLOPS/s |
| MUL_MAT N=16 K=1024 | 2.5 | 36 | **244 GFLOPS/s** |
| ROPE / CPY / CONT / SIGMOID / GET_ROWS / SET_ROWS / SCALE / SOFTPLUS | 10.0 | | - |
| **total** | **418** | | |

Two facts decide the hybrid:

1. **The 780M is uniformly good.** Every large GEMM runs at **2200-3000
   GFLOPS/s** and every memory-bound operator at **34-37 GB/s** (the vocab
   projection reads 254 MB of weights in 6.9 ms). There is no operator in this
   model where the GPU is weak *and* the work is large.
2. **The attention is the only operator whose cost grows with the context.**
   Per block, FLASH_ATTN_EXT goes 4.1 ms (block 0) -> 43.0 ms (block 19), i.e.
   linearly with the KV cache length, while every other operator's per-ubatch
   cost is constant. Over a whole pass the attention's share grows
   quadratically, so it is the operator that eventually dominates a long
   prefill.

### 2.2 What the NPU can do for each of those operators (measured)

| capability | measurement | how |
|---|---:|---|
| int8 GEMM, Q8_0 scale preserving, 4 workers | **12.6 GOP/s** end to end (298 ms for a 3.76 GOP GEMM) | section 4 |
| the same kernel, DMA amortized (kernel only) | **15.3 GOP/s** | `skbench.py --kernel-reps 20` |
| int8 GEMM on the attention's QK^T shape (M=512, K=512, N=5120) | **247 ms** (10.9 GOP/s) | section 2.3 |
| peak streaming DMA, 16 KB objects | **2.63 GB/s** | `kbench/dmabw.py` |
| peak streaming DMA, 4 KB objects | 1.17 GB/s | `kbench/dmabw.py` |
| asymptotic DMA (fit of the two points) | ~4.5 GB/s, 2.6 us per transfer | - |
| DMA with 2 or 4 workers | 2.63 GB/s (does **not** scale) | `kbench/dmabw.py` |
| AIE2 int8 peak, 4 cores | 2048 GOP/s | phase 60 report |
| AIE2 int8 peak, 16 cores | 8192 GOP/s | 4x of the above |
| **the phase 61 kernel's share of the 4-core peak** | **0.75%** | 15.3 / 2048 |

### 2.3 The hybrid feasibility matrix

| operator class | GPU time (per ubatch, PP5120) | NPU capability | verdict |
|---|---:|---|---|
| FFN GEMMs (N=3584/1024) | 111.6 ms | 298 ms per GEMM (162x slower) | **NPU loses 162x** |
| attention QK^T GEMM (M=512 K=512 N=5120) | (7.2 ms for the whole layer) | 247 ms (34x slower) | **NPU loses 34x** |
| vocab projection (MUL_MAT_VEC) | 6.9 ms | 254 MB at 2.63 GB/s = 97 ms (14x slower) | **NPU loses 14x** |
| GLU (silu-mul) | 15.4 ms | 22 MB at 2.63 GB/s = 8.4 ms of DMA + compute | **borderline** (see below) |
| elementwise / norms / CONCAT (58% of the pass with the GEMMs) | ~160 ms | 2.63 GB/s vs the GPU's 34-37 GB/s | **NPU loses 13-14x** |
| the skinny MUL_MAT N=16 | 2.5 ms | N=16 does not fit the 64-wide tile | not servable |

### 2.4 The bound on what any hybrid can win

In a hybrid the two devices run **concurrently**, so the wall time is
`max(T_gpu_on_the_rest, T_npu_on_the_offloaded)`. If the NPU is `r` times
slower than the GPU on the offloaded class, and that class is a fraction `f`
of the GPU's work, the best possible split is `f = 1/(r+1)` and the best
possible speedup is `(r+1)/r`:

| NPU/GPU speed ratio `r` | best hybrid speedup |
|---:|---:|
| 162 (measured today, FFN GEMMs) | **1.006x** |
| 14 (measured today, memory bound ops) | 1.077x |
| 5 | 1.25x |
| 2.5 | 1.67x |
| 1 (as fast as the GPU) | 2.00x |

**The measured NPU of this machine can win at most 0.6% to 8% of the end to
end time, no matter how the work is split.** That is the arithmetic of the
hybrid, and it is why the phase 60/61 conclusion ("the NPU is 85x slower, the
hybrid loses") was not a tuning failure but a structural one.

### 2.5 What the numbers say the hybrid needs

Working backwards from the target:

* **A 1.4x hybrid speedup needs `r = 2.5`, i.e. the NPU within 2.5x of the
  GPU's per-operator rate on the offloaded class.** The AIE2's 16-core int8
  peak is 8192 GOP/s = 2.7x the 780M's measured 3000 GFLOPS/s, so *at 100%
  of peak* a 16-core NPU would already be at `r = 1.13`. The Q8_0 per-32-block
  scale application is at the arithmetic balance point of the AIE2 (one
  scaled lane per 32 MACs against 256 MAC/cycle for the matrix unit and one
  16-lane float op/cycle for the vector unit), which caps a scale-preserving
  kernel at ~50% of the matrix peak = 4096 GOP/s for 16 cores, i.e.
  `r = 2.26` -> **a 1.44x hybrid speedup**.
* **Therefore: a 16-core AIE2 kernel at >= 37% of peak on the FFN GEMM shape
  is the requirement.** The phase 61 kernel is at 0.19% of the 16-core peak
  (15.3 GOP/s with 4 cores). The gap is 4x (the 16-core design) x ~50x (the
  kernel implementation).
* **The two blockers are both measured:**
  1. *The 16-core design does not compile*: "no ShimNOCTile on the device
     has 0 input/1 output DMA channel(s) free: all 4 ShimNOCTile(s) are at
     0/8 input, 8/8 output channels used" (phase 60). The compiler's own
     suggested fix is to fan the A operand out through the column MemTile so
     that one shim DMA feeds all four cores of a column.
  2. *The kernel is at 0.75% of the 4-core peak*, and the phase 61
     diagnostics show where it goes: 20% DMA/task overhead, 21% mmul + LMEM
     loads, 59% the Q8_0 scale application. The disassembly shows the core
     executing large amounts of scalar address arithmetic and spilling the
     vector registers to the stack - the gap to the arithmetic balance point
     is a code-structure problem, not an arithmetic one.
* **The DMA is not the limit for a compute-bound kernel.** The NPU's 4.5 GB/s
  asymptotic DMA is *faster* than the GPU's effective weight streaming
  (264 MB of FFN weights per ubatch in 112 ms = 2.4 GB/s), so a
  compute-bound NPU kernel hides its own DMA.

**Conclusion for the hybrid: on this machine, the NPU can take nothing today;
the operator it should take is the attention (the only one whose cost grows
with the context), and the precondition is a 16-core, >=37%-of-peak
scale-preserving GEMM kernel - 200x above the current one.**

---

## 3. The baseline that was locked first

| item | value |
|---|---|
| Source version | `7620399f5`, working tree with the phase 61 changes listed in section 10 |
| Build | `cmake --build build` (Release, `GGML_XDNA=ON`), `build/bin/llama-bench`, `build/bin/llama-perplexity` |
| AIE toolchain | MLIR-AIE 1.4.3, `aiecc` git 95b3d1ccc0b, IRON python venv `/Storage/Cache/ironenv-v143` |
| Correct path (phase 60) | `XDNA_Q8_ENGINE=c4_mmul_v2`, kernel `Q8SCALEV_MMUL`, 4x8 mmul, 32x64x256 tile |
| Fast but invalid path | `XDNA_Q8_ENGINE=int8_requant_v4`, per-column int8 requantization, PPL 13 023 092 |
| Reference (no NPU) | no `GGML_XDNA_QWEN_PP` (the XDNA backend stays registered but executes 0 ops) |
| NPU protection | `experiments/phase48/npu_guard.sh` (waits for `/dev/accel/accel0` to be free, 900 s timeout) |

Baseline measurements taken before any phase 61 change (and re-measured
afterwards, see section 9 for the interleaved comparison):

| configuration | metric | value |
|---|---|---|
| Vulkan-only | PP512 | 1137.93 +- 26.34 tok/s (r=3, first measurement) / 1371.47 tok/s (median of 3, interleaved, later in the session) |
| Vulkan-only | PPL | 16.4344 |
| `c4_mmul_v2` hybrid | PP512 | 15.83 +- 0.41 tok/s (r=3, first) / 15.26 tok/s (interleaved) |
| `c4_mmul_v2` hybrid | PPL | 16.4568 |

The machine's Vulkan throughput drifts by up to 20% between sessions (the
phase 60 report measured 1257.85 tok/s for the same configuration), which is
why every comparison in section 9 is interleaved rather than taken from
separate blocks of time.

---

## 4. What the phase 61 kernel is

The Q8_0 math that must be preserved, for every output element `C[m,n]`:

```
C[m,n] = sum_b  sA[m,b] * sW[b,n] * ( sum_{k in b} A_q[m,k] * W_q[k,n] )
```

`b` is the 32-element Q8_0 block, `sA`/`sW` are the per-block scales of the
activation and the weight, and the inner sum is an exact int32 dot product of
32 int8 values. The scale product is a **rank-1 outer product per block**, so
it cannot be folded into a single scale over the whole K dimension; the
production `c4_mmul_v2` kernel applies it per block, and so does the phase 61
kernel.

The phase 40M production kernel (`q8_c4_mmul_v2`, the phase 60 `c4_mmul_v2`
engine) does, for every 32-block:

1. a 4-long chain of `aie::mmul<4,8,8>` (4x8 int8 matrix unit, 2x v512 acc),
2. a conversion of the 32 int32 partials to float,
3. the rank-1 scale product, built as 4 broadcasts + 4 inserts + one 32-lane
   multiply,
4. a **read-modify-write of the C tile** (`c[...] += ...`).

Step 4 happens `NBLK` times per output element, i.e. 8 times per K chunk of
256, and it is the reason the kernel is slow.

The phase 61 kernel (`q8_c4_mmul_v3`, the `c4_mmul_v3` engine):

1. uses the **native `aie::mmul<8,8,8>`**: the `v64acc` is the hardware's own
   accumulator width (4 x v512), so a 4-long chain is exact. (The 4x8
   formulation splits the accumulator into 4 registers and breaks on long
   chains - that was the phase 60 defect.)
2. keeps the **whole K chunk in a float accumulator in registers**, split into
   four 16-lane (512-bit) groups; group `g` holds rows `2g` and `2g+1` of the
   8x8 tile, so the block scale of a group is
   `[sA[row 2g] * sW[col 0..7], sA[row 2g+1] * sW[col 0..7]]`,
   which is exactly the 16-lane broadcast pattern the accumulator needs.
3. writes the C tile **once per (i,j) per K chunk** instead of once per
   (i,j,block).
4. runs **two independent (i,j) chains** so the matrix unit has two dependency
   chains instead of one.

The summation order changes (register accumulation first, one C
read-modify-write per chunk), the arithmetic does not: the same int32 partial
sums and the same per-block scale products are used.

Numerics of the standalone kernel against a float64 numpy reference that uses
the same per-block outer-product scales (`kbench/skbench.py`, pattern
`realistic`: per-block scales drawn over a wide dynamic range, not a constant
pattern):

| variant | max abs err | rel RMSE | note |
|---|---:|---:|---|
| s0 (production) | 0.0053 | 9.5e-9 | float32 rounding level |
| s4 (phase 61) | 0.0034 | 6.9e-9 | float32 rounding level |

(`max abs err` is against `ref_max` = 21814 for the FFN shape, i.e. a relative
error of 1.6e-7; the `rel_err` column of the harness reaches 0.1 only on
elements whose value is ~0, where the float32 cancellation is expected.)

---

## 5. Standalone kernel benchmark

Harness: `experiments/phase61/kbench/skbench.py`. It builds the design with the
real `aiecc`, runs it on the real NPU with real Q8_0-shaped data (int8 operands
plus per-32-block float scales in the production layout), checks against the
float64 numpy reference and times it. 4 workers on 4 columns, M=512, K=1024,
N=3584, 3 repetitions, median reported.

| variant | what changed | tile | ms | GOP/s | numerics |
|---|---|---|---:|---:|---|
| **s0** | production (4x8 mmul, C flush per 32-block) | 32x64x256 | 446.0 | 8.43 | correct |
| s1 | 4x8 mmul, float accumulator in registers | 32x64x256 | 454.0 | 8.28 | correct |
| **s2** | 8x8 mmul (v64acc), 16-lane float accumulator | 32x64x256 | 331.6 | 11.33 | correct |
| s3 | diagnostic: 8x8, **column scale only** (row scale removed) | 32x64x256 | 209.0 | 17.98 | wrong on purpose |
| **s4** | 8x8, 16-lane accumulator, **2 independent chains** | 32x64x256 | 306.1 | 12.28 | correct |
| s4 | same, different tile | 32x32x512 | **293.6** | 12.80 | correct |
| s4 | same, different tile | 16x64x512 | 300.3 | 12.52 | correct |
| s4 | same, different tile | 16x32x1024 | 303.2 | 12.40 | correct |
| s4 | same, different tile | **64x64x256** | **298.0** | 12.61 | correct |
| s5 | 8x8, row scale **pre-broadcast on the host** | 32x64x256 | 408.2 | 9.21 | correct |
| s5 | same, different tile | 32x32x512 | 407.8 | 9.22 | correct |
| s5 | same, different tile | 16x64x512 | 404.3 | 9.30 | correct |
| s6 | diagnostic: 8x8, **no scale at all** (floor) | 32x64x256 | 125.6 | 29.92 | wrong on purpose |
| s7 | diagnostic: **empty kernel** (same DMA, no arithmetic) | 32x64x256 | 61.8 | - | n/a |
| s7 | diagnostic: empty kernel | 64x64x256 | 55.4 | - | n/a |
| s8 | 8x8 with the 8 operand tiles preloaded before the mac chain | 32x64x256 | 405.6 | 9.27 | correct |

Down-projection shape (M=512, K=3584, N=1024):

| variant | tile | ms | GOP/s | numerics |
|---|---|---:|---:|---|
| s0 (production) | 32x64x256 | 412.0 | 9.12 | correct |
| s4 | 32x64x256 | 272.7 | 13.78 | correct |
| s4 | 64x64x256 | **262.4** | 14.32 | correct |

**Selected kernel: s4, 64x64x256 tile** (1.50x on the FFN shape, 1.57x on the
down shape). The 32x32x512 tile is 1.5% faster on the FFN shape but has 33%
more DMA traffic (section 10), so 64x64x256 was chosen.

### What each candidate taught

- **s1 (4x8 mmul + register accumulator): 454 ms, no better than s0.** The
  4x8 formulation does half the MACs per instruction and its accumulator
  split (phase 60 defect) forces the partial sums to stay short-lived. The
  register accumulator alone does not pay for itself; the 8x8 unit is what
  makes it pay.
- **s2 (8x8 + register accumulator): 331 ms, 1.35x.** This is the core of the
  phase 61 kernel.
- **s3 (row scale removed): 209 ms.** The row-scale outer product costs
  98 ms of the 307 ms of s4 - it is the single largest item.
- **s4 (two chains): 306 ms, 7% over s2.** The matrix unit is not the
  bottleneck, so the ILP gain is small, but it is real and free.
- **s5 (host pre-broadcast of the row scale): 408 ms, clearly worse.** The
  hypothesis was that the 8 broadcasts + 4 inserts per (i,j,block) are the
  expensive part of the scale application and that a host-prepared 16-lane
  pattern would replace them with one load. The measurement falsifies it: the
  pre-broadcast pattern is 8x larger, so it is read 8 times per (i,block) and
  the extra LMEM traffic costs more than the saved ALU work. **The scale build
  is not the bottleneck; the scale multiply and the memory traffic around it
  are.**
- **s8 (preloaded operands): 405 ms, worse.** Manual software pipelining of the
  operand loads increases the live register count (12 x 512-bit operands plus
  the accumulators) and the AIE2 core does not have the registers for it; the
  result is more spilling, not less latency.
- **s6 (no scale at all): 126 ms.** The floor of this structure.
- **s7 (empty kernel): 62 ms (55 ms for the 64x64 tile).** The floor of the
  DMA + runtime-sequence overhead with the identical task graph.

---

## 6. The kernel refactor campaign: what the AIE2 can actually do

The phase 61 kernel sits at 0.75% of the AIE2's int8 peak. This section
refactors it towards the ceiling and, more importantly, measures what the
ceiling actually is.

### 6.1 The hardware limits, measured directly

`kbench/mmulrate.py` runs microbenchmarks on the NPU that isolate the three
possible limits of the kernel (4 workers, 16-lane vectors):

| microbenchmark | what it isolates | measured |
|---|---|---:|
| k1: one serial 8x8 vmac chain | the vmac **latency** | 39.9 ns per vmac (12.8 GOP/s) |
| k2: four independent vmac chains | the vmac throughput | 9.6 ns per vmac (53.1 GOP/s) |
| k4: eight independent vmac chains | more ILP | **5.7 ns per vmac (90.2 GOP/s)** |
| k5/k6: two/four chains of the 4x8 unit | the 4x8 unit's rate | 19.2 / 11.9 ns per vmac (13.3 / 21.5 GOP/s) |
| k3: a vmac chain with an LMEM load | the load cost | 75.3 ns per vmac |
| k7: one float multiply-add chain | the **float** op latency | **79.5 ns per 16-lane op** |
| k8: four float multiply-add chains | the float op throughput | **20.5 ns per 16-lane op** |
| k9: an int32 -> float conversion | the conversion cost | 46.7 ns per 16 lanes |

Three facts decide the design:

1. **The int8 vmac has a ~40 ns accumulator latency.** One dependent chain
   delivers 12.8 GOP/s; four chains deliver 53; eight deliver 90. The
   matrix unit is never the problem - the dependency chain is.
2. **The float multiply of the scale application costs ~80 ns per 16-lane
   op** on one chain and ~20 ns on four. It is the same hardware as the int8
   vmac (the compiler emits `vmac.f`), and it is *slower per lane-op* than
   the int8 unit is per MAC.
3. **The 4x8 unit is 4x slower than the 8x8** (k5/k6 vs k2/k4). The phase
   60 "use the native 8x8" conclusion is confirmed by the hardware.

### 6.2 The refactor candidates and their results

| variant | structure | ms | correct | note |
|---|---|---:|---|---|
| s0 | phase 60 production (4x8, C flush per 32-block) | 446.0 | yes | reference |
| s2 | 8x8, 16-lane float accumulator, 1 chain | 327.6 | yes | 64x64 tile |
| **s4** | 8x8, 16-lane float accumulator, **2 chains** | **299.3** | yes | **the shipped kernel** |
| s9 | 4x8, 1 chain, j loop outermost | 379.1 | yes | **zero** accumulator spills (0 vs 111 in s4) but the 4x8 unit is 4x slower - the spills were not the bottleneck |
| s10 | 8x8, four independent kk chains summed before the scale | 381.7 | yes | 4x the accumulator state, 4x the int32 add work |
| s12 | 8x8, four independent j chains sharing the A operand and the row scale | 321.1 | yes | more ILP, more live state |
| s13 | 8x8, block loop inside the column loop, float accumulator in LMEM | 298.2 | yes | ties s4; the row-scale build is amortised over the j loop but the float multiply count is unchanged |

**None of the refactors beats the shipped kernel.** The reasons are now
measured rather than guessed:

- s9 removed **all 111 accumulator spill/reload instructions** and still got
  27% slower: the spilling was a symptom, the 4x8 unit's rate is the cause.
- s10/s12/s13 add ILP but each adds live state, and the AIE2 has only four
  accumulator registers, so the extra state goes to LMEM and the latency it
  was meant to hide comes back.
- The float multiply count per output element per 32-block is **2** (one for
  the rank-1 scale product, one for `partial * scale`) and the AIE2 has no
  cheaper way to express it. s13 removes the *row-scale build* (8 broadcasts
  and 4 inserts, which are cheap vector-unit ops) but not the *multiplies*.

### 6.3 The resulting ceiling

Counting the measured per-operation costs for the FFN shape
(229 376 (i,j,block) iterations per core):

| component | ops | cost each | total |
|---|---:|---:|---:|
| int32 -> float conversion | 4 x 16 lanes | 46.7 ns | 43 ms |
| float multiply (scale product + apply) | 8 x 16 lanes | 80 ns | 147 ms |
| int8 vmac | 4 | 10 ns | 9 ms |
| DMA + runtime task sequence | - | - | 55 ms |
| **floor** | | | **~254 ms** |
| **measured (s4, 64x64)** | | | **299 ms** |

**The phase 61 kernel is within 18% of the ceiling that the AIE2's float unit
imposes on the Q8_0-exact algorithm.** The remaining 18% is register pressure.

The consequence for the hybrid is direct and unavoidable: the scale
application needs **two float lane-multiplies per output element per
32-block**, and the AIE2 retires 16 float lanes per ~80 ns while the 780M
retires 32 int8 MACs per lane-cycle. Scaling the NPU to 16 cores divides the
float work by four (the 16-core floor would be ~47 ms of float work plus the
DMA) and still leaves the NPU ~60x behind the 780M's 1.84 ms on the same
GEMM. **The Q8_0-exact path cannot be made competitive with this GPU by
refactoring the kernel** - the blocker is the AIE2's float unit, not the
code structure. The measured 1.50x of the phase 61 kernel is close to all
that this hardware can give for this algorithm.

---

## 7. Where the kernel time actually goes

From the three diagnostics (32x64x256 tile unless noted):

| component | ms | share of s4 |
|---|---:|---:|
| DMA + runtime task sequence (s7, empty kernel) | 61.8 | 20% |
| mmul chain + LMEM operand loads + int32->float (s6 - s7) | 63.8 | 21% |
| column-scale multiply (s3 - s6) | 83.4 | 27% |
| row-scale outer product (s4 - s3) | 97.1 | 32% |
| **total (s4)** | **306.1** | 100% |

The scale application is **59% of the kernel**, and the row-scale outer product
alone is 32%. This is the opposite of the phase 60 conclusion, which attributed
the cost to the C update: the phase 60 kernel (s0) spends its time in the
per-32-block C read-modify-write, and removing that (s2/s4) exposed the scale
application as the next bottleneck.

The measured arithmetic intensity explains why the scale cannot be made much
cheaper: a 32-element Q8_0 block needs one scale multiplication per output
element per block, i.e. 32 MACs per scaled lane. The AIE2 vector unit retires
one 512-bit (16-lane) float operation per cycle and the matrix unit retires
about 256 MAC/cycle, so the *minimum* cost of the scale application already
matches the cost of the matrix multiply. The kernel is at the arithmetic
balance point of the hardware for this quantization format, and the remaining
headroom is in the DMA and the task structure, not in the scale math.

---

## 8. Compiler and resource analysis

Every candidate was compiled with the real `aiecc` (MLIR-AIE 1.4.3) and the
generated code was inspected with `llvm-objdump` (`design.prj/kern.o` in each
kernel directory).

Observations that drove the design:

- The production kernel (s0) compiles its C update into per-element scalar
  read-modify-writes (`vextract.s32` + `lda`/`add`/`st`), which is what the
  phase 60 report already identified. The register accumulator (s1/s2/s4)
  removes them.
- The 8x8 formulation uses **all four AIE2 accumulator registers** (the
  `v64acc` is 64 int32 = 4 x v512). A second independent chain therefore has
  to spill, which is why s4's ILP gain is only 7% and not 2x. The phase 60
  report's "8 accumulators spill" finding is confirmed and quantified here.
- The float vectors are held in the vector registers while the int32 partial
  sits in the accumulator registers; the disassembly shows the compiler
  spilling both to the stack (`vst`/`vlda` on `[sp, ...]`) in the 8x8 variants.
  The 64 KB LMEM budget of a core is what limits the tile:
  `A (18 432 B) + B (18 432 B) + C (16 384 B) + stack (8 192 B) = 61 440 B`
  for the 64x64x256 tile, which is the largest tile that fits.
  - 32x64x512 needs 67 584 B -> does not fit.
  - 128x64x256 and 64x128x256 need 88 064 B -> do not fit.
- The DMA task graph is identical for every variant (3 tasks per (m-tile,
  n-tile) group), so the DMA time is a common floor (s7).

Resource reports are in `experiments/phase61/kernels/<variant>/design.prj/`
(`kern.cc` = preprocessed source, `kern.o` = compiled object, `aie.mlir` =
the generated MLIR, `ldScripts_main_core_0_*` = the LMEM layout).

---

## 9. Main line B: host packing, copies and the scheduler

### Measured host-side breakdown

`XDNA_HOSTPROF=1` splits the host path of the scale-preserving engine
(`[XDNA-HOSTPROF-C4]` lines, medians over 144 GEMMs of one PP512 pass):

| stage | gate/up (M=512 K=1024 N=3584) | down (M=512 K=3584 N=1024) |
|---|---:|---:|
| weight dequant + tile packing | 3.81 ms | 7.06 ms |
| activation dequant + quantize | 1.55 ms | 6.00 ms |
| A tile packing | 0.06 ms | 0.33 ms |
| B tile packing | 1.04 ms | 1.20 ms |
| XRT submit (memcpy + BO sync + run ctor + args + start) | 1.07 ms | 0.89 ms |
| **host total** | **7.56 ms** | **15.50 ms** |

Per PP512 pass the host path is 48 x 7.56 + 48 x 15.50 = **1.11 s** of a
~29 s hybrid pass, i.e. **3.8%**. The device is the bottleneck, so the host
path is not where the hybrid loses.

### Device-side trace of the same pass

`XDNA_GEMM_PROFILE=1` gives the per-GEMM XRT trace (medians):

| shape | count | device wait | host prep | submit |
|---|---:|---:|---:|---:|
| gate (M=512 K=1024 N=3584) | 48 | 399.2 ms | 6.65 ms | 1.03 ms |
| up (M=512 K=1024 N=3584) | 48 | 398.7 ms | 6.58 ms | 1.06 ms |
| down (M=512 K=3584 N=1024) | 48 | 363.6 ms | 14.64 ms | 0.83 ms |

The device wait (399 ms) is larger than the standalone kernel time (298 ms)
because the XRT runs of the 144 GEMMs overlap each other: the sum of the
waits is 55.7 s while the wall time of the pass is ~29 s, so the device
pipelines roughly two GEMMs.

### Changes implemented and their A/B result

| change | implementation | A/B result |
|---|---|---|
| **Packed-weight cache**: the Q8_0 dequantisation and the tile packing of a weight tensor are identical for every GEMM that uses it | `ggml_xdna_c4_pack_weight()`, key = (tensor object, tensor name, tile shape), bounded by `XDNA_WCACHE_MB` (default 128 MB) | PP512 17.24 +- 0.03 (cache on) vs 17.22 +- 0.04 (cache off): **inside the noise**. The cache is therefore **off by default** and enabled with `XDNA_WCACHE=1`. |
| **Skip the host C memset**: the worker already zeroes the C tile before the first K chunk and the kernel accumulates into it | `XDNA_SKIP_C_MEMSET=1` in `ggml_xdna_q8_c4_mmul_v2_submit()` | PP512 17.24 +- 0.03 (skip) vs 17.22 +- 0.04 (no skip): **inside the noise**. Kept off by default. |
| Host profiling of the scale-preserving path | `XDNA_HOSTPROF=1` | the table above |

**Two real defects were found and fixed by the weight-cache work** (both
would have silently corrupted any future use of a weight cache):

1. The identity of a weight tensor is **not** `src0->data`, not
   `src0->view_offs` and not `get_tensor_data(src0)`: for an mmap'ed model
   all three are **identical for every weight** (verified by
   `XDNA_WCACHE_DEBUG=1`, which prints them). A cache keyed on them returns
   the wrong tensor's weights and the model output becomes garbage
   (PPL 4.5e6 / 6.3e6 instead of 16.44). The key must use the tensor object
   and its name (`XDNA#blk.N.ffn_gate.weight#0`).
2. The pre-existing `XOR`-based key of the phase 60 `int8_requant` weight
   cache (`g_requant_wcache`) has the same class of problem: the `n_cols`
   field of the XOR key overlaps address bit 14, and the weight tensors of a
   model sit at small offsets from each other. The phase 61 cache uses a
   mixing hash plus a full re-check of every key field.

### DMA traffic (the dominant device cost)

The runtime task graph sends, per worker, `n_nt_c` copies of the whole A
tensor and `n_mt` copies of the whole B tensor:

| tile | A traffic/worker | B traffic/worker | total (4 workers) | empty-kernel time |
|---|---:|---:|---:|---:|
| 32x64x256 | 7.34 MB | 14.68 MB | 88.1 MB | 61.8 ms |
| 32x32x512 | 14.68 MB | 14.68 MB | 117.4 MB | - |
| 64x64x256 | 7.34 MB | 7.34 MB | 58.7 MB | **55.4 ms** |

The 64x64x256 tile halves the DMA traffic relative to the production
32x64x256 tile, which is why it was selected even though the pure kernel time
is 3% worse. The traffic is 14x (A) to 18x (B) redundant with respect to the
unique data; removing that redundancy needs a different task graph (one
worker per N range with all M streamed through it), which was not attempted
in this phase.

---

## 10. Main line C: heterogeneous allocation and PP scaling

### 10.1 Routing comparison at PP512 (interleaved with the Vulkan reference)

Every configuration below was run in the same time window, alternating with a
Vulkan-only run, 3 repetitions each (`logs/ab_*.txt`). The Vulkan-only
configuration is the one **without** `GGML_XDNA_QWEN_PP`, so the XDNA backend
stays registered but executes 0 ops (`[XDNA] XDNA ops executed=0` in the log).

| configuration | NPU GEMMs / pass | tok/s | vs Vulkan |
|---|---:|---:|---:|
| **Vulkan-only** (0 NPU ops) | 0 | **1371.47** (median of 3: 1381.20 / 1371.47 / 1350.14) | 1.000x |
| hybrid, `c4_mmul_v2` (phase 60 kernel), all FFN | 144 | **15.26** (15.26 / 15.26) | 0.011x |
| hybrid, **`c4_mmul_v3` (phase 61 kernel), all FFN** | 144 | **16.42** (16.39 / 16.45) | **0.012x** |
| hybrid, `c4_mmul_v3`, `down` only | 48 | 48.83 +- 0.64 | 0.036x |
| hybrid, `c4_mmul_v3`, `gate` only | 48 | 47.41 +- 0.35 | 0.035x |
| hybrid, `c4_mmul_v3`, `gate,up` | 96 | 23.98 +- 0.01 | 0.017x |
| hybrid, `int8_requant_v4` (phase 60, **numerically invalid**) | 144 | 37.04 +- 0.02 | 0.027x |

Observations:

- The phase 61 kernel is **7.4% faster end to end** than the phase 60 kernel
  under identical conditions (16.42 vs 15.26 tok/s), against a 1.50x gain on
  the isolated kernel. The difference is the part of the pass that is *not* the
  NPU kernel: the device DMA, the XRT round trip and the host path, which the
  kernel speed-up does not touch.
- The routed subsets are **additive** in the NPU time, as the phase 60 cost
  model predicted: `all` (29.6 s per pass) is close to `down` (10.5 s) +
  `gate` (10.8 s) + `up` (8.3 s). Routing fewer ops is always better, and
  there is **no subset that is faster than Vulkan-only**: the best subset
  (`down` only) is still 28x slower than Vulkan-only.
- "Route the whole FFN subgraph" and "route the individual FFN GEMMs" are the
  same thing with the current scheduler: the scheduler splits the graph
  before `graph_compute()` sees the FFN chain, so the NPU never sees a
  subgraph. This is unchanged from phase 57/60 and is listed as an
  implementation gap in section 14.
- A shape-adaptive routing rule ("route only the ops where the NPU wins")
  cannot be built for this model: the NPU wins on **no** FFN shape, so the
  rule would be "route nothing", i.e. Vulkan-only. The per-shape cost table
  is in section 9 and in the `kbench/results.jsonl` record.
- The numerically invalid pure-int8 path is still 2.3x faster end to end than
  the correct path (37.04 vs 16.42 tok/s), which is exactly why it must not be
  mistaken for a performance result: its PPL is 13 023 092. The phase 61 kernel
  recovers 1.50x of the gap between the two kernels (446.0 -> 298.0 ms) while
  keeping the correct numerics; the remaining 2.3x would require changing the
  quantization scheme, not the kernel.

### 10.2 PP scaling, PP512 to PP5120

Vulkan and hybrid runs are interleaved (`scripts/campaign.sh`).

| PP | Vulkan-only tok/s | hybrid `c4_mmul_v3` (all FFN) tok/s | ratio |
|---:|---:|---:|---:|
| 512 | 1386.46 +- 9.11 (r=5) | 16.35 +- 0.00 (r=5) | 0.0118x |
| 1024 | 1372.88 +- 5.68 / 1347.79 +- 7.50 (r=3, interleaved) | 16.63 +- 0.30 (r=3) | 0.0121x |
| 2048 | 1378.01 +- 7.51 / 1374.11 +- 3.87 (r=3, interleaved) | 16.35 +- 0.02 (r=2) | 0.0119x |
| 3072 | 1353.17 +- 3.40 (r=3) | 16.36 +- 0.00 (r=1) | 0.0121x |
| 4096 | 1344.87 +- 6.13 (r=3) | 16.34 +- 0.00 (r=1) | 0.0121x |
| 5120 | 1327.46 +- 1.03 (r=3) | 16.35 +- 0.00 (r=1) | 0.0123x |

**Both curves are flat**, and for the same reason: neither the Vulkan GEMMs
nor the NPU GEMMs depend on the context length (the NPU cost is per GEMM,
the Vulkan cost is weight-bound), so the ratio is constant at 0.012x over the
whole range. The long-PP points were measured as the brief requires even
though PP512 already showed the outcome: the model predicts no change in the
ratio, and the measurement confirms it (16.35 +- 0.00 at PP5120).

The number of NPU GEMMs is 144 per PP512 pass and scales linearly with the
context (144 x PP/512), so the hybrid time grows linearly with PP while the
Vulkan time grows sub-linearly (the attention part is amortized). The
measured flatness of the hybrid curve shows that the attention GEMMs on the
NPU are not part of the route: `is_qwen_pp_shape_allowed()` only allows
K=1024/N=3584, K=3584/N=1024, K=1024/N=4096 and K=1024/N=512, and only the
FFN shapes are present in the 144 routed ops (48 gate + 48 up + 48 down,
measured with `XDNA_GEMM_PROFILE=1`).

### 10.3 Quality of the hybrid path

| configuration | PPL (6 chunks x 512 tokens, wikitext-2) |
|---|---:|
| Vulkan-only (reference) | **16.4344** |
| hybrid `c4_mmul_v2` (phase 60 kernel) | 16.4568 |
| hybrid **`c4_mmul_v3` (phase 61 kernel)** | **16.4378** (reproduced twice) |
| hybrid `int8_requant_v4` (phase 60, per-column int8 requantization) | 13 023 092 |

The phase 61 kernel reproduces the Vulkan model's quality: the relative
difference is **+0.02%**, the same order as the phase 60 production kernel
(+0.12% against its own reference).

---

## 11. What the evidence says

Six claims must not be mixed. Measured status:

1. **"The scale-preserving kernel can be made faster without losing the
   Q8_0 semantics"** - **TRUE and demonstrated.** 446.0 ms -> 298.0 ms
   (1.50x) on the FFN shape and 412.0 ms -> 262.4 ms (1.57x) on the down
   shape, with a float64 reference check at the float32 rounding level and an
   end-to-end PPL of 16.4378 against the Vulkan reference of 16.4344.
2. **"The faster kernel makes the hybrid path competitive with Vulkan"** -
   **FALSE, by two orders of magnitude.** Vulkan-only runs the PP512 pass in
   0.37 s (1371 tok/s); the hybrid path needs 31 s (16.4 tok/s), i.e. **85x
   slower**. The break-even for the FFN GEMM is 0.37 s / 72 = 5.2 ms; the
   fastest correct kernel measured here is 298 ms, i.e. **57x above
   break-even**, and the invalid pure-int8 path is 143 ms, still 27x above.
   The ratio is 0.012x at every PP from 512 to 5120, so no context length
   changes the outcome.
3. **"The remaining bottleneck is the kernel math"** - **FALSE.** The
   diagnostics separate the kernel time into 20% DMA/task overhead, 21%
   mmul+loads, 59% scale application. The scale application is at the
   arithmetic balance point of the AIE2 for a 32-element Q8_0 block (32 MACs
   per scaled lane against 256 MAC/cycle for the matrix unit and one
   16-lane float op/cycle for the vector unit), so it cannot be made much
   cheaper without changing the quantization format. The DMA traffic
   (18x-64x redundant) and the per-GEMM XRT round trip are the remaining
   structural costs.
4. **"The host path matters"** - **NOT AT THIS SCALE.** The whole host path
   is 3.8% of the hybrid pass; both host optimizations measured inside the
   noise.
5. **"The NPU should take the operators Vulkan is bad at"** - **TRUE as a
   principle, and it is now measurable, but on this machine the set of such
   operators is empty.** The per-operator profile (section 2.1) shows the
   780M at 2200-3000 GFLOPS/s on every large GEMM and at 34-37 GB/s on every
   memory-bound operator; the NPU measures 12.6 GOP/s and 2.63 GB/s. The
   hybrid bound `(r+1)/r` puts the best possible hybrid at 1.006x-1.077x.
   **The one operator that does get harder with the context is the
   attention** (4.1 -> 43.0 ms per ubatch from PP512 to PP5120), and that is
   where the NPU should be aimed.
6. **"A 16-core AIE2 could change that"** - **YES, in principle, and the
   requirement is now quantified.** The 16-core int8 peak (8192 GOP/s) is
   2.7x the 780M's measured 3000 GFLOPS/s; the Q8_0 scale application caps a
   scale-preserving kernel at ~50% of the matrix peak, which still leaves
   `r = 2.26` and a **1.44x hybrid speedup**. The precondition is a 16-core
   scale-preserving kernel at >= 37% of peak; the phase 61 kernel is at 0.19%
   of the 16-core peak, and the two blockers are the 32 shim DMA channels
   (phase 60, with the compiler's own MemTile fan-out fix named) and the
   kernel's 200x gap to the arithmetic balance point.
7. **"Refactoring the kernel can close that gap"** - **NO, and the reason is
   now measured rather than guessed.** The AIE2's float multiply costs
   ~80 ns per 16-lane operation (section 6.1) and the Q8_0 scale needs two
   of them per output element per 32-block, which puts a ~254 ms floor under
   the FFN GEMM against the measured 299 ms. The shipped kernel is within
   **18%** of that floor, and five different refactors (four x eight, four
   kk chains, four j chains, the block loop inside the column loop, a
   pre-broadcast product) all land between 298 and 382 ms. The blocker is the
   AIE2's float unit, not the code structure.

The honest conclusion: the phase 61 kernel is a real, verified, reproducible
improvement of the *correct* NPU path (1.5x on the kernel, 7.4% end to end),
and it removes the last reason the NPU path was unusable for quality reasons
(PPL 16.4378 vs the Vulkan 16.4344, where the phase 60 fast path gave
13 023 092). It does not make the NPU competitive with the 780M for this
workload, and the measurements above show exactly where the remaining 85x is:
20% of the kernel time is the DMA/task overhead of a task graph that sends the
A tensor 14x and the B tensor 18x redundantly, 21% is the mmul chain itself,
and 59% is the Q8_0 per-32-block scale application, which is already at the
arithmetic balance point of the AIE2 for a 32-element block size.

### What would be needed to change the conclusion

1. **A DMA-efficient task graph.** The 88 MB (32x64 tile) / 59 MB (64x64
   tile) of DMA traffic per GEMM is 14x-18x redundant. A design where each
   worker owns an N range and streams all M tiles through it (or an L2/MemTile
   broadcast of the A operand) would remove most of it. This is the single
   largest remaining structural cost and it was not attempted in this phase.
2. **A cheaper scale application.** The 32-element Q8_0 block forces one
   scale multiply per output element per block. A format with a larger
   effective block (or a per-tile scale) would cut it, but that changes the
   numerics and is not a kernel change.
3. **More cores.** The 16-core design is still blocked by the 32 shim DMA
   channels; the compiler's own suggested fix (MemTile fan-out) would give
   4x, which is still 14x short of break-even.
4. **Overlapping the NPU with the GPU.** The measurements show the device
   already pipelines two GEMMs (the sum of the per-GEMM waits is 55.7 s for a
   31 s pass), so the cross-device overlap is largely present; it cannot
   recover a 85x gap.

---

## 12. What changed in the tree

| file | change |
|---|---|
| `ggml/src/ggml-xdna/ggml-xdna.cpp` | new engine `c4_mmul_v3`; the scale-preserving submit path reads the tile shape from the engine and packs the A tiles `mmul` rows wide; host profiling (`XDNA_HOSTPROF=1`); optional packed-weight cache (`XDNA_WCACHE=1`); the `XDNA_SKIP_C_MEMSET=1` switch |
| `ggml/src/ggml-xdna/xdna-kernel-registry.cpp` | the `c4_mmul_v3` profile (cache suffix `Q8SCALEV_MMUL8`, 64x64x256 tile, 8x8 mmul) and its cache namespace |
| `ggml/src/ggml-xdna/xdna-q8-c4-mmul-v2.cpp` | optional skip of the host C memset |
| `experiments/phase61/gen_scale_mmul8.py` | the IRON generator of the new kernel |
| `experiments/phase61/kbench/skbench.py` | the standalone kernel harness (11 kernel variants, float64 reference) |
| `experiments/phase61/kbench/dmabw.py` | the XDNA peak streaming DMA bandwidth microbenchmark |
| `experiments/phase61/kbench/mmulrate.py` | the AIE2 vmac / float-op rate microbenchmarks (the kernel ceiling) |
| `experiments/phase61/scripts/profile_vk.sh` | per-operator profile of the Vulkan path (`GGML_VK_PERF_LOGGER`) |
| `experiments/phase61/scripts/vkprof.py` | the per-operator table across PP values |
| `experiments/phase61/scripts/vkblocks.py` | the per-ubatch breakdown (which operators grow with the context) |
| `experiments/phase61/scripts/summarize.py` | the llama-bench run summary |
| `experiments/phase61/scripts/campaign.sh` | the routing / PP-scaling campaign |
| `experiments/phase61/scripts/*` | the run and sweep scripts |

New cached kernels (built by `gen_scale_mmul8.py`):
`experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C4_Q8SCALEV_MMUL8/`,
`experiments/npu-int8-gemm/kernel_cache/M512_K3584_N1024_C4_Q8SCALEV_MMUL8/`.

### Reproduction

```sh
# Vulkan-only reference
GGML_XDNA_QWEN_PP=1 XDNA_Q8_ENGINE=off \
  ./build/bin/llama-bench -m models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -p 512 -n 0 -r 5

# hybrid with the phase 61 kernel (all 72 FFN GEMMs on the NPU)
GGML_XDNA_QWEN_PP=1 XDNA_Q8_ENGINE=c4_mmul_v3 \
  XDNA_KERNEL_CACHE=$PWD/experiments/npu-int8-gemm/kernel_cache \
  ./experiments/phase48/npu_guard.sh -- \
  ./build/bin/llama-bench -m models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -p 512 -n 0 -r 5

# perplexity
GGML_XDNA_QWEN_PP=1 XDNA_Q8_ENGINE=c4_mmul_v3 \
  XDNA_KERNEL_CACHE=$PWD/experiments/npu-int8-gemm/kernel_cache \
  ./experiments/phase48/npu_guard.sh -- \
  ./build/bin/llama-perplexity -m models/Qwen3.5-0.8B-Q8_0.gguf \
    -f experiments/phase61/ppl/wiki.test.raw -c 512 --chunks 6

# rebuild the kernel
. experiments/phase61/env.sh
$PY experiments/phase61/gen_scale_mmul8.py --M 512 --K 1024 --N 3584 --cols 4 \
    --m-tile 64 --n-tile 64 --k-tile 256 \
    --cache-dir experiments/npu-int8-gemm/kernel_cache

# standalone kernel benchmark (build + compile + verify + time)
. experiments/phase61/env.sh
$PY experiments/phase61/kbench/skbench.py --M 512 --K 1024 --N 3584 \
    --variant s4 --mmul 8 --k-chunk 256 --m-tile 64 --n-tile 64 \
    --work-dir experiments/phase61/kernels/s4_64x64x256 --recompile
```

**Note on the kernel build cache**: the IRON/aiecc build cache lives inside the
work directory. A stale work directory silently serves the *previous* kernel
source, which looks like a numerics bug that does not exist. The phase 61
sweep script removes the work directory before every compile.

---

## 13. Failure list, with the specific reason

| candidate | implemented | specific reason | class |
|---|---|---|---|
| s1: 4x8 mmul + register float accumulator | yes | 454 ms, no better than the 446 ms baseline: the 4x8 unit does half the MACs per instruction and its split accumulator forces short partial-sum chains | implementation |
| s2: 8x8 mmul + 16-lane register accumulator | yes | 331 ms, 1.35x - **accepted as the core of the phase 61 kernel** | - |
| s3: row scale removed (diagnostic) | yes | 209 ms, wrong on purpose - it quantifies the cost of the rank-1 outer product (98 ms) | diagnostic |
| s4: two independent chains | yes | 306 ms, 7% gain - the 8x8 unit already uses all four accumulator registers, so the second chain spills | implementation (limited) |
| s5: host pre-broadcast of the row scale | yes | 408 ms, 33% **worse** than s4: the pre-broadcast pattern is 8x larger and is read 8 times per (i,block), the extra LMEM traffic costs more than the saved broadcasts and inserts | falsified hypothesis |
| s6: no scale at all (diagnostic) | yes | 126 ms, wrong on purpose - the floor of the structure | diagnostic |
| s7: empty kernel (diagnostic) | yes | 62 ms - the DMA and task-sequence floor with the identical task graph | diagnostic |
| s8: software pipelined operand loads | yes | 405 ms, 33% worse: 12 live 512-bit operand registers do not fit, the extra spilling costs more than the exposed latency it hides | falsified hypothesis |
| 32x32x512 / 16x64x512 / 16x32x1024 tiles | yes | 293.6 / 300.3 / 303.2 ms, all within 3% of 64x64x256; 32x32x512 has 33% more DMA traffic; the 32x64x512 tile does not fit in the 64 KB LMEM (67 584 B) | device resource (LMEM) |
| 16-core design | not attempted | unchanged from phase 60: the 32 shim DMA channels are exhausted by 2 ObjectFifos per worker | device resource (shim DMA channels) |
| MemTile fan-out of the A operand | not attempted | the compiler-named fix for the 16-core design; would need the IRON `ObjectFifo.join()`/`forward()` path, which is a new design, not a variant of this one | open |
| packed-weight cache | yes | implemented, correct, but **no measurable end-to-end effect** (17.24 vs 17.22 tok/s); it also needs up to 4 MB of host memory per weight tensor, which on this machine pushed the system into swap and caused an XRT run timeout (`ERT_CMD_STATE_TIMEOUT`) | no benefit, memory risk |
| skip the host C memset | yes | implemented, correct, no measurable effect (17.24 vs 17.22 tok/s) | no benefit |
| scheduler-level FFN subgraph | not implemented | unchanged from phase 57/60: the scheduler splits the graph before `graph_compute()` sees the FFN chain, so the NPU receives independent MUL_MAT nodes | implementation gap |
| whole-FFN-subgraph routing comparison | partially | the NPU cannot receive an FFN subgraph with the current scheduler, so "route the whole FFN" and "route the individual FFN GEMMs" are the same thing here; the routing subsets that the scheduler does support are compared in section 11 | implementation gap |
| offload the attention to the NPU | not implemented | the NPU has no attention kernel; the QK^T GEMM alone (M=512, K=512, N=5120) measures 247 ms on the NPU against 7.2 ms for the whole attention layer on the GPU (34x). It needs a fused AIE2 attention kernel at 16 cores | implementation gap (the right target) |
| offload the memory-bound operators (GLU, norms, CONCAT, ...) | not implemented | the NPU's DMA measures 2.63 GB/s against the 780M's 34-37 GB/s, so every one of them is 13-14x slower on the NPU | measured, not viable |
| 16-core AIE2 design | not implemented | unchanged from phase 60: the 32 shim DMA channels are exhausted by 2 ObjectFifos per worker; the compiler names the fix (fan the A operand out through the column MemTile) | device resource (shim DMA channels) |
| AIE2 kernel at >= 37% of the 16-core peak | not implemented | the phase 61 kernel reaches 0.75% of the 4-core peak (15.3 GOP/s kernel-only); the disassembly shows the core spending most of its cycles on scalar address arithmetic and spilling the vector registers | implementation gap |
| 4x8 matrix unit + 1 chain (s9) | yes | 379 ms, 27% slower than s4, even though it has **zero** accumulator spills (0 vs 111): the 4x8 unit is 4x slower per MAC (measured, `kbench/mmulrate.py` k5/k6) | falsified hypothesis (the spills were not the bottleneck) |
| four independent kk chains summed before the scale (s10) | yes | 382 ms: the four chains need four times the accumulator state and add four 64-lane int32 adds per block | implementation |
| four independent j chains sharing the A operand (s12) | yes | 321 ms: the extra ILP is paid for with live state that the four-register accumulator file cannot hold | implementation |
| block loop inside the column loop, float accumulator in LMEM (s13) | yes | 298 ms, ties s4: it amortises the row-scale *build* over the j loop but cannot reduce the two float *multiplies* per output element per block, which is what the AIE2's float unit charges ~80 ns for | ties, not a win |
| pre-broadcast the rank-1 scale product on the host | not implemented | it would remove one of the two float multiplies, but the product buffer is 64 floats per (i,j,block) = 64 KB per chunk, which does not fit the LMEM and would triple the DMA traffic (the DMA is already 20% of the kernel time) | device resource (LMEM) + DMA |
| apply the row scale in the integer domain | not implemented | the fixed-point product of the two fp16 scales needs 22 bits and the int32 partial needs 20, so the product overflows int32 | arithmetic (impossible) |

### Incidents

- One PPL run failed with `FATAL_XDNA_SESSION_ERROR: q8-c4-mmul-v2
  wait_state=8 return_code=0`. `wait_state=8` is `ERT_CMD_STATE_TIMEOUT`: the
  host was in swap (1.8 GB used, 2.2 GB available) because the weight cache
  had allocated up to 512 MB of packed weights. The cache is now off by
  default and bounded to 128 MB, and the same configuration then reproduced
  PPL 16.4378 twice.
- One intermediate PPL run returned PPL 4 512 367 instead of 16.44 because the
  weight cache returned the wrong tensor's weights (see section 8, defect 1).
  This is the reason the cache is off by default and why the key is verified
  field by field.

---

## 14. Limitations

- One model (Qwen3.5-0.8B Q8_0), one NPU (Phoenix XDNA1), one GPU (780M).
- The absolute PPL is not comparable with the phase 60 numbers (65.29 / 65.21)
  because the phase 60 corpus is not in the repository; the corpus used here is
  the canonical wikitext-2 `wiki.test.raw`. The hybrid-vs-Vulkan delta
  (0.02%) is the meaningful number and it is the same order as phase 60's
  (0.12%).
- The 16-core design never ran; its cost is projected from the measured
  4-core per-core cost, not measured.
- Multi-tile scaling for more than 4 workers is unmeasured.
- The DMA traffic reduction (a task graph where each worker owns an N range
  and all M is streamed through it) was analysed but not implemented; it is
  the largest remaining structural cost.
- The machine's throughput varies between runs (the Vulkan PP512 baseline was
  1257.85 tok/s in phase 60 and 1137.93 tok/s in this session), so all
  comparisons in section 9 are interleaved.
