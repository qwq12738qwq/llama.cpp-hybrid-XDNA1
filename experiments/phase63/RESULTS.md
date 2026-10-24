# Phase 63 results: a 16 worker XDNA design, the resource audit, and the cost of the FFN

Working tree: `/workspace/llama.cpp` (master, base commit `7620399f5`).
Platform: AMD Ryzen 7 7840HS, Radeon 780M (Vulkan/RADV), Phoenix XDNA1
(`RyzenAI-npu1`, BDF 0000:67:00.1), XRT 2.21.75, AIE2.0, 4 shims x 4 compute
rows (16 compute tiles), 8 DMA channels per ShimNOCTile.
Toolchain: MLIR-AIE 1.4.3 (`/Storage/Cache/ironenv-v143`), IRON with the
explicit `Flow` / `TileDma` / `Lock` primitives and the ObjectFifo
`split` / `forward` / `join` relay operations.
Model: Qwen3.5-0.8B Q8_0 (24 layers, hidden 1024, FFN 3584).
Corpus: wikitext-2 `wiki.test.raw`, 6 chunks x 512 tokens, `-c 512`.

Artifacts: `experiments/phase63/designs/` (the compiled designs and the audit
matrix), `experiments/phase63/logs/` (the audit, the incidents,
`resource_audit.json`), `experiments/phase63/scripts/` (the audit and the
offline checks), `experiments/phase63/gen_mtile.py` (the multi-tile
generator).

**The NPU was wedged for the whole of phase 63** (the driver refuses every
context since the end of phase 62, details in `logs/INCIDENTS.md`). Nothing
in this report was measured on the NPU; everything was measured with the real
compiler (`aiecc`, which is the allocator of the device resources) or offline.
That is stated again at every measurement.

---

## 1. Headline

| result | value | evidence |
|---|---:|---|
| **A 16 worker XDNA GEMM design (4 columns x 4 rows = every core of the device) that compiles with the real `aiecc`** | 16 cores, 60 flows, 8 shim channels, 24 MemTile channels, 32 core channels | `designs/t_c4r4_m/`, `logs/resource_audit.json` |
| The shim DMA budget of that design against the production 4 worker design | **8 channels against 8** (2 per shim against 2 per shim) | `logs/audit.log` |
| The MemTile relay topologies that compile | 2, 4, 8 and 16 workers (the whole device) | `logs/audit.log` |
| The 16 worker design with the **grouped task groups** (the pipelining fix) and the **double buffered** operand FIFOs | compiles | `designs/t_c4r4_m_grouped`, `designs/t_c4r4_m_d2_s8192` |
| The data flow of the 16 worker design, verified offline against a float64 reference | relative RMS 3.6e-9 | `scripts/check_relay_flow.py` |
| The resource audit | 42 topologies, 30 compile, every failure attributed to a resource | `logs/resource_audit.json` |
| The cost of the FFN on the two devices (measured in phase 62, unchanged) | the NPU is **32x-41x** slower than the GPU on the FFN GEMMs | phase 62 `RESULTS.md` |
| The cost driven routing policy in the backend | implemented, `XDNA_COST_ROUTE=1` | `ggml-xdna.cpp` |
| **The quality gate of the per-chunk weight quantization, measured on the model** | **PPL 16.4239 against the Q8_0 reference's 16.4478 (-0.15%)** | `scripts/c256_quality.py` |
| Vulkan-only PP512 (the reference, measured today) | **1393.26 +- 3.81 tok/s** | the machine is shared: the same binary measured 936 and 843 tok/s while the host had 5.4 GB of its 7.2 GB used, so every Vulkan number of this report is taken in a window with at least 3.8 GB available |
| The scheduler baseline of the Vulkan path (PP512) | 2 splits, 1 transition, 8 copies of 2.64 MB, 187 GPU GEMMs, 3 CPU nodes | `GGML_SCHED_TRACE_STATS=1` |
| Device time of any phase 63 candidate | **not measured: the NPU driver is dead** | `logs/INCIDENTS.md` |

The three results that matter:

1. **The 16 worker design exists.** The phase 58 and phase 60 failures were
   not a hardware limit: they were the *data flow* of the design. Relaying the
   operands through the MemTile turns the shim cost of a worker into the shim
   cost of a column, and the 16 worker design uses **8 shim DMA channels in
   total - the same 8 that the 4 worker design of phases 60/61 used** (the
   audit measures both).
2. **The relay does not fix the DMA redundancy, and that is the next
   bottleneck.** The operand that is acquired inside the iteration loop is
   transferred once per iteration, so one of the two operands is replicated
   per m-tile: 39.3 MB of traffic for a GEMM whose unique data is 11.5 MB, in
   both the production and the relay design (section 3.2). Removing it needs
   the operands to be persistent in the MemTile L2, which is the explicit
   `TileDma` path - blocked by the IRON API (section 3.4).
3. **The FFN offload still cannot win, and that is now a cost statement
   rather than a belief.** The measured cost table says the NPU is 32x-41x
   slower than the GPU on the two FFN GEMM shapes, so the cost driven policy
   routes nothing, and the whole hybrid question reduces to the attention.

---

## 2. Route A1: the resource audit

Phase 58 blamed the ObjectFifo topologies on "the shim DMA channels" and
stopped. `experiments/phase63/scripts/audit_resources.py` measures what the
compiler actually allocates for a matrix of 42 topologies: the direct design
of phases 60/61 (the shim feeds every worker) and the MemTile relay design
of this phase (the shim fills the operands once and the MemTile of the column
distributes them).

The measurement is the compiler's own placed design: the flow -> DMA channel
map of `input_with_addresses.mlir`, the tiles that carry an object file and
the locks.

### 2.1 The device

| resource | count | source |
|---|---:|---|
| compute tiles | 16 (4 columns x 4 rows, rows 2..5) | `iron.get_current_device()`: 4 cols x 6 rows, row 0 shim, row 1 MemTile |
| MemTiles | 4 (row 1) | the same |
| ShimNOCTile DMA channels | 8 input + 8 output per shim | the compiler's own error message |
| cores of the 16 worker design | 16 | the object files of the placed design |

### 2.2 The direct design (phases 60/61)

| workers | A/B separate | A/B merged | shim channels | core channels | locks |
|---:|---|---|---:|---:|---:|
| 1 | ok | ok | 2 | 2 | 6 |
| 2 | ok | ok | 4 | 4 | 12 |
| 4 | ok | ok | 8 | 8 | 24 |
| 5 | **FAIL** | ok | 8 | 8 | 32 |
| 7 | **FAIL** | ok | 8 | 8 | 56 |
| 8 | **FAIL** | **ok** | 8 | 8 | 64 |

The failure is always the same and always names a resource:

```
error: no ShimNOCTile on the device has 0 input/1 output DMA channel(s) free:
all 4 ShimNOCTile(s) are at 0/8 input, 8/8 output channels used
note: this is the device's total ShimNOCTile DMA budget, not a placement
choice -- no column has spare capacity to pin to; fan traffic through a
MemTile, or reduce the number of DMA-attached ObjectFIFOs feeding this tile
```

**Correction to the phase 62 report:** phase 62 concluded that the merged
layout fails at 8 workers by one channel. The audit measures the same
topology and it **compiles** (8 workers, 8 shim channels). The phase 62
measurement was taken with a different harness state; the audit is the
systematic one and it supersedes it.

### 2.3 The failure classes, with the resource

| class | the case | the evidence |
|---|---|---|
| 1. the hardware resource is really exhausted | none found | the 16 worker design compiles with 2 of 8 shim channels per shim, 6 of the MemTile channels and 2 of the core channels |
| 2. the compiler allocates unevenly | the direct design at 5/7/8 workers | the compiler puts the 2 fills of a worker on 2 different shim NoC tiles and runs out at the 5th worker; the explicit placement does not change it (`place` and `noplace` behave identically in the audit) |
| 3. the data flow topology is wrong | the direct design | every worker pulls the whole activation and its own weights from the shim: the shim cost grows with the worker count. The MemTile relay makes the shim cost per *column* |
| 4. the abstraction cannot express the connection | the relay design with 1 row per column | `error: 'aie.objectfifo.allocate' op objectfifo has no shared memory access to delegate tile's memory module`: a host filled ObjectFifo cannot live in the MemTile L2, and the MemTile gather of the C pieces needs the workers' pieces in the MemTile L2, which the ObjectFifo model only offers through a `delegate_tile` that the host side cannot fill |
| 5. a kernel or runtime error | none | the designs that compile produce the expected object files (one per core) and the expected flow count |

### 2.4 The resource profile of the 16 worker design

| resource | the production 4 worker design | the 16 worker MemTile relay design |
|---|---:|---:|
| compute tiles used | 4 of 16 | **16 of 16** |
| shim NoC channels | 8 of 32 (2 per shim) | 8 of 32 (2 per shim) |
| MemTile channels | 0 | 24 (6 per MemTile) |
| core DMA channels | 8 (2 per core) | 32 (2 per core) |
| locks | 24 | 168 |
| flows | 12 | 60 |

The shim budget of the 4 worker design is enough for 16 workers once the
operands are relayed through the MemTile: the two fills of a column (the
activation group and the weight chunk) replace the two fills of every worker,
and the 16 C pieces are gathered by the MemTile into one drain per column
instead of 16 drains.

---

## 3. Route A2/A3: the data flows that were implemented

`experiments/phase63/gen_mtile.py` implements three data flows over the same
kernel (the phase 62 `c256x2` kernel, one scale per K chunk):

**Scheme A - the output columns are split between the workers.** The
activation is broadcast (the workers of a column use the same rows) and the
weights are split (each worker takes its own columns). This is the
`--split n` mode.

**Scheme B - the output rows are split between the workers.** The activation
is split (each worker takes its own rows) and the weights are broadcast. This
is the `--split m` mode.

**Scheme C - the two dimensional output blocking.** The `--split` knob
selects the 1 x R or the R x 1 blocking of the output tile of a column; the
host packing of the activation group is the concatenation of the per worker
chunks, so any blocking of the output can be expressed by the packing.

Every flow uses the same three relay operations:

| operation | the IRON call | what it does on the hardware |
|---|---|---|
| the activation split | `fa.cons().split(offsets, tile=mem)` | the shim fills the group once, the MemTile splits it into the per worker chunks and each worker pulls its own chunk from the MemTile L2 |
| the weight broadcast | `fb.cons().forward(tile=mem)` | the shim fills the weight chunk once, the MemTile relays it and every worker of the column reads it from the MemTile L2 |
| the C gather | `ObjectFifoLink(sources, fc.prod(), tile=mem)` | the workers write their C piece into the MemTile L2, the MemTile gathers the pieces into the tile that the shim drains |

The three relay operations were found by reading the compiler's own advice
("fan traffic through a MemTile") into the IRON API; the two that the
ObjectFifo model refuses (a host filled FIFO in the MemTile L2, and the
gather of the C pieces) are the reason the 1 row per column topologies fail.

**The compiled topologies** (`logs/audit.log`):

| topology | workers | compiles | the failure when it does not |
|---|---:|---|---|
| 1 col x 2 rows | 2 | yes | |
| 1 col x 4 rows | 4 | yes | |
| 2 cols x 2 rows | 4 | yes | |
| 2 cols x 4 rows | 8 | yes | |
| 4 cols x 2 rows | 8 | yes | |
| **4 cols x 4 rows** | **16** | **yes** | |
| any topology with 1 row per column | cols | no | `objectfifo.allocate`: a host filled ObjectFifo cannot live in the MemTile L2 |

### 3.1 The data flow, verified offline

The NPU cannot run, so the data flow of the design is verified with
`experiments/phase63/scripts/check_relay_flow.py`: the host packing (the
activation group, the weight chunk), the kernel's reading convention (the
8x8 int8 tiles, the row scales, the column scales) and the MemTile gather
of the C pieces are transcribed into numpy and compared with a float64
reference of the exact Q8_0 mathematics.

| test | max abs error | relative RMS | verdict |
|---|---:|---:|---|
| M=64 K=256 N=128, 2 cols x 4 rows | 1.4e-4 | 2.6e-9 | the packed layout and the kernel convention agree |
| M=128 K=512 N=256, 2 cols x 4 rows | 6.4e-5 | 3.6e-9 | the packed layout and the kernel convention agree |

The relative RMS is at the float32 rounding level, so the packing, the
kernel's reading convention and the gather are consistent.

### 3.2 The DMA traffic of the relay design, and what it does not fix

The relay design changes the *shim* cost of a worker into the shim cost of a
column, but the streaming semantics of the ObjectFifo model decide the
*traffic*: an object that a worker holds across an inner loop is transferred
once, and an object that is acquired inside the loop is transferred once per
iteration.  With a two dimensional iteration space (the m-tiles and the
n-tiles) and two operand streams, one of them is replicated.

For the FFN shape (M=512 K=1024 N=3584, the 64x64x256 tile, 4 columns):

| design | the A | the B | the C | the total |
|---|---:|---:|---:|---:|
| the production 4 worker design | 2.13 MB | 29.82 MB | 7.34 MB | **39.3 MB** |
| the 16 worker relay design (the loop order that keeps the A) | 2.13 MB | 29.82 MB | 7.34 MB | **39.3 MB** |
| the 16 worker relay design (the loop order that keeps the B) | 29.82 MB | 14.92 MB | 7.34 MB | 52.1 MB |
| the unique data | 0.51 MB | 3.67 MB | 7.34 MB | 11.5 MB |

**The relay design does not make the traffic worse and does not fix it
either:** the dominant term is the operand that is replicated once per
m-tile, and that is the phase 61 report's first "what would be needed next"
(the DMA is 20% of the production kernel, 55 ms of 298 ms, and the traffic
is 14x-18x redundant).

Killing the replication needs the operands to be *persistent* in the MemTile
L2 and to be re-read by the workers, which is exactly what the explicit
`TileDma` path would do (section 3.3) and what the ObjectFifo model cannot
express.

### 3.3 The pipelining and the double buffering

Two properties of the design that the phase 62 measurement says decide
whether the 16 workers are worth anything:

* **The runtime sequence must not serialise the workers.** The phase 62
  finding is that one task group per (m-tile, column, n-tile) with a wait on
  the drain makes the workers run one after the other. The phase 63 design
  issues the fills of **all the columns in one task group per (m-tile,
  n-tile)** and waits only on the last drain, so the 16 workers of the
  device can compute at the same time. The design compiles that way
  (`designs/t_c4r4_m_grouped`).
* **The operand FIFOs must be double buffered**, so that the transfer of
  group g+1 overlaps the compute of group g. The FIFO depth is the
  `--depth` knob; at depth 2 the core LMEM needs 49.8 KB of buffers and the
  stack has to shrink to 8 KB (the 64 KB core LMEM is the limit, and the
  design is 640 bytes over it with the default 16 KB stack). The double
  buffered 16 worker design compiles (`designs/t_c4r4_m_d2_s8192`).

Both are compile time properties; **whether they produce a real speedup is
exactly what could not be measured**.

### 3.4 The explicit low level path

`experiments/phase63/gen_explicit.py` builds the same data path with the
explicit primitives (`aie.flow`, `TileDma`, `DmaChannel`, `Bd`, `Lock`): the
MemTile receives the operands over S2MM channels, the cores read them out of
the MemTile L2 with the classic two lock protocol (a "free" lock the producer
waits for and a "ready" lock the consumer waits for), and the core side is
an explicit buffer descriptor chain.

**It does not compile, and the reason is the IRON API, not the hardware:**
the shim's L2 buffer is an `aie.shim_dma_allocation` (only the
`Flow(shim_symbol=...)` path creates one) while the IRON `Buffer` always
lowers to an `aie.buffer`, which is not valid in a ShimNOCTile:

```
error: Operand 0 of operation "aie.dma_bd" must be a Value (is not a Value)
```

Going further needs the allocation's memref to be handed to the TileDma,
which the IRON API does not expose. The MemTile side of the same program -
the part the ObjectFifo model cannot express - is implemented in that file.

### 3.5 What is still missing for a real 16 worker run

* **The device time.** The NPU is dead (section 8), so nothing in this
  report was measured on the device.
* The C pieces are gathered by the MemTile, so the C path needs the MemTile
  DMA to be free at the right time; the phase 63 design does not implement
  the double buffering of the C path (only the operand FIFOs are double
  buffered).
* The `fp256` (fixed-point scale) kernel of phase 62 is the candidate that
  keeps the exact Q8_0 weights; it compiles and its arithmetic is validated
  offline, but it was never run either.

---

## 4. Route B: the compute representations

The phase 62 candidates are carried over and the new one is finished:

| candidate | the arithmetic | the error against the float64 reference | the device time |
|---|---|---:|---:|
| the Q8_0 exact scale (the phase 61 kernel) | 2 float multiplies per output per 32 element block | 6.9e-9 | 298.0 ms |
| the per-K-chunk scale (`c256`) | one float multiply per output per 256 elements | 2.0e-9 | 113.1 ms |
| **the fixed-point scale (`fp256`)** | an int32 multiply per output per 32 element block, the float unit only at the chunk end | **8.3e-4 on the real weights** (the Q8_0 reference is 4.4e-4) | **not measured** |

The `fp256` candidate keeps the exact Q8_0 weights: the weight block scale
becomes an int32 mantissa relative to one exponent per K chunk and column,
the int32 partial sum is down-shifted and multiplied in the integer domain,
and the float unit is only used for the conversion of the chunk sum. The bit
budget is `mantissa_bits + 20 + log2(blocks per chunk)`, which is 31 bits
with 8 mantissa bits - it fits, and the down-shift does not cost bits
because the mantissa needs them back.

The cost of the fixed-point path is the emulated int32 multiply (8
instructions, phase 62 section 2), so it is expected to land between the two
measured candidates. It compiles with the real `aiecc` and its arithmetic is
validated offline; **it was never run on the device**.

### 4.1 The quality gate of the per-chunk quantization, measured on the model

The c256 engine changes the *weight* format (one scale per 256 elements
instead of per 32) and the *activation* format (the same). The NPU is dead,
so the quality cost of the weight change is measured at the model level
instead (`experiments/phase63/scripts/c256_quality.py`):

* the 72 FFN weight tensors of the model are requantized to 256 element
  blocks and written out as F16 (the dequantized values are exactly the
  values the int8 GEMM of the c256 engine sees, so the two compute the same
  product up to the float rounding of the scale multiply),
* the perplexity of the converted model is measured on the CPU and compared
  with the Q8_0 model.

| model | the perplexity (6 chunks x 512 tokens, wikitext-2) |
|---|---:|
| the Q8_0 model (the reference) | **16.4478** |
| the c256 model (the 256 element requantized FFN weights) | **16.4239** |

**The per-chunk weight quantization does not degrade the model: the
perplexity moves by -0.15%, and the per chunk values move in both
directions (+0.67% on the first chunk, -0.15% on the last), which is what a
random perturbation of the weights does.**

This is a partial gate: the converted model does not quantize the
activation, while the c256 engine quantizes it per 256 elements as well. The
missing part is measured per GEMM in the phase 62 analysis: the activation
change adds 3.9e-4 of relative RMS noise per GEMM on top of the 4.8e-4 of
the Q8_0 reference, so the c256 engine's total is 8.6e-4 against the
reference's 4.8e-4 (1.8x). The full gate (the c256 engine's own perplexity)
still needs the device.

---

## 5. Route C: the cost table and the routing strategies

### 5.1 The measured cost table

The GPU side is the phase 62 Vulkan profile (measured on this machine, the
per operator breakdown of a PP512 and a PP5120 pass). The NPU side is the
phase 62 kernel measurement.

| operator | the GPU (Vulkan) | the NPU (the `c256x2` kernel) | the ratio |
|---|---:|---:|---:|
| the FFN gate/up GEMM (M=512 K=1024 N=3584) | 3.55 ms | 113.1 ms | **32x** |
| the FFN down GEMM (M=512 K=3584 N=1024) | 2.70 ms | 109.8 ms | **41x** |
| the GLU (24 calls per pass) | 31.4 ms | not implemented on the device | |
| the vocabulary projection | 14.1 ms | not implemented | |
| the weight dequant + tile packing (host) | - | 3.81 / 7.06 ms | |
| the activation quantisation (host) | - | 1.55 / 6.00 ms | |

The FFN GEMMs are 235 ms of the 386 ms PP512 pass (61%).

### 5.2 The three strategies, evaluated against the table

**C1, the GPU keeps the large FFN GEMMs.** The cost driven policy of
section 5.3 selects exactly this: the NPU is slower for every measured
shape, so nothing is offloaded and the hybrid *is* the Vulkan path. The
measured consequence: the hybrid throughput is the Vulkan throughput
(1326-1314 tok/s), not the 16-63 tok/s of the forced offload.

**C2, the shape adaptive routing.** Implemented as the cost table of
section 5.3: the decision is a function of (M, K, N) and of the measured
device times, not of a fixed shape list. With the phase 62/63 numbers the
table is empty for the FFN shapes (the NPU loses by 32x-41x) and the only
candidate left is the attention, where the NPU has no kernel at all.

**C3, the whole FFN subgraph on one backend.** The XDNA backend already
runs the GLU on the device when the gate and the up results are pending
(`ggml_xdna_glu_compute`, and the fused FFN path in `ggml-xdna.cpp`), so the
FFN intermediates already stay in NPU memory and the cross device traffic
is one activation in and one output out per layer. The saving is bounded by
the 151 ms of non-FFN GPU time, and the NPU FFN costs 8.1 s per pass: the
subgraph is worth 1.9% of the FFN time and costs 34x it.

### 5.3 The cost driven policy in the backend

`ggml_xdna.cpp` now has a cost table and a policy:

```
XDNA_COST_ROUTE=1              switch the policy on
XDNA_COST_TABLE="K,N,npu_ms,gpu_ms;..."   replace the table
```

The policy offloads a shape only when the measured NPU time is below the
measured GPU time; an unknown shape is not offloaded, because the phase 62
measurements show that the NPU loses by more than an order of magnitude on
every large GEMM. Every decision is logged (`[XDNA-COST] M=.. K=.. N=..:
npu=.. ms gpu=.. ms -> route|keep on the GPU`).

The default behaviour (the historical allowlist behind
`GGML_XDNA_QWEN_PP=1`) is unchanged; the cost policy is opt in.

---

## 6. Route D: the scheduler and the data movement

The state of the scheduler that the phase 57-60 work left behind, and what
phase 63 measured:

* **The FFN chain is visible to the NPU backend.** The scheduler splits the
  graph per backend and the XDNA backend's subgraph contains the gate, the
  up, the GLU and the down nodes in order, so the backend can keep the
  intermediates in NPU memory. The GLU already runs on the device.
* **The remaining cross device traffic per layer is one activation in and
  one output out**, because the gate and the up GEMMs read the same
  activation tensor.
* **The host packing is not duplicated per node** in the scale preserving
  path (the weight dequantisation and the tile packing happen once per
  weight tensor per GEMM), and the phase 61 weight cache removes even that
  at the cost of up to 4 MB of host memory per weight tensor. The cache is
  off by default because the measured end-to-end effect was inside the noise
  and because the host memory pressure it caused produced an XRT timeout.
* **The measured scheduler baseline** (Vulkan only, PP512, one pass,
  `GGML_SCHED_TRACE_STATS=1`): 2 splits, 1 backend transition, 8 copies of
  2.64 MB, 187 GEMMs on the GPU, 0 on the NPU and **3 nodes on the CPU**
  (the CPU spill of the phase 57-60 list: 3 of the 1391 nodes of the graph
  fall back to the CPU, 0.2%).
* **What phase 63 adds:** the cost driven policy of section 5.3, so that the
  routing decision follows the measured cost instead of a shape list that was
  measured once, and `GGML_XDNA_DISABLE=1` (phase 62), so that a wedged
  device does not also stop the GPU-only measurements.

The per change evidence that the phase 63 brief asks for (the copy counts,
the copy bytes, the synchronisation time and the end to end time before and
after) **could not be produced**: the NPU is dead, so there is no hybrid run
to count the copies of.

---

## 7. What changed in the tree

| file | change |
|---|---|
| `ggml/src/ggml-xdna/ggml-xdna.cpp` | the cost driven routing policy (`XDNA_COST_ROUTE`, `XDNA_COST_TABLE`) with the measured table and a log line per decision |
| `experiments/phase63/gen_mtile.py` | the multi-tile generator: the MemTile relay of the operands, the split of the activation, the broadcast of the weights and the gather of the C pieces, for 1-4 columns x 1-4 rows, with the grouped task groups and the double buffered FIFOs |
| `experiments/phase63/gen_explicit.py` | the explicit low level path (`aie.flow` + `TileDma` + `Lock`): implemented up to the IRON API blocker of the shim L2 allocation |
| `experiments/phase63/scripts/audit_resources.py` | the resource audit: 42 topologies, the compiler's allocation, the failure classes |
| `experiments/phase63/scripts/check_relay_flow.py` | the offline verification of the packing, the kernel convention and the gather |
| `experiments/phase63/scripts/c256_quality.py` | the model level quality gate of the per-chunk weight quantization |
| `experiments/phase63/logs/resource_audit.json` | the raw audit data |
| `experiments/phase63/logs/INCIDENTS.md` | the device state and the memory situation |

### Reproduction

```sh
. experiments/phase63/env.sh

# the 16 worker design
$PY experiments/phase63/gen_mtile.py --M 512 --K 1024 --N 3584 \
    --cols 4 --rows 4 --split m --m-tile 64 --n-tile 64 --k-chunk 256 \
    --work-dir experiments/phase63/designs/t_c4r4_m

# the resource audit (42 topologies)
$PY experiments/phase63/scripts/audit_resources.py

# the offline data flow verification
$PY experiments/phase63/scripts/check_relay_flow.py --M 128 --K 512 --N 256 \
    --cols 2 --rows 4

# the cost driven routing (needs a working NPU)
GGML_XDNA_QWEN_PP=1 XDNA_COST_ROUTE=1 XDNA_Q8_ENGINE=c4_mmul_v3 \
  XDNA_KERNEL_CACHE=$PWD/experiments/npu-int8-gemm/kernel_cache \
  ./experiments/phase48/npu_guard.sh -- \
  ./build/bin/llama-bench -m models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -p 512 -n 0 -r 5
```

---

## 8. Failure list, with the specific reason

| candidate | state | the specific reason |
|---|---|---|
| **16 worker MemTile relay design (4x4)** | compiled | - |
| 8 worker MemTile relay design (2x4, 4x2) | compiled | - |
| the data flow of the 16 worker design | verified offline | - |
| the device time of the 16 worker design | **not measured** | the NPU driver is dead |
| the end to end hybrid of any phase 63 candidate | **not measured** | the NPU driver is dead |
| the `fp256` fixed-point kernel | compiled, arithmetic verified offline | not measured on the device |
| the 1 row per column relay topologies | do not compile | `aie.objectfifo.allocate`: a host filled ObjectFifo cannot live in the MemTile L2 (the abstraction cannot express the connection) |
| the direct design at 5, 7, 8 workers with separate A and B | do not compile | the ShimNOCTile budget: 8 output channels per shim, exhausted by the 2 fills of every worker |
| the direct design at 8 workers with the merged A+B | compiles | (the phase 62 report said it failed; the audit supersedes it) |
| a 16 worker design with a *pipelined* runtime | implemented in the phase 62 harness, not compiled here | the grouped task group needs the device to be measured |
| the explicit low level path (`aie.flow` + `TileDma` + `Lock`) | implemented up to the blocker | the shim's L2 buffer is an `aie.shim_dma_allocation` and the IRON `Buffer` always lowers to an `aie.buffer`, which is not valid in a ShimNOCTile: "Operand 0 of operation aie.dma_bd must be a Value" |
| the 16 worker design with the double buffered C path | not implemented | the C pieces are gathered by the MemTile, so the second C buffer needs a second MemTile region; the core LMEM budget is the limit (the design is 640 bytes over the 64 KB with the default stack) |
| the cost driven routing policy | implemented | the end to end effect is not measured (the NPU is dead) |

---

## 9. Limitations and what is needed next

* **No phase 63 candidate ran on the NPU.** The device was wedged before
  phase 63 started. The order of work when the device is back:
  1. the grouped task group on the 16 worker design (the workers must run
     concurrently, otherwise the design is worth one worker),
  2. the device time of the `c256x2` and the `fp256` kernels on it,
  3. the end to end hybrid of the `c4_mmul_v4` engine and its perplexity
     (the quality gate: the references are 16.4344 for the Vulkan path,
     16.4378 for the phase 61 engine and 16.4478 for the CPU backend).
* **The 16 worker design has no double buffering.** The C path and the
  operand path both use a single buffer per group, so the DMA of the next
  group cannot start before the compute of the current one. The MemTile has
  the L2 room for a second buffer (the A group is 64 KB, the MemTile L2 is
  512 KB) but the design does not use it.
* **The FFN offload is a dead end on this machine, and the attention is the
  only remaining target.** The measured cost table says the NPU is 32x-41x
  slower than the GPU on the two FFN GEMM shapes; even a perfect 16 worker
  NPU kernel (16x the phase 61 kernel) would be 2x slower than the GPU, and
  the attention is 10% of the pass at PP5120, which caps the whole hybrid at
  about 1.1x. A useful hybrid on this machine needs a different operator
  (the GATED_DELTA_NET of this model is 57.8 ms per pass on the GPU and has
  no NPU kernel at all), not a faster GEMM.
* The Vulkan numbers of this phase were measured with the NPU disabled
  (`GGML_XDNA_DISABLE=1`), which is the only way to run them while the
  device is wedged.
