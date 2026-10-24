# Phase 50 instruction-count evidence (AIE core ELF disassembly, block loop)

Method: `llvm-objdump -d` on `elfs_main_core_0_2.elf`, block loop identified
by hardware-loop registers (`movxm ls/le`). Counts are per (bb,i,j) block
(32 outputs). `spills` = vector loads/stores with `[sp,...]` (local-memory
stack). All builds M512 K1024 N3584 1x4 unless noted.

| variant | bundles | spills | vmac.f | vmov | vconv | vmsc | note |
|---|---|---|---|---|---|---|---|
| prod P32 (phase46 build) | 206 | 31 | 48 | 53 | 15 | 10 | baseline, 433 ms |
| s1 (row-wise, 1x to_float) | 245 | 110 | 64 | 90 | 27 | 18 | worse |
| s3 (2x to_float 16-wide) | 251 | 129 | 64 | 91 | 27 | 18 | worse (double 32-lane machinery) |
| s4 (manual filter conv) | 301 | 159 | 80 | 104 | 33 | 22 | worse |
| s6 (scalar-scale, normal install) | 206 | 31 | 48 | 53 | 15 | 10 | identical to P32 (compiler canonicalizes) |
| s7c (manual accum bias conv) | 490 | 145 | 80 | 144 | 54 | 36 | WRONG (max_err 4.8e9) + slow (977 ms) |
| s8 (pre-broadcast + prod C) | 225 | 84 | 48 | 90 | 27 | 18 | worse |
| s9 (store/load bias conv) | 354 | 227 | 80 | 146 | 51 | 34 | worse (aliasing) |
| s10 (full redesign, contig C) | 152 | 31 | 32 | 31 | 15 | 10 | 26% fewer; undrainable (4D tap) |
| s11 (s10 scale + prod C) | 225 | 84 | 48 | 90 | 27 | 18 | correct but 505 ms (slower) |
| s12 (= s10 kernel, linear drain) | 152 | 31 | 32 | 31 | 15 | 10 | 359 ms device, CORRECT |
| accfloat (to_float on accum) | 250 | 38 | 48 | 77 | 27 | 18 | worse |
| d0/mmul_only (MAC + int sink) | 6* | 0 | - | - | - | - | *kk loop only; 57.8 ms total |
| d3 (= s1-style full) | 238 | 132 | 64 | 92 | 27 | 18 | worse |

Key inferences (all cross-checked against the AIE trace: MAC 7 cycles,
float 214 cycles per block):
- `aie::to_float` int32->fp32 lowers to ~25 bf16-magic bundles per 32 values
  (vconv.bf16.fp32 + vmsc.f + vadd/vups). It is the single biggest chunk.
- fp32 8-lane ops lower to bf16-pair `vmac.f` at ~2 cycles each (48 vmac.f
  for 24 fp32 ops in P32; 32 in s10/s12 for 16 fp32 ops). Hardware rate.
- The `insert<8>`-based 32-wide scale construction costs ~20 bundles in P32;
  s10/s12 16-lane scale path removes it (vmac.f 48 -> 32).
- Remaining vmov/spills (~58 in s12) are bank-pressure fallout; every attempt
  to cut live values further increased them (compiler canonicalization +
  bf16 bank constraints).
- `-O3` gives byte-identical output to `-O2` for this kernel.
