# Phase 62: AIE2 op rates, re-measured

Harness: `experiments/phase62/kbench/rate62.py` (4096 iterations, median of 5 runs,
one worker on one column, 16-lane vectors).

| kernel | op | ns per 16-lane chain-iteration (1 chain) | (4 chains) |
|---|---|---:|---:|
| i1 | int32 mul + add (operands in registers) | 49.3 | 9.8 |
| i2 | int32 mul + add, 4 chains | - | 9.8 |
| i3 | int32 downshift + mul + add | 57.5 | - |
| i4 | int32 downshift + add | 60.9 | - |
| f1 | float mul + add (operands loaded from LMEM each iteration) | 79.3 | 19.8 |
| f2 | float mul + add, 4 chains | - | 19.8 |
| f3 | int32 -> float conversion (operands from LMEM) | 45.8 | 15.1 |
| f4 | int32 -> float conversion, 4 chains | - | 15.1 |
| f5 | two dependent float multiplies (operands in registers) | 41.2 | - |

The phase 61 numbers (k7 79.5 ns, k8 20.5 ns, k9 46.7 ns) reproduce.

## What the numbers say

1. **The float matrix unit retires one 16-lane op every ~22 cycles** (19.8 ns
   per chain-iteration with 4 chains, 2 instructions each). The integer
   vector path retires one instruction every ~2.4 cycles. **A float op costs
   about 4x an integer instruction.**
2. `aie::mul` on int32 is **software emulated**: the disassembly of i1 shows
   4 `vshuffle` + 1 `vmul` + 3 `vmac` (the 16x16 element matrix ops) for one
   16-lane int32 multiply. There is no native 32-bit lane multiply in the
   AIE2 vector unit, so the fixed-point route pays 8 instructions per scale
   multiply.
3. **An LMEM load costs ~19 ns of latency** (f1 - f5: the same float multiply
   is 41 ns with the operands in registers and 79 ns with them loaded from
   LMEM). The phase 61 kernel issues 186 `vlda` per function body and 80
   `vst` (spills), so the LMEM traffic, not the arithmetic, is a large part of
   its cost.
4. The int32 -> float conversion (45.8 ns latency, 15.1 ns at 4 chains) is
   pure overhead in the Q8_0 kernel: it exists only to feed the float scale
   multiply.
