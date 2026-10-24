#!/usr/bin/env python3
"""Phase 50: run the precompiled TRACE variant and dump AIE event trace.

Uses the audited phase40 trace flow (HostRuntime + TraceConfig).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
from gen_q8_c4_mmul_v2 import pack_A, pack_B, M_TILE, N_TILE, KC
from bench_q8_c4_mmul_v2 import run_kernel_trace

M, K, N = 512, 1024, 3584
TRACE_SIZE = 8192
work = "/tmp/p50_trace"
xp = os.path.join(work, "design.xclbin")
ip = os.path.join(work, "design.insts.bin")

rng = np.random.default_rng(42)
nb = K // 32
A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
As = rng.uniform(0.001, 0.02, size=(M, nb)).astype(np.float32)
Bs = rng.uniform(0.001, 0.02, size=(nb, N)).astype(np.float32)
Ap = pack_A(A_i8, As, M, K, M_TILE, KC)
Bp = pack_B(B_i8, Bs, K, N, N_TILE, KC)

t0 = time.perf_counter()
C, ms = run_kernel_trace(xp, ip, Ap, Bp, M, K, N, TRACE_SIZE,
                         os.path.join(work, "trace.txt"),
                         os.path.join(work, "trace.json"))
print(f"[trace] wall_ms={ms:.1f} C00={C.ravel()[0]:.4f}")
print(f"[trace] files: {os.path.getsize(os.path.join(work,'trace.txt'))}B txt, "
      + (f"{os.path.getsize(os.path.join(work,'trace.json'))}B json"
         if os.path.exists(os.path.join(work, 'trace.json')) else "no json"))
