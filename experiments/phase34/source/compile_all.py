#!/usr/bin/env python3
"""Phase 34: compile all tile counts for scaling sweep."""

import os
import subprocess
import sys

M = 512
K = 2048
N = 2048
WORKDIR = "/tmp/p34_sweep"

_TILE_COUNT_TO_ROWS_COLS = {
    1:  (1, 1),
    2:  (1, 2),
    4:  (1, 4),
    8:  (2, 4),
    16: (4, 4),
}
_TILE_COUNTS = [1, 2, 4, 8, 16]

os.makedirs(WORKDIR, exist_ok=True)

for tiles in _TILE_COUNTS:
    outdir = os.path.join(WORKDIR, f"t{tiles}")
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "gen_phase21.py"),
        "--M", str(M), "--K", str(K), "--N", str(N),
        "-m", "64", "-k", "64", "-n", "64",
        "--n-aie-rows", str(_TILE_COUNT_TO_ROWS_COLS[tiles][0]),
        "--n-aie-cols", str(_TILE_COUNT_TO_ROWS_COLS[tiles][1]),
        "--work-dir", outdir,
    ]
    print(f"\n=== Compiling tiles={tiles} ===")
    subprocess.run(cmd, check=True)

print("\nAll kernels compiled.")
