#!/usr/bin/env python3
"""Phase 22: fused GEMM+ReLU correctness sweep on the whole-array XDNA kernel.

For each K in the list, generate a fused (relu) whole-array int8 GEMM at the
Qwen ffn_gate-like shape and compare against the CPU reference. A run that does
not return within --timeout is recorded as HANG (the kernel / NPU firmware can
stall on some (K, N) combinations even though the same xclbin ran earlier).

Writes results/phase22_fused.csv with columns:
  M,K,N,activation,status,mismatches,run_ms,note

Usage:
  python fused_sweep.py --K 1024,2048,4096,8192 --N 3584,4096
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "..", "phase21", "gen_gemm.py")
CHECK = os.path.join(HERE, "..", "phase21", "fused_check.py")
PY = sys.executable


def run(cmd, timeout):
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return cp.returncode, cp.stdout, cp.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", default="1024,2048,4096,8192")
    ap.add_argument("--N", default="3584,4096")
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--workroot", default="/tmp/p22_fused")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "phase22_fused.csv"))
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    for K in [int(x) for x in a.K.split(",")]:
        for N in [int(x) for x in a.N.split(",")]:
            wd = os.path.join(a.workroot, f"K{K}_N{N}_{a.activation}")
            xc = os.path.join(wd, "design.xclbin")
            ins = os.path.join(wd, "design.insts.bin")
            rc, out, err = run([PY, GEN, "--M", str(a.M), "--K", str(K), "--N", str(N),
                                "-m", "64", "-k", "64", "-n", "32", "--n-aie-cols", "4",
                                "--dtype-in", "i8", "--dtype-out", "i32",
                                "--activation", a.activation, "--work-dir", wd],
                               timeout=max(120.0, a.timeout))
            if rc != 0:
                rows.append([a.M, K, N, a.activation, "COMPILE_FAIL", "", "", err[-120:]])
                continue
            rc, out, err = run([PY, CHECK, xc, ins, str(a.M), str(K), str(N), "--relu"],
                               timeout=a.timeout)
            if rc == 124:
                rows.append([a.M, K, N, a.activation, "HANG", "", "", "no return in %gs" % a.timeout])
            elif rc == 0:
                mism = out.split("mismatches=")[1].split("/")[0] if "mismatches=" in out else ""
                ms = out.split("ms=")[1].split()[0] if "ms=" in out else ""
                rows.append([a.M, K, N, a.activation, "PASS", mism, ms, ""])
            else:
                mism = out.split("mismatches=")[1].split("/")[0] if "mismatches=" in out else ""
                rows.append([a.M, K, N, a.activation, "FAIL", mism, "", out[-120:]])
            print(rows[-1], flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "K", "N", "activation", "status", "mismatches", "run_ms", "note"])
        w.writerows(rows)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
