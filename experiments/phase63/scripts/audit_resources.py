#!/usr/bin/env python3
"""Phase 63: the resource audit of the XDNA topologies.

Phase 58 and phase 60 blamed the 8 and 16 worker designs on "the shim DMA
channels" and stopped there.  This script measures what the compiler actually
allocates for a matrix of topologies, so that every failure can be attributed
to a concrete resource instead of an error message.

Two families of topologies are audited:

  * "direct": the phase 60/61 design, the shim feeds every worker directly
    (experiments/phase62/kbench/kbench62.py),
  * "relay": the phase 63 design, the shim fills the operands once and the
    MemTile of the column broadcasts them to the workers
    (experiments/phase63/gen_mtile.py).

For every topology the script records
  * whether the design compiles,
  * the compiler's error (if any),
  * the flow -> DMA channel map of the placed design (the compiler's own
    allocation),
  * the number of shim-attached DMA endpoints (the ones that cost the
    ShimNOCTile channels),
  * the tiles that carry an object file (the workers that really exist).

usage: audit_resources.py [--work-dir DIR]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def analyze(work_dir):
    """Read the placed design and count the resources the compiler used."""
    out = {"flows": [], "cores": [], "shim_channels": {}, "mem_channels": {},
           "core_channels": {}, "locks": 0}
    placed = None
    for name in ("input_with_addresses.mlir", "aie.mlir"):
        p = os.path.join(work_dir, "design.prj", name)
        if os.path.exists(p):
            placed = p
            break
    if placed is None:
        return out
    txt = open(placed, errors="ignore").read()
    flows = re.findall(r"aie\.flow\(([^)]*)\)", txt)
    out["flows"] = flows
    for f in flows:
        m = re.match(r"\s*(%\S+), DMA : (\d+), (%\S+), DMA : (\d+)", f)
        if not m:
            continue
        s, sc, d, dc = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        for tile, ch in ((s, sc), (d, dc)):
            if "shim_noc" in tile:
                out["shim_channels"].setdefault(tile, set()).add(ch)
            elif "mem_tile" in tile:
                out["mem_channels"].setdefault(tile, set()).add(ch)
            elif "tile_" in tile and "shim" not in tile and "mem" not in tile:
                out["core_channels"].setdefault(tile, set()).add(ch)
    for k in ("shim_channels", "mem_channels", "core_channels"):
        out[k] = {kk: sorted(v) for kk, v in out[k].items()}
    out["cores"] = sorted(set(re.findall(r"%tile_(\d+)_(\d+)", txt)))
    out["locks"] = len(re.findall(r"aie\.lock", txt))
    return out


def run(cmd, work_dir):
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    dt = round(time.time() - t0, 2)
    # the harness continues to the run after a successful compile, and the
    # run fails when the device is down: the compile result is what matters
    out = p.stdout + p.stderr
    m = re.search(r"compile (ok|FAILED)", out)
    ok = bool(m) and m.group(1) == "ok"
    err = ""
    if not ok:
        m2 = re.search(r"(error:.*?)(?:\n\n|\Z)", out, re.S)
        err = (m2.group(1) if m2 else out[-1200:]).strip()[:1200]
    res = {"compile_ok": ok, "compile_s": dt, "error": err}
    res.update(analyze(work_dir))
    res["n_flows"] = len(res["flows"])
    res["n_shim_ch"] = sum(len(v) for v in res["shim_channels"].values())
    res["n_mem_ch"] = sum(len(v) for v in res["mem_channels"].values())
    res["n_core_ch"] = sum(len(v) for v in res["core_channels"].values())
    res["n_cores"] = len(res["cores"])
    res.pop("flows", None)
    return res


def direct_cmd(kbench, work_dir, wcols, layout, place):
    cmd = [sys.executable, kbench,
           "--M", "512", "--K", "1024", "--N", "3584",
           "--wcols", str(wcols), "--m-tile", "64", "--n-tile", "64",
           "--k-chunk", "256", "--variant", "c256", "--layout", layout,
           "--repeat", "1", "--work-dir", work_dir, "--recompile",
           "--tag", f"direct_c{wcols}_{layout}"]
    if not place:
        cmd.append("--no-place-cols")
    return cmd


def relay_cmd(gen, work_dir, cols, rows, split, merge):
    cmd = [sys.executable, gen,
           "--M", "512", "--K", "1024", "--N", "3584",
           "--cols", str(cols), "--rows", str(rows), "--split", split,
           "--m-tile", "64", "--n-tile", "64", "--k-chunk", "256",
           "--work-dir", work_dir, "--tag", f"relay_c{cols}r{rows}_{split}"]
    if merge:
        cmd.append("--merge")
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=os.path.join(HERE, "..", "designs", "audit"))
    ap.add_argument("--kbench", default=os.path.join(ROOT, "experiments", "phase62",
                                                     "kbench", "kbench62.py"))
    ap.add_argument("--gen", default=os.path.join(ROOT, "experiments", "phase63",
                                                  "gen_mtile.py"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "logs", "resource_audit.json"))
    a = ap.parse_args()
    os.makedirs(a.work_dir, exist_ok=True)

    results = []
    cases = []
    # the direct (phase 60/61) topologies
    for wcols in (1, 2, 4, 5, 7, 8):
        for layout in ("sep", "mrg"):
            for place in (True, False):
                cases.append(("direct", wcols, layout, place, False))
    # the MemTile relay topologies (phase 63)
    for cols in (1, 2, 4):
        for rows in (1, 2, 4):
            for split in ("m", "n"):
                cases.append(("relay", cols, rows, split, False))

    for fam, x, y, z, merge in cases:
        if fam == "direct":
            tag = f"{fam}_c{x}_{y}_{'place' if z else 'noplace'}"
            cmd = direct_cmd(a.kbench, os.path.join(a.work_dir, tag), x, y, z)
            desc = f"direct  wcols={x} {y} {'place' if z else 'noplace'}"
        else:
            tag = f"{fam}_c{x}r{y}_{z}"
            cmd = relay_cmd(a.gen, os.path.join(a.work_dir, tag), x, y, z, merge)
            desc = f"relay   {x} cols x {y} rows split {z}"
        r = run(cmd, os.path.join(a.work_dir, tag))
        r.update({"family": fam, "tag": tag, "desc": desc})
        results.append(r)
        print(f"{desc:38s} {'ok  ' if r['compile_ok'] else 'FAIL'} "
              f"cores={r['n_cores']:2d} flows={r['n_flows']:3d} "
              f"shim_ch={r['n_shim_ch']:3d} mem_ch={r['n_mem_ch']:2d} "
              f"core_ch={r['n_core_ch']:3d} locks={r['locks']:3d} "
              f"({r['compile_s']}s)")
        if not r["compile_ok"]:
            print("        " + r["error"].splitlines()[0][:150])
        sys.stdout.flush()

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
