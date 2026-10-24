#!/usr/bin/env python3
"""Phase 61: summarize the llama-bench runs of the phase 61 logs."""
import glob
import os
import re
import statistics
import sys

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

rows = []
for f in sorted(glob.glob(os.path.join(LOGS, "*.txt"))):
    base = os.path.basename(f)[:-4]
    if base.startswith(("ppl_", "dbg", "mem", "health", "c4v3_gp", "c4v3_prof", "c4v3_hp", "c4v3_wc")):
        continue
    txt = open(f, errors="ignore").read()
    for line in txt.splitlines():
        if "qwen35" not in line or "pp" not in line:
            continue
        m = re.search(r"pp(\d+)\s*\|\s*([\d.]+)\s*.\s*([\d.]+)\s*\|", line)
        if m:
            rows.append((base, int(m.group(1)), float(m.group(2)), float(m.group(3))))

print("%-28s %6s %10s %8s" % ("run", "pp", "tok/s", "sd"))
for base, pp, ts, sd in rows:
    print("%-28s %6d %10.2f %8.2f" % (base, pp, ts, sd))

# group by (pp, configuration keyword)
groups = {}
for base, pp, ts, sd in rows:
    key = (pp, re.sub(r"_r\d+$", "", base))
    groups.setdefault(key, []).append(ts)
print()
print("%6s  %-34s %8s %8s %8s" % ("pp", "configuration", "median", "n", "spread"))
for (pp, cfg), vals in sorted(groups.items()):
    print("%6d  %-34s %8.2f %8d %8.2f" % (pp, cfg, statistics.median(vals), len(vals),
                                            max(vals) - min(vals)))
