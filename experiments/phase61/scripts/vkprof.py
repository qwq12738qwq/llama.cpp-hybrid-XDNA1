#!/usr/bin/env python3
"""Phase 61: per-operator breakdown of the Vulkan path, sorted by total time.

Reads the GGML_VK_PERF_LOGGER output ("<OP> ...: N x T us = TOTAL us") from
the logs and prints the per-operator totals, optionally for several PP values
side by side so that the operators whose cost grows with the context length
are visible.
"""
import collections
import glob
import os
import re
import sys

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

ROW = re.compile(r"^(?P<name>[A-Za-z_0-9 ]+?)(?:\s+(?P<extra>.*?))?:\s+(?P<n>\d+) x (?P<t>[\d.]+) us = (?P<total>[\d.]+) us")


def parse(path):
    ops = collections.defaultdict(float)
    counts = {}
    for line in open(path, errors="ignore"):
        line = line.strip()
        m = ROW.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        extra = (m.group("extra") or "").strip()
        key = name + (" " + extra if extra else "")
        ops[key] += float(m.group("total"))
        counts[key] = counts.get(key, 0) + int(m.group("n"))
    return ops, counts


def main():
    pps = [int(a) for a in sys.argv[1:]] or [512, 1024, 2048, 4096, 5120]
    data = {}
    for pp in pps:
        path = os.path.join(LOGS, "vkprof_pp%d.txt" % pp)
        if not os.path.exists(path):
            continue
        data[pp] = parse(path)
    if not data:
        print("no profiles found")
        return

    allops = sorted({k for pp in data for k in data[pp][0]},
                    key=lambda k: -max(data[pp][0].get(k, 0) for pp in data))
    hdr = "%-52s" % "operator" + "".join("%12s" % ("pp%d" % pp) for pp in data)
    print(hdr)
    print("-" * len(hdr))
    for op in allops:
        row = "%-52s" % op[:52]
        for pp in data:
            row += "%12.1f" % (data[pp][0].get(op, 0.0) / 1000.0)
        print(row)


if __name__ == "__main__":
    main()
