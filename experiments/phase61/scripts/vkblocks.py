#!/usr/bin/env python3
"""Phase 61: per-ubatch operator breakdown of the Vulkan path.

The GGML_VK_PERF_LOGGER prints one block per graph_compute (one per ubatch).
This script splits the log into blocks and prints the last block (the one with
the longest KV cache) plus the sum over all blocks, so that the operators
whose cost grows with the context length are visible.
"""
import collections
import re
import sys
import os

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
ROW = re.compile(r"^(?P<name>[A-Za-z_0-9 ]+?)(?:\s+(?P<extra>.*?))?:\s+(?P<n>\d+) x (?P<t>[\d.]+) us = (?P<total>[\d.]+) us")


def blocks(path):
    txt = open(path, errors="ignore").read()
    out = []
    cur = collections.defaultdict(float)
    cnt = collections.Counter()
    flops = {}
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("Total time:"):
            out.append((dict(cur), dict(cnt), flops))
            cur = collections.defaultdict(float)
            cnt = collections.Counter()
            flops = {}
            continue
        m = ROW.match(line)
        if not m:
            continue
        key = m.group("name").strip()
        extra = (m.group("extra") or "").strip()
        if extra:
            key += " " + extra
        cur[key] += float(m.group("total"))
        cnt[key] += int(m.group("n"))
        g = re.search(r"\(([\d.]+) GFLOPS/s\)", line)
        if g:
            flops[key] = float(g.group(1))
    return out


def main():
    pp = int(sys.argv[1]) if len(sys.argv) > 1 else 5120
    path = os.path.join(LOGS, "vkprof_pp%d.txt" % pp)
    bs = blocks(path)
    if not bs:
        print("no data")
        return
    last, lastc, lastf = bs[-1]
    print("PP%d: %d blocks; LAST block (longest KV cache), times in ms" % (pp, len(bs)))
    print("%-56s %8s %8s %10s" % ("operator", "ms", "calls", "GFLOPS/s"))
    tot = 0.0
    for k in sorted(last, key=lambda k: -last[k]):
        tot += last[k]
        g = lastf.get(k, 0.0)
        print("%-56s %8.2f %8d %10s" % (k[:56], last[k] / 1000.0, lastc.get(k, 0),
                                       ("%.0f" % g) if g else "-"))
    print("%-56s %8.2f" % ("TOTAL (sum of node intervals)", tot / 1000.0))

    # per-block totals: does any op grow with the context?
    print()
    print("per-block totals (ms) for the ops that change across blocks:")
    keys = [k for k in last if k.startswith("FLASH_ATTN")]
    watch = keys + ["GATED_DELTA_NET", "CONCAT", "SSM_CONV_SILU SSM_CONV"]
    hdr = "%-34s" % "block" + "".join("%14s" % k[:13] for k in watch)
    print(hdr)
    for i, (cur, c, f) in enumerate(bs):
        if i % max(1, len(bs) // 10) and i != len(bs) - 1:
            continue
        row = "%-34s" % ("block %d" % i)
        for k in watch:
            row += "%14.2f" % (cur.get(k, 0.0) / 1000.0)
        print(row)


if __name__ == "__main__":
    main()
