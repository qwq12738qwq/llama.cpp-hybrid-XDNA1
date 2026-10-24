#!/usr/bin/env python3
"""Parallel range downloader for ModelScope single files.

Splits the target into N chunks, fetches each with its own HTTP Range
connection (ModelScope throttles per connection), and resumes per chunk:
a chunk is skipped when the target region already has the right byte count.
Verifies the final size and the GGUF magic.

usage: dl_parallel.py <url> <out> [nchunks=16]
"""
import os
import sys
import threading
import urllib.request

URL = sys.argv[1]
OUT = sys.argv[2]
NCH = int(sys.argv[3]) if len(sys.argv) > 3 else 16

# total size via a 1-byte range probe
req = urllib.request.Request(URL, headers={"Range": "bytes=0-0", "User-Agent": "curl/8"})
r = urllib.request.urlopen(req, timeout=30)
cr = r.headers.get("Content-Range")
TOTAL = int(cr.split("/")[-1]) if cr else int(r.headers.get("Content-Length"))
print(f"total {TOTAL/1e6:.1f} MB, {NCH} chunks", flush=True)

CH = (TOTAL + NCH - 1) // NCH
with open(OUT, "ab") as f:
    f.truncate(TOTAL)

lock = threading.Lock()
done = [0]
fail = []


def fetch(i):
    start, end = i * CH, min(TOTAL, (i + 1) * CH) - 1
    expected = end - start + 1
    part = f"{OUT}.part{i}"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > expected:                       # corrupted by an earlier retry
        os.remove(part)
        have = 0
    if have == expected:
        with lock:
            done[0] += have
        return
    for attempt in range(6):
        hdr = {"User-Agent": "curl/8", "Range": f"bytes={start+have}-{end}"}
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, headers=hdr), timeout=60) as rr:
                if rr.status != 206:
                    raise RuntimeError(f"server returned {rr.status} for a Range request")
                with open(part, "ab") as f:
                    while True:
                        b = rr.read(1 << 20)
                        if not b:
                            break
                        if f.tell() + len(b) > expected:
                            raise RuntimeError("overrun")
                        f.write(b)
                        with lock:
                            done[0] += len(b)
            sz = os.path.getsize(part)
            if sz == expected:
                return
            have = sz                        # resume from the true file size
        except Exception as e:
            sz = os.path.getsize(part) if os.path.exists(part) else 0
            have = min(sz, expected)
            print(f"chunk {i} attempt {attempt}: {e} (have {have})", flush=True)
    fail.append(i)


os.makedirs("/tmp", exist_ok=True)
ths = [threading.Thread(target=fetch, args=(i,), daemon=True) for i in range(NCH)]
rep = threading.Thread(daemon=True)


def reporter():
    while True:
        import time
        time.sleep(20)
        with lock:
            d = done[0]
        print(f"progress {d/1e6:.0f}/{TOTAL/1e6:.0f} MB  ({100*d/TOTAL:.1f}%)", flush=True)
        if d >= TOTAL:
            break


threading.Thread(target=reporter, daemon=True).start()
for t in ths:
    t.start()
for t in ths:
    t.join()

if fail:
    print(f"FAILED chunks: {fail}", flush=True)
    sys.exit(1)

# assemble: chunks already at correct offsets in separate part files
with open(OUT, "r+b") as f:
    for i in range(NCH):
        start = i * CH
        end = min(TOTAL, (i + 1) * CH)
        part = f"{OUT}.part{i}"
        data = open(part, "rb").read()
        if len(data) != end - start:
            print(f"chunk {i} short {len(data)} != {end-start}", flush=True)
            sys.exit(1)
        f.seek(start)
        f.write(data)
        os.remove(part)

sz = os.path.getsize(OUT)
head = open(OUT, "rb").read(4)
if sz != TOTAL or head != b"GGUF":
    print(f"VERIFY FAIL size={sz} magic={head!r}", flush=True)
    sys.exit(1)
print(f"DONE {OUT} {sz/1e6:.1f} MB verified GGUF", flush=True)
