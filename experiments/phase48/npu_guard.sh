#!/usr/bin/env bash
# Phase 48: wait until the XDNA device is free, then run a command.
#
# The NPU has no compute queue sharing in this driver stack, so two processes
# opening /dev/accel/accel0 can wedge the device. Nothing may touch the NPU
# while another process holds it open.
#
# usage: npu_guard.sh [--dev /dev/accel/accel0] [--timeout 900] -- cmd args...
set -u

DEV=/dev/accel/accel0
TIMEOUT=900

while [ $# -gt 0 ]; do
  case "$1" in
    --dev)     DEV="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --)        shift; break ;;
    *)         break ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "usage: $0 [--dev D] [--timeout S] -- cmd..." >&2
  exit 1
fi

deadline=$(( $(date +%s) + TIMEOUT ))
while :; do
  holders=$(python3 - "$DEV" <<'PY'
import os, sys
dev = sys.argv[1]
try:
    st = os.stat(dev)
except FileNotFoundError:
    print("MISSING")
    sys.exit(0)
key = (st.st_dev, st.st_ino)
me = os.getpid()
out = []
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == me:
        continue
    fdd = f'/proc/{pid}/fd'
    try:
        for fd in os.listdir(fdd):
            try:
                s = os.stat(f'{fdd}/{fd}')
            except OSError:
                continue
            if (s.st_dev, s.st_ino) == key:
                out.append(pid)
                break
    except OSError:
        continue
print(",".join(sorted(set(out), key=int)))
PY
)
  case "$holders" in
    ""|MISSING) ;;
    *) echo "npu_guard: waiting for $DEV holders: $holders" >&2; sleep 5 ;;
  esac
  [ "$holders" = "" ] || [ "$holders" = "MISSING" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "npu_guard: timed out after ${TIMEOUT}s waiting for $DEV" >&2
    exit 75
  fi
  sleep 5
done

if [ ! -e "$DEV" ]; then
  echo "npu_guard: $DEV does not exist, refusing to run" >&2
  exit 76
fi

exec "$@"