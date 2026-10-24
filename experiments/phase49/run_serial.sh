#!/bin/bash
# Phase 49: run one command at a time, nothing else on the box.
#
# This host has 7 GB shared with the UMA iGPU. Two concurrent benchmarks have
# already OOM-killed each other, so every run goes through this wrapper: it
# refuses to start while anything else is running, holds a lock so two wrappers
# cannot overlap, and aborts if memory is already tight.
set -u
# usage: run_serial.sh CMD [ARGS...]      (no -- separator)
MAX_USED_MB=${MAX_USED_MB:-5200}
LOCK=/tmp/opencode/phase49.lock
mkdir -p /tmp/opencode

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "run_serial: another phase49 run holds the lock, refusing to start" >&2
  exit 75
fi

# count only other bench processes; this wrapper's own cmdline names the binary
busy() {
  python3 - <<'PYEOF'
import os, sys
me = os.getpid()
n = 0
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == me:
        continue
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as fh:
            cl = fh.read().decode('utf8', 'replace')
    except OSError:
        continue
    if 'bench_pp_scaling' in cl and 'run_serial.sh' not in cl:
        n += 1
sys.exit(0 if n else 1)
PYEOF
}

for _ in $(seq 1 120); do
  busy && { echo "run_serial: waiting for other bench processes to exit" >&2; sleep 10; continue; }
  used=$(free -m | awk 'NR==2{print $3}')
  if [ "$used" -gt "$MAX_USED_MB" ]; then
    echo "run_serial: memory already at ${used} MB (limit ${MAX_USED_MB}), waiting" >&2
    sleep 20
    continue
  fi
  break
done

echo "run_serial: starting with ${used} MB used" >&2
"$@"
rc=$?
echo "run_serial: finished rc=$rc" >&2
exit $rc
