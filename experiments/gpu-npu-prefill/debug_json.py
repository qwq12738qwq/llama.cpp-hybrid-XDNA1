#!/usr/bin/env python3
import subprocess, json, sys

cmd = ['bash', '-c', 'source /root/src/mlir-aie-venv/bin/activate && cd /experiments/gpu-npu-prefill && python3 npu_candidate_bench.py --op GEMM --M 1024 --K 128 --N 1024 --iters 1']
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd='/experiments/gpu-npu-prefill')
combined = result.stdout + result.stderr

last_json_start = -1
for i in range(len(combined) - 1, -1, -1):
    if combined[i] == '{':
        last_json_start = i
        break

if last_json_start >= 0:
    json_str = combined[last_json_start:]
    brace_count = 0
    end_idx = -1
    in_string = False
    escape_next = False
    for i, c in enumerate(json_str):
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    if end_idx > 0:
        data = json.loads(json_str[:end_idx])
        print('Keys:', list(data.keys())[:5])
        print('Status value:', repr(data.get('status')))
        print('Has status:', 'status' in data)
    else:
        print('No end brace found')
        print('JSON str first 300:', json_str[:300])
else:
    print('No JSON start found')
