#!/usr/bin/env python3
import subprocess, json, sys

# Test direct GPU
try:
    r = subprocess.run(['./gpu_bench', '--M', '64', '--K', '64', '--N', '64', '--iters', '1', '--warmup', '0'],
                       capture_output=True, text=True, cwd='/experiments/gpu-npu-prefill', timeout=10)
    print('Direct GPU RC:', r.returncode, file=sys.stderr)
    print('Direct GPU out:', r.stdout[:100], file=sys.stderr)
except Exception as e:
    print('Direct GPU error:', e, file=sys.stderr)

# Test NPU via subprocess
cmd = [
    'bash', '-c',
    'source /root/src/mlir-aie-venv/bin/activate && cd /experiments/gpu-npu-prefill && python3 npu_candidate_bench.py --op GEMM --M 1024 --K 128 --N 1024 --iters 1'
]
print('Running NPU subprocess...', file=sys.stderr)
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd='/experiments/gpu-npu-prefill')
print('NPU RC:', result.returncode, file=sys.stderr)
print('NPU STDOUT len:', len(result.stdout), file=sys.stderr)
print('NPU STDERR len:', len(result.stderr), file=sys.stderr)
print('NPU STDERR:', result.stderr[:500], file=sys.stderr)

# Parse
combined = result.stdout + result.stderr
last_json_start = -1
for i in range(len(combined) - 1, -1, -1):
    if combined[i] == '{':
        last_json_start = i
        break
print('JSON start:', last_json_start, file=sys.stderr)
if last_json_start >= 0:
    try:
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
            print('Parsed status:', data.get('status'), file=sys.stderr)
        else:
            print('No end brace found', file=sys.stderr)
    except Exception as e:
        print('Parse error:', e, file=sys.stderr)
