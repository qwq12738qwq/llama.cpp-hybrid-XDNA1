#!/usr/bin/env python3
"""Transform V1 MLIR for DMA pipelining."""
import os

def parse_and_transform(mlir_text):
    lines = mlir_text.split('\n')

    seq_start = None
    seq_end = None
    for i, line in enumerate(lines):
        if 'aie.runtime_sequence' in line:
            seq_start = i
        if seq_start is not None and i > seq_start + 2:
            stripped = line.strip()
            if stripped == '}':
                depth = 0
                for j in range(seq_start, i + 1):
                    depth += lines[j].count('{') - lines[j].count('}')
                if depth <= 0:
                    seq_end = i
                    break

    print(f"runtime_sequence: lines {seq_start+1} to {seq_end+1}")

    tile_starts = []
    for i, line in enumerate(lines[seq_start+1:seq_end], start=seq_start+1):
        if 'dma_configure_task_for @A_L3L2_0' in line:
            tile_starts.append(i)

    n_tiles = len(tile_starts)
    print(f"Found {n_tiles} tiles")

    tiles = []
    for idx, start in enumerate(tile_starts):
        end = tile_starts[idx + 1] if idx + 1 < len(tile_starts) else seq_end
        tiles.append(lines[start:end])

    tile_parts = []
    for t_lines in tiles:
        a_part, b_part, c_part, af_part = [], [], [], []
        state = 'a'
        for line in t_lines:
            stripped = line.strip()
            if state == 'a':
                a_part.append(line)
                if stripped.startswith('aiex.dma_start_task') and 'A_' not in stripped:
                    state = 'b'
            elif state == 'b':
                b_part.append(line)
                if stripped.startswith('aiex.dma_start_task') and 'B_' not in stripped:
                    state = 'c'
            elif state == 'c':
                c_part.append(line)
                if stripped.startswith('aiex.dma_start_task'):
                    state = 'af'
            elif state == 'af':
                af_part.append(line)
        tile_parts.append({'a': a_part, 'b': b_part, 'c': c_part, 'af': af_part})

    pipelined = []
    # Prefetch tile 0 A+B
    pipelined.extend(tile_parts[0]['a'])
    pipelined.extend(tile_parts[0]['b'])

    for i in range(n_tiles):
        pipelined.extend(tile_parts[i]['c'])
        if i > 0:
            pipelined.extend(tile_parts[i-1]['af'])
        if i < n_tiles - 1:
            pipelined.extend(tile_parts[i+1]['a'])
            pipelined.extend(tile_parts[i+1]['b'])

    pipelined.extend(tile_parts[-1]['af'])

    before = '\n'.join(lines[:seq_start + 1])
    after = '\n'.join(lines[seq_end:])
    new_mlir = before + '\n' + '\n'.join(pipelined) + '\n    ' + after

    n_await = new_mlir.count('dma_await_task')
    n_free = new_mlir.count('dma_free_task')
    n_configure = new_mlir.count('dma_configure_task_for')
    n_start = new_mlir.count('dma_start_task')
    print(f"Result: {n_configure} configure, {n_start} start, {n_await} await, {n_free} free")
    return new_mlir

v1 = "/Storage/Cache/llama.cpp/experiments/phase20/v2_pipe.prj/design.prj/aie_v1.mlir"
with open(v1) as f:
    mlir = parse_and_transform(f.read())
out = "/Storage/Cache/llama.cpp/experiments/phase20/v2_pipe.prj/design.prj/aie_pipelined.mlir"
with open(out, 'w') as f:
    f.write(mlir)
print(f"Written to {out}")
