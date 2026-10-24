#!/usr/bin/env python3
"""Transform V1 MLIR runtime_sequence for DMA pipelining.
Current pattern per tile:
  configure+start A_i, B_i, C_i -> await C_i -> free A_i, B_i, C_i

Pipelined pattern:
  configure+start A_0, B_0
  for i in 0..383:
    configure+start C_i
    if i > 0: await C_{i-1}, free A_{i-1}, B_{i-1}, C_{i-1}
    configure+start A_{i+1}, B_{i+1}
  await C_383, free A_383, B_383, C_383

This overlaps A/B prefetch for tile i+1 with C writeback for tile i."""
import re
import os
import sys

def transform_sequence(mlir_text):
    """Parse and restructure the runtime_sequence for pipelining."""
    lines = mlir_text.split('\n')

    # Find the runtime_sequence block
    seq_start = None
    seq_end = None
    for i, line in enumerate(lines):
        if 'aie.runtime_sequence' in line:
            seq_start = i
        if seq_start is not None and seq_end is None:
            if line.strip().startswith('}') and i > seq_start + 2:
                # Check if this closes the runtime_sequence
                # Count braces
                brace_count = 0
                for j in range(seq_start, i + 1):
                    brace_count += lines[j].count('{') - lines[j].count('}')
                if brace_count == 0:
                    seq_end = i
                    break

    if seq_start is None or seq_end is None:
        print("ERROR: Could not find runtime_sequence")
        return mlir_text

    seq_body = '\n'.join(lines[seq_start:seq_end + 1])
    print(f"Found runtime_sequence: lines {seq_start+1}-{seq_end+1}")

    # Parse individual DMA operations
    # Each tile has: configure+start A, configure+start B, configure+start C, await C, free A, free C
    # Pattern: %N = aiex.dma_configure_task_for @X_L3L2 { ... }
    #          aiex.dma_start_task(%N)
    # For C: aiex.dma_await_task(%N_c)
    #          aiex.dma_free_task(%N_a)
    #          aiex.dma_free_task(%N_b)
    #          aiex.dma_free_task(%N_c)

    # Extract the function signature line
    sig_line = lines[seq_start]

    # Extract the body (between first { and last })
    # Find inner content
    inner_start = seq_start + 1
    inner_end = seq_end - 1
    inner_lines = lines[inner_start:inner_end + 1]
    inner_text = '\n'.join(inner_lines)

    # Parse tiles: each tile block has configure(A), start(A), configure(B), start(B), configure(C), start(C), await(C), free(A), free(B), free(C)
    tile_pattern = re.compile(
        r'(\s*%(?P<a_id>\d+)\s*=\s*aiex\.dma_configure_task_for\s+@A_L3L2_0\s*\{[^}]*\}\s*\n'
        r'\s*aiex\.dma_start_task\(%(?P<a_id2>\d+)\)\s*\n'
        r'\s*%(?P<b_id>\d+)\s*=\s*aiex\.dma_configure_task_for\s+@B_L3L2_0\s*\{[^}]*\}\s*\n'
        r'\s*aiex\.dma_start_task\(%(?P<b_id2>\d+)\)\s*\n'
        r'\s*%(?P<c_id>\d+)\s*=\s*aiex\.dma_configure_task_for\s+@C_L2L3_0\s*\{[^}]*\}\s*\}\s*\{issue_token\s*=\s*true\}\s*\n'
        r'\s*aiex\.dma_start_task\(%(?P<c_id2>\d+)\)\s*\n'
        r'\s*aiex\.dma_await_task\(%(?P<c_id3>\d+)\)\s*\n'
        r'\s*aiex\.dma_free_task\(%(?P<a_free>\d+)\)\s*\n'
        r'\s*aiex\.dma_free_task\(%(?P<b_free>\d+)\)\s*\n'
        r'\s*aiex\.dma_free_task\(%(?P<c_free>\d+)\))',
        re.DOTALL
    )

    tiles = list(tile_pattern.finditer(inner_text))
    print(f"Found {len(tiles)} tile iterations")

    if len(tiles) == 0:
        print("ERROR: No tiles found with regex pattern")
        return mlir_text

    # Verify IDs match
    for t in tiles:
        g = t.groupdict()
        assert g['a_id'] == g['a_id2'], f"A ID mismatch: {g['a_id']} vs {g['a_id2']}"
        assert g['b_id'] == g['b_id2'], f"B ID mismatch: {g['b_id']} vs {g['b_id2']}"
        assert g['c_id'] == g['c_id2'] == g['c_id3'], f"C ID mismatch"
        assert g['a_free'] == g['a_id'], f"A free mismatch"
        assert g['b_free'] == g['b_id'], f"B free mismatch"
        assert g['c_free'] == g['c_id'], f"C free mismatch"

    # Extract the configure+start blocks for A, B, C separately
    tile_data = []
    for t in tiles:
        full = t.group(0)
        a_configure_start = re.search(
            r'(%\d+\s*=\s*aiex\.dma_configure_task_for\s+@A_L3L2_0\s*\{[^}]*\}\s*\n\s*aiex\.dma_start_task\(%\d+\))',
            full, re.DOTALL).group(1)
        b_configure_start = re.search(
            r'(%\d+\s*=\s*aiex\.dma_configure_task_for\s+@B_L3L2_0\s*\{[^}]*\}\s*\n\s*aiex\.dma_start_task\(%\d+\))',
            full, re.DOTALL).group(1)
        c_configure_start = re.search(
            r'(%\d+\s*=\s*aiex\.dma_configure_task_for\s+@C_L2L3_0\s*\{[^}]*\}\s*\}\s*\{issue_token\s*=\s*true\}\s*\n\s*aiex\.dma_start_task\(%\d+\))',
            full, re.DOTALL).group(1)
        c_id = t.group('c_id')
        a_id = t.group('a_id')
        b_id = t.group('b_id')
        tile_data.append({
            'a': a_configure_start, 'a_id': a_id,
            'b': b_configure_start, 'b_id': b_id,
            'c': c_configure_start, 'c_id': c_id,
        })

    # Build pipelined sequence
    # Phase 1: Prefetch A_0, B_0
    pipelined = []
    pipelined.append(tile_data[0]['a'])
    pipelined.append(tile_data[0]['b'])

    # Phase 2: For each tile i
    for i in range(len(tile_data)):
        t = tile_data[i]
        # Start C drain for current tile
        pipelined.append(t['c'])

        # Await C and free A/B/C for PREVIOUS tile (if not first)
        if i > 0:
            prev = tile_data[i - 1]
            pipelined.append(f'      aiex.dma_await_task(%{prev["c_id"]})')
            pipelined.append(f'      aiex.dma_free_task(%{prev["a_id"]})')
            pipelined.append(f'      aiex.dma_free_task(%{prev["b_id"]})')
            pipelined.append(f'      aiex.dma_free_task(%{prev["c_id"]})')

        # Prefetch A/B for NEXT tile (if not last)
        if i < len(tile_data) - 1:
            next_t = tile_data[i + 1]
            pipelined.append(next_t['a'])
            pipelined.append(next_t['b'])

    # Phase 3: Await last tile's C and free
    last = tile_data[-1]
    pipelined.append(f'      aiex.dma_await_task(%{last["c_id"]})')
    pipelined.append(f'      aiex.dma_free_task(%{last["a_id"]})')
    pipelined.append(f'      aiex.dma_free_task(%{last["b_id"]})')
    pipelined.append(f'      aiex.dma_free_task(%{last["c_id"]})')

    new_inner = '\n'.join(pipelined)

    # Reconstruct the file
    before = '\n'.join(lines[:seq_start + 1])
    after = '\n'.join(lines[seq_end:])

    new_mlir = before + '\n' + new_inner + '\n' + '    ' + after
    return new_mlir


def main():
    v1_mlir = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3072_C1_Tm64_Tk64_Tn64/design.prj/aie.mlir"
    with open(v1_mlir) as f:
        mlir_text = f.read()

    new_mlir = transform_sequence(mlir_text)

    out_dir = "/Storage/Cache/llama.cpp/experiments/phase20/v2_pipe.prj/design.prj"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "aie_pipelined.mlir")
    with open(out_path, 'w') as f:
        f.write(new_mlir)
    print(f"Written pipelined MLIR to {out_path}")

    # Verify structure
    n_await = new_mlir.count('dma_await_task')
    n_free = new_mlir.count('dma_free_task')
    n_start = new_mlir.count('dma_start_task')
    n_configure = new_mlir.count('dma_configure_task_for')
    print(f"  dma_configure_task_for: {n_configure}")
    print(f"  dma_start_task: {n_start}")
    print(f"  dma_await_task: {n_await}")
    print(f"  dma_free_task: {n_free}")
    print(f"  Expected: {384*3} configure, {384*3} start, {384} await, {384*3} free")


if __name__ == "__main__":
    main()
