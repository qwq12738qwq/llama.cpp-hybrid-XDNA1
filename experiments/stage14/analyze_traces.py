#!/usr/bin/env python3
"""
Analyze Qwen PP traces and construct critical path model.
"""

import json
import sys
from collections import defaultdict

def load_trace(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_trace(trace):
    """Analyze a single trace and return operation timing summary."""
    events = trace['events']
    
    # Group events by operation (match start/end pairs)
    ops = defaultdict(list)
    for e in events:
        key = (e['backend'], e['op_name'], e['graph_index'], e['layer'], e['M'], e['K'], e['N'])
        ops[key].append(e)
    
    # Calculate durations
    op_durations = []
    for key, evts in ops.items():
        start_evts = [e for e in evts if e['event_type'] == 'start']
        end_evts = [e for e in evts if e['event_type'] == 'end']
        
        if start_evts and end_evts:
            # Use the first start and last end
            start = min(e['timestamp_ns'] for e in start_evts)
            end = max(e['timestamp_ns'] for e in end_evts)
            duration_ns = end - start
            
            backend, op_name, graph_idx, layer, M, K, N = key
            start_evt = start_evts[0]
            op_durations.append({
                'backend': backend,
                'op_name': op_name,
                'op_type': start_evt['op_type'],
                'graph_index': graph_idx,
                'layer': layer,
                'M': M,
                'K': K,
                'N': N,
                'dtype_a': start_evt['dtype_a'],
                'dtype_b': start_evt['dtype_b'],
                'dtype_c': start_evt['dtype_c'],
                'tensor_size': start_evt['tensor_size_bytes'],
                'xdna_compatible': start_evt['xdna_compatible'],
                'duration_ns': duration_ns,
                'duration_ms': duration_ns / 1_000_000,
            })
    
    # Sort by start time
    # We need to sort by start timestamp
    start_times = {}
    for key, evts in ops.items():
        start_evts = [e for e in evts if e['event_type'] == 'start']
        if start_evts:
            start_times[key] = min(e['timestamp_ns'] for e in start_evts)
    
    # Sort op_durations by start time
    op_durations.sort(key=lambda x: start_times.get((x['backend'], x['op_name'], x['graph_index'], x['layer'], x['M'], x['K'], x['N']), 0))
    
    return op_durations

def print_trace_summary(context, durations):
    print(f"\n=== Context {context} ===")
    print(f"Total operations: {len(durations)}")
    
    # Group by backend
    by_backend = defaultdict(list)
    for op in durations:
        by_backend[op['backend']].append(op)
    
    for backend, ops in by_backend.items():
        total_time = sum(o['duration_ms'] for o in ops)
        print(f"\n  Backend: {backend} - Total: {total_time:.2f} ms, Ops: {len(ops)}")
        
        # Group by op type
        by_op = defaultdict(list)
        for o in ops:
            by_op[o['op_name']].append(o)
        
        for op_name, op_list in sorted(by_op.items(), key=lambda x: -sum(o['duration_ms'] for o in x[1])):
            total = sum(o['duration_ms'] for o in op_list)
            count = len(op_list)
            avg = total / count if count > 0 else 0
            shapes = set((o['M'], o['K'], o['N']) for o in op_list)
            print(f"    {op_name}: {total:.2f} ms ({count} ops, avg {avg:.2f} ms), shapes: {shapes}")

def aggregate_by_context():
    contexts = [512, 1024, 4096, 8192, 16384, 32768, 65536]
    all_data = {}
    
    for ctx in contexts:
        filepath = f"/workspace/llama.cpp/qwen_pp_trace_ctx{ctx}_iter0.json"
        try:
            trace = load_trace(filepath)
            durations = analyze_trace(trace)
            all_data[ctx] = durations
            print_trace_summary(ctx, durations)
        except FileNotFoundError:
            print(f"Trace file not found for context {ctx}")
    
    return all_data

def print_critical_path_analysis(all_data):
    print("\n\n=== CRITICAL PATH ANALYSIS ===")
    
    # For each context, find the critical path
    contexts = sorted(all_data.keys())
    
    for ctx in contexts:
        durations = all_data[ctx]
        print(f"\nContext {ctx}:")
        
        # Total time per backend
        by_backend = defaultdict(float)
        for op in durations:
            by_backend[op['backend']] += op['duration_ms']
        
        print(f"  Backend breakdown:")
        for backend, total in sorted(by_backend.items(), key=lambda x: -x[1]):
            print(f"    {backend}: {total:.2f} ms")
        
        # Top 10 operations by time
        print(f"  Top 10 operations:")
        sorted_ops = sorted(durations, key=lambda x: -x['duration_ms'])
        for i, op in enumerate(sorted_ops[:10]):
            print(f"    {i+1}. {op['backend']} {op['op_name']} M={op['M']} K={op['K']} N={op['N']}: {op['duration_ms']:.2f} ms")

    # Scaling analysis
    print("\n\n=== CONTEXT SCALING ANALYSIS ===")
    
    # Find operations that scale with context
    op_scaling = defaultdict(list)  # (backend, op_name) -> list of (ctx, total_time)
    
    for ctx in contexts:
        durations = all_data[ctx]
        by_op = defaultdict(float)
        for op in durations:
            key = (op['backend'], op['op_name'])
            by_op[key] += op['duration_ms']
        
        for key, total in by_op.items():
            op_scaling[key].append((ctx, total))
    
    print("Operations with significant scaling:")
    for key, points in op_scaling.items():
        if len(points) >= 2:
            # Calculate ratio from smallest to largest context
            ctxs = [p[0] for p in points]
            times = [p[1] for p in points]
            min_ctx = min(ctxs)
            max_ctx = max(ctxs)
            min_time = times[ctxs.index(min_ctx)]
            max_time = times[ctxs.index(max_ctx)]
            
            if min_time > 0:
                ratio = max_time / min_time
                if ratio > 1.5:  # Only show operations that scale significantly
                    print(f"  {key[0]} {key[1]}: {min_ctx}->{max_ctx} ratio = {ratio:.2f}x ({min_time:.1f}ms -> {max_time:.1f}ms)")

if __name__ == '__main__':
    all_data = aggregate_by_context()
    print_critical_path_analysis(all_data)