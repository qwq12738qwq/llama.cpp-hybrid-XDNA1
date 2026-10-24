#!/usr/bin/env python3
"""orchestrator.py -- GPU-NPU overlap benchmark.

Measures:
  A. GPU only          -- GPU submits, waits
  B. NPU only          -- NPU submits, waits
  C. GPU + NPU overlap -- both submit simultaneously, then both wait

Synchronization (Case C):
  - GPU binary initialized and blocks on a named pipe (FIFO) for "GO"
  - NPU pre-warmed, blocks on threading.Event for GO
  - Orchestrator waits for BOTH to signal "ready", then triggers both
    in the same Python call (FIFO write + Event set)
  - Wall time measured from trigger to both-complete

This guarantees:
  GPU submit -> NPU submit -> wait GPU -> wait NPU
  (NOT the pseudo-overlap of: GPU submit -> wait GPU -> NPU submit -> wait NPU)
"""
import os
import sys
import time
import subprocess
import threading
import statistics
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_BIN = str(SCRIPT_DIR / "gpu_burn")

# Default shapes
GPU_W = 1024
GPU_H = 1024
GPU_K = 1024
NPU_M = 128
NPU_K = 512
NPU_N = 4096


def make_fifos(tag):
    base = f"/tmp/gpunpu_{tag}_{os.getpid()}"
    ready = f"{base}.ready"
    go = f"{base}.go"
    result = f"{base}.result"
    for p in [ready, go, result]:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
        os.mkfifo(p, 0o666)
    return ready, go, result


def cleanup_fifos(paths):
    for p in paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def run_gpu_standalone(gpu_w, gpu_h, gpu_k, warmup, read_back=False):
    """Case A: Run GPU binary in standalone mode. Returns timing dict."""
    cmd = [GPU_BIN, "--width", str(gpu_w), "--height", str(gpu_h),
           "--kdim", str(gpu_k), "--warmup", str(warmup), "--iters", "1"]
    if read_back:
        cmd.append("--read-back")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
    if proc.returncode != 0:
        print(f"GPU binary failed:\n{proc.stderr}", file=sys.stderr)
        return None
    lines = proc.stdout.strip().split('\n')
    csv = [l for l in lines if l and ',' in l][-1].strip()
    parts = csv.split(',')
    return {
        "wall_start_ns": int(parts[0]),
        "wall_end_ns": int(parts[1]),
        "gpu_exec_ns": int(parts[2]),
        "ts_period": float(parts[3]),
        "ts_start": int(parts[4]),
        "ts_end": int(parts[5]),
    }


def run_overlap(gpu_w, gpu_h, gpu_k, npu_m, npu_k, npu_n,
                n_iters, gpu_warmup, read_back=False):
    """Case C: GPU + NPU true overlap.

    Pre-warms GPU binary and NPU workload, then triggers both simultaneously.
    Returns list of result dicts with timing for each iteration.
    """
    from npu_workload import NPUWorkload

    # Pre-create and warm up NPU workload (compile + BO + warmup)
    npu_w = NPUWorkload(M=npu_m, K=npu_k, N=npu_n)
    npu_w.warmup()

    results = []

    for iter_idx in range(n_iters):
        ready_fifo, go_fifo, result_fifo = make_fifos(f"v{iter_idx}")

        try:
            # Start GPU subprocess with FIFO-based sync
            gpu_cmd = [
                GPU_BIN, "--width", str(gpu_w), "--height", str(gpu_h),
                "--kdim", str(gpu_k), "--warmup", str(gpu_warmup),
                "--ready-fifo", ready_fifo,
                "--go-fifo", go_fifo,
                "--result-fifo", result_fifo,
            ]
            if read_back:
                gpu_cmd.append("--read-back")

            gpu_proc = subprocess.Popen(
                gpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(SCRIPT_DIR), text=True
            )

            # Wait for GPU READY
            ready_fd = os.open(ready_fifo, os.O_RDONLY)
            ready_buf = b""
            while b"\n" not in ready_buf:
                chunk = os.read(ready_fd, 256)
                if not chunk:
                    break
                ready_buf += chunk
            os.close(ready_fd)
            ready_str = ready_buf.decode().strip()
            gpu_init_ms = float(ready_str.split()[1]) if " " in ready_str else 0.0

            # Start NPU thread (already warmed, just needs GO)
            go_event = threading.Event()
            npu_result_box = [None]

            def npu_thread_fn():
                go_event.wait()
                npu_result_box[0] = npu_w.run()

            npu_thread = threading.Thread(target=npu_thread_fn)
            npu_thread.start()

            # Both are ready: GPU finished warmup, NPU is pre-warmed.
            # Start GPU result reader BEFORE signaling - prevents deadlock
            # (GPU blocks on open(result_fifo, O_WRONLY) until reader opens O_RDONLY)
            gpu_result_box = [None]
            def gpu_reader():
                try:
                    result_fd = os.open(result_fifo, os.O_RDONLY)
                    buf = b""
                    while b"\n" not in buf:
                        chunk = os.read(result_fd, 1024)
                        if not chunk:
                            break
                        buf += chunk
                    os.close(result_fd)
                    csv = buf.decode().strip()
                    p = csv.split(',')
                    gpu_result_box[0] = {
                        "wall_start_ns": int(p[0]),
                        "wall_end_ns": int(p[1]),
                        "gpu_exec_ns": int(p[2]),
                        "ts_period": float(p[3]),
                        "ts_start": int(p[4]),
                        "ts_end": int(p[5]),
                    } if len(p) >= 6 else None
                except Exception as e:
                    gpu_result_box[0] = None

            gpu_reader_thread = threading.Thread(target=gpu_reader)
            gpu_reader_thread.start()

            # --- SIGNAL BOTH SIMULTANEOUSLY ---
            wall_start = time.perf_counter_ns()

            go_fd = os.open(go_fifo, os.O_WRONLY)
            os.write(go_fd, b"GO\n")
            os.close(go_fd)

            go_event.set()

            # --- WAIT FOR BOTH ---
            gpu_proc.wait()
            npu_thread.join()
            gpu_reader_thread.join()

            wall_end = time.perf_counter_ns()

            gpu_r = gpu_result_box[0]

            npu_r = npu_result_box[0]

            results.append({
                "overall_wall_start_ns": wall_start,
                "overall_wall_end_ns": wall_end,
                "overall_wall_ns": wall_end - wall_start,
                "gpu_result": gpu_r,
                "npu_result": npu_r,
                "gpu_init_ms": gpu_init_ms,
            })
            if (iter_idx + 1) % 3 == 0:
                wall_ms = (wall_end - wall_start) / 1e6
                print(f"  {iter_idx + 1}/{n_iters}  wall={wall_ms:.2f}ms", flush=True)

        finally:
            cleanup_fifos([ready_fifo, go_fifo, result_fifo])
            try:
                gpu_proc.kill()
                gpu_proc.wait()
            except:
                pass

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU-NPU overlap benchmark")
    parser.add_argument("--gpu-width", type=int, default=GPU_W)
    parser.add_argument("--gpu-height", type=int, default=GPU_H)
    parser.add_argument("--gpu-kdim", type=int, default=GPU_K)
    parser.add_argument("--npu-M", type=int, default=NPU_M)
    parser.add_argument("--npu-K", type=int, default=NPU_K)
    parser.add_argument("--npu-N", type=int, default=NPU_N)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--overlap-iters", type=int, default=10)
    parser.add_argument("--read-back", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(GPU_BIN):
        print(f"GPU binary not found at {GPU_BIN}", file=sys.stderr)
        sys.exit(1)

    print("=== GPU-NPU Overlap Benchmark ===")
    print(f"GPU: {args.gpu_height}x{args.gpu_kdim}x{args.gpu_width} FP32 GEMM")
    print(f"NPU: {args.npu_M}x{args.npu_K}x{args.npu_N} INT8 GEMM")
    print(f"Iters: {args.iters}, warmup: {args.warmup}")
    print()

    # --- Case A: GPU only ---
    print("--- Case A: GPU only ---", flush=True)
    gpu_results = []
    for i in range(args.iters):
        r = run_gpu_standalone(args.gpu_width, args.gpu_height, args.gpu_kdim,
                               args.warmup, args.read_back)
        if r:
            gpu_results.append(r)
        if (i + 1) % 5 == 0 or i == args.iters - 1:
            print(f"  {i + 1}/{args.iters}", flush=True)

    gpu_exec_ms = [r["gpu_exec_ns"] / 1e6 for r in gpu_results]
    gpu_wall_ms = [(r["wall_end_ns"] - r["wall_start_ns"]) / 1e6 for r in gpu_results]

    T_gpu = statistics.median(gpu_exec_ms)
    T_gpu_wall = statistics.median(gpu_wall_ms)
    print(f"  gpu_exec median={T_gpu:.2f}ms  wall median={T_gpu_wall:.2f}ms")

    # --- Case B: NPU only ---
    print("\n--- Case B: NPU only ---", flush=True)
    from npu_workload import NPUWorkload
    w = NPUWorkload(M=args.npu_M, K=args.npu_K, N=args.npu_N)
    w.warmup()

    npu_results = []
    for i in range(args.iters):
        r = w.run()
        npu_results.append(r)
        if (i + 1) % 5 == 0 or i == args.iters - 1:
            mm = r["mismatches"]
            print(f"  {i + 1}/{args.iters}  (mismatch={mm})", flush=True)

    npu_exec_ns_valid = [r["npu_exec_ns"] for r in npu_results if r["npu_exec_ns"] is not None]
    npu_exec_ms = [ns / 1e6 for ns in npu_exec_ns_valid]
    npu_wall_ms = [r["wall_exec_ns"] / 1e6 for r in npu_results]

    T_npu = statistics.median(npu_exec_ms) if npu_exec_ms else 0
    T_npu_wall = statistics.median(npu_wall_ms)
    print(f"  npu_exec median={T_npu:.2f}ms  wall median={T_npu_wall:.2f}ms")

    # --- Case C: Overlap ---
    print(f"\n--- Case C: GPU + NPU overlap ---", flush=True)
    overlap_results = run_overlap(
        args.gpu_width, args.gpu_height, args.gpu_kdim,
        args.npu_M, args.npu_K, args.npu_N,
        args.overlap_iters, args.warmup, args.read_back
    )

    overlap_walls = [r["overall_wall_ns"] / 1e6 for r in overlap_results]
    T_overlap = statistics.median(overlap_walls)

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("=== RESULTS ===")

    print("\nGPU only:")
    print(f"  gpu_ms = {T_gpu:.2f}")
    print(f"  wall_ms = {T_gpu_wall:.2f}")

    print("\nNPU only:")
    print(f"  npu_ms = {T_npu:.2f}")
    print(f"  wall_ms = {T_npu_wall:.2f}")

    T_sequential = T_gpu + T_npu
    print("\nSequential:")
    print(f"  wall_ms = {T_sequential:.2f}")

    print("\nGPU + NPU overlap:")
    print(f"  wall_ms = {T_overlap:.2f}")

    # Execution intervals from overlap runs (using wall time per component)
    gpu_wall_in_overlap = []
    npu_wall_in_overlap = []
    for r in overlap_results:
        gr = r.get("gpu_result")
        nr = r.get("npu_result")
        if gr:
            gpu_wall_in_overlap.append((gr["wall_end_ns"] - gr["wall_start_ns"]) / 1e6)
        if nr:
            npu_wall_in_overlap.append(nr["wall_exec_ns"] / 1e6)

    if gpu_results and npu_results and overlap_results:
        g_starts = [r["gpu_result"]["wall_start_ns"] for r in overlap_results if r.get("gpu_result")]
        g_ends = [r["gpu_result"]["wall_end_ns"] for r in overlap_results if r.get("gpu_result")]
        n_starts = [r["npu_result"]["wall_start_ns"] for r in overlap_results if r.get("npu_result")]
        n_ends = [r["npu_result"]["wall_end_ns"] for r in overlap_results if r.get("npu_result")]

        if g_starts and n_starts:
            g_start = statistics.median(g_starts)
            g_end = statistics.median(g_ends)
            n_start = statistics.median(n_starts)
            n_end = statistics.median(n_ends)

            print(f"\nGPU execution interval:")
            print(f"  [{g_start}, {g_end}]  ({T_gpu:.2f} ms)")

            print(f"\nNPU execution interval:")
            print(f"  [{n_start}, {n_end}]  ({T_npu:.2f} ms)")

            overlap_start = max(g_start, n_start)
            overlap_end = min(g_end, n_end)
            actual_overlap_ns = max(0, overlap_end - overlap_start)
            actual_overlap_ms = actual_overlap_ns / 1e6

            print(f"\nActual overlap:")
            print(f"  {actual_overlap_ms:.2f} ms")

            overlap_efficiency = (T_gpu + T_npu - T_overlap) / min(T_gpu, T_npu) * 100
            print(f"\nOverlap efficiency:")
            print(f"  {overlap_efficiency:.1f}%")

            # Slowdown analysis
            gpu_overlap_exec = statistics.median(gpu_wall_in_overlap) if gpu_wall_in_overlap else T_gpu_wall
            npu_overlap_exec = statistics.median(npu_wall_in_overlap) if npu_wall_in_overlap else T_npu_wall

            gpu_slowdown = ((gpu_overlap_exec - T_gpu_wall) / T_gpu_wall * 100) if T_gpu_wall > 0 else 0
            npu_slowdown = ((npu_overlap_exec - T_npu_wall) / T_npu_wall * 100) if T_npu_wall > 0 else 0

            print(f"\nGPU slowdown under overlap:")
            print(f"  {gpu_slowdown:.1f}%  ({T_gpu_wall:.2f}ms -> {gpu_overlap_exec:.2f}ms)")

            print(f"\nNPU slowdown under overlap:")
            print(f"  {npu_slowdown:.1f}%  ({T_npu_wall:.2f}ms -> {npu_overlap_exec:.2f}ms)")

            all_correct = all(
                r.get("npu_result", {}).get("mismatches", 1) == 0
                for r in overlap_results
            )
            print(f"\nNPU correctness: {'PASS' if all_correct else 'FAIL'}")

            # Conclusion: use actual interval overlap + efficiency as criteria
            intervals_overlap = (n_start < g_end and g_start < n_end)
            hiding_ratio = actual_overlap_ms / min(T_gpu, T_npu) * 100 if actual_overlap_ms > 0 else 0
            any_benefit = T_overlap < T_sequential

            if not intervals_overlap:
                conclusion = "NO_OVERLAP"
            elif not any_benefit:
                # Intervals overlap but overhead exceeds benefit
                conclusion = "NO_OVERLAP"
            elif hiding_ratio >= 80:
                conclusion = "REAL_OVERLAP"
            elif hiding_ratio >= 30:
                conclusion = "PARTIAL_OVERLAP"
            else:
                conclusion = "PARTIAL_OVERLAP"

            print(f"\nConclusion:")
            print(f"  {conclusion}")
            print(f"\n(T_gpu={T_gpu:.2f}ms, T_npu={T_npu:.2f}ms, "
                  f"T_seq={T_sequential:.2f}ms, T_overlap={T_overlap:.2f}ms, "
                  f"eff={overlap_efficiency:.1f}%)")
        else:
            print("\nInterval data not available")
            print("\nConclusion: INCOMPLETE")
    else:
        print("\nInsufficient data for analysis")
        print("\nConclusion: INCOMPLETE")

    print("\n" + "=" * 60)
    print("Timing note: perf_counter_ns() monotonic clock. GPU timestamps")
    print("via Vulkan query pool. NPU timing via XRT npu_time attribute.")
    print("OS scheduling jitter may affect sub-millisecond precision.")


if __name__ == "__main__":
    main()
