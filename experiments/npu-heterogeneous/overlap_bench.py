#!/usr/bin/env python3
"""overlap_bench.py -- GPU + NPU true overlap benchmark.

Measures:
  A. GPU only          -- GPU submits, waits
  B. NPU only          -- NPU submits, waits
  C. GPU + NPU overlap -- both submit simultaneously, then both wait

Protocol:
  - GPU binary blocks on FIFO for "GO" signal
  - NPU workload pre-warmed, blocks on threading.Event for GO
  - Orchestrator signals both simultaneously
  - Wall time measured from trigger to both-complete

Usage:
    python3 overlap_bench.py --gpu-M 1024 --gpu-K 4096 --gpu-N 4096 \\
                             --npu-M 128 --npu-K 512 --npu-N 4096 \\
                             --iters 10 --warmup 3
"""
import argparse, os, subprocess, threading, time, statistics, json, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_BENCH = os.path.join(SCRIPT_DIR, "gpu_bench")
GPU_BURN = os.path.join(SCRIPT_DIR, "gpu_burn")

def _ensure_bins():
    for bin_path in [GPU_BENCH, GPU_BURN]:
        if not os.path.exists(bin_path):
            build_script = os.path.join(SCRIPT_DIR, "build_gpu_bench.sh")
            if os.path.exists(build_script):
                subprocess.run(["bash", build_script], check=True, capture_output=True)
            else:
                raise FileNotFoundError(f"GPU binary not found at {bin_path}")

def _make_fifos(tag):
    base = f"/tmp/gpunpu_het_{tag}_{os.getpid()}"
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

def _cleanup_fifos(paths):
    for p in paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass

def _read_fifo(path):
    fd = os.open(path, os.O_RDONLY)
    buf = b""
    while b"\n" not in buf:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        buf += chunk
    os.close(fd)
    return buf.decode().strip()

def _write_fifo(path, data):
    fd = os.open(path, os.O_WRONLY)
    os.write(fd, data.encode())
    os.close(fd)

def run_gpu_standalone(M, K, N, iters=5, warmup=2):
    cmd = [GPU_BENCH, "--M", str(M), "--K", str(K), "--N", str(N),
           "--iters", str(iters), "--warmup", str(warmup)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=120)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                data["status"] = "OK"
                return data
            except json.JSONDecodeError:
                continue
    return None

def run_overlap(gpu_M, gpu_K, gpu_N, npu_M, npu_K, npu_N, n_iters=10, gpu_warmup=2):
    from npu_bench import NPUBenchmark

    _ensure_bins()

    npu_bench = NPUBenchmark(npu_M, npu_K, npu_N)
    npu_bench.warmup()

    results = []

    for iter_idx in range(n_iters):
        ready_fifo, go_fifo, result_fifo = _make_fifos(f"v{iter_idx}")
        try:
            # Start GPU subprocess (gpu_burn with FIFO-based sync)
            gpu_cmd = [GPU_BURN,
                       "--width", str(gpu_N), "--height", str(gpu_M), "--kdim", str(gpu_K),
                       "--warmup", str(gpu_warmup),
                       "--ready-fifo", ready_fifo,
                       "--go-fifo", go_fifo,
                       "--result-fifo", result_fifo]
            gpu_proc = subprocess.Popen(
                gpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=SCRIPT_DIR, text=True
            )

            # Wait for GPU READY
            ready_str = _read_fifo(ready_fifo)

            # Start NPU thread
            go_event = threading.Event()
            npu_result_box = [None]

            def npu_thread_fn():
                go_event.wait()
                npu_result_box[0] = npu_bench.run_iteration(new_input=False)

            npu_thread = threading.Thread(target=npu_thread_fn)
            npu_thread.start()

            # Start GPU result reader BEFORE signaling
            gpu_result_box = [None]
            def gpu_reader():
                try:
                    result_str = _read_fifo(result_fifo)
                    p = result_str.split(',')
                    if len(p) >= 6:
                        gpu_result_box[0] = {
                            "wall_start_ns": int(p[0]),
                            "wall_end_ns": int(p[1]),
                            "gpu_exec_ns": int(p[2]),
                            "ts_period": float(p[3]),
                            "ts_start": int(p[4]),
                            "ts_end": int(p[5]),
                        }
                except Exception:
                    gpu_result_box[0] = None

            gpu_reader_thread = threading.Thread(target=gpu_reader)
            gpu_reader_thread.start()

            # --- SIGNAL BOTH SIMULTANEOUSLY ---
            wall_start = time.perf_counter_ns()
            _write_fifo(go_fifo, "GO\n")
            go_event.set()

            # --- WAIT FOR BOTH ---
            gpu_proc.wait(timeout=60)
            npu_thread.join(timeout=60)
            gpu_reader_thread.join(timeout=60)
            wall_end = time.perf_counter_ns()

            gpu_r = gpu_result_box[0]
            npu_r = npu_result_box[0]

            results.append({
                "overall_wall_ns": wall_end - wall_start,
                "gpu_result": gpu_r,
                "npu_result": npu_r,
            })
            if (iter_idx + 1) % 3 == 0:
                print(f"  overlap {iter_idx + 1}/{n_iters}: wall={(wall_end-wall_start)/1e6:.2f}ms",
                      flush=True)

        finally:
            _cleanup_fifos([ready_fifo, go_fifo, result_fifo])
            try:
                gpu_proc.kill()
                gpu_proc.wait(timeout=2)
            except:
                pass

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-M", type=int, default=1024)
    parser.add_argument("--gpu-K", type=int, default=4096)
    parser.add_argument("--gpu-N", type=int, default=4096)
    parser.add_argument("--npu-M", type=int, default=128)
    parser.add_argument("--npu-K", type=int, default=512)
    parser.add_argument("--npu-N", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gpu-only-iters", type=int, default=5)
    parser.add_argument("--npu-only-iters", type=int, default=10)
    args = parser.parse_args()

    print("=== GPU+NPU Overlap Benchmark ===", flush=True)
    print(f"GPU: {args.gpu_M}x{args.gpu_K}x{args.gpu_N}", flush=True)
    print(f"NPU: {args.npu_M}x{args.npu_K}x{args.npu_N}", flush=True)

    # Case A: GPU only
    print("\n--- Case A: GPU only ---", flush=True)
    gpu_results = []
    for i in range(args.gpu_only_iters):
        r = run_gpu_standalone(args.gpu_M, args.gpu_K, args.gpu_N)
        if r:
            gpu_results.append(r)
        if (i + 1) % 3 == 0:
            print(f"  {i + 1}/{args.gpu_only_iters}", flush=True)

    gpu_exec_ms = [r["compute_ms"] for r in gpu_results if "compute_ms" in r]
    gpu_wall_ms = [(r["wall_end_ns"] - r["wall_start_ns"]) / 1e6 for r in gpu_results
                   if "wall_start_ns" in r and "wall_end_ns" in r]
    T_gpu_exec = statistics.median(gpu_exec_ms) if gpu_exec_ms else 0
    T_gpu_wall = statistics.median(gpu_wall_ms) if gpu_wall_ms else 0

    # Case B: NPU only
    print("\n--- Case B: NPU only ---", flush=True)
    from npu_bench import NPUBenchmark
    npu_bench = NPUBenchmark(args.npu_M, args.npu_K, args.npu_N)
    npu_bench.warmup()
    npu_results = []
    for i in range(args.npu_only_iters):
        r = npu_bench.run_iteration(new_input=(i % 3 == 0))
        npu_results.append(r)
        if (i + 1) % 3 == 0:
            print(f"  {i + 1}/{args.npu_only_iters}  mm={r['mismatches']}", flush=True)

    npu_exec_ms = [r["npu_us"] / 1000.0 for r in npu_results if r.get("npu_us") is not None]
    npu_wall_ms = [r["wall_us"] / 1000.0 for r in npu_results]
    T_npu_exec = statistics.median(npu_exec_ms) if npu_exec_ms else 0
    T_npu_wall = statistics.median(npu_wall_ms) if npu_wall_ms else 0

    # Case C: Overlap
    print(f"\n--- Case C: GPU+NPU overlap ---", flush=True)
    overlap_results = run_overlap(
        args.gpu_M, args.gpu_K, args.gpu_N,
        args.npu_M, args.npu_K, args.npu_N,
        args.iters, args.warmup
    )

    overlap_walls = [r["overall_wall_ns"] / 1e6 for r in overlap_results]
    T_overlap = statistics.median(overlap_walls) if overlap_walls else 0

    # Analysis
    print("\n" + "=" * 60)
    print("=== RESULTS ===")
    print(f"\nGPU only:   exec={T_gpu_exec:.2f}ms  wall={T_gpu_wall:.2f}ms")
    print(f"NPU only:   exec={T_npu_exec:.2f}ms  wall={T_npu_wall:.2f}ms")
    print(f"Sequential: {T_gpu_exec + T_npu_exec:.2f}ms")
    print(f"Overlap:    {T_overlap:.2f}ms")

    T_sequential = T_gpu_exec + T_npu_exec
    if T_sequential > 0 and T_overlap > 0:
        speedup = T_sequential / T_overlap
        efficiency = (T_sequential - T_overlap) / min(T_gpu_exec, T_npu_exec) * 100
        print(f"\nSpeedup:    {speedup:.2f}x")
        print(f"Efficiency: {efficiency:.1f}%")

        gpu_slowdowns = []
        npu_slowdowns = []
        for r in overlap_results:
            gr = r.get("gpu_result")
            nr = r.get("npu_result")
            if gr and "gpu_exec_ns" in gr:
                gpu_slowdowns.append(gr["gpu_exec_ns"] / 1e6)
            if nr and nr.get("npu_us") is not None:
                npu_slowdowns.append(nr["npu_us"] / 1000.0)

        if gpu_slowdowns:
            med_gpu_slow = statistics.median(gpu_slowdowns)
            gpu_slowdown_pct = ((med_gpu_slow - T_gpu_exec) / T_gpu_exec * 100) if T_gpu_exec > 0 else 0
            print(f"GPU slowdown under overlap: {gpu_slowdown_pct:.1f}%")

        if npu_slowdowns:
            med_npu_slow = statistics.median(npu_slowdowns)
            npu_slowdown_pct = ((med_npu_slow - T_npu_exec) / T_npu_exec * 100) if T_npu_exec > 0 else 0
            print(f"NPU slowdown under overlap: {npu_slowdown_pct:.1f}%")

        all_correct = all(r.get("npu_result", {}).get("mismatches", 1) == 0
                         for r in overlap_results)
        print(f"NPU correctness: {'PASS' if all_correct else 'FAIL'}")

        if T_overlap < T_sequential and efficiency >= 30:
            conclusion = "REAL_OVERLAP"
        elif T_overlap < T_sequential:
            conclusion = "PARTIAL_OVERLAP"
        else:
            conclusion = "NO_OVERLAP"
        print(f"Conclusion: {conclusion}")

    result = {
        "gpu_M": args.gpu_M, "gpu_K": args.gpu_K, "gpu_N": args.gpu_N,
        "npu_M": args.npu_M, "npu_K": args.npu_K, "npu_N": args.npu_N,
        "gpu_exec_ms": T_gpu_exec, "gpu_wall_ms": T_gpu_wall,
        "npu_exec_ms": T_npu_exec, "npu_wall_ms": T_npu_wall,
        "sequential_ms": T_sequential,
        "overlap_ms": T_overlap,
        "overlap_results": overlap_results,
    }
    if T_sequential > 0 and T_overlap > 0:
        result["speedup"] = T_sequential / T_overlap
        result["efficiency_pct"] = (T_sequential - T_overlap) / min(T_gpu_exec, T_npu_exec) * 100

    out_path = os.path.join(SCRIPT_DIR, "results", "overlap_result.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
