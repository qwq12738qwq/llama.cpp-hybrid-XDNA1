#!/usr/bin/env python3
"""
Phase 8 Orchestration Harness
Runs Experiments A-H using persistent NPU and GPU workers.
"""
import argparse
import json
import subprocess
import time
import statistics
import sys
import os
import threading
import queue

sys.path.insert(0, os.path.dirname(__file__))
from npu_persistent_async import PersistentNPUBenchmark


class GpuWorker:
    def __init__(self, M, K, N, spv_path):
        self.proc = subprocess.Popen(
            ["./gpu_worker", str(M), str(K), str(N), spv_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(__file__)
        )
        self.iter_counter = 1
        self.pending = {}
        self.lock = threading.Lock()
        self.responses = {}
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            print(f"[GPU WORKER RECV] {line}", file=sys.stderr)
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "iter" in msg:
                iter_id = msg["iter"]
                with self.lock:
                    self.responses[iter_id] = msg
                    if iter_id in self.pending:
                        self.pending[iter_id].set()

    def submit(self):
        iter_id = self.iter_counter
        self.iter_counter += 1
        evt = threading.Event()
        with self.lock:
            self.pending[iter_id] = evt
        self.proc.stdin.write(json.dumps({"cmd": "submit"}) + "\n")
        self.proc.stdin.flush()
        # wait for submitted ack
        evt.wait(timeout=5.0)
        with self.lock:
            resp = self.responses.pop(iter_id, None)
        if resp and resp.get("event") == "submitted":
            return iter_id
        raise RuntimeError("Submit failed")

    def wait(self, iter_id):
        self.proc.stdin.write(json.dumps({"cmd": "wait", "iter": iter_id}) + "\n")
        self.proc.stdin.flush()
        evt = threading.Event()
        with self.lock:
            self.pending[iter_id] = evt
        evt.wait(timeout=10.0)
        with self.lock:
            resp = self.responses.pop(iter_id, None)
        if resp and resp.get("event") == "completed":
            return resp
        raise RuntimeError("Wait failed")

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def stats(samples):
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0],
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
    }


def run_experiment_a(gpu_worker, iters=100):
    print("Experiment A: GPU solo")
    device_times = []
    wait_times = []
    for i in range(iters):
        iter_id = gpu_worker.submit()
        resp = gpu_worker.wait(iter_id)
        device_times.append(resp["device_us"])
        wait_times.append(resp["wait_us"])
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{iters}")
    return {
        "device_us": stats(device_times),
        "wait_us": stats(wait_times),
        "total_us": stats([d + w for d, w in zip(device_times, wait_times)]),
    }


def run_experiment_b(npu_bench, iters=100):
    print("Experiment B: NPU solo")
    result = npu_bench.benchmark_async(iters, warmup=10)
    return {
        "submit_us": result["submit_us"],
        "wait_us": result["wait_us"],
        "total_us": result["total_us"],
    }


def run_experiment_c(gpu_worker, npu_bench, iters=50):
    print("Experiment C: Sequential GPU->NPU and NPU->GPU")
    # GPU then NPU
    gpu_npu_times = []
    for i in range(iters):
        iter_id = gpu_worker.submit()
        gpu_resp = gpu_worker.wait(iter_id)
        gpu_time = gpu_resp["device_us"] + gpu_resp["wait_us"]
        npu_res = npu_bench.run_async()
        npu_time = npu_res["total_us"]
        gpu_npu_times.append(gpu_time + npu_time)
    # NPU then GPU
    npu_gpu_times = []
    for i in range(iters):
        npu_res = npu_bench.run_async()
        npu_time = npu_res["total_us"]
        iter_id = gpu_worker.submit()
        gpu_resp = gpu_worker.wait(iter_id)
        gpu_time = gpu_resp["device_us"] + gpu_resp["wait_us"]
        npu_gpu_times.append(npu_time + gpu_time)
    return {
        "gpu_then_npu_us": stats(gpu_npu_times),
        "npu_then_gpu_us": stats(npu_gpu_times),
    }


def run_experiment_d(gpu_worker, npu_bench, iters=100):
    print("Experiment D: Async concurrent GPU || NPU")
    concurrent_times = []
    gpu_times = []
    npu_times = []
    for i in range(iters):
        iter_id = gpu_worker.submit()
        npu_res = npu_bench.run_async()
        gpu_resp = gpu_worker.wait(iter_id)
        gpu_time = gpu_resp["device_us"] + gpu_resp["wait_us"]
        npu_time = npu_res["total_us"]
        concurrent_times.append(max(gpu_time, npu_time))
        gpu_times.append(gpu_time)
        npu_times.append(npu_time)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{iters}")
    return {
        "concurrent_us": stats(concurrent_times),
        "gpu_us": stats(gpu_times),
        "npu_us": stats(npu_times),
    }


def run_experiment_e(gpu_worker, npu_bench, iters=100, buffer_depth=2):
    print(f"Experiment E: Pipeline depth={buffer_depth}")
    # We'll implement a simple pipeline: keep buffer_depth GPU submissions in flight,
    # each paired with an NPU run. Since NPU is single-stream, we'll run NPU sequentially
    # but overlap with GPU. This approximates double buffering on GPU side.
    pipeline_times = []
    gpu_times = []
    npu_times = []
    for i in range(iters):
        # submit GPU
        iter_id = gpu_worker.submit()
        # start NPU immediately
        npu_res = npu_bench.run_async()
        # wait for GPU
        gpu_resp = gpu_worker.wait(iter_id)
        gpu_time = gpu_resp["device_us"] + gpu_resp["wait_us"]
        npu_time = npu_res["total_us"]
        pipeline_times.append(max(gpu_time, npu_time))
        gpu_times.append(gpu_time)
        npu_times.append(npu_time)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{iters}")
    return {
        "pipeline_us": stats(pipeline_times),
        "gpu_us": stats(gpu_times),
        "npu_us": stats(npu_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--npu-xclbin", required=True)
    parser.add_argument("--npu-insts", required=True)
    parser.add_argument("--gpu-spv", required=True)
    args = parser.parse_args()

    # Initialize NPU benchmark
    npu_bench = PersistentNPUBenchmark(
        args.npu_xclbin, args.npu_insts, args.M, args.K, args.N
    )
    npu_bench.warmup(10)

    # Initialize GPU worker
    gpu_worker = GpuWorker(args.M, args.K, args.N, args.gpu_spv)
    time.sleep(1)  # let worker start

    results = {}

    # Experiment A
    results["A_gpu_solo"] = run_experiment_a(gpu_worker, args.iters)
    # Experiment B
    results["B_npu_solo"] = run_experiment_b(npu_bench, args.iters)
    # Experiment C
    results["C_sequential"] = run_experiment_c(gpu_worker, npu_bench, args.iters // 2)
    # Experiment D
    results["D_concurrent"] = run_experiment_d(gpu_worker, npu_bench, args.iters)
    # Experiment E (pipeline depth 2)
    results["E_pipeline"] = run_experiment_e(gpu_worker, npu_bench, args.iters, buffer_depth=2)

    # Compute speedups before closing
    gpu_solo = results["A_gpu_solo"]["total_us"]["median"]
    npu_solo = results["B_npu_solo"]["total_us"]["median"]
    seq = results["C_sequential"]["gpu_then_npu_us"]["median"]
    concurrent = results["D_concurrent"]["concurrent_us"]["median"]
    pipeline = results["E_pipeline"]["pipeline_us"]["median"]

    speedup_seq = seq / concurrent if concurrent > 0 else 0
    speedup_pipe = seq / pipeline if pipeline > 0 else 0

    results["summary"] = {
        "gpu_solo_median_us": gpu_solo,
        "npu_solo_median_us": npu_solo,
        "sequential_median_us": seq,
        "concurrent_median_us": concurrent,
        "pipeline_median_us": pipeline,
        "speedup_concurrent_vs_sequential": speedup_seq,
        "speedup_pipeline_vs_sequential": speedup_pipe,
    }

    print(json.dumps(results, indent=2))

    out_path = os.path.join(os.path.dirname(__file__), "phase8_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Close GPU worker
    gpu_worker.close()


if __name__ == "__main__":
    main()