#!/usr/bin/env python3
"""
Phase 8-E: Pipeline Depth Sweep
Tests 1, 2, 3 buffer pipeline to measure effective overlap.
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


def run_pipeline_depth(gpu_worker, npu_bench, iters=100, buffer_depth=2):
    """Run pipeline with specified buffer depth.
    
    Buffer depth 1: Sequential (submit GPU, wait GPU, submit NPU, wait NPU)
    Buffer depth 2: Double buffer (submit GPU i, start NPU i, wait GPU i, start GPU i+1, wait NPU i)
    Buffer depth 3: Triple buffer (similar with 3 in-flight)
    """
    print(f"Pipeline depth={buffer_depth}")
    
    pipeline_times = []
    gpu_times = []
    npu_times = []
    
    if buffer_depth == 1:
        # Sequential baseline
        for i in range(iters):
            iter_id = gpu_worker.submit()
            gpu_resp = gpu_worker.wait(iter_id)
            gpu_time = gpu_resp["device_us"] + gpu_resp["wait_us"]
            npu_res = npu_bench.run_async()
            npu_time = npu_res["total_us"]
            pipeline_times.append(gpu_time + npu_time)
            gpu_times.append(gpu_time)
            npu_times.append(npu_time)
    else:
        # Pipeline with double/triple buffering
        # Keep buffer_depth-1 GPU submissions in flight
        in_flight = []
        
        for i in range(iters):
            # Submit GPU
            iter_id = gpu_worker.submit()
            in_flight.append({"iter_id": iter_id, "gpu_submitted": True, "npu_started": False})
            
            # Start NPU for the oldest in-flight that hasn't started NPU
            for item in in_flight:
                if not item["npu_started"]:
                    item["npu_res"] = npu_bench.run_async()
                    item["npu_started"] = True
                    break
            
            # If we have buffer_depth items in flight, wait for the oldest
            if len(in_flight) >= buffer_depth:
                oldest = in_flight.pop(0)
                gpu_resp = gpu_worker.wait(oldest["iter_id"])
                gpu_time = gpu_resp["device_us"] + gpu_resp["wait_us"]
                npu_time = oldest["npu_res"]["total_us"]
                pipeline_times.append(max(gpu_time, npu_time))
                gpu_times.append(gpu_time)
                npu_times.append(npu_time)
        
        # Drain remaining
        for item in in_flight:
            gpu_resp = gpu_worker.wait(item["iter_id"])
            gpu_time = gpu_resp["device_us"] + gpu_resp["wait_us"]
            npu_time = item["npu_res"]["total_us"]
            pipeline_times.append(max(gpu_time, npu_time))
            gpu_times.append(gpu_time)
            npu_times.append(npu_time)
    
    if (len(pipeline_times)) % 20 == 0:
        print(f"  {len(pipeline_times)}/{iters}")
    
    return {
        "pipeline_us": stats(pipeline_times),
        "gpu_us": stats(gpu_times),
        "npu_us": stats(npu_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--npu-xclbin", required=True)
    parser.add_argument("--npu-insts", required=True)
    parser.add_argument("--gpu-spv", required=True)
    args = parser.parse_args()

    npu_bench = PersistentNPUBenchmark(
        args.npu_xclbin, args.npu_insts, args.M, args.K, args.N
    )
    npu_bench.warmup(20)

    gpu_worker = GpuWorker(args.M, args.K, args.N, args.gpu_spv)
    time.sleep(1)

    results = {}
    
    # GPU solo baseline
    print("GPU Solo baseline")
    device_times = []
    wait_times = []
    for i in range(args.iters):
        iter_id = gpu_worker.submit()
        resp = gpu_worker.wait(iter_id)
        device_times.append(resp["device_us"])
        wait_times.append(resp["wait_us"])
    results["gpu_solo"] = {
        "device_us": stats(device_times),
        "wait_us": stats(wait_times),
        "total_us": stats([d + w for d, w in zip(device_times, wait_times)]),
    }
    
    # NPU solo baseline
    print("NPU Solo baseline")
    results["npu_solo"] = npu_bench.benchmark_async(args.iters, warmup=20)
    
    # Pipeline depths
    for depth in [1, 2, 3]:
        results[f"pipeline_depth_{depth}"] = run_pipeline_depth(
            gpu_worker, npu_bench, args.iters, buffer_depth=depth
        )
    
    gpu_worker.close()

    # Compute speedups
    gpu_solo = results["gpu_solo"]["total_us"]["median"]
    npu_solo = results["npu_solo"]["total_us"]["median"]
    seq = results["pipeline_depth_1"]["pipeline_us"]["median"]
    
    results["summary"] = {
        "gpu_solo_median_us": gpu_solo,
        "npu_solo_median_us": npu_solo,
        "sequential_median_us": seq,
    }
    
    for depth in [1, 2, 3]:
        pipe = results[f"pipeline_depth_{depth}"]["pipeline_us"]["median"]
        speedup = seq / pipe if pipe > 0 else 0
        gpu_concurrent = results[f"pipeline_depth_{depth}"]["gpu_us"]["median"]
        gpu_slowdown = gpu_concurrent / gpu_solo - 1.0
        results["summary"][f"depth{depth}_pipeline_median_us"] = pipe
        results["summary"][f"depth{depth}_speedup"] = speedup
        results["summary"][f"depth{depth}_gpu_slowdown"] = gpu_slowdown

    print(json.dumps(results, indent=2))

    out_path = os.path.join(os.path.dirname(__file__), "phase8e_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()