#!/usr/bin/env python3
"""
Phase 8: Persistent NPU benchmark with genuine asynchronous XRT execution.
Uses pyxrt.run.start() / wait() for true async submission.
"""
import argparse
import json
import time
import statistics
import numpy as np
import pyxrt


class PersistentNPUBenchmark:
    def __init__(self, xclbin_path, insts_path, M, K, N, seed=42):
        self.M, self.K, self.N = M, K, N
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.dev = pyxrt.device(0)
        self.xclbin = pyxrt.xclbin(xclbin_path)
        self.dev.register_xclbin(self.xclbin)
        self.uuid = self.xclbin.get_uuid()
        self.hw_ctx = pyxrt.hw_context(self.dev, self.uuid)
        self.kernel = pyxrt.kernel(self.hw_ctx, "MLIR_AIE")

        self.insts = np.fromfile(insts_path, dtype=np.uint32)
        self.inst_bo = pyxrt.bo(
            self.dev, self.insts.nbytes, pyxrt.bo.flags.cacheable,
            self.kernel.group_id(1)
        )
        np.frombuffer(self.inst_bo.map(), dtype=np.uint32)[:] = self.insts
        self.inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        self.A = self._make_bo((M, K), np.int8, 3)
        self.B = self._make_bo((K, N), np.int8, 4)
        self.C = self._make_bo((M, N), np.int32, 5)

        self.A_data = self.rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        self.B_data = self.rng.integers(-128, 127, size=(K, N), dtype=np.int8)
        self._load_inputs()

    def _make_bo(self, shape, dtype, group_idx):
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        return pyxrt.bo(self.dev, nbytes, pyxrt.bo.flags.host_only, self.kernel.group_id(group_idx))

    def _load_inputs(self):
        np.frombuffer(self.A.map(), dtype=np.int8).reshape(self.M, self.K)[:] = self.A_data
        np.frombuffer(self.B.map(), dtype=np.int8).reshape(self.K, self.N)[:] = self.B_data
        np.frombuffer(self.C.map(), dtype=np.int32).reshape(self.M, self.N).fill(0)
        for bo in (self.A, self.B, self.C):
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def _verify_output(self):
        self.C.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        got = np.frombuffer(self.C.map(), dtype=np.int32).reshape(self.M, self.N).copy()
        ref = self.A_data.astype(np.int32) @ self.B_data.astype(np.int32)
        mismatches = int(np.count_nonzero(got != ref))
        max_err = int(np.max(np.abs(got - ref))) if mismatches else 0
        return mismatches, max_err

    def warmup(self, iters=5):
        for _ in range(iters):
            run = pyxrt.run(self.kernel)
            for i, v in enumerate((3, self.inst_bo, self.insts.nbytes, self.A, self.B, self.C)):
                run.set_arg(i, v)
            run.start()
            run.wait()
        return self._verify_output()

    def run_async(self):
        run = pyxrt.run(self.kernel)
        for i, v in enumerate((3, self.inst_bo, self.insts.nbytes, self.A, self.B, self.C)):
            run.set_arg(i, v)

        t_submit = time.perf_counter_ns()
        run.start()
        submit_us = (time.perf_counter_ns() - t_submit) / 1000.0

        t_wait = time.perf_counter_ns()
        state = run.wait()
        wait_us = (time.perf_counter_ns() - t_wait) / 1000.0

        total_us = submit_us + wait_us
        return {
            "submit_us": submit_us,
            "wait_us": wait_us,
            "total_us": total_us,
            "state": str(state),
        }

    def run_sync(self):
        run = pyxrt.run(self.kernel)
        for i, v in enumerate((3, self.inst_bo, self.insts.nbytes, self.A, self.B, self.C)):
            run.set_arg(i, v)

        t0 = time.perf_counter_ns()
        run.start()
        run.wait()
        return (time.perf_counter_ns() - t0) / 1000.0

    def benchmark_async(self, iters=100, warmup=10):
        self.warmup(warmup)

        submit_times = []
        wait_times = []
        total_times = []
        states = []

        for i in range(iters):
            if i % 20 == 0:
                self.A_data = self.rng.integers(-128, 127, size=(self.M, self.K), dtype=np.int8)
                np.frombuffer(self.A.map(), dtype=np.int8).reshape(self.M, self.K)[:] = self.A_data
                self.A.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            r = self.run_async()
            submit_times.append(r["submit_us"])
            wait_times.append(r["wait_us"])
            total_times.append(r["total_us"])
            states.append(r["state"])

        mismatches, max_err = self._verify_output()

        return {
            "status": "OK" if mismatches == 0 else "MISMATCH",
            "M": self.M, "K": self.K, "N": self.N,
            "iters": iters,
            "submit_us": self._stats(submit_times),
            "wait_us": self._stats(wait_times),
            "total_us": self._stats(total_times),
            "state_counts": {s: states.count(s) for s in set(states)},
            "mismatches": mismatches,
            "max_abs_error": max_err,
        }

    def benchmark_sync(self, iters=100, warmup=10):
        self.warmup(warmup)

        times = []
        for i in range(iters):
            if i % 20 == 0:
                self.A_data = self.rng.integers(-128, 127, size=(self.M, self.K), dtype=np.int8)
                np.frombuffer(self.A.map(), dtype=np.int8).reshape(self.M, self.K)[:] = self.A_data
                self.A.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            t = self.run_sync()
            times.append(t)

        mismatches, max_err = self._verify_output()

        return {
            "status": "OK" if mismatches == 0 else "MISMATCH",
            "M": self.M, "K": self.K, "N": self.N,
            "iters": iters,
            "wall_us": self._stats(times),
            "mismatches": mismatches,
            "max_abs_error": max_err,
        }

    @staticmethod
    def _stats(samples):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xclbin", required=True)
    parser.add_argument("--insts", required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--mode", choices=["async", "sync"], default="async")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bench = PersistentNPUBenchmark(args.xclbin, args.insts, args.M, args.K, args.N, args.seed)

    if args.mode == "async":
        result = bench.benchmark_async(args.iters, args.warmup)
    else:
        result = bench.benchmark_sync(args.iters, args.warmup)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()