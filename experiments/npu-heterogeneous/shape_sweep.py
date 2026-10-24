#!/usr/bin/env python3
"""shape_sweep.py -- Phase 7A: Generic MUL_MAT operator characterization.

Systematically benchmarks MUL_MAT across CPU, GPU, and NPU backends
for a curated set of shapes. Records detailed timing breakdown,
correctness, and compiler/hardware limitations.

Usage:
    # Full sweep
    python3 shape_sweep.py --iters 10 --warmup 3

    # Specific shape
    python3 shape_sweep.py --M 2 --K 1024 --N 16 --iters 10

    # Overlap test only
    python3 shape_sweep.py --overlap-only --gpu-M 1024 --gpu-K 4096 --gpu-N 4096 \\
                           --npu-M 128 --npu-K 512 --npu-N 4096
"""
import argparse, json, sys, os, time, subprocess, statistics
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Curated shape sets for systematic characterization
# Format: (M, K, N, label)
# Small M: decode-like, GEMV-like
# Medium M: short prefill
# Large M: long prefill / batched
SHAPE_SETS = {
    "decode_smallN": [
        (1, 1024, 16,  "M1_K1024_N16"),
        (2, 1024, 16,  "M2_K1024_N16"),
        (4, 1024, 16,  "M4_K1024_N16"),
        (8, 1024, 16,  "M8_K1024_N16"),
        (16, 1024, 16, "M16_K1024_N16"),
    ],
    "decode_mediumN": [
        (1, 1024, 128, "M1_K1024_N128"),
        (2, 1024, 128, "M2_K1024_N128"),
        (8, 1024, 128, "M8_K1024_N128"),
    ],
    "prefill_small": [
        (32, 1024, 128,  "M32_K1024_N128"),
        (64, 1024, 256,  "M64_K1024_N256"),
        (128, 1024, 512, "M128_K1024_N512"),
    ],
    "prefill_largeK": [
        (64, 2048, 256,  "M64_K2048_N256"),
        (128, 4096, 512, "M128_K4096_N512"),
        (256, 4096, 1024, "M256_K4096_N1024"),
    ],
    "square_large": [
        (512, 4096, 4096, "M512_K4096_N4096"),
        (1024, 4096, 4096, "M1024_K4096_N4096"),
    ],
}

def _merge_dicts(a, b):
    out = dict(a)
    out.update(b)
    return out

def run_cpu(M, K, N, iters=10, warmup=3):
    from cpu_bench import benchmark
    return benchmark(M, K, N, iters=iters, warmup=warmup)

def run_gpu(M, K, N, iters=10, warmup=2):
    from gpu_bench_wrapper import run_gpu_bench
    return run_gpu_bench(M, K, N, iters=iters, warmup=warmup)

def run_npu(M, K, N, iters=10, warmup=3):
    from npu_bench import NPUBenchmark
    try:
        bench = NPUBenchmark(M, K, N)
        bench.warmup()
        result = bench.benchmark(iters=iters, warmup_iters=warmup)
        return result
    except (ValueError, AssertionError, RuntimeError) as e:
        return {"status": "BLOCKED", "M": M, "K": K, "N": N, "error": str(e)}

def run_transfer(M, K, N, iters=20):
    from transfer_bench import benchmark_transfer
    return benchmark_transfer(M, K, N, iters=iters)

def shape_label(M, K, N):
    return f"M{M}_K{K}_N{N}"

def benchmark_shape(M, K, N, iters=10, warmup=3, skip_gpu=False, skip_npu=False):
    label = shape_label(M, K, N)
    print(f"\n{'='*60}")
    print(f"Shape: {label}")
    print(f"{'='*60}")

    result = {"M": M, "K": K, "N": N, "label": label}

    # CPU
    print(f"  [CPU]   M={M} K={K} N={N}...", flush=True)
    cpu_r = run_cpu(M, K, N, iters=iters, warmup=warmup)
    result["cpu"] = cpu_r
    print(f"         compute={cpu_r.get('compute_ms_median',0):.2f}ms  "
          f"total={cpu_r.get('total_ms_median',0):.2f}ms  "
          f"gops={cpu_r.get('gops',0):.1f}")

    # GPU
    if not skip_gpu:
        print(f"  [GPU]   M={M} K={K} N={N}...", flush=True)
        gpu_r = run_gpu(M, K, N, iters=iters, warmup=warmup)
        result["gpu"] = gpu_r
        if gpu_r.get("status") == "OK":
            print(f"         compute={gpu_r.get('compute_ms',0):.2f}ms  "
                  f"total={gpu_r.get('total_ms',0):.2f}ms  "
                  f"h2d={gpu_r.get('h2d_ms',0):.2f}ms  "
                  f"d2h={gpu_r.get('d2h_ms',0):.2f}ms")
        else:
            print(f"         {gpu_r.get('status','UNKNOWN')}: {gpu_r.get('error','')[:80]}")
    else:
        result["gpu"] = {"status": "SKIPPED"}

    # NPU
    if not skip_npu:
        print(f"  [NPU]   M={M} K={K} N={N}...", flush=True)
        npu_r = run_npu(M, K, N, iters=iters, warmup=warmup)
        result["npu"] = npu_r
        if npu_r.get("status") == "OK":
            print(f"         npu={npu_r.get('npu_ms_median',0):.2f}ms  "
                  f"wall={npu_r.get('wall_ms_median',0):.2f}ms  "
                  f"h2d={npu_r.get('h2d_ms_median',0):.2f}ms  "
                  f"d2h={npu_r.get('d2h_ms_median',0):.2f}ms  "
                  f"mm={npu_r.get('mismatches',-1)}")
        else:
            print(f"         {npu_r.get('status','UNKNOWN')}: {npu_r.get('error','')[:80]}")
    else:
        result["npu"] = {"status": "SKIPPED"}

    # Transfer (only if NPU ran successfully)
    if result.get("npu", {}).get("status") == "OK":
        print(f"  [XFER]   M={M} K={K} N={N}...", flush=True)
        xfer_r = run_transfer(M, K, N, iters=min(iters * 2, 40))
        result["transfer"] = xfer_r
        if xfer_r.get("status") == "OK":
            print(f"         h2d={xfer_r['h2d_A_us']['median']:.1f}us  "
                  f"d2h={xfer_r['d2h_A_us']['median']:.1f}us  "
                  f"bw={xfer_r.get('h2d_A_bw_gbps',0):.2f}GB/s")
    else:
        result["transfer"] = {"status": "SKIPPED"}

    # Save individual result
    out_file = os.path.join(RESULTS_DIR, f"{label}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result

def run_overlap_for_shape(M, K, N, iters=5):
    label = shape_label(M, K, N)
    print(f"\n{'='*60}")
    print(f"Overlap test: {label}")
    print(f"{'='*60}")

    try:
        from overlap_bench import run_overlap
    except ImportError:
        print("  overlap_bench.py not available, skipping overlap test")
        return {"status": "SKIPPED", "reason": "module unavailable"}

    # Use the shape as both GPU and NPU workload
    # For NPU, ensure shape is NPU-compatible (divisible by 64, K <= 512)
    npu_M = M if M % 64 == 0 else 64
    npu_K = K if K <= 512 and K % 64 == 0 else 512
    npu_N = N if N % 64 == 0 else 64

    results = run_overlap(
        gpu_M=M, gpu_K=K, gpu_N=N,
        npu_M=npu_M, npu_K=npu_K, npu_N=npu_N,
        n_iters=iters, gpu_warmup=2
    )

    out_file = os.path.join(RESULTS_DIR, f"{label}_overlap.json")
    with open(out_file, "w") as f:
        json.dump({"shape": {"M": M, "K": K, "N": N},
                   "npu_shape": {"M": npu_M, "K": npu_K, "N": npu_N},
                   "results": results}, f, indent=2)

    return results

def run_shape_set(set_name, shapes, iters=10, warmup=3, skip_gpu=False, skip_npu=False,
                  run_overlap_flag=False, overlap_shapes=None):
    print(f"\n{'#'*60}")
    print(f"# Shape set: {set_name}  ({len(shapes)} shapes)")
    print(f"{'#'*60}")

    all_results = []
    for M, K, N, label in shapes:
        try:
            r = benchmark_shape(M, K, N, iters=iters, warmup=warmup,
                                skip_gpu=skip_gpu, skip_npu=skip_npu)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"M": M, "K": K, "N": N, "label": label, "error": str(e)})

    # Run overlap for selected shapes
    if run_overlap_flag:
        overlap_targets = overlap_shapes or shapes[:3]  # default: first 3 shapes
        for M, K, N, label in overlap_targets:
            try:
                run_overlap_for_shape(M, K, N, iters=5)
            except Exception as e:
                print(f"  Overlap ERROR for {label}: {e}")

    # Save set summary
    set_file = os.path.join(RESULTS_DIR, f"set_{set_name}.json")
    with open(set_file, "w") as f:
        json.dump({"set_name": set_name, "results": all_results}, f, indent=2)

    return all_results

def main():
    parser = argparse.ArgumentParser(description="Phase 7A: MUL_MAT shape characterization")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations per shape")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--set", default=None, help="Run specific shape set only")
    parser.add_argument("--M", type=int, default=None, help="Single M value")
    parser.add_argument("--K", type=int, default=None, help="Single K value")
    parser.add_argument("--N", type=int, default=None, help="Single N value")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-npu", action="store_true")
    parser.add_argument("--overlap", action="store_true", help="Run overlap tests")
    parser.add_argument("--overlap-only", action="store_true")
    parser.add_argument("--gpu-M", type=int, default=1024)
    parser.add_argument("--gpu-K", type=int, default=4096)
    parser.add_argument("--gpu-N", type=int, default=4096)
    parser.add_argument("--npu-M", type=int, default=128)
    parser.add_argument("--npu-K", type=int, default=512)
    parser.add_argument("--npu-N", type=int, default=4096)
    args = parser.parse_args()

    if args.overlap_only:
        from overlap_bench import main as overlap_main
        sys.argv = ["overlap_bench",
                    "--gpu-M", str(args.gpu_M), "--gpu-K", str(args.gpu_K), "--gpu-N", str(args.gpu_N),
                    "--npu-M", str(args.npu_M), "--npu-K", str(args.npu_K), "--npu-N", str(args.npu_N),
                    "--iters", "10", "--warmup", "3"]
        overlap_main()
        return

    if args.M is not None and args.K is not None and args.N is not None:
        # Single shape mode
        r = benchmark_shape(args.M, args.K, args.N, args.iters, args.warmup,
                            args.skip_gpu, args.skip_npu)
        print(json.dumps(r, indent=2))
        return

    # Sweep mode
    if args.set and args.set in SHAPE_SETS:
        sets_to_run = {args.set: SHAPE_SETS[args.set]}
    else:
        sets_to_run = SHAPE_SETS

    print("=" * 60)
    print("Phase 7A: MUL_MAT Operator Characterization")
    print(f"Sets: {list(sets_to_run.keys())}")
    print(f"Iters: {args.iters}, Warmup: {args.warmup}")
    print(f"Skip GPU: {args.skip_gpu}, Skip NPU: {args.skip_npu}")
    print(f"Overlap: {args.overlap}")
    print("=" * 60)

    all_results = {}
    for set_name, shapes in sets_to_run.items():
        results = run_shape_set(
            set_name, shapes,
            iters=args.iters, warmup=args.warmup,
            skip_gpu=args.skip_gpu, skip_npu=args.skip_npu,
            run_overlap_flag=args.overlap
        )
        all_results[set_name] = results

    # Save full sweep summary
    summary_file = os.path.join(RESULTS_DIR, "sweep_summary.json")
    with open(summary_file, "w") as f:
        json.dump({"config": {
            "iters": args.iters, "warmup": args.warmup,
            "skip_gpu": args.skip_gpu, "skip_npu": args.skip_npu,
            "sets": list(sets_to_run.keys()),
        }, "results": all_results}, f, indent=2)

    print(f"\nSweep complete. Results in {RESULTS_DIR}")
    print(f"Summary: {summary_file}")

if __name__ == "__main__":
    main()
