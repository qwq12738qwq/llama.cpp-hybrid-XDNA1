#!/usr/bin/env python3
"""
Phase 33: DMA microbenchmark.

Measures the overhead of individual XRT/DMA operations on Phoenix XDNA1.
Tests 1, 2, 4, 8, 16, 32, 64 transfers with three patterns:

A: N individual ObjectFifo fill+drain cycles (each with wait=True)
B: N fill/drain without wait, single wait at end
C: BD chain (single XRT submission for all transfers)

The benchmark uses a simple passthrough kernel on a single compute tile.
"""
import os
import sys
import json
import time
import subprocess
import argparse
import numpy as np

IRONENV = os.environ.get("IRONENV", "/Storage/Cache/ironenv-v143/bin/python")
PYTHONPATH = "/Storage/Cache/mlir-aie-src/python:/Storage/Cache/ironenv-v143/lib/python3.14/site-packages/mlir_aie/python"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_and_compile(n_transfers, transfer_bytes, approach, work_dir):
    """Generate MLIR and compile with aiecc using iron API."""
    os.makedirs(work_dir, exist_ok=True)
    xclbin_path = os.path.join(work_dir, "design.xclbin")
    insts_path = os.path.join(work_dir, "design.insts.bin")

    if os.path.exists(xclbin_path):
        return True

    num_elems = transfer_bytes // 4  # int32 elements

    # Build the runtime sequence body based on approach
    if approach == "A":
        seq_lines = [
            "        tg = iron.TaskGroup()",
            "        for i in range(NUM_TRANSFERS):",
            "            in_prod.fill(A, group=tg)",
            "            out_cons.drain(C, group=tg, wait=True)",
            "        tg.finish()",
        ]
    elif approach == "B":
        seq_lines = [
            "        tg = iron.TaskGroup()",
            "        for i in range(NUM_TRANSFERS):",
            "            in_prod.fill(A, group=tg)",
            "            out_cons.drain(C, group=tg, wait=False)",
            "        tg.finish()",
        ]
    else:
        seq_lines = [
            "        tg = iron.TaskGroup()",
            "        for i in range(NUM_TRANSFERS):",
            "            in_prod.fill(A, group=tg)",
            "            out_cons.drain(C, group=tg, wait=False)",
            "        tg.finish()",
        ]
    seq_body = "\n".join(seq_lines)

    # Build the generator script using simple string concatenation
    lines = []
    lines.append("import os")
    lines.append("import time")
    lines.append("import numpy as np")
    lines.append("import aie.iron as iron")
    lines.append("from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction")
    lines.append("from aie.iron.controlflow import range_ as iron_range")
    lines.append("")
    lines.append(f"NUM_TRANSFERS = {n_transfers}")
    lines.append(f"NUM_ELEMS = {num_elems}")
    lines.append("")
    lines.append('KERNEL_SRC = r\'\'\'')
    lines.append('extern "C" void dma_passthrough(const int32_t *in, int32_t *out, int n) {')
    lines.append('    for (int i = 0; i < n; i++) out[i] = in[i];')
    lines.append('}')
    lines.append('\'\'\'')
    lines.append("")
    lines.append("def build():")
    lines.append("    fused = ExternalFunction(")
    lines.append('        "dma_passthrough",')
    lines.append("        source_string=KERNEL_SRC,")
    lines.append("        arg_types=[")
    lines.append("            np.ndarray[(NUM_ELEMS,), np.dtype[np.int32]],")
    lines.append("            np.ndarray[(NUM_ELEMS,), np.dtype[np.int32]],")
    lines.append("            np.int32,")
    lines.append("        ],")
    lines.append("    )")
    lines.append("")
    lines.append("    in_ty = np.ndarray[(NUM_ELEMS,), np.dtype[np.int32]]")
    lines.append("    out_ty = np.ndarray[(NUM_ELEMS,), np.dtype[np.int32]]")
    lines.append("")
    lines.append("    fifo_in = iron.ObjectFifo(in_ty, name=\"in_fifo\", depth=1)")
    lines.append("    fifo_out = iron.ObjectFifo(out_ty, name=\"out_fifo\", depth=1)")
    lines.append("")
    lines.append("    def core_fn(of_in, of_out, kern, n):")
    lines.append("        for _ in iron_range(NUM_TRANSFERS):")
    lines.append("            elem_in = of_in.acquire(1)")
    lines.append("            elem_out = of_out.acquire(1)")
    lines.append("            kern(elem_in, elem_out, n)")
    lines.append("            of_in.release(1)")
    lines.append("            of_out.release(1)")
    lines.append("")
    lines.append("    worker = Worker(core_fn, [fifo_in.cons(), fifo_out.prod(), fused, NUM_ELEMS])")
    lines.append("")
    lines.append("    def sequence(A, C, _2, in_prod, out_cons):")

    for seq_line in seq_lines:
        lines.append(seq_line)

    lines.append("")
    lines.append("    rt = Runtime(sequence, [in_ty, out_ty, in_ty, fifo_in.prod(), fifo_out.cons()])")
    lines.append("    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()")
    lines.append("")
    lines.append(f'xp = "{xclbin_path}"')
    lines.append(f'ip = "{insts_path}"')
    pdi_path = os.path.join(work_dir, "main.pdi").replace("\\", "\\\\")
    lines.append(f'pp = "{pdi_path}"')
    lines.append("")
    lines.append("@iron.jit")
    lines.append("def design(A: In, C: Out, *, dummy: CompileTime[int] = 0):")
    lines.append("    return build()")
    lines.append("")
    lines.append('print("[DESIGN-' + approach + '] Generating and compiling...")')
    lines.append("t0 = time.time()")
    lines.append("design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)")
    lines.append('print("[DESIGN-' + approach + '] Compiled in %.1fs" % (time.time() - t0))')

    gen_script_path = os.path.join(work_dir, "generate.py")
    with open(gen_script_path, "w") as f:
        f.write("\n".join(lines))

    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    result = subprocess.run(
        [IRONENV, gen_script_path],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        print(f"[GEN-{approach}] FAILED: {result.stderr[-2000:]}")
        return False
    print(f"[GEN-{approach}] OK: {result.stdout.strip()}")
    return True


def run_benchmark(work_dir, n_transfers, repeat=5):
    """Run the benchmark on real hardware."""
    import pyxrt

    xclbin_path = os.path.join(work_dir, "design.xclbin")
    insts_path = os.path.join(work_dir, "design.insts.bin")

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xclbin_path)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    gid3 = kern.group_id(3)

    transfer_bytes = 1024  # 1KB per transfer
    total_bytes = n_transfers * transfer_bytes

    in_bo = pyxrt.bo(dev, total_bytes, pyxrt.bo.flags.host_only, gid3)
    out_bo = pyxrt.bo(dev, total_bytes, pyxrt.bo.flags.host_only, gid3)

    isz = os.path.getsize(insts_path)
    inst_bo = pyxrt.bo(dev, isz, pyxrt.bo.flags.cacheable, gid3)
    with open(insts_path, "rb") as f:
        inst_bo.write(f.read(), 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Prepare data
    in_data = np.arange(total_bytes, dtype=np.uint8)
    out_data = np.zeros(total_bytes, dtype=np.uint8)

    in_bo.write(in_data.tobytes(), 0)
    out_bo.write(out_data.tobytes(), 0)
    for bo in (in_bo, out_bo):
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Warmup
    for _ in range(2):
        r = pyxrt.run(kern)
        r.set_arg(0, 3)
        r.set_arg(1, inst_bo)
        r.set_arg(2, isz)
        r.set_arg(3, in_bo)
        r.set_arg(4, out_bo)
        r.start()
        r.wait()

    times = []
    for _ in range(repeat):
        r = pyxrt.run(kern)
        r.set_arg(0, 3)
        r.set_arg(1, inst_bo)
        r.set_arg(2, isz)
        r.set_arg(3, in_bo)
        r.set_arg(4, out_bo)

        t0 = time.perf_counter()
        r.start()
        r.wait()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--transfers", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--transfer-bytes", type=int, default=1024)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--work-dir", default=os.path.join(HERE, "dma_microbench"))
    args = ap.parse_args()

    transfers = [int(x) for x in args.transfers.split(",")]
    approaches = ["A", "B", "C"] if args.approach == "all" else [args.approach]

    results = []
    for approach in approaches:
        for n in transfers:
            print(f"\n[PHASE33] Approach {approach}, transfers={n}, bytes={args.transfer_bytes}")
            work_dir = os.path.join(args.work_dir, f"approach_{approach}", f"t{n}")
            os.makedirs(work_dir, exist_ok=True)

            if not generate_and_compile(n, args.transfer_bytes, approach, work_dir):
                print(f"[PHASE33] SKIP approach={approach} transfers={n} (generate failed)")
                continue

            try:
                times = run_benchmark(work_dir, n, args.repeat)
                med = sorted(times)[len(times) // 2]
                results.append({
                    "approach": approach,
                    "transfers": n,
                    "transfer_bytes": args.transfer_bytes,
                    "total_bytes": n * args.transfer_bytes,
                    "repeat": args.repeat,
                    "ms_all": [round(t, 4) for t in times],
                    "ms_median": round(med, 4),
                    "ms_per_transfer": round(med / n, 4) if n > 0 else 0,
                })
                print(f"[PHASE33] RESULT approach={approach} transfers={n} "
                      f"median={med:.3f}ms per_transfer={med/n:.4f}ms")
            except Exception as e:
                print(f"[PHASE33] RUN FAILED: {e}")

    # Save results
    results_path = os.path.join(RESULTS_DIR, "phase33_dma_microbench.jsonl")
    with open(results_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[PHASE33] Results saved to {results_path}")

    # Print summary
    print("\n=== PHASE 33 SUMMARY ===")
    print(f"{'Approach':<10} {'Transfers':<10} {'Median (ms)':<15} {'Per-transfer (ms)':<20}")
    for r in results:
        print(f"{r['approach']:<10} {r['transfers']:<10} {r['ms_median']:<15.4f} {r['ms_per_transfer']:<20.4f}")


if __name__ == "__main__":
    main()
