#!/usr/bin/env python3
"""Phase 18-J/K: Build XDNA Capability Model and Vulkan vs XDNA Cost Model.

Measures XDNA performance across M/K/N shapes and outputs a JSON capability model
that can be used to decide: XDNA-only, Vulkan-only, or N-partition for any MUL_MAT.
"""
import json, os, sys, time, subprocess
import numpy as np

try:
    import pyxrt
except ImportError:
    sys.exit("ERROR: pyxrt not available")

KC = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"
GEN = "/workspace/llama.cpp/experiments/npu-int8-gemm/gen_xdna_gemm.py"
OUT = "/home/ubuntu12738/phase18"

# Vulkan reference times (measured from Phase 17B and hetero test)
# These are per-MUL_MAT ggml-backend times for Q8_0 x F32 -> F32
VULKAN_REF = {
    # (M, K, N): vulkan_time_ms  (interpolated from known data points)
    # From Phase 17B: M=512, K=1024, N=3072 => 34ms
    # Scaling: time proportional to M*K*N (for large enough shapes)
    # From hetero test: Vulkan handles N=1024 of N=3072 in ~11ms
}

def compile_if_needed(M, K, N, nc=1, tk=64, tn=64):
    wd = os.path.join(KC, f"M{M}_K{K}_N{N}_C{nc}_Tm64_Tk{tk}_Tn{tn}")
    xp = os.path.join(wd, "design.xclbin")
    ip = os.path.join(wd, "design.insts.bin")
    if os.path.exists(xp) and os.path.exists(ip):
        return xp, ip, 0
    cmd = [sys.executable, GEN, "--M", str(M), "--K", str(K), "--N", str(N),
           "--cols", str(nc), "--tile-m", "64", "--tile-k", str(tk), "--tile-n", str(tn), "--work-dir", wd]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    ct = time.time() - t0
    if r.returncode != 0 or not os.path.exists(xp):
        return None, None, 0
    return xp, ip, ct

def xrt_bench(xp, ip, M, K, N, ni=3):
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xp)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    in_sz=M*K; wt_sz=K*N; out_sz=M*N*4
    gid3=kern.group_id(3)
    in_bo=pyxrt.bo(dev,in_sz,pyxrt.bo.flags.host_only,gid3)
    wt_bo=pyxrt.bo(dev,wt_sz,pyxrt.bo.flags.host_only,gid3)
    out_bo=pyxrt.bo(dev,out_sz,pyxrt.bo.flags.host_only,gid3)
    isz=os.path.getsize(ip)
    gid1=kern.group_id(1)
    inst_bo=pyxrt.bo(dev,isz,pyxrt.bo.flags.cacheable,gid1)
    with open(ip,"rb") as f: inst_bo.write(f.read(),0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    rng=np.random.RandomState(42)
    inp=rng.randint(-127,128,size=(M,K),dtype=np.int8)
    wt=rng.randint(-127,128,size=(K,N),dtype=np.int8)
    # warmup
    for _ in range(1):
        in_bo.write(inp.tobytes(),0); wt_bo.write(wt.tobytes(),0)
        out_bo.write(b"\x00"*out_sz,0)
        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run=pyxrt.run(kern)
        run.set_arg(0,3); run.set_arg(1,inst_bo); run.set_arg(2,isz)
        run.set_arg(3,in_bo); run.set_arg(4,wt_bo); run.set_arg(5,out_bo)
        run.start(); run.wait()
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    times=[]
    for _ in range(ni):
        t0=time.time()
        in_bo.write(inp.tobytes(),0); wt_bo.write(wt.tobytes(),0)
        out_bo.write(b"\x00"*out_sz,0)
        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wt_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t1=time.time()
        run=pyxrt.run(kern)
        run.set_arg(0,3); run.set_arg(1,inst_bo); run.set_arg(2,isz)
        run.set_arg(3,in_bo); run.set_arg(4,wt_bo); run.set_arg(5,out_bo)
        run.start(); run.wait()
        t2=time.time()
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        t3=time.time()
        times.append(((t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000,(t3-t0)*1000))
    out=np.frombuffer(out_bo.read(out_sz,0),dtype=np.int32).reshape(M,N)
    cpu=inp.astype(np.int32) @ wt.astype(np.int32)
    exact=bool(np.array_equal(cpu,out))
    avg_tot=np.mean([t[3] for t in times])
    avg_kern=np.mean([t[1] for t in times])
    flops=2.0*M*K*N
    gops=flops/(avg_tot/1000)/1e9 if avg_tot>0 else 0
    del inst_bo, in_bo, wt_bo, out_bo, kern, ctx, xb, dev
    return {"total_ms":round(avg_tot,2),"kernel_ms":round(avg_kern,2),
            "gops":round(gops,2),"bit_exact":exact,"flops":flops}

def estimate_vulkan_time(M, K, N):
    """Estimate Vulkan time based on measured reference point.
    Reference: M=512, K=1024, N=3072 => 34ms for Q8_0 x F32
    Scaling: roughly linear with M*K*N for large shapes.
    """
    ref_M, ref_K, ref_N, ref_ms = 512, 1024, 3072, 34.0
    ref_ops = ref_M * ref_K * ref_N
    ops = M * K * N
    # Linear scaling with problem size
    return ref_ms * ops / ref_ops

def main():
    os.makedirs(OUT, exist_ok=True)
    capability = []

    # Scan M x K x N space
    M_vals = [64, 128, 256, 512]
    K_vals = [1024, 2048, 4096]
    N_vals = [512, 1024, 2048, 3072, 4096, 6144]

    print("Phase 18-J/K: Building XDNA Capability Model")
    print(f"Shapes to test: {len(M_vals)}x{len(K_vals)}x{len(N_vals)} = {len(M_vals)*len(K_vals)*len(N_vals)}")

    for M in M_vals:
        for K in K_vals:
            for N in N_vals:
                # Check AIE2 alignment
                if M % 4 or K % 8 or N % 8: continue
                # Check tile alignment
                if M % 64 or K % 64 or N % 64: continue

                print(f"  M={M} K={K} N={N}...", end="", flush=True)
                xp, ip, ct = compile_if_needed(M, K, N)
                if not xp:
                    print(" SKIP")
                    continue

                try:
                    perf = xrt_bench(xp, ip, M, K, N, ni=3)
                except Exception as e:
                    print(f" ERROR: {e}")
                    continue

                vk_est = estimate_vulkan_time(M, K, N)
                ratio = perf["total_ms"] / vk_est if vk_est > 0 else float("inf")

                cap = {
                    "M": M, "K": K, "N": N,
                    "xdna_total_ms": perf["total_ms"],
                    "xdna_kernel_ms": perf["kernel_ms"],
                    "xdna_gops": perf["gops"],
                    "bit_exact": perf["bit_exact"],
                    "vulkan_est_ms": round(vk_est, 2),
                    "xdna_vk_ratio": round(ratio, 3),
                    "compile_time_s": ct,
                }
                capability.append(cap)
                faster = "XDNA faster" if ratio < 1 else "VK faster"
                print(f" {perf['total_ms']:>7.1f}ms  VK~{vk_est:>7.1f}ms  ratio={ratio:.3f}  {faster}")

    # Save capability model
    jp = os.path.join(OUT, "phase18_capability_model.json")
    with open(jp, "w") as f:
        json.dump(capability, f, indent=2)
    print(f"\nJSON: {jp}")

    # Summary
    if capability:
        print(f"\n{'='*90}")
        print(f"XDNA CAPABILITY MODEL ({len(capability)} shapes)")
        print(f"{'='*90}")
        print(f"{'M':>5} {'K':>5} {'N':>5} {'XDNA':>8} {'VK_est':>8} {'ratio':>7} {'GOPS':>7} {'exact':>5}")
        for c in sorted(capability, key=lambda x: (x["M"], x["K"], x["N"])):
            faster = "*" if c["xdna_vk_ratio"] < 1 else " "
            print(f"{c['M']:>5} {c['K']:>5} {c['N']:>5} {c['xdna_total_ms']:>8.1f} "
                  f"{c['vulkan_est_ms']:>8.1f} {c['xdna_vk_ratio']:>7.3f} "
                  f"{c['xdna_gops']:>7.2f} {c['bit_exact']:>5} {faster}")

        # Find shapes where XDNA is faster
        faster_shapes = [c for c in capability if c["xdna_vk_ratio"] < 1]
        slower_shapes = [c for c in capability if c["xdna_vk_ratio"] >= 1]
        print(f"\nXDNA faster: {len(faster_shapes)}/{len(capability)} shapes")
        print(f"VK faster: {len(slower_shapes)}/{len(capability)} shapes")
        if faster_shapes:
            print("\nXDNA-faster shapes:")
            for c in sorted(faster_shapes, key=lambda x: x["xdna_vk_ratio"]):
                print(f"  M={c['M']} K={c['K']} N={c['N']}: ratio={c['xdna_vk_ratio']:.3f}")

if __name__ == "__main__":
    main()
