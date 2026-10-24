#!/usr/bin/env python3
"""Phase 18: XDNA capability scan - one M value per invocation to avoid XRT context leak."""
import json, os, sys, time, subprocess
import numpy as np

try:
    import pyxrt
except ImportError:
    sys.exit("ERROR: pyxrt not available")

KC = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"
GEN = "/workspace/llama.cpp/experiments/npu-int8-gemm/gen_xdna_gemm.py"
OUT = "/home/ubuntu12738/phase18"

def compile_if_needed(M, K, N, nc=1, tk=64):
    wd = os.path.join(KC, f"M{M}_K{K}_N{N}_C{nc}_Tm64_Tk{tk}_Tn64")
    xp = os.path.join(wd, "design.xclbin")
    ip = os.path.join(wd, "design.insts.bin")
    if os.path.exists(xp) and os.path.exists(ip):
        return xp, ip, 0
    cmd = [sys.executable, GEN, "--M", str(M), "--K", str(K), "--N", str(N),
           "--cols", str(nc), "--tile-m", "64", "--tile-k", str(tk), "--tile-n", "64", "--work-dir", wd]
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
    return {"total_ms":round(avg_tot,2),"kernel_ms":round(avg_kern,2),
            "gops":round(gops,2),"bit_exact":exact}

def estimate_vk(M, K, N):
    return 34.0 * (M*K*N) / (512*1024*3072)

def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    os.makedirs(OUT, exist_ok=True)
    capability = []
    K_vals = [1024, 2048, 4096]
    N_vals = [512, 1024, 2048, 3072, 4096, 6144]

    print(f"Phase 18 Capability Scan: M={M}")
    for K in K_vals:
        for N in N_vals:
            if M % 4 or K % 8 or N % 8: continue
            if M % 64 or K % 64 or N % 64: continue
            print(f"  M={M} K={K} N={N}...", end="", flush=True)
            xp, ip, ct = compile_if_needed(M, K, N)
            if not xp: print(" SKIP"); continue
            try:
                perf = xrt_bench(xp, ip, M, K, N, ni=3)
            except Exception as e:
                print(f" ERROR: {e}"); continue
            vk = estimate_vk(M, K, N)
            ratio = perf["total_ms"] / vk if vk > 0 else 999
            cap = {"M":M,"K":K,"N":N,"xdna_ms":perf["total_ms"],"vk_ms":round(vk,2),
                   "ratio":round(ratio,3),"gops":perf["gops"],"exact":perf["bit_exact"]}
            capability.append(cap)
            faster = "XDNA<" if ratio < 1 else "VK<"
            print(f" {perf['total_ms']:>7.1f}ms  VK~{vk:>7.1f}ms  ratio={ratio:.3f}  {faster}")
            sys.stdout.flush()

    jp = os.path.join(OUT, f"phase18_cap_M{M}.json")
    with open(jp, "w") as f: json.dump(capability, f, indent=2)
    print(f"\nSaved: {jp}")
    # Also append to master
    mp = os.path.join(OUT, "phase18_capability_model.json")
    existing = []
    if os.path.exists(mp):
        with open(mp) as f: existing = json.load(f)
    existing.extend(capability)
    with open(mp, "w") as f: json.dump(existing, f, indent=2)

if __name__ == "__main__":
    main()
