#!/usr/bin/env python3
"""Phase 20: Single-shape benchmark (run per-shape to avoid hw_context conflict)."""
import sys, pyxrt, numpy as np, time, os, json

XCLBIN_DIR = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

def bench(M, K, N, n_cols=1, tm=64, tk=64, tn=64, nw=10, nr=50):
    base = f"{XCLBIN_DIR}/M{M}_K{K}_N{N}_C{n_cols}_Tm{tm}_Tk{tk}_Tn{tn}"
    xp = f"{base}/design.xclbin"
    ip = f"{base}/design.insts.bin"
    if not os.path.exists(xp):
        return None

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xp)
    dev.register_xclbin(xb)
    ib = os.path.getsize(ip)
    with open(ip, "rb") as f:
        id_ = f.read()
    hw = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(hw, "MLIR_AIE")
    g1, g3 = kern.group_id(1), kern.group_id(3)
    ibo = pyxrt.bo(dev, ib, pyxrt.bo.flags.cacheable, g1)
    ibo.write(id_, 0)
    ibo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    np.random.seed(42)
    inp_i8 = np.random.randint(-127, 128, size=M*K, dtype=np.int8)
    wgt_i8 = np.random.randint(-127, 128, size=K*N, dtype=np.int8)
    ref = (inp_i8.reshape(M,K).astype(np.int32) @ wgt_i8.reshape(K,N).astype(np.int32)).astype(np.int32)

    def run_one():
        inb = pyxrt.bo(dev, max(M*K,64), pyxrt.bo.flags.host_only, g3)
        wtb = pyxrt.bo(dev, max(K*N,64), pyxrt.bo.flags.host_only, g3)
        otb = pyxrt.bo(dev, max(M*N*4,64), pyxrt.bo.flags.host_only, g3)
        t0 = time.perf_counter()
        inb.write(inp_i8.tobytes(), 0)
        wtb.write(wgt_i8.tobytes(), 0)
        otb.write(b'\x00'*(M*N*4), 0)
        inb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wtb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        otb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t1 = time.perf_counter()
        r = pyxrt.run(kern)
        r.set_arg(0,3); r.set_arg(1,ibo); r.set_arg(2,ib)
        r.set_arg(3,inb); r.set_arg(4,wtb); r.set_arg(5,otb)
        r.start(); r.wait()
        t2 = time.perf_counter()
        otb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        result = np.frombuffer(otb.read(M*N*4,0), dtype=np.int32).copy()
        t3 = time.perf_counter()
        return result, (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000

    for _ in range(nw):
        run_one()
    kt, tt, check = [], [], None
    for i in range(nr):
        res, up, k, dl = run_one()
        kt.append(k); tt.append(up+k+dl)
        if i == 0: check = res

    err = int(np.max(np.abs(check - ref.flatten())))
    ka = np.array(kt); ta = np.array(tt)
    flops = 2.0*M*K*N; bmove = M*K+K*N+M*N*4
    return {
        "M":M,"K":K,"N":N,"n_cols":n_cols,
        "correct":err==0,"max_err":err,
        "kernel_ms":float(np.mean(ka)),"kernel_p50":float(np.percentile(ka,50)),
        "kernel_p95":float(np.percentile(ka,95)),
        "total_ms":float(np.mean(ta)),
        "GOPS_kernel":flops/(np.mean(ka)*1e-3)/1e9,
        "GOPS_total":flops/(np.mean(ta)*1e-3)/1e9,
        "FLOPs":flops,"bytes_moved":bmove,
        "AI":flops/bmove,
        "DMA_GB_s":bmove/(np.mean(ka)*1e-3)/1e9,
    }

if __name__ == "__main__":
    M,K,N = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
    nc = int(sys.argv[4]) if len(sys.argv)>4 else 1
    tm = int(sys.argv[5]) if len(sys.argv)>5 else 64
    tk = int(sys.argv[6]) if len(sys.argv)>6 else 64
    tn = int(sys.argv[7]) if len(sys.argv)>7 else 64
    r = bench(M,K,N,nc,tm,tk,tn)
    if r:
        print(json.dumps(r))
    else:
        print('{"error":"kernel not found"}')
