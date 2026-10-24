#!/usr/bin/env python3
"""Compare V1 vs V2 pipelined kernel via direct XRT benchmark."""
import sys, pyxrt, numpy as np, time, os, json

def bench_xclbin(xclbin_path, insts_path, label, M=512, K=1024, N=3072, nw=10, nr=50):
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xclbin_path)
    dev.register_xclbin(xb)
    with open(insts_path, "rb") as f:
        id_ = f.read()
    ib = len(id_)
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
        inb.write(inp_i8.tobytes(), 0)
        wtb.write(wgt_i8.tobytes(), 0)
        otb.write(b'\x00'*(M*N*4), 0)
        inb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        wtb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        otb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        r = pyxrt.run(kern)
        r.set_arg(0,3); r.set_arg(1,ibo); r.set_arg(2,ib)
        r.set_arg(3,inb); r.set_arg(4,wtb); r.set_arg(5,otb)
        t0 = time.perf_counter()
        r.start(); r.wait()
        t1 = time.perf_counter()
        otb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        result = np.frombuffer(otb.read(M*N*4,0), dtype=np.int32).reshape(M,N).copy()
        return result, (t1-t0)*1000

    for _ in range(nw):
        run_one()

    check = None
    kt = []
    for i in range(nr):
        res, k = run_one()
        kt.append(k)
        if i == 0:
            check = res

    kt_arr = np.array(kt)
    max_err = int(np.max(np.abs(check - ref)))
    n_mismatch = int(np.sum(check != ref))
    flops = 2.0 * M * K * N
    gops = flops / (np.mean(kt_arr) * 1e-3) / 1e9

    print(f"  {label}: {np.mean(kt_arr):.2f} ms, {gops:.2f} GOPS, "
          f"err={max_err}, mismatch={n_mismatch}/{M*N}")

    del kern, hw, dev
    return {"label": label, "ms": float(np.mean(kt_arr)), "p50": float(np.percentile(kt_arr, 50)),
            "p95": float(np.percentile(kt_arr, 95)), "GOPS": gops, "max_err": max_err,
            "correct": max_err == 0}


M, K, N = 512, 1024, 3072
V1_XCLBIN = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3072_C1_Tm64_Tk64_Tn64/design.xclbin"
V1_INSTS = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3072_C1_Tm64_Tk64_Tn64/design.insts.bin"
V2_XCLBIN = "/Storage/Cache/llama.cpp/experiments/phase20/v2_pipe.prj/design.xclbin"
V2_INSTS = "/Storage/Cache/llama.cpp/experiments/phase20/v2_pipe.prj/design.insts.bin"

print(f"V1 vs V2 Pipelined comparison")
print(f"V1 insts: {os.path.getsize(V1_INSTS)} bytes")
print(f"V2 insts: {os.path.getsize(V2_INSTS)} bytes")

results = []
r1 = bench_xclbin(V1_XCLBIN, V1_INSTS, "V1 baseline")
results.append(r1)

# V2 needs new device context (hw_context conflict)
r2 = bench_xclbin(V2_XCLBIN, V2_INSTS, "V2 pipelined")
results.append(r2)

print(f"\nSpeedup: {r2['GOPS']/r1['GOPS']:.3f}x")

with open("/Storage/Cache/llama.cpp/experiments/phase20/phase20_v2_pipe.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to phase20_v2_pipe.json")
