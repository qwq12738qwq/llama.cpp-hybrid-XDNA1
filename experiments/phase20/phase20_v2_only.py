#!/usr/bin/env python3
"""Test V2 pipelined kernel only."""
import pyxrt, numpy as np, time, os, json

M, K, N = 512, 1024, 3072
V2_XCLBIN = "/Storage/Cache/llama.cpp/experiments/phase20/v2_pipe.prj/design.xclbin"
V2_INSTS = "/Storage/Cache/llama.cpp/experiments/phase20/v2_pipe.prj/design.insts.bin"

dev = pyxrt.device(0)
xb = pyxrt.xclbin(V2_XCLBIN)
dev.register_xclbin(xb)
with open(V2_INSTS, "rb") as f:
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

inb = pyxrt.bo(dev, max(M*K,64), pyxrt.bo.flags.host_only, g3)
wtb = pyxrt.bo(dev, max(K*N,64), pyxrt.bo.flags.host_only, g3)
otb = pyxrt.bo(dev, max(M*N*4,64), pyxrt.bo.flags.host_only, g3)

inb.write(inp_i8.tobytes(), 0)
wtb.write(wgt_i8.tobytes(), 0)
otb.write(b'\x00'*(M*N*4), 0)
inb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
wtb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
otb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

# Warmup
print("Warming up...")
for _ in range(3):
    r = pyxrt.run(kern)
    r.set_arg(0,3); r.set_arg(1,ibo); r.set_arg(2,ib)
    r.set_arg(3,inb); r.set_arg(4,wtb); r.set_arg(5,otb)
    r.start(); r.wait()

# Benchmark
print("Benchmarking V2 pipelined...")
kt = []
for i in range(20):
    otb.write(b'\x00'*(M*N*4), 0)
    otb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    r = pyxrt.run(kern)
    r.set_arg(0,3); r.set_arg(1,ibo); r.set_arg(2,ib)
    r.set_arg(3,inb); r.set_arg(4,wtb); r.set_arg(5,otb)
    t0 = time.perf_counter()
    r.start(); r.wait()
    t1 = time.perf_counter()
    kt.append((t1-t0)*1000)

otb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
got = np.frombuffer(otb.read(M*N*4,0), dtype=np.int32).reshape(M,N).copy()

max_err = int(np.max(np.abs(got - ref)))
n_mismatch = int(np.sum(got != ref))

kt_arr = np.array(kt)
flops = 2.0 * M * K * N
gops = flops / (np.mean(kt_arr) * 1e-3) / 1e9

print(f"V2 Pipelined M={M} K={K} N={N}")
print(f"  Correctness: max_err={max_err}, mismatches={n_mismatch}/{M*N}")
print(f"  Kernel: {np.mean(kt_arr):.2f} ms (p50={np.percentile(kt_arr,50):.2f}, p95={np.percentile(kt_arr,95):.2f})")
print(f"  GOPS:   {gops:.2f}")
print(f"  V1 was: 53.96 GOPS -> speedup: {gops/53.96:.2f}x")

result = {"label": "V2 pipelined", "ms": float(np.mean(kt_arr)),
          "GOPS": gops, "max_err": max_err, "correct": max_err == 0}
with open("/Storage/Cache/llama.cpp/experiments/phase20/phase20_v2_pipe.json", "w") as f:
    json.dump(result, f, indent=2)
print("Saved.")
