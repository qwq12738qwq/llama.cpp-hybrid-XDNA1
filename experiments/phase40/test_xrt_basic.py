import sys
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase40")
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase39")
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase20g")

import numpy as np
import os
import time
import pyxrt

M, K, N = 512, 1024, 3584
cache_dir = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

q8v_xc = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.xclbin"
q8v_ins = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.insts.bin"

print(f"xclbin exists: {os.path.exists(q8v_xc)}")
print(f"insts exists: {os.path.exists(q8v_ins)}")

if not os.path.exists(q8v_xc):
    sys.exit(1)

print("loading xclbin...")
t0 = time.perf_counter()
xclbin = pyxrt.xclbin(q8v_xc)
uuid = xclbin.get_uuid()
device = pyxrt.device(0)
device.register_xclbin(xclbin)
hwctx = pyxrt.hw_context(device, uuid)
kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
t1 = time.perf_counter()
print(f"xclbin loaded in {(t1-t0)*1000:.2f}ms")

gid1 = kernel.group_id(1)
gid3 = kernel.group_id(3)
print(f"gid1={gid1}, gid3={gid3}")

inst_bytes = os.path.getsize(q8v_ins)
print(f"inst_bytes={inst_bytes}")

with open(q8v_ins, "rb") as f:
    insts = np.frombuffer(f.read(), dtype=np.uint32)
print(f"insts loaded: {len(insts)} u32s")

inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
inst_map = np.frombuffer(inst_bo.map(), dtype=np.uint32)
inst_map[:] = insts
inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
print("inst_bo synced")

a_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.host_only, gid3)
b_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.host_only, gid3)
c_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.host_only, gid3)
print("BOs created")

run = pyxrt.run(kernel)
run.set_arg(0, np.uint64(3))
run.set_arg(1, inst_bo)
run.set_arg(2, np.uint32(inst_bytes))
run.set_arg(3, a_bo)
run.set_arg(4, b_bo)
run.set_arg(5, c_bo)
print("run configured")

t0 = time.perf_counter()
run.start()
print("run started")
run.wait()
t1 = time.perf_counter()
print(f"run.wait() completed in {(t1-t0)*1000:.2f}ms")
