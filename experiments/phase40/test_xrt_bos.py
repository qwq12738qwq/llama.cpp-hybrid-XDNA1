import pyxrt
import numpy as np
import os
import time

M, K, N = 512, 1024, 3584
cache_dir = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

q8v_xc = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.xclbin"
q8v_ins = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.insts.bin"

device = pyxrt.device(0)
xclbin = pyxrt.xclbin(q8v_xc)
uuid = xclbin.get_uuid()
device.register_xclbin(xclbin)
hwctx = pyxrt.hw_context(device, uuid)
kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
gid1 = kernel.group_id(1)
gid3 = kernel.group_id(3)

inst_bytes = os.path.getsize(q8v_ins)
print(f"inst_bytes={inst_bytes}")

with open(q8v_ins, "rb") as f:
    insts = np.frombuffer(f.read(), dtype=np.uint32)
print(f"insts loaded: {len(insts)} u32s")

print("creating inst_bo...")
inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
print(f"inst_bo: {inst_bo}")

print("mapping inst_bo...")
inst_map = np.frombuffer(inst_bo.map(), dtype=np.uint32)
print(f"inst_map: {len(inst_map)} u32s")

print("copying insts...")
inst_map[:] = insts
print("insts copied")

print("syncing inst_bo...")
inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
print("inst_bo synced")

print("creating a_bo...")
a_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.host_only, gid3)
print(f"a_bo: {a_bo}")

print("creating b_bo...")
b_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.host_only, gid3)
print(f"b_bo: {b_bo}")

print("creating c_bo...")
c_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.host_only, gid3)
print(f"c_bo: {c_bo}")

print("all BOs created")
