import pyxrt
import numpy as np
import os
import time

M, K, N = 512, 1024, 3584
cache_dir = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

q8v_xc = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.xclbin"
q8v_ins = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.insts.bin"

print(f"xclbin exists: {os.path.exists(q8v_xc)}")
print(f"insts exists: {os.path.exists(q8v_ins)}")

print("creating device...")
device = pyxrt.device(0)
print(f"device: {device}")

print("loading xclbin...")
xclbin = pyxrt.xclbin(q8v_xc)
print(f"xclbin loaded")

print("getting uuid...")
uuid = xclbin.get_uuid()
print(f"uuid: {uuid}")

print("registering xclbin...")
device.register_xclbin(xclbin)
print("xclbin registered")

print("creating hw context...")
hwctx = pyxrt.hw_context(device, uuid)
print(f"hwctx: {hwctx}")

print("creating kernel...")
kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
print(f"kernel: {kernel}")

print("getting group ids...")
gid1 = kernel.group_id(1)
gid3 = kernel.group_id(3)
print(f"gid1={gid1}, gid3={gid3}")
