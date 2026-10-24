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
with open(q8v_ins, "rb") as f:
    insts = np.frombuffer(f.read(), dtype=np.uint32)

inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
inst_map = np.frombuffer(inst_bo.map(), dtype=np.uint32)
inst_map[:] = insts
inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

a_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.host_only, gid3)
b_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.host_only, gid3)
c_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.host_only, gid3)

run = pyxrt.run(kernel)
run.set_arg(0, np.uint64(3))
run.set_arg(1, inst_bo)
run.set_arg(2, np.uint32(inst_bytes))
run.set_arg(3, a_bo)
run.set_arg(4, b_bo)
run.set_arg(5, c_bo)
print("run configured")

print("starting run...")
t0 = time.perf_counter()
run.start()
print("run started")

print("waiting...")
run.wait()
t1 = time.perf_counter()
print(f"run completed in {(t1-t0)*1000:.2f}ms")
