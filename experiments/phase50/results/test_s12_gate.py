import os, sys, time
import numpy as np
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase50")
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase40")
from gen_q8_c4_mmul_v2 import ref_q8_gemm, M_TILE, N_TILE, KC, NBLK
import gen_s10  # pack functions (same layout as s12)

M, K, N, n_cols = 512, 1024, 3584, 4
MA, NA = M_TILE//4, N_TILE//8
n_mt, n_nt_c, n_kc = M//M_TILE, (N//n_cols)//N_TILE, K//KC

rng = np.random.default_rng(42)
A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
As = rng.uniform(0.001, 0.02, size=(M, K//32)).astype(np.float32)
Bs = rng.uniform(0.001, 0.02, size=(K//32, N)).astype(np.float32)
C_ref = ref_q8_gemm(A_i8, As, B_i8, Bs)

import pyxrt
xp = "/tmp/p50_s12/design.xclbin"
ip = xp.replace(".xclbin", ".insts.bin")
Ap = gen_s10.pack_A_s10(A_i8, As, M, K)
Bp = gen_s10.pack_B_s10(B_i8, Bs, K, N)
xclbin = pyxrt.xclbin(xp)
device = pyxrt.device(0)
device.register_xclbin(xclbin)
hwctx = pyxrt.hw_context(device, xclbin.get_uuid())
kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
gid1, gid3 = kernel.group_id(1), kernel.group_id(3)
inst_bytes = os.path.getsize(ip)
with open(ip, "rb") as f: insts = np.frombuffer(f.read(), dtype=np.uint32)
inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
np.frombuffer(inst_bo.map(), dtype=np.uint32)[:] = insts
inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
a_bo = pyxrt.bo(device, Ap.nbytes, pyxrt.bo.flags.host_only, gid3)
b_bo = pyxrt.bo(device, Bp.nbytes, pyxrt.bo.flags.host_only, gid3)
c_bo = pyxrt.bo(device, M*N*4, pyxrt.bo.flags.host_only, gid3)
np.frombuffer(a_bo.map(), dtype=np.uint8)[:] = np.frombuffer(np.ascontiguousarray(Ap), dtype=np.uint8)
np.frombuffer(b_bo.map(), dtype=np.uint8)[:] = np.frombuffer(np.ascontiguousarray(Bp), dtype=np.uint8)
a_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
b_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

def run_once():
    c_map = np.frombuffer(c_bo.map(), dtype=np.float32)
    c_map[:] = 0
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t0 = time.perf_counter()
    run = pyxrt.run(kernel)
    run.set_arg(0, np.uint64(3)); run.set_arg(1, inst_bo); run.set_arg(2, np.uint32(inst_bytes))
    run.set_arg(3, a_bo); run.set_arg(4, b_bo); run.set_arg(5, c_bo)
    run.start(); run.wait()
    t_wait = (time.perf_counter()-t0)*1000
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    scratch = np.frombuffer(c_bo.map(), dtype=np.float32).copy()
    # host scatter: scratch[g*2048 + i*256 + j*32 + rr*8+lane] -> C[mt*32+i*4+rr][ntb+j*8+lane]
    t0 = time.perf_counter()
    C_dev = np.zeros((M, N), np.float32)
    for mt in range(n_mt):
        for col in range(n_cols):
            for nt in range(n_nt_c):
                g = (mt*n_cols*n_nt_c + col*n_nt_c + nt)
                tile = scratch[g*2048:(g+1)*2048].reshape(MA, NA, 4, 8)
                ntb = (col*n_nt_c + nt)*N_TILE
                for i in range(MA):
                    for j in range(NA):
                        for rr in range(4):
                            C_dev[mt*32+i*4+rr, ntb+j*8:ntb+j*8+8] = tile[i, j, rr, :]
    t_scat = (time.perf_counter()-t0)*1000
    return C_dev, t_wait, t_scat

for _ in range(2): run_once()  # warmup
waits, scats, errs = [], [], []
for it in range(5):
    C_dev, tw, ts = run_once()
    d = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
    waits.append(tw); scats.append(ts); errs.append(float(d.max()))
    print(f"iter={it} wait={tw:.1f}ms scatter={ts:.1f}ms total={tw+ts:.1f}ms max_err={d.max():.3e}")
print(f"WAIT med={sorted(waits)[2]:.1f} SCATTER med={sorted(scats)[2]:.1f} TOTAL med={sorted(waits)[2]+sorted(scats)[2]:.1f} max_err={max(errs):.3e}")
