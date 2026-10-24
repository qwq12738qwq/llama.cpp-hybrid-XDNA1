import os
import time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
from aie.iron.controlflow import range_ as iron_range

NUM_TRANSFERS = 1
NUM_ELEMS = 256

KERNEL_SRC = r'''
extern "C" void dma_passthrough(const int32_t *in, int32_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = in[i];
}
'''

def build():
    fused = ExternalFunction(
        "dma_passthrough",
        source_string=KERNEL_SRC,
        arg_types=[
            np.ndarray[(NUM_ELEMS,), np.dtype[np.int32]],
            np.ndarray[(NUM_ELEMS,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    in_ty = np.ndarray[(NUM_ELEMS,), np.dtype[np.int32]]
    out_ty = np.ndarray[(NUM_ELEMS,), np.dtype[np.int32]]

    fifo_in = iron.ObjectFifo(in_ty, name="in_fifo", depth=1)
    fifo_out = iron.ObjectFifo(out_ty, name="out_fifo", depth=1)

    def core_fn(of_in, of_out, kern, n):
        for _ in iron_range(NUM_TRANSFERS):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kern(elem_in, elem_out, n)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_fn, [fifo_in.cons(), fifo_out.prod(), fused, NUM_ELEMS])

    def sequence(A, C, _2, in_prod, out_cons):
        tg = iron.TaskGroup()
        for i in range(NUM_TRANSFERS):
            in_prod.fill(A, group=tg)
            out_cons.drain(C, group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [in_ty, out_ty, in_ty, fifo_in.prod(), fifo_out.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

xp = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/dma_microbench/approach_A/t1/design.xclbin"
ip = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/dma_microbench/approach_A/t1/design.insts.bin"
pp = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/dma_microbench/approach_A/t1/main.pdi"

@iron.jit
def design(A: In, C: Out, *, dummy: CompileTime[int] = 0):
    return build()

print("[DESIGN-A] Generating and compiling...")
t0 = time.time()
design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
print("[DESIGN-A] Compiled in %.1fs" % (time.time() - t0))