import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Runtime, Worker, ExternalFunction, In, Out, CompileTime
from aie.utils import config

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void tint64(const int8_t *a, int64_t *c) {
    aie::vector<int8_t,8> av = aie::load_v<8>(a);
    aie::vector<int64_t,8> cv = aie::to_vector<int64_t>(av);
    aie::store_v(c, cv);
}
"""

def main():
    iron.set_current_device(iron.device.from_name("npu", n_cols=4))
    kern = ExternalFunction("tint64", source_string=KERNEL_SRC,
        arg_types=[np.ndarray[(8,),np.dtype[np.int8]], np.ndarray[(8,),np.dtype[np.int64]]],
        include_dirs=[config.cxx_header_path()])
    a_ty=np.ndarray[(8,),np.dtype[np.int8]]; c_ty=np.ndarray[(8,),np.dtype[np.int64]]
    def core_fn(of_a,of_c,kern):
        ea=of_a.acquire(1); ec=of_c.acquire(1); kern(ea,ec); of_a.release(1); of_c.release(1)
    fa=ObjectFifo(a_ty,name="a",depth=1); fc=ObjectFifo(c_ty,name="c",depth=1)
    w=Worker(core_fn,[fa.cons(),fc.prod(),kern])
    A_ty=np.ndarray[(8,),np.dtype[np.int8]]; C_ty=np.ndarray[(8,),np.dtype[np.int64]]
    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int]=0):
        pass
    # build manually
    def sequence(A,C,*fifos):
        ap=fifos[0]; cp=fifos[1]; ap.fill(A); cp.drain(C,wait=True)
    rt=Runtime(sequence,[A_ty,C_ty,fa.prod(),fc.cons()])
    prog=iron.Program(iron.get_current_device(),rt,workers=[w])
    xp="/tmp/p51_int64test.xclbin"; ip="/tmp/p51_int64test.insts"
    compiled=prog.resolve_program()  # not enough
    print("resolve OK, now compile via aiecc not done")
    # Use design.compile path instead:
PYEOF
echo "skip"