import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Runtime, Worker, ExternalFunction, In, Out, CompileTime
from aie.utils import config

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void tint64mul(const int32_t *a, const int32_t *b, int64_t *c) {
    aie::vector<int32_t,8> av = aie::load_v<8>(a);
    aie::vector<int32_t,8> bv = aie::load_v<8>(b);
    aie::vector<int64_t,8> cv = aie::mul(av, bv);  // may not exist
    aie::store_v(c, cv);
}
"""

def build():
    kern = ExternalFunction("tint64mul", source_string=KERNEL_SRC,
        arg_types=[np.ndarray[(8,),np.dtype[np.int32]], np.ndarray[(8,),np.dtype[np.int32]], np.ndarray[(8,),np.dtype[np.int64]]],
        include_dirs=[config.cxx_header_path()])
    a_ty=np.ndarray[(8,),np.dtype[np.int32]]; b_ty=np.ndarray[(8,),np.dtype[np.int32]]; c_ty=np.ndarray[(8,),np.dtype[np.int64]]
    def core_fn(of_a,of_b,of_c,kern):
        ea=of_a.acquire(1); eb=of_b.acquire(1); ec=of_c.acquire(1); kern(ea,eb,ec); of_a.release(1); of_b.release(1); of_c.release(1)
    fa=ObjectFifo(a_ty,name="a",depth=1); fb=ObjectFifo(b_ty,name="b",depth=1); fc=ObjectFifo(c_ty,name="c",depth=1)
    w=Worker(core_fn,[fa.cons(),fb.cons(),fc.prod(),kern])
    A_ty=np.ndarray[(8,),np.dtype[np.int32]]; B_ty=np.ndarray[(8,),np.dtype[np.int32]]; C_ty=np.ndarray[(8,),np.dtype[np.int64]]
    def sequence(A,B,C,*fifos):
        ap=fifos[0]; bp=fifos[1]; cp=fifos[2]
        ap.fill(A); bp.fill(B); cp.drain(C,wait=True)
    rt=Runtime(sequence,[A_ty,B_ty,C_ty,fa.prod(),fb.prod(),fc.cons()])
    return iron.Program(iron.get_current_device(),rt,workers=[w]).resolve_program()

iron.set_current_device(iron.device.from_name("npu",n_cols=4))
prog=build()
print("BUILD OK")
