import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Runtime, Worker, ExternalFunction
from aie.utils import config

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void tconv(const int64_t *a, float *c) {
    aie::vector<int64_t,8> av = aie::load_v<8>(a);
    aie::vector<float,8> cv = aie::to_float(av);
    aie::store_v(c, cv);
}
"""

def build():
    kern = ExternalFunction("tconv", source_string=KERNEL_SRC,
        arg_types=[np.ndarray[(8,),np.dtype[np.int64]], np.ndarray[(8,),np.dtype[np.float32]]],
        include_dirs=[config.cxx_header_path()])
    a_ty=np.ndarray[(8,),np.dtype[np.int64]]; c_ty=np.ndarray[(8,),np.dtype[np.float32]]
    def core_fn(of_a,of_c,kern):
        ea=of_a.acquire(1); ec=of_c.acquire(1); kern(ea,ec); of_a.release(1); of_c.release(1)
    fa=ObjectFifo(a_ty,name="a",depth=1); fc=ObjectFifo(c_ty,name="c",depth=1)
    w=Worker(core_fn,[fa.cons(),fc.prod(),kern])
    A_ty=np.ndarray[(8,),np.dtype[np.int64]]; C_ty=np.ndarray[(8,),np.dtype[np.float32]]
    def sequence(A,C,*fifos):
        ap=fifos[0]; cp=fifos[1]; ap.fill(A); cp.drain(C,wait=True)
    rt=Runtime(sequence,[A_ty,C_ty,fa.prod(),fc.cons()])
    return iron.Program(iron.get_current_device(),rt,workers=[w]).resolve_program()

iron.set_current_device(iron.device.from_name("npu",n_cols=4))
prog=build()
print("BUILD OK")
