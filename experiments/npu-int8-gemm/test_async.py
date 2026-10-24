#!/usr/bin/env python3
"""
Test XDNA async execution with N-parallel kernel
"""

import sys
import os
import numpy as np

sys.path.insert(0, '/workspace/llama.cpp')

# Use the persistent engine directly
from ctypes import CDLL, c_int, c_char_p, c_void_p, c_int8, c_int32, POINTER, byref

# Load the XDNA library
import ctypes as _ct
_B = '/workspace/llama.cpp/build-vk/bin/'
# libggml-xdna.so needs ggml_backend_* from libggml-base; pull deps in globally first
for _dep in ('libggml-base.so', 'libggml.so'):
    try:
        _ct.CDLL(_B + _dep, mode=_ct.RTLD_GLOBAL)
    except OSError:
        pass
libxdna = CDLL(_B + 'libggml-xdna.so', mode=_ct.RTLD_GLOBAL)

# Define function signatures
libxdna.ggml_xdna_persistent_engine_create.argtypes = [c_int, c_int, c_int, c_int]
libxdna.ggml_xdna_persistent_engine_create.restype = c_void_p

libxdna.ggml_xdna_persistent_engine_init_xrt_c.argtypes = [c_void_p, c_char_p, c_char_p]
libxdna.ggml_xdna_persistent_engine_init_xrt_c.restype = c_int

libxdna.ggml_xdna_persistent_engine_submit_c.argtypes = [c_void_p, POINTER(c_int8), POINTER(c_int8), POINTER(c_int32), POINTER(c_void_p)]
libxdna.ggml_xdna_persistent_engine_submit_c.restype = c_int

libxdna.ggml_xdna_persistent_engine_wait_c.argtypes = [c_void_p, c_void_p, POINTER(c_int32)]
libxdna.ggml_xdna_persistent_engine_wait_c.restype = c_int

libxdna.ggml_xdna_persistent_engine_free.argtypes = [c_void_p]

def test_n_parallel():
    """Test N-parallel async execution"""
    M, K, N = 64, 1024, 64
    m_tile, k_tile, n_tile = 64, 64, 64
    n_cols = 1
    
    print(f"Testing M={M}, K={K}, N={N}, n_cols={n_cols}")
    
    # Create engine
    engine = libxdna.ggml_xdna_persistent_engine_create(M, K, N, 0)  # full-K: engine->k must equal K
    if not engine:
        print("FAIL: Failed to create engine")
        return False
    
    # Initialize XRT
    xclbin = b"/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M64_K1024_N64/design.xclbin"
    insts = b"/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M64_K1024_N64/design.insts.bin"
    
    if not libxdna.ggml_xdna_persistent_engine_init_xrt_c(engine, xclbin, insts):
        print("FAIL: Failed to init XRT")
        libxdna.ggml_xdna_persistent_engine_free(engine)
        return False
    
    print("Engine initialized successfully")
    
    # Generate test data
    np.random.seed(42)
    input_data = np.random.randint(-128, 127, size=(M, K), dtype=np.int8)
    weight_data = np.random.randint(-128, 127, size=(K, N), dtype=np.int8)
    
    # CPU reference
    output_cpu = np.zeros((M, N), dtype=np.int32)
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(K):
                acc += int(input_data[m, k]) * int(weight_data[k, n])
            output_cpu[m, n] = acc
    
    # Prepare output buffer
    output_xdna = np.zeros((M, N), dtype=np.int32)
    
    # Submit async
    run_ptr = c_void_p()
    input_flat = input_data.ravel()
    weight_flat = weight_data.ravel()
    output_flat = output_xdna.ravel()
    
    print("Submitting async...")
    ret = libxdna.ggml_xdna_persistent_engine_submit_c(
        engine,
        input_flat.ctypes.data_as(POINTER(c_int8)),
        weight_flat.ctypes.data_as(POINTER(c_int8)),
        output_flat.ctypes.data_as(POINTER(c_int32)),
        byref(run_ptr)
    )
    
    if not ret:
        print("FAIL: Submit failed")
        libxdna.ggml_xdna_persistent_engine_free(engine)
        return False
    
    print(f"Submitted, run_ptr={run_ptr.value}")
    
    # Wait for completion
    print("Waiting...")
    ret = libxdna.ggml_xdna_persistent_engine_wait_c(engine, run_ptr, output_flat.ctypes.data_as(POINTER(c_int32)))
    
    if not ret:
        print("FAIL: Wait failed")
        libxdna.ggml_xdna_persistent_engine_free(engine)
        return False
    
    print("Completed")
    
    # Verify
    mismatches = np.sum(output_xdna != output_cpu)
    if mismatches == 0:
        print("PASS: Output matches CPU reference")
    else:
        print(f"FAIL: {mismatches} mismatches")
        print(f"  output_xdna[:5, :5] = {output_xdna[:5, :5]}")
        print(f"  output_cpu[:5, :5] = {output_cpu[:5, :5]}")
    
    libxdna.ggml_xdna_persistent_engine_free(engine)
    return mismatches == 0

if __name__ == "__main__":
    success = test_n_parallel()
    sys.exit(0 if success else 1)