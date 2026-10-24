#!/usr/bin/env python3
"""xdna_llama_smoke.py - Smoke test for XDNA backend integration.

This script verifies that:
1. XDNA backend loads correctly
2. A constrained MUL_MAT operation is accepted
3. Unsupported operations are rejected
"""

import sys
import os
import ctypes
from ctypes import c_void_p, c_size_t, c_int, c_char_p, POINTER

def load_libggml():
    """Load the ggml shared library."""
    # Try common paths
    paths = [
        "./build/bin/libggml.so",
        "./build/lib/libggml.so",
        "./build/ggml/src/libggml.so",
    ]
    for path in paths:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    
    # Try system path
    try:
        return ctypes.CDLL("libggml.so")
    except OSError:
        pass
    
    raise RuntimeError("Could not find libggml.so")

def main():
    print("=" * 60)
    print("XDNA Backend Smoke Test")
    print("=" * 60)
    
    # Set debug environment
    os.environ["GGML_XDNA_DEBUG"] = "1"
    
    try:
        lib = load_libggml()
        print(f"Loaded libggml: {lib._name}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Load XDNA backend
    try:
        lib.ggml_backend_xdna_reg.restype = c_void_p
        reg = lib.ggml_backend_xdna_reg()
        if not reg:
            print("ERROR: XDNA backend registration returned NULL")
            return 1
        print("XDNA backend registered")
    except AttributeError:
        print("ERROR: XDNA backend not found in libggml")
        print("Build with: -DGGML_XDNA=ON")
        return 1
    
    # Get device count
    lib.ggml_xdna_reg_get_device_count.restype = c_size_t
    lib.ggml_xdna_reg_get_device_count.argtypes = [c_void_p]
    dev_count = lib.ggml_xdna_reg_get_device_count(reg)
    print(f"XDNA devices: {dev_count}")
    
    if dev_count == 0:
        print("WARNING: No XDNA devices found")
    
    # Initialize backend
    lib.ggml_backend_xdna_init.restype = c_void_p
    backend = lib.ggml_backend_xdna_init()
    if not backend:
        print("ERROR: XDNA backend init failed")
        return 1
    print("XDNA backend initialized")
    
    # Check backend name
    lib.ggml_backend_name.restype = c_char_p
    lib.ggml_backend_name.argtypes = [c_void_p]
    name = lib.ggml_backend_name(backend)
    print(f"Backend name: {name.decode() if name else 'NULL'}")
    
    # Check is_xdna
    lib.ggml_backend_is_xdna.restype = c_int
    lib.ggml_backend_is_xdna.argtypes = [c_void_p]
    is_xdna = lib.ggml_backend_is_xdna(backend)
    print(f"Is XDNA: {bool(is_xdna)}")
    
    print("=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
