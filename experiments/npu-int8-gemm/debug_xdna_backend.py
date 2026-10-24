#!/usr/bin/env python3
"""Debug XDNA backend with simple test."""
import sys
sys.path.insert(0, '/workspace/llama.cpp/build')
sys.path.insert(0, '/workspace/llama.cpp/ggml/src/ggml-xdna')

import ctypes
import numpy as np

# Load the XDNA backend library
lib = ctypes.CDLL('/workspace/llama.cpp/build/bin/libxdna.so')
ggml_lib = ctypes.CDLL('/workspace/llama.cpp/build/bin/libggml.so')

# This is complex - let me just run the existing test with more debug
# by modifying the test to print chunk outputs

print("Run the test with debug by modifying test-qwen-ssm-beta.cpp")
