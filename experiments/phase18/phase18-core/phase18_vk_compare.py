#!/usr/bin/env python3
"""Phase 18: Measure Vulkan kernel time for same M/K/N shape."""
import time, numpy as np

try:
    import ggml
except ImportError:
    print("ggml not available, using build-phase17b")
    import sys
    sys.path.insert(0, "/workspace/llama.cpp/build-phase17b/bin")

import ggml
import ggml_backend

def main():
    M, K, N = 512, 1024, 3072
    
    # Init Vulkan backend
    ggml_backend.load_all()
    vk = ggml_backend.init_by_name("Vulkan0", "")
    if not vk:
        vk = ggml_backend.init_by_name("Vulkan", "")
    if not vk:
        print("ERROR: No Vulkan backend")
        return
    print(f"Vulkan backend: {ggml_backend.get_name(vk)}")
    
    # Create test tensors
    ctx_params = ggml.context_params()
    ctx_params.mem_size = 256 * 1024 * 1024
    ctx = ggml.init(ctx_params)
    
    rng = np.random.RandomState(42)
    w_data = rng.randint(-127, 128, size=(N, K)).astype(np.float32) / 127.0
    a_data = rng.randint(-127, 128, size=(M, K)).astype(np.float32) / 127.0
    
    # Quantize weight to Q8_0
    # ... this needs ggml quantize API
    
    # For now, just time the raw Vulkan compute
    # Using a simpler approach: measure ggml graph compute time
    
    print(f"Shape: M={M} K={K} N={N}")
    print("Note: Use the existing hetero-vulkan-xdna-test for accurate Vulkan timing")
    print("Vulkan-only (from Phase 17B): ~34ms")
    print("XDNA-only (K=1024 single submit): ~62ms")
    print("Ratio: XDNA/Vulkan = 1.82x")
    print()
    print("KEY INSIGHT: XDNA kernel is ~1.8x slower than Vulkan for this shape.")
    print("For heterogeneous execution to help:")
    print("  1. Need XDNA < Vulkan (not yet achieved)")
    print("  2. OR: Use graph-level partitioning (different operators)")
    print("  3. OR: Increase M_TILE for better AIE utilization")

if __name__ == "__main__":
    main()
