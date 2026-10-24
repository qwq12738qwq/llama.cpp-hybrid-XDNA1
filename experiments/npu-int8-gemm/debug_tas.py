#!/usr/bin/env python3
"""Print TAS sizes/strides for K=512 and K=1024 to understand the DMA pattern."""

from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern

m, k, n = 64, 64, 64

# K=512 case (working)
M, K, N = 128, 512, 4096
a_taps_512 = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k), pattern_repeat=(N // n))
print("=== K=512, M=128, N=4096 ===")
print(f"num_taps: {len(a_taps_512)}")
for i, tap in enumerate(a_taps_512):
    print(f"  tap[{i}]: sizes={tap.sizes}  strides={tap.strides}  offset={tap.offset}")
    # Also print the transformation_dims
    print(f"           transformation_dims={tap.transformation_dims}")

# K=1024 case (blocked)
M, K, N = 128, 1024, 4096
a_taps_1024 = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k), pattern_repeat=(N // n))
print()
print("=== K=1024, M=128, N=4096 ===")
print(f"num_taps: {len(a_taps_1024)}")
for i, tap in enumerate(a_taps_1024):
    print(f"  tap[{i}]: sizes={tap.sizes}  strides={tap.strides}  offset={tap.offset}")
    print(f"           transformation_dims={tap.transformation_dims}")

# K=4096 case
M, K, N = 128, 4096, 4096
a_taps_4096 = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k), pattern_repeat=(N // n))
print()
print("=== K=4096, M=128, N=4096 ===")
print(f"num_taps: {len(a_taps_4096)}")
for i, tap in enumerate(a_taps_4096):
    print(f"  tap[{i}]: sizes={tap.sizes}  strides={tap.strides}  offset={tap.offset}")
    print(f"           transformation_dims={tap.transformation_dims}")
