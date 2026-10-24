#!/usr/bin/env python3
"""Print TAS for B tensor and also the L2L1 tiler for both K=512 and K=1024."""

from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern

m, k, n = 64, 64, 64

# B tensor TAS for K=512
K = 512
N = 4096
b_tap_512 = TensorTiler2D.group_tiler((K, N), (k, n), (K // k, N // n), tile_group_col_major=True)[0]
print("=== B tensor K=512 ===")
print(f"  sizes={b_tap_512.sizes}  strides={b_tap_512.strides}  offset={b_tap_512.offset}")
print(f"  transformation_dims={b_tap_512.transformation_dims}")

# L2L1 A tiler for K=512
tap_A_512 = TensorTiler2D.group_tiler((m, k), (m//4, k//8), (m//4, k//8))[0]
print("=== L2L1 A tile K=512 ===")
print(f"  sizes={tap_A_512.sizes}  strides={tap_A_512.strides}  offset={tap_A_512.offset}")

# L2L1 B tiler for K=512
tap_B_512 = TensorTiler2D.group_tiler((k, n), (k//8, n//8), (k//8, n//8))[0]
print("=== L2L1 B tile K=512 ===")
print(f"  sizes={tap_B_512.sizes}  strides={tap_B_512.strides}  offset={tap_B_512.offset}")

print()

# B tensor TAS for K=1024
K = 1024
b_tap_1024 = TensorTiler2D.group_tiler((K, N), (k, n), (K // k, N // n), tile_group_col_major=True)[0]
print("=== B tensor K=1024 ===")
print(f"  sizes={b_tap_1024.sizes}  strides={b_tap_1024.strides}  offset={b_tap_1024.offset}")
print(f"  transformation_dims={b_tap_1024.transformation_dims}")

# L2L1 B tiler for K=1024
tap_B_1024 = TensorTiler2D.group_tiler((k, n), (k//8, n//8), (k//8, n//8))[0]
print("=== L2L1 B tile K=1024 ===")
print(f"  sizes={tap_B_1024.sizes}  strides={tap_B_1024.strides}  offset={tap_B_1024.offset}")

print()

# B tensor TAS for K=4096
K = 4096
b_tap_4096 = TensorTiler2D.group_tiler((K, N), (k, n), (K // k, N // n), tile_group_col_major=True)[0]
print("=== B tensor K=4096 ===")
print(f"  sizes={b_tap_4096.sizes}  strides={b_tap_4096.strides}  offset={b_tap_4096.offset}")
print(f"  transformation_dims={b_tap_4096.transformation_dims}")

# Also check C tensor TAS for K=1024
M = 128
K = 1024
c_taps_1024 = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))
print()
print("=== C tensor K=1024 ===")
print(f"num_taps: {len(c_taps_1024)}")
for i, tap in enumerate(c_taps_1024):
    print(f"  tap[{i}]: sizes={tap.sizes}  strides={tap.strides}  offset={tap.offset}")
    print(f"           transformation_dims={tap.transformation_dims}")
