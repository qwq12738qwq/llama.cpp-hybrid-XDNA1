from aie.helpers.taplib import TensorTiler2D

m, k = 64, 64
M, K = 128, 1024

# A with default (col-major)
a_taps_col = TensorTiler2D.group_tiler(
    (M, K), (m, k), (1, K // k), pattern_repeat=64
)
print("=== A K=1024 col-major ===")
for i, tap in enumerate(a_taps_col):
    print(f"  tap[{i}]: sizes={tap.sizes}  strides={tap.strides}  offset={tap.offset}")

# A with row-major
a_taps_row = TensorTiler2D.group_tiler(
    (M, K), (m, k), (1, K // k), pattern_repeat=64, tile_group_col_major=False
)
print("\n=== A K=1024 row-major ===")
for i, tap in enumerate(a_taps_row):
    print(f"  tap[{i}]: sizes={tap.sizes}  strides={tap.strides}  offset={tap.offset}")
