from aie.helpers.taplib import TensorTiler2D

m, k, n = 64, 64, 64

# B tensor TAS for K=1024, column-major (original)
K = 1024
N = 4096
b_tap_1024_col = TensorTiler2D.group_tiler((K, N), (k, n), (K // k, N // n), tile_group_col_major=True)[0]
print("=== B tensor K=1024 col-major ===")
print(f"  sizes={b_tap_1024_col.sizes}  strides={b_tap_1024_col.strides}  offset={b_tap_1024_col.offset}")
print(f"  transformation_dims={b_tap_1024_col.transformation_dims}")

# B tensor TAS for K=1024, row-major
b_tap_1024_row = TensorTiler2D.group_tiler((K, N), (k, n), (K // k, N // n), tile_group_col_major=False)[0]
print("\n=== B tensor K=1024 row-major ===")
print(f"  sizes={b_tap_1024_row.sizes}  strides={b_tap_1024_row.strides}  offset={b_tap_1024_row.offset}")
print(f"  transformation_dims={b_tap_1024_row.transformation_dims}")
