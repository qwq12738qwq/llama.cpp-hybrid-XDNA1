from aie.helpers.taplib import TensorAccessPattern
import numpy as np

# Test custom TAS for B K=1024
tap = TensorAccessPattern(
    tensor_dims=(1024, 4096),
    offset=0,
    sizes=[2, 64, 64, 64],
    strides=[2097152, 4096, 64, 1],
)

access_order = tap.access_order()
unique, counts = np.unique(access_order, return_counts=True)

# Find elements accessed more than once
overlap_elements = unique[counts > 1]
print(f"Elements accessed more than once: {len(overlap_elements)}")
if len(overlap_elements) > 0:
    print(f"First few: {overlap_elements[:10]}")
    print(f"Counts: {counts[counts > 1][:10]}")

# Find elements not accessed
all_elements = set(range(1024*4096))
accessed_elements = set(access_order.flatten())
not_accessed = all_elements - accessed_elements
print(f"Elements not accessed: {len(not_accessed)}")
if len(not_accessed) > 0:
    print(f"First few: {list(not_accessed)[:10]}")
