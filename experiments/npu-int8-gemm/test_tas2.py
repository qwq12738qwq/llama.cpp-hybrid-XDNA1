from aie.helpers.taplib import TensorAccessPattern
import numpy as np

# Test custom TAS for B K=1024, decomposed into 2 chunks of 512
# Correct strides: each dimension must not overlap with inner dimensions
tap = TensorAccessPattern(
    tensor_dims=(1024, 4096),
    offset=0,
    sizes=[2, 64, 512, 64],
    strides=[2097152, 32768, 64, 1],
)

access_order = tap.access_order()
print(f"Access order shape: {access_order.shape}")
unique, counts = np.unique(access_order, return_counts=True)
print(f"Unique elements accessed: {len(unique)}")
print(f"Min access count: {np.min(counts)}")
print(f"Max access count: {np.max(counts)}")
overlaps = np.sum(counts > 1)
print(f"Elements accessed more than once: {overlaps}")

# Check if all elements are accessed exactly once
expected = 1024 * 4096
print(f"Expected total elements: {expected}")
print(f"Actual unique elements: {len(unique)}")
print(f"Match: {len(unique) == expected and overlaps == 0}")
