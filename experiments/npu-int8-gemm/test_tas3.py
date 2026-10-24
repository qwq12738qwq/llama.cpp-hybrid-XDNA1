from aie.helpers.taplib import TensorAccessPattern
import numpy as np

# Test custom TAS for B K=1024, decomposed into 2 chunks
# Using sizes=[2, 64, 64, 64] with correct strides
tap = TensorAccessPattern(
    tensor_dims=(1024, 4096),
    offset=0,
    sizes=[2, 64, 64, 64],
    strides=[2097152, 4096, 64, 1],
)

access_order = tap.access_order()
unique, counts = np.unique(access_order, return_counts=True)
print(f"Unique elements accessed: {len(unique)}")
print(f"Min access count: {np.min(counts)}")
print(f"Max access count: {np.max(counts)}")
overlaps = np.sum(counts > 1)
print(f"Elements accessed more than once: {overlaps}")
print(f"Match: {len(unique) == 1024*4096 and overlaps == 0}")
