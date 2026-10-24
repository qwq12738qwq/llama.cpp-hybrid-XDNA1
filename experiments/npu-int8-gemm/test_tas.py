from aie.helpers.taplib import TensorAccessPattern
import numpy as np

# Test custom TAS for B K=1024, decomposed into 2 chunks of 512
tap = TensorAccessPattern(
    tensor_dims=(1024, 4096),
    offset=0,
    sizes=[2, 64, 512, 64],
    strides=[2097152, 4096, 64, 1],
)

access_order = tap.access_order()
print(f"Access order shape: {access_order.shape}")
print(f"Total accesses: {np.max(access_order) + 1}")

# Find unique accesses
unique, counts = np.unique(access_order, return_counts=True)
print(f"Unique elements accessed: {len(unique)}")
print(f"Min access count: {np.min(counts)}")
print(f"Max access count: {np.max(counts)}")

# Check for overlaps (elements accessed more than once)
overlaps = np.sum(counts > 1)
print(f"Elements accessed more than once: {overlaps}")

# Print first few accesses
print(f"\nFirst 10 accesses: {access_order.flatten()[:10]}")
print(f"Last 10 accesses: {access_order.flatten()[-10:]}")
