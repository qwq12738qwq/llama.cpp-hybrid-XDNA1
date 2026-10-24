# Quick fix: replace all bo.write(data, offset, size) with map+memcpy+sync
import re

path = "/workspace/llama.cpp/tools/phase18-core/phase18_kernel_search.py"
with open(path) as f:
    code = f.read()

# Fix inst_bo.write - use map+memcpy+sync pattern
old = """    with open(insts_path, \"rb\") as f:
        inst_bo.write(f.read(), 0, inst_sz)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)"""

new = """    with open(insts_path, \"rb\") as f:
        inst_data = f.read()
    import ctypes
    mapped = inst_bo.map()
    ctypes.memmove(mapped, inst_data, len(inst_data))
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)"""

code = code.replace(old, new)

# Fix bo.write calls for input/weight/output data
# Replace in_bo.write(input_i8.tobytes(), 0, in_bytes) etc.
for pattern, replacement in [
    ("in_bo.write(input_i8.tobytes(), 0, in_bytes)", "m = in_bo.map(); import ctypes; ctypes.memmove(m, input_i8.tobytes(), in_bytes)"),
    ("wt_bo.write(weight_i8.tobytes(), 0, wt_bytes)", "m = wt_bo.map(); import ctypes; ctypes.memmove(m, weight_i8.tobytes(), wt_bytes)"),
    ("out_bo.write(b\"\\x00\" * out_bytes, 0, out_bytes)", "m = out_bo.map(); import ctypes; ctypes.memset(m, 0, out_bytes)"),
]:
    code = code.replace(pattern, replacement)

# Fix bo.read - use map+frombuffer
old_read = """        raw = out_bo.read(out_bytes, 0)"""
new_read = """        raw = bytes(out_bo.read(out_bytes, 0))"""
code = code.replace(old_read, new_read)

with open(path, "w") as f:
    f.write(code)
print("FIXED")
