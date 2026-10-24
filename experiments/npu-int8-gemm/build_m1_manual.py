#!/usr/bin/env python3
"""Build M=1 kernel using iron API, then fix linker and re-link manually."""
import os, sys, subprocess, shutil, re

WORK_DIR = "/home/ubuntu12738/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/k1024_m1_n16"
PRJ = os.path.join(WORK_DIR, "design.prj")
LLVM_BIN = "/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/llvm-aie/bin"

# Step 1: Run iron API to generate MLIR and compile kernel
print("[STEP 1] Running iron API to generate MLIR and compile kernel...")
env = os.environ.copy()
env["PATH"] = "/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/llvm-aie/bin:" + env["PATH"]
env["PYTHONPATH"] = "/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python"

# Run the gen_m1_kernel.py to generate MLIR and compile
result = subprocess.run([
    "/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/bin/python3",
    "/home/ubuntu12738/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/gen_m1_kernel.py"
], capture_output=True, text=True, env=env)
print(f"[STEP 1] Return code: {result.returncode}")
print(f"[STEP 1] stdout: {result.stdout[-1000:]}")
print(f"[STEP 1] stderr: {result.stderr[-1000:]}")

# Step 2: Find the compiled object file
obj_src = os.path.join(PRJ, "objects_main_core_0_2", "objects_main_core_0_2.o")
obj_dst = os.path.join(PRJ, "matmul_scalar_i8_i32_05f2508a.o")
if os.path.exists(obj_src):
    shutil.copy2(obj_src, obj_dst)
    print(f"[STEP 2] Copied {obj_src} -> {obj_dst}")
else:
    print(f"[STEP 2] ERROR: Source object not found at {obj_src}")
    sys.exit(1)

# Step 3: Fix the linker script
ld_path = os.path.join(PRJ, "ldScripts_main_core_0_2.ld.script")
with open(ld_path) as f:
    content = f.read()
# Replace INPUT line
import re
content = re.sub(
    r'INPUT\([^)]*\)',
    f'INPUT({obj_dst})',
    content
)
with open(ld_path, "w") as f:
    f.write(content)
print(f"[STEP 3] Fixed linker script INPUT -> {obj_dst}")

# Step 4: Re-link the ELF manually
print("[STEP 4] Re-linking ELF...")
elf_path = os.path.join(PRJ, "elfs_main_core_0_2.elf")
cmd = [
    "/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/llvm-aie/bin/clang",
    "-target", "aie2-none-unknown-elf",
    "-fuse-ld=lld", "-nostdlib",
    "-Wl,-T", ld_path,
    "-o", elf_path
]
result = subprocess.run(cmd, capture_output=True, text=True, env=env)
print(f"[STEP 4] Link return code: {result.returncode}")
print(f"[STEP 4] Link stderr: {result.stderr[-1000:]}")
if result.returncode != 0:
    print("[STEP 4] Link FAILED")
    sys.exit(1)
print("[STEP 4] Link SUCCESS")

# Step 5: Run aiecc from the ELF stage to produce xclbin
# Use aiecc with the already generated intermediates
print("[STEP 5] Running aiecc to produce xclbin...")
aiecc = "/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/bin/aiecc"
cmd = [
    "PYTHONPATH=/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python",
    aiecc,
    "--output-dir", WORK_DIR,
    "--tmpdir", PRJ,
    "--xclbin-name", os.path.join(WORK_DIR, "design.xclbin"),
    "--pdi-name", "main.pdi",
    "--elf-name", "matmul_i8_i32_m1_k1024_n16.o",
    os.path.join(PRJ, "aie.mlir")
]
# Actually, let's just run aiecc again - it should use the existing intermediates
result = subprocess.run(cmd, capture_output=True, text=True, env=env, shell=True)
print(f"[STEP 5] Return code: {result.returncode}")
print(f"[STEP 5] stdout: {result.stdout[-2000:]}")
print(f"[STEP 5] stderr: {result.stderr[-2000:]}")

# Check if xclbin was created
xclbin_path = os.path.join(WORK_DIR, "design.xclbin")
if os.path.exists(xclbin_path):
    print(f"[SUCCESS] xclbin created: {xclbin_path} ({os.path.getsize(xclbin_path)} bytes)")
else:
    print(f"[FAIL] xclbin not found at {xclbin_path}")

