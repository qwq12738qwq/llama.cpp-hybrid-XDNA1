#!/usr/bin/env python3
"""Fix linker script INPUT path and re-run link + xclbin packaging for M=1 kernel."""
import os, subprocess, sys, json, time

OUT_DIR = "/home/ubuntu12738/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/k1024_m1_n16"
PRJ = os.path.join(OUT_DIR, "design.prj")
LLVM = "/home/ubuntu12738/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/llvm-aie/bin"

# Fix the linker script to point at the actual .o
ld_path = os.path.join(PRJ, "ldScripts_main_core_0_2.ld.script")
with open(ld_path) as f:
    content = f.read()
# Replace any INPUT(...) line with the correct object
import re
content = re.sub(
    r'INPUT\([^)]*\)',
    f'INPUT({PRJ}/objects_main_core_0_2/objects_main_core_0_2.o)',
    content
)
with open(ld_path, "w") as f:
    f.write(content)
print(f"[FIX] Linker script INPUT -> {PRJ}/objects_main_core_0_2/objects_main_core_0_2.o")

# Link the ELF
env = os.environ.copy()
env["PATH"] = LLVM + ":" + env["PATH"]
elf_path = os.path.join(PRJ, "elfs_main_core_0_2.elf")
cmd = [os.path.join(LLVM, "clang"), "-fuse-ld=lld", "-nostdlib", "-Wl,-T", ld_path, "-o", elf_path]
print(f"[FIX] Linking: {' '.join(cmd)}")
r = subprocess.run(cmd, capture_output=True, text=True, env=env)
print(f"[FIX] Link stdout: {r.stdout[-500:]}")
print(f"[FIX] Link stderr: {r.stderr[-500:]}")
print(f"[FIX] Link return: {r.returncode}")

if r.returncode != 0:
    print("[FIX] Link FAILED")
    sys.exit(1)

# Now run aiecc from the ELF stage to produce xclbin
# We need to re-run aiecc but skip the compile steps. Let's use aiecc with --checkpoint
# Actually, let's just run aiecc again - it should now use the fixed linker script
# But aiecc regenerates the linker script... so let's use the aiecc --xclbinutil path

# Alternative: use xclbinutil directly
# First, let's check what aiecc does after linking
print("[FIX] ELF created. Now running aiecc again to produce xclbin...")
