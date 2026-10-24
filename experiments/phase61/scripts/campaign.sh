#!/usr/bin/env bash
# Phase 61 measurement campaign: routing comparison and PP scaling.
#
# Vulkan and hybrid runs are interleaved so that a machine level drift hits
# both configurations equally.
set -u
ROOT=/workspace/llama.cpp
S=$ROOT/experiments/phase61/scripts

run () {  # tag engine route pp reps [extra env]
  $S/run_pp.sh "$@"
}

echo "########## PP512 ##########"
run vulkan_pp512_r5        off              none 512 5
run c4v2_all_pp512_r3     c4_mmul_v2       all  512 3
run c4v3_all_pp512_r5     c4_mmul_v3       all  512 5
run vulkan_pp512_r5b      off              none 512 5
run c4v3_down_pp512_r5    c4_mmul_v3       down 512 5
run c4v3_gate_pp512_r3    c4_mmul_v3       gate 512 3
run c4v3_up_pp512_r3      c4_mmul_v3       up   512 3
run c4v3_gateup_pp512_r3  c4_mmul_v3       gate,up 512 3
run vulkan_pp512_r5c      off              none 512 5
run c4v3_all_pp512_r5b    c4_mmul_v3       all  512 5

echo "########## PP1024 ##########"
run vulkan_pp1024_r3      off              none 1024 3
run c4v3_all_pp1024_r3    c4_mmul_v3       all  1024 3
run c4v3_down_pp1024_r3   c4_mmul_v3       down 1024 3
run vulkan_pp1024_r3b     off              none 1024 3

echo "########## PP2048 ##########"
run vulkan_pp2048_r3      off              none 2048 3
run c4v3_all_pp2048_r2    c4_mmul_v3       all  2048 2
run vulkan_pp2048_r3b     off              none 2048 3

echo "########## PP3072 / 4096 / 5120 ##########"
run vulkan_pp3072_r3      off              none 3072 3
run c4v3_all_pp3072_r1    c4_mmul_v3       all  3072 1
run vulkan_pp4096_r3      off              none 4096 3
run c4v3_all_pp4096_r1    c4_mmul_v3       all  4096 1
run vulkan_pp5120_r3      off              none 5120 3
run c4v3_all_pp5120_r1    c4_mmul_v3       all  5120 1

echo "########## DONE ##########"
