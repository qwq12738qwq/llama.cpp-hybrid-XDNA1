#!/usr/bin/env bash
# Phase 62 baseline: re-measure the phase 61 reference points before any change.
# Vulkan and hybrid runs are interleaved so a machine level drift hits both.
set -u
ROOT=/workspace/llama.cpp
S=$ROOT/experiments/phase62/scripts

echo "########## PP512 interleaved baseline ##########"
$S/run_pp.sh vulkan_pp512_r5   off        none 512 5
$S/run_pp.sh c4v3_all_pp512_r5 c4_mmul_v3 all  512 5
$S/run_pp.sh vulkan_pp512_r5b  off        none 512 5
$S/run_pp.sh c4v3_all_pp512_r5b c4_mmul_v3 all 512 5
$S/run_pp.sh vulkan_pp512_r5c  off        none 512 5

echo "########## PPL baseline ##########"
$S/run_ppl.sh vulkan off none
$S/run_ppl.sh c4v3  c4_mmul_v3 all

echo "########## BASELINE DONE ##########"
