#!/usr/bin/env python3
# Phase 30 STEP 3: quantify Vulkan FA KV read geometry from the shader layout.
#
# Model (Qwen3.5-0.8B-Q8_0, from GGUF metadata):
#   full-attention layers = 6   (block_count=24, full_attention_interval=4)
#   HSK = HSV = 256 (attention.key_length / value_length)
#   n_head = 8, n_head_kv = 2  -> gqa_ratio = 4
#   ubatch N = 512 (llama.cpp default), Br = 16, Bc = 64  (Phase 29 VK-TRACE-ATTN)
#   split_k = 1, split_kv = KV  (Phase 29) -> every query tile scans the full KV
#   dispatch_x = Tr * Br, Tr = CEIL_DIV(N, Br) = 32
#
# Shader fact (ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp):
#   K/V are f16, read as f16vec4. Non-staged addressing:
#     data_kv4[k_offset/4 + (j*Bc + c*cols_per_iter + col_tid)*k_stride/4 + d*D_split + d_tid]
#   k_stride = nb11 (elements) = HSK for a token-major (contiguous) KV cache.
#   => head dim is contiguous per token; consecutive tokens are k_stride apart.
#   Each workgroup (Br=16 query rows, 1 head) walks j = 0..KV/Bc over the whole KV.
#
# So per op (per layer):
#   dup read    = Tr * n_head    * KV * (HSK+HSV) * 2   (what the shader issues)
#   unique read = Tr * n_head_kv * KV * (HSK+HSV) * 2   (ideal L2 dedup of gqa 4x)
# Per ubatch multiply by N_LAYERS=6.

import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE29 = os.path.join(HERE, "..", "phase29", "results", "phase29_attention_efficiency.csv")

N = 512
N_HEAD = 8
N_HEAD_KV = 2
HSK = 256
HSV = 256
N_LAYERS = 6
BYTES = 2
BR = 16
TR = -(-N // BR)   # 32

rows = []
with open(PHASE29) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["kv_len"]), float(r["flash_ms_6ops"])))

out = []
for kv, ms in rows:
    per_op_dup = TR * N_HEAD * kv * (HSK + HSV) * BYTES
    per_op_unique = TR * N_HEAD_KV * kv * (HSK + HSV) * BYTES
    total_dup = per_op_dup * N_LAYERS
    total_unique = per_op_unique * N_LAYERS
    # unique KV working set for one layer (both kv heads), K+V f16
    ws_mb = (N_HEAD_KV * kv * (HSK + HSV) * BYTES) / (1024 * 1024)
    us_per_tok = ms * 1000.0 / kv
    out.append({
        "kv_len": kv,
        "flash_ms_6ops": round(ms, 4),
        "us_per_kvtoken": round(us_per_tok, 4),
        "kv_read_dup_GB": round(total_dup / 1e9, 5),
        "kv_read_unique_GB": round(total_unique / 1e9, 5),
        "l2_working_set_MB": round(ws_mb, 3),
        "eff_bw_dup_GBps": round(total_dup / (ms * 1e-3) / 1e9, 2),
        "eff_bw_unique_GBps": round(total_unique / (ms * 1e-3) / 1e9, 2),
    })

dest = os.path.join(HERE, "results", "phase30_kv_access_geometry.csv")
with open(dest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

print("wrote", dest)
for o in out:
    if o["kv_len"] in (512, 4096, 16384, 32768, 65536):
        print(o)
print()
print("780M PHOENIX LPDDR5 ceiling ~100-120 GB/s. eff_bw_unique is the best case")
print("(perfect gqa L2 dedup); eff_bw_dup is the worst case (no dedup).")
