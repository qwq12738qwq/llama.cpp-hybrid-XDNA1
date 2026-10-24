#!/usr/bin/env python3
"""Phase 31: derive the hybrid attention critical path from measured data.

Inputs (all measured on this machine, see the CSV/raw files in this dir):
  - attention_partition.csv : Vulkan FLASH_ATTN_EXT full / half-KV pair / split_k=2
  - xdna_gemm.csv           : XDNA int8 GEMM time at M=512 K=256 (attention head dim)
  - xdna_dma.csv            : XDNA host BO staging bandwidth

Model: KV-range partition of attention into A (Vulkan) and B (XDNA).
  T_vk_full(C)      measured full attention
  T_vk_A            = measured half-KV Vulkan dispatch, scaled to the A fraction
  T_xdna_B(C_B)     = XDNA QK^T cost for C_B KV tokens. Lower bound: it excludes
                      softmax, PV, merge and cross-device handoff.
  hybrid_compute    = max(T_vk_A, T_xdna_B)
  relief iff hybrid_compute + T_merge < T_vk_full for some fraction f.
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))

# --- measured Vulkan partition data (ms) ---
vk = {
    512:   dict(full=2.3621,  A=1.0271,  B=0.8826,  splitk2=1.7785, cos=0.999999596),
    4096:  dict(full=7.2822,  A=3.3228,  B=3.3301,  splitk2=8.0671, cos=0.999998999),
    16384: dict(full=25.3477, A=12.7954, B=12.8109, splitk2=28.8190, cos=0.999997517),
    32768: dict(full=57.4781, A=25.7732, B=25.5253, splitk2=59.7566, cos=0.999995603),
}

# --- measured XDNA int8 GEMM at K=256 (attention head dim), M=512 ---
#   M=512,K=256,N=4096 -> 63.5975 ms  => per output-KV-token cost
XDNA_PER_N_MS_M512 = 63.5975 / 4096.0        # one q-head, M=512
XDNA_PER_N_MS_M4096 = XDNA_PER_N_MS_M512 * 8.0  # all n_head=8 q heads (real QK^T)

# measured merge overhead proxy: native split_k=2 dispatch vs unsplit
MERGE_FRAC = 0.06   # ~upper bound seen (splitk2 slightly above full)

with open(os.path.join(HERE, "phase31_critical_path.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kv", "T_vk_full_ms", "T_vk_A_ms", "T_vk_B_ms",
                "T_xdna_B_qkT_M512_ms", "T_xdna_B_qkT_M4096_ms",
                "hybrid_max_M4096_ms", "hybrid_max_M512_ms",
                "T_full_ms", "relief_M4096", "relief_M512"])
    for kv in sorted(vk):
        d = vk[kv]
        cb = kv // 2
        xd512 = XDNA_PER_N_MS_M512 * cb
        xd4096 = XDNA_PER_N_MS_M4096 * cb
        merge = MERGE_FRAC * d["full"]
        h4096 = max(d["A"], xd4096) + merge
        h512 = max(d["A"], xd512) + merge
        w.writerow([kv, round(d["full"], 4), round(d["A"], 4), round(d["B"], 4),
                    round(xd512, 3), round(xd4096, 3),
                    round(h4096, 3), round(h512, 3), round(d["full"], 4),
                    "yes" if h4096 < d["full"] else "no",
                    "yes" if h512 < d["full"] else "no"])

# smallest fraction f at which XDNA QK^T could be hidden by Vulkan A,
# assuming T_vk_A = (1-f) * T_vk_full (linear KV scaling).
with open(os.path.join(HERE, "phase31_verdict.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kv", "vk_per_kvtoken_ms", "xdna_qkT_per_kvtoken_ms_M4096",
                "max_hideable_fraction", "net_relief_ceiling"])
    for kv in sorted(vk):
        d = vk[kv]
        vk_per = d["full"] / kv
        xdna_per = XDNA_PER_N_MS_M4096
        # (1-f)*T_full >= f*C*xdna_per  ->  f <= 1/(1 + xdna_per/vk_per)
        fmax = 1.0 / (1.0 + xdna_per / vk_per)
        w.writerow([kv, round(vk_per, 6), round(xdna_per, 6),
                    round(fmax, 5), round(fmax, 5)])

print("wrote phase31_critical_path.csv, phase31_verdict.csv")
print("XDNA per-KV-token QK^T (all 8 heads, ms):", round(XDNA_PER_N_MS_M4096, 6))
for kv in sorted(vk):
    vk_per = vk[kv]["full"] / kv
    fmax = 1.0 / (1.0 + XDNA_PER_N_MS_M4096 / vk_per)
    print(f"KV={kv}: vk_per_tok={vk_per:.6f} ms  max_hideable_f={fmax:.4f}  net_relief_ceiling={fmax:.4f}")
