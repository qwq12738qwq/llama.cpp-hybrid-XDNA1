import os, sys
sys.path.insert(0, '/workspace/llama.cpp/gguf-py')
import numpy as np
from gguf import GGUFReader

def dequant_q8_0(raw):
    # raw: (rows, 34*(cols//32)) uint8
    rows = raw.shape[0]
    cols = (raw.shape[1]//34)*32
    out = np.zeros((rows, cols), dtype=np.float32)
    for ridx in range(rows):
        for i in range(cols//32):
            block = raw[ridx, i*34:(i+1)*34]
            d = np.frombuffer(block[0:2], dtype=np.float16)[0].astype(np.float32)
            qs = block[2:34].astype(np.int8)
            out[ridx, i*32:(i+1)*32] = qs.astype(np.float32)*d
    return out

def requant_per_row(W):
    rows, cols = W.shape
    scales = np.zeros(rows, dtype=np.float32)
    Q = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        amax = np.max(np.abs(W[r]))
        s = amax/127.0 if amax>0 else 1.0
        scales[r]=s
        Q[r]=np.clip(np.round(W[r]/s),-127,127).astype(np.int8)
    return Q, scales

# Load real weight
r = GGUFReader('/workspace/llama.cpp/models/Qwen3.5-0.8B-Q8_0.gguf')
t = [x for x in r.tensors if x.name=='blk.0.ffn_gate.weight'][0]
W = dequant_q8_0(t.data)  # (3584, 1024) float, but stored as (ne1, ne0)=(3584,1024)? 
print("W shape", W.shape)
# W is (3584, 1024): 3584 output rows, 1024 input cols (K). For GEMM: C[m,n] = sum_k A[m,k]*B[k,n].
# Here weight B is (K, N) = (1024, 3584) in ggml (ne0=K, ne1=N). Our W is (N, K)=(3584,1024). Transpose to (K,N).
W = W.T.copy()  # (1024, 3584) = (K, N)
K, N = W.shape
M = 512
rng = np.random.default_rng(0)
# Synthetic activation A (M, K) in float, then Q8_0 per-32 quantize
A_f = rng.standard_normal((M, K)).astype(np.float32)*0.5
# Quantize A to Q8_0 per-32
def quant_q8_0_per32(A):
    M,K = A.shape
    nb = K//32
    out = np.zeros((M, K), dtype=np.int8)
    scales = np.zeros((M, nb), dtype=np.float32)
    for m in range(M):
        for b in range(nb):
            seg = A[m, b*32:(b+1)*32]
            amax = np.max(np.abs(seg))
            s = amax/127.0 if amax>0 else 1.0
            scales[m,b]=s
            out[m, b*32:(b+1)*32] = np.clip(np.round(seg/s),-127,127).astype(np.int8)
    return out, scales
Aq, Ascales = quant_q8_0_per32(A_f)

# Reference GEMM in Q8_0 semantics: C[m,n] = sum over 32-blocks of (Ascale[m,b]*Bscale[n,b]) * sum_i Aq[m,b*32+i]*Bq_int (weight dequant)
# Weight W is already dequantized float. Let's make weight Q8_0 too.
Wq, Wscales = quant_q8_0_per32(W.T)  # W (K,N) -> (N,K)? need (rows=K? no)
# Careful: weight is (K, N) meaning K input features, N output features.
# Q8_0 quantizes along K (input dim) per row of N. So each output column n has K values -> K//32 blocks.
# requant: W is (K,N). We want per-output-column (n) scale over K. Transpose to (N,K).
Wt = W.T.copy()  # (N, K)
Wq_t, Wscales_t = quant_q8_0_per32(Wt)  # (N, K) int8, (N, K//32) scales
Wq = Wq_t.T.copy()  # (K, N) int8
Wscales = Wscales_t.T.copy()  # (K//32? no) 
# Wscales_t is (N, K//32). For GEMM we need per (output n, block b): Wscales_t[n,b].
# Keep as (N, nbK).

# Reference: dequant weight float
Wref = Wq.astype(np.float32)
for n in range(N):
    for b in range(K//32):
        Wref[b*32:(b+1)*32, n] = Wq[b*32:(b+1)*32, n].astype(np.float32)*Wscales_t[n,b]

# Reference GEMM output
Cref = A_f @ Wref  # (M, N)

# --- Approach 1: per-row re-quantization of both A and W ---
# A: per-row scale (M scales). Aq_row, Ascale_row
Aq_row, Ascale_row = requant_per_row(A_f)  # Aq (M,K) int8, Ascale (M,)
# W: per-column scale (N scales) over K. W is (K,N). Per "row" of W^T = per column of W.
Wt2 = W.T.copy()  # (N, K)
Wq_row_t, Wscale_row_t = requant_per_row(Wt2)  # (N,K) int8, (N,) scales
Wq_row = Wq_row_t.T.copy()  # (K, N) int8
Wscale_col = Wscale_row_t  # (N,) scale per output col

# GEMM in int8 -> int32
Cint = Aq_row.astype(np.int32) @ Wq_row.astype(np.int32)  # (M, N) int32
# Apply scales: C[m,n] = Ascale_row[m] * Wscale_col[n] * Cint[m,n]
Cperrow = (Ascale_row[:,None] * Wscale_col[None,:]) * Cint.astype(np.float32)

err = np.abs(Cref - Cperrow)
print("\nPer-row requant GEMM vs Q8_0-ref:")
print("  max_err=%.4f mean_err=%.4f rel=%.4f" % (err.max(), err.mean(), err.mean()/(np.abs(Cref).mean()+1e-9)))

# --- Approach 2: per-32 for A (keep Q8_0), per-row for W only ---
# A stays Q8_0 (per-32). W per-row. Need to handle A's per-32 scales.
# C[m,n] = sum_b Ascale[m,b]*Wscale_col[n] * sum_i Aq[m,b*32+i]*Wq[b*32+i,n]
# = Wscale_col[n] * sum_b Ascale[m,b] * (Aq_block @ Wq_block)
C2 = np.zeros((M,N), dtype=np.float32)
for b in range(K//32):
    Ablk = Aq[:, b*32:(b+1)*32].astype(np.int32)  # (M,32)
    Wblk = Wq[b*32:(b+1)*32, :].astype(np.int32)  # (32,N)
    dot = Ablk @ Wblk  # (M,N) int32
    C2 += (Ascales[:,b][:,None] * Wscale_col[None,:]) * dot.astype(np.float32)
err2 = np.abs(Cref - C2)
print("A per-32 + W per-row GEMM vs Q8_0-ref:")
print("  max_err=%.4f mean_err=%.4f rel=%.4f" % (err2.max(), err2.mean(), err2.mean()/(np.abs(Cref).mean()+1e-9)))

# Baseline: pure Q8_0 GEMM (per-32 both)
Cq8 = np.zeros((M,N), dtype=np.float32)
for b in range(K//32):
    Ablk = Aq[:, b*32:(b+1)*32].astype(np.int32)
    Wblk = Wq8b = Wq[b*32:(b+1)*32,:].astype(np.int32) if False else None
# use dequant weights already Q8_0
Wq8 = Wq.copy()
# recompute Q8_0 weight quantization properly
Wq8_t, Wscales8_t = quant_q8_0_per32(Wt)
Wq8 = Wq8_t.T.copy()
Cq8b = np.zeros((M,N), dtype=np.float32)
for b in range(K//32):
    Ablk = Aq[:, b*32:(b+1)*32].astype(np.int32)
    Wblk = Wq8[b*32:(b+1)*32, :].astype(np.int32)
    dot = Ablk @ Wblk
    Cq8b += (Ascales[:,b][:,None] * Wscales8_t.T[b,:,None] if False else Ascales[:,b][:,None]*Wscales8_t.T[b][:,None]) * dot.astype(np.float32)
# simpler:
Cq8c = np.zeros((M,N), dtype=np.float32)
for b in range(K//32):
    Ablk = Aq[:, b*32:(b+1)*32].astype(np.int32)
    Wblk = Wq8[b*32:(b+1)*32, :].astype(np.int32)
    dot = Ablk @ Wblk
    Cq8c += (Ascales[:, b][:, None] * Wscales8_t.T[b, :][None, :]) * dot.astype(np.float32)
errq8 = np.abs(Cref - Cq8c)
print("Q8_0 per-32 both GEMM vs dequant-ref (should be ~0):")
print("  max_err=%.4f mean_err=%.4f" % (errq8.max(), errq8.mean()))
PYEOF
