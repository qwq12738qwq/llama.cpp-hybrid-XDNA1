import os, sys
sys.path.insert(0, '/workspace/llama.cpp/gguf-py')
import numpy as np
from gguf import GGUFReader

def dequant_q8_0_row(raw_row):
    # raw_row: 1088 bytes = 32 blocks of 34
    n_blocks = len(raw_row)//34
    out = np.zeros(n_blocks*32, dtype=np.float32)
    for i in range(n_blocks):
        block = raw_row[i*34:(i+1)*34]
        d = np.frombuffer(block[0:2], dtype=np.float16)[0].astype(np.float32)
        qs = block[2:34].astype(np.int8)
        out[i*32:(i+1)*32] = qs.astype(np.float32) * d
    return out

def requant_per_row(W_f32):
    # W_f32: (rows, cols) float. Re-quantize to int8 with one scale per row.
    rows, cols = W_f32.shape
    scales = np.zeros(rows, dtype=np.float32)
    Q = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        amax = np.max(np.abs(W_f32[r]))
        s = amax / 127.0 if amax > 0 else 1.0
        scales[r] = s
        Q[r] = np.clip(np.round(W_f32[r] / s), -127, 127).astype(np.int8)
    return Q, scales

def requant_per_block32(W_f32):
    # standard Q8_0 per-32
    rows, cols = W_f32.shape
    assert cols % 32 == 0
    nb = cols // 32
    scales = np.zeros((rows, nb), dtype=np.float32)
    Q = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        for b in range(nb):
            seg = W_f32[r, b*32:(b+1)*32]
            amax = np.max(np.abs(seg))
            s = amax/127.0 if amax>0 else 1.0
            scales[r,b] = s
            Q[r, b*32:(b+1)*32] = np.clip(np.round(seg/s), -127, 127).astype(np.int8)
    return Q, scales

r = GGUFReader('/workspace/llama.cpp/models/Qwen3.5-0.8B-Q8_0.gguf')
t = [x for x in r.tensors if x.name=='blk.0.ffn_gate.weight'][0]
raw = t.data  # (3584, 1088) uint8
rows = raw.shape[0]
cols = 1024
W = np.zeros((rows, cols), dtype=np.float32)
for ridx in range(rows):
    W[ridx] = dequant_q8_0_row(raw[ridx])

# Original Q8_0 reconstruction (per-32) error
Q32, S32 = requant_per_block32(W)
W32 = Q32.astype(np.float32)
for ridx in range(rows):
    for b in range(cols//32):
        W32[ridx, b*32:(b+1)*32] *= S32[ridx,b]
err32 = np.abs(W - W32)
print("per-32 Q8_0 roundtrip: max_err=%.6f mean_err=%.6f rel=%.6f" % (err32.max(), err32.mean(), err32.mean()/ (np.abs(W).mean()+1e-9)))

# Per-row re-quantization
Qrow, Srow = requant_per_row(W)
Wrow = Qrow.astype(np.float32) * Srow[:,None]
err_row = np.abs(W - Wrow)
print("per-row requant: max_err=%.6f mean_err=%.6f rel=%.6f" % (err_row.max(), err_row.mean(), err_row.mean()/(np.abs(W).mean()+1e-9)))

# Per-8 (one scale per 8 values) 
def requant_per_group(W_f32, g):
    rows, cols = W_f32.shape
    assert cols % g == 0
    ng = cols // g
    scales = np.zeros((rows, ng), dtype=np.float32)
    Q = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        for b in range(ng):
            seg = W_f32[r, b*g:(b+1)*g]
            amax = np.max(np.abs(seg))
            s = amax/127.0 if amax>0 else 1.0
            scales[r,b] = s
            Q[r, b*g:(b+1)*g] = np.clip(np.round(seg/s), -127, 127).astype(np.int8)
    return Q, scales
for g in [64, 128, 256, 1024]:
    Qg, Sg = requant_per_group(W, g)
    Wg = Qg.astype(np.float32)
    for r in range(rows):
        for b in range(cols//g):
            Wg[r, b*g:(b+1)*g] *= Sg[r,b]
    errg = np.abs(W - Wg)
    print("group-%d requant: max_err=%.6f mean_err=%.6f rel=%.6f" % (g, errg.max(), errg.mean(), errg.mean()/(np.abs(W).mean()+1e-9)))
