# Qwen3.5-0.8B GGML Graph Audit - Phase 6B

**Date:** 2026-09-08
**Model:** /models/Qwen3.5-0.8B-Q8_0.gguf
**Quantization:** Q8_0
**Architecture:** 24 layers, 1024 embedding, 3584 FFN hidden, 16 heads, 128 head dim
**Attention Pattern:** Linear attention (gated delta net) for layers 0-2, 4-6, 8-10, 12-14, 16-18, 20-22; Full attention for layers 3, 7, 11, 15, 19, 23

---

## Graph Structure Summary

The graph is built for **prefill** (M=2 tokens: "Hello" + BOS) and **decode** (M=1 token per step).

All weights are **Q8_0** quantized. Input tensors are **f32** (dequantized on-the-fly by ggml_mul_mat).

### Operation Types

| Layer Type | Layers | Operations |
|------------|--------|------------|
| Linear Attention | 18 (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22) | attn_qkv, attn_gate, ssm_beta, ssm_alpha, ssm_out |
| Full Attention | 6 (3,7,11,15,19,23) | attn_q, attn_k, attn_v, attn_output |
| FFN (all layers) | 24 | ffn_up, ffn_gate, ffn_down |
| Output | - | token_embd (LM head) |

---

## Detailed Operation Catalog

### Linear Attention Layers (18 layers)

Each linear attention layer has 5 MUL_MAT operations:

| Operation | Weight Tensor | Weight Shape (ne[0], ne[1]) | Input Shape (ne[0], ne[1]) | M | K | N | Quant |
|-----------|---------------|----------------------------|---------------------------|---|---|---|-------|
| attn_qkv | blk.{il}.attn_qkv.weight | [1024, 6144] | [1024, M] | M | 1024 | 6144 | Q8_0 |
| attn_gate | blk.{il}.attn_gate.weight | [1024, 2048] | [1024, M] | M | 1024 | 2048 | Q8_0 |
| ssm_beta | blk.{il}.ssm_beta.weight | [1024, 16] | [1024, M] | M | 1024 | 16 | Q8_0 |
| ssm_alpha | blk.{il}.ssm_alpha.weight | [1024, 16] | [1024, M] | M | 1024 | 16 | Q8_0 |
| ssm_out | blk.{il}.ssm_out.weight | [2048, 1024] | [2048, M] | M | 2048 | 1024 | Q8_0 |

### Full Attention Layers (6 layers: 3, 7, 11, 15, 19, 23)

Each full attention layer has 4 MUL_MAT operations:

| Operation | Weight Tensor | Weight Shape (ne[0], ne[1]) | Input Shape (ne[0], ne[1]) | M | K | N | Quant |
|-----------|---------------|----------------------------|---------------------------|---|---|---|-------|
| attn_q | blk.{il}.attn_q.weight | [1024, 4096] | [1024, M] | M | 1024 | 4096 | Q8_0 |
| attn_k | blk.{il}.attn_k.weight | [1024, 512] | [1024, M] | M | 1024 | 512 | Q8_0 |
| attn_v | blk.{il}.attn_v.weight | [1024, 512] | [1024, M] | M | 1024 | 512 | Q8_0 |
| attn_output | blk.{il}.attn_output.weight | [2048, 1024] | [2048, M] | M | 2048 | 1024 | Q8_0 |

### FFN Layers (All 24 layers)

Each layer has 3 MUL_MAT operations:

| Operation | Weight Tensor | Weight Shape (ne[0], ne[1]) | Input Shape (ne[0], ne[1]) | M | K | N | Quant |
|-----------|---------------|----------------------------|---------------------------|---|---|---|-------|
| ffn_up | blk.{il}.ffn_up.weight | [1024, 3584] | [1024, M] | M | 1024 | 3584 | Q8_0 |
| ffn_gate | blk.{il}.ffn_gate.weight | [1024, 3584] | [1024, M] | M | 1024 | 3584 | Q8_0 |
| ffn_down | blk.{il}.ffn_down.weight | [3584, 1024] | [3584, M] | M | 3584 | 1024 | Q8_0 |

### Output Layer

| Operation | Weight Tensor | Weight Shape (ne[0], ne[1]) | Input Shape (ne[0], ne[1]) | M | K | N | Quant |
|-----------|---------------|----------------------------|---------------------------|---|---|---|-------|
| LM Head | token_embd.weight | [1024, 248320] | [1024, M] | M | 1024 | 248320 | Q8_0 |

---

## Dimension Analysis

### Prefill (M=2 tokens)
- Input: [1024, 2] f32
- All MUL_MAT operations have M=2 for prefill

### Decode (M=1 token)
- Input: [1024, 1] f32
- All MUL_MAT operations have M=1 for decode

### Current XDNA Backend Constraints
- **Supported:** M=1, K=192, N=128, Q8_0
- **Actual Qwen:** M=1/2, K=1024/2048/3584, N=16/512/1024/3584/4096/6144/248320

### Mismatch Summary
| Dimension | XDNA Current | Qwen Actual | Gap |
|-----------|--------------|-------------|-----|
| M | 1 | 1 (decode), 2 (prefill) | ✅ Decode OK |
| K | 192 | 1024, 2048, 3584 | ❌ 5-18x larger |
| N | 128 | 16, 512, 1024, 3584, 4096, 6144, 248320 | ❌ Mostly larger |
| Quant | Q8_0 | Q8_0 | ✅ Match |

---

## XDNA Target Candidates (Phase 6C Analysis)

### Smallest Operations by N Dimension
1. **ssm_beta / ssm_alpha** (linear attention): K=1024, N=16 - Smallest N, but K=1024 ≠ 192
2. **attn_k / attn_v** (full attention): K=1024, N=512 - Moderate N, K=1024 ≠ 192
3. **ssm_out / attn_output**: K=2048, N=1024 - Larger
4. **ffn_up / ffn_gate**: K=1024, N=3584 - Large N
5. **attn_qkv**: K=1024, N=6144 - Largest in linear attention
6. **attn_q**: K=1024, N=4096 - Largest in full attention
7. **ffn_down**: K=3584, N=1024 - Largest K
8. **LM Head**: K=1024, N=248320 - Largest N

### Recommended First Target
**ssm_beta / ssm_alpha** (linear attention layers)
- Smallest N=16
- M=1 (decode) or M=2 (prefill)
- K=1024 (requires XDNA engine extension from K=192 to K=1024)
- 18 occurrences per forward pass
- Q8_0 weights match XDNA INT8 kernel

**Alternative: attn_k / attn_v** (full attention layers)
- N=512
- 6 occurrences per forward pass
- Same K=1024 requirement

---

## Next Steps (Phase 6C-6D)

1. **Extend XDNA persistent engine** to support K=1024 (tiling or larger kernel)
2. **Implement weight conversion** from Q8_0 GGUF layout to XDNA-packed INT8
3. **Target ssm_beta/alpha** as first real operation (smallest N=16)
4. **Verify correctness** against CPU reference
5. **Measure decode performance** (M=1)
6. **Extend to prefill** (M>1) if decode works

---

## Phase 6B Status: COMPLETE

Graph captured successfully. All MUL_MAT operations documented with dimensions, quantization, and layer mapping.