#!/usr/bin/env python3
"""
Extract MUL_MAT shapes from Qwen3.5-0.8B model graph
"""

import json

def analyze_qwen_model():
    # Qwen3.5-0.8B architecture
    # 24 layers
    # hidden_size = 1024
    # num_attention_heads = 16
    # num_key_value_heads = 8
    # intermediate_size = 3072 (FFN up projection)
    # vocab_size = 151936
    
    hidden_size = 1024
    num_heads = 16
    num_kv_heads = 8
    head_dim = 64
    intermediate_size = 3072
    vocab_size = 151936
    num_layers = 24
    
    q_dim = num_heads * head_dim  # 1024
    kv_dim = num_kv_heads * head_dim  # 512
    
    shapes = []
    
    # Context lengths for PP
    context_lengths = [4096, 8192, 16384, 32768, 65536]
    
    for ctx_len in context_lengths:
        # Prefill shapes (M = ctx_len)
        shapes.append({
            "op_name": "q_proj",
            "layer": "all",
            "M": ctx_len,
            "K": hidden_size,
            "N": q_dim,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": num_layers,
            "context_length": ctx_len,
            "phase": "prefill"
        })
        shapes.append({
            "op_name": "k_proj",
            "layer": "all",
            "M": ctx_len,
            "K": hidden_size,
            "N": kv_dim,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": num_layers,
            "context_length": ctx_len,
            "phase": "prefill"
        })
        shapes.append({
            "op_name": "v_proj",
            "layer": "all",
            "M": ctx_len,
            "K": hidden_size,
            "N": kv_dim,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": num_layers,
            "context_length": ctx_len,
            "phase": "prefill"
        })
        shapes.append({
            "op_name": "o_proj",
            "layer": "all",
            "M": ctx_len,
            "K": q_dim,
            "N": hidden_size,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": num_layers,
            "context_length": ctx_len,
            "phase": "prefill"
        })
        shapes.append({
            "op_name": "up_proj",
            "layer": "all",
            "M": ctx_len,
            "K": hidden_size,
            "N": intermediate_size,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": num_layers,
            "context_length": ctx_len,
            "phase": "prefill"
        })
        shapes.append({
            "op_name": "gate_proj",
            "layer": "all",
            "M": ctx_len,
            "K": hidden_size,
            "N": intermediate_size,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": num_layers,
            "context_length": ctx_len,
            "phase": "prefill"
        })
        shapes.append({
            "op_name": "down_proj",
            "layer": "all",
            "M": ctx_len,
            "K": intermediate_size,
            "N": hidden_size,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": num_layers,
            "context_length": ctx_len,
            "phase": "prefill"
        })
        shapes.append({
            "op_name": "ssm_beta",
            "layer": "all",
            "M": ctx_len,
            "K": hidden_size,
            "N": 16,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": num_layers,
            "context_length": ctx_len,
            "phase": "prefill"
        })
        shapes.append({
            "op_name": "ssm_alpha",
            "layer": "all",
            "M": ctx_len,
            "K": hidden_size,
            "N": 16,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": num_layers,
            "context_length": ctx_len,
            "phase": "prefill"
        })
    
    # Decode shapes (M = 1)
    decode_shapes = [
        {"op_name": "q_proj", "M": 1, "K": hidden_size, "N": q_dim, "occurrence": num_layers},
        {"op_name": "k_proj", "M": 1, "K": hidden_size, "N": kv_dim, "occurrence": num_layers},
        {"op_name": "v_proj", "M": 1, "K": hidden_size, "N": kv_dim, "occurrence": num_layers},
        {"op_name": "o_proj", "M": 1, "K": q_dim, "N": hidden_size, "occurrence": num_layers},
        {"op_name": "up_proj", "M": 1, "K": hidden_size, "N": intermediate_size, "occurrence": num_layers},
        {"op_name": "gate_proj", "M": 1, "K": hidden_size, "N": intermediate_size, "occurrence": num_layers},
        {"op_name": "down_proj", "M": 1, "K": intermediate_size, "N": hidden_size, "occurrence": num_layers},
        {"op_name": "ssm_beta", "M": 1, "K": hidden_size, "N": 16, "occurrence": num_layers},
        {"op_name": "ssm_alpha", "M": 1, "K": hidden_size, "N": 16, "occurrence": num_layers},
    ]
    
    for s in decode_shapes:
        shapes.append({
            **s,
            "layer": "all",
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "context_length": 1,
            "phase": "decode"
        })
    
    # Add LM head (vocab projection)
    shapes.append({
        "op_name": "lm_head",
        "layer": "final",
        "M": 1,  # decode
        "K": hidden_size,
        "N": vocab_size,
        "dtype": "Q8_0",
        "quantization": "Q8_0",
        "occurrence": 1,
        "context_length": 1,
        "phase": "decode"
    })
    
    # For PP, the LM head is applied to the last token only typically
    for ctx_len in context_lengths:
        shapes.append({
            "op_name": "lm_head",
            "layer": "final",
            "M": 1,  # usually only last token for causal LM
            "K": hidden_size,
            "N": vocab_size,
            "dtype": "Q8_0",
            "quantization": "Q8_0",
            "occurrence": 1,
            "context_length": ctx_len,
            "phase": "prefill"
        })
    
    # Calculate estimated FLOPs and memory
    for s in shapes:
        s["flops"] = 2 * s["M"] * s["K"] * s["N"] * s["occurrence"]
        s["weight_bytes"] = s["K"] * s["N"] * 1  # Q8_0 = 1 byte per weight + scale
        s["input_bytes"] = s["M"] * s["K"] * 1  # Q8_0
        s["output_bytes"] = s["M"] * s["N"] * 4  # F32 output
    
    # Sort by total FLOPs descending
    shapes.sort(key=lambda x: x["flops"], reverse=True)
    
    return shapes


def main():
    shapes = analyze_qwen_model()
    
    # Save to JSON
    output = {
        "model": "Qwen3.5-0.8B",
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 64,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "shapes": shapes
    }
    
    with open('/workspace/llama.cpp/experiments/npu-int8-gemm/qwen_xdna_shape_matrix.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(shapes)} shapes to qwen_xdna_shape_matrix.json")
    
    # Print summary
    print("\n=== Shape Matrix Summary ===")
    for s in shapes[:25]:
        print(f"  {s['op_name']:12s} M={s['M']:6d} K={s['K']:5d} N={s['N']:6d} "
              f"ctx={s['context_length']:6d} phase={s['phase']:6s} "
              f"occ={s['occurrence']:2d} FLOPs={s['flops']/1e9:.2f}G")
    
    # Group by unique (M,K,N) tuples
    unique_shapes = {}
    for s in shapes:
        key = (s['M'], s['K'], s['N'], s['phase'])
        if key not in unique_shapes:
            unique_shapes[key] = {
                "M": s['M'], "K": s['K'], "N": s['N'], "phase": s['phase'],
                "ops": [], "total_occurrence": 0, "context_lengths": set()
            }
        unique_shapes[key]["ops"].append(s['op_name'])
        unique_shapes[key]["total_occurrence"] += s['occurrence']
        unique_shapes[key]["context_lengths"].add(s['context_length'])
    
    print("\n=== Unique Shape Tuples ===")
    for key, info in sorted(unique_shapes.items(), key=lambda x: 2*x[1]['M']*x[1]['K']*x[1]['N']*x[1]['total_occurrence'], reverse=True):
        print(f"  M={info['M']:6d} K={info['K']:5d} N={info['N']:6d} phase={info['phase']:6s} "
              f"ops={info['ops']} total_occ={info['total_occurrence']} "
              f"ctxs={sorted(info['context_lengths'])}")

if __name__ == "__main__":
    main()