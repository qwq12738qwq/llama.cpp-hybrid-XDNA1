# Phase 7B Qwen Candidate Matrix

| operator | shape | phase | count | GPU time | CPU time | XDNA support | XDNA time | persistent XDNA time | transfer | weight residency | packing possible | K tiling possible | overlap opportunity | critical-path contribution | 20% relevance | decision | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SSM | M512_K1024_N16 | prefill | 36 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |
| ATTENTION | M512_K1024_N2048 | prefill | 18 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |
| FFN | M512_K1024_N3584 | prefill | 48 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |
| ATTENTION | M512_K1024_N6144 | prefill | 24 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |
| SSM/ATTN | M512_K2048_N1024 | prefill | 24 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=2048>512 |
| FFN | M512_K3584_N1024 | prefill | 24 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=3584>512 |
| EMBEDDING | M0_K1024_N248320 | prefill | 1 | N/A (blocked) | N/A | XDNA_BLOCKED_BY_K | N/A (blocked) | N/A (blocked) | N/A | N/A | NO | NO | NONE | 0.1% | NO | BLOCKED | K=1024>512 |
