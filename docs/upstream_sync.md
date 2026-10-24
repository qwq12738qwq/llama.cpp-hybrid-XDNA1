# Upstream sync record (measured, not merged)

- Local base: 7620399f5 (master, Sep 6 2026) + untracked ggml-xdna backend
  (in src/ggml-xdna/) and the 11 tracked-file diffs (patches/).
- Upstream tip measured: 8e5a58df8 (Oct 23 2026), built clean in a detached
  worktree (Vulkan only), same machine / model / sandwich protocol.
- Result: +17.4% uniform on Ling-3.0-tiny pp curve (567 vs 483 tok/s @16384;
  full table in ../README.md section 3); long-PP slope unchanged.
- Prime movers: #27952 RDNA3 int8 coopmat, #29988 FA shmem fix, #30003
  sparse FA (coopmat2; model-gated), MoE MUL_MAT_ID work.
- Intentionally NOT merged into the sealed tree (see README section 6);
  rebase onto upstream master is the recommended action for any future work.
