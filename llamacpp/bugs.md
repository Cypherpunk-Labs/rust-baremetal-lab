# Confirmed bugs (fixed) and validated facts — SmolLM-135M ggml forward

All of the following were confirmed via Rust testcases (`parity/tests/`) and cross-checked
against an independent ground truth: the committed Burn kernel (`burn/kernel`, causal
attention, grouped GQA) run on the host, which predicts token `7042` (" Paris") for the
prompt "The capital of France is".

## THE bug that broke generation (now fixed)

**`values` was reshaped to `[hidden, seq]` without a true transpose.** The attention
`values` tensor lives in memory as `[head_dim, seq, n_head]` (d fastest, then token s,
then head h). The forward converted it for the output projection with
`ggml_reshape_2d(ctx, values, hidden, seq)` — but ggml reshape does NOT move data, it
re-interprets the buffer. The resulting `[hidden, seq]` had `(h, d, s)` order scrambled
(seq varying at stride `hd`, head at stride `hd*seq`), so `o = Wo @ values` multiplied
against a head/seq-swapped array. Logits were wrong-but-nonzero, and greedy argmax
produced garbage (`token 258` etc.).

The handwritten `Ref` in `parity/src/lib.rs` had the SAME wrong reshape, so ggml vs Ref
"parity" passed (16/16 argmax, layer0 to 1e-6) while BOTH disagreed with the burn
ground truth (`7042`). **This is why the earlier 16/16 claim was misleading: it compared
ggml against a mirror of itself, not against a correct oracle.**

**Fix** (in `model/src/forward.rs` forward + layer0_diag, and mirrored in the Ref):
```
values = permute(values, 0, 2, 1, 3);  // [hd, seq, nh] -> [hd, nh, seq]
values = cont(values);                  // materialize the transposed order
values = reshape_2d(values, hidden, seq); // now correctly (h*hd + d) + s*hidden
```

Verified result: `full_logits_parity` maxdiff dropped 0.026 -> 0.0002 with **0** elements
over 1e-2 (was 19645), argmax still 16/16, and the ggml production forward now generates:
> "The capital of France is **Paris. It is the largest city in France and the second
> largest in the world. It is also** ..."

## Other validated facts (confirmed correct, do not regress)

- **`ggml_new_graph_custom(ctx, 4096, false)` is required.** Default node capacity (2048)
  is too small for the 30-layer graph and silently produces all-zero logits.
- **Causal mask:** scores are `[ne0=key, ne1=query, ne2=head]`; `ggml_diag_mask_inf(scores, 0)`
  masks key > query — exactly the causal mask the burn kernel applies. Both ggml and the
  committed burn use it. (The earlier "generation is causal for free" note was wrong: the
  mask is required and is now in place.)
- **`ggml_mul_mat(a, b)` for attention** with `a = K [hd, seq_k, nh]`, `b = Q [hd, seq_q, nh]`
  gives `[ne0=seq_k, ne1=seq_q, ne2=nh]` with `result[i0=key, i1=query, h] = sum_d K[d,key]Q[d,query]`.
  `ggml_soft_max` softmaxes over ne0 (keys) = correct attention.
- **GQA grouping.** SmolLM groups kv heads: query head `h` uses kv head `h / (n_head/n_kv)`.
  `ggml_repeat` interleaves (`h % n_kv`) — wrong for this model. Use `gqa_group_expand`
  (get_rows with idx[p]=p/rep). Confirmed by `repeat_head_ordering` (repeat gives [0,1,0,1])
  and by parity vs burn.
- **RoPE layout.** Apply `ggml_rope` in `[hd, n_head, seq]` (seq on axis 2), ROPE_MODE=2
  (NEOX), then `permute(0,2,1,3)` + `cont` to `[hd, seq, n_head]`. Matches burn's
  rotate_half + inv_freq convention.
- **Tied LM head.** `logits = embed^T @ x` via `ggml_mul_mat(embed, x)` -> `[vocab, seq]`.
- **Every flat dump needs a shape+layout annotation.** ggml reshape/permute don't
  relayout memory; a "reshape" that is really a transpose scrambles logical axes. The
  values-transpose bug above is exactly this class of error.

## Process rules (from the earlier loop, kept)

- Tests are Rust testcases in `parity/tests/`; no python.
- Scratch data goes in a gitignored dir, nothing generated is committed.
- Validate against the committed `burn/` kernel output (the trusted oracle), never against
  a hand-written mirror alone.