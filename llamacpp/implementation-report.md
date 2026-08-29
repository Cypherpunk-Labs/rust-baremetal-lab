# Implementation Report — ggml forward parity for SmolLM-135M

## Status: SOLVED (verified working generation)

The ggml CPU forward for SmolLM-135M now generates coherent text on the host,
verified against the committed Burn kernel as ground truth:

```
PROMPT: The capital of France is
GGML-GENERATED:  Paris. It is the largest city in France and the second
largest in the world. It is also ...
```

The burn kernel predicts `7042` (= " Paris") as the first continuation token; the ggml
path now predicts the identical token (before the fix it produced garbage token `258`).

## The root cause (was never solved until now)

`model/src/forward.rs` computed attention `values` as `[head_dim, seq, n_head]` and then
converted it to `[hidden, seq]` for the output projection with a plain
`ggml_reshape_2d`. Because ggml's reshape re-interprets the buffer without moving data,
the resulting `[hidden, seq]` had the head/token axes scrambled, and `o = Wo @ values`
read the wrong elements. Logits were wrong but finite, so greedy decoding produced
plausible-looking tokens that were garbage.

The hand-written pure-Rust reference (`parity/src/lib.rs`) had the same wrong reshape, so
ggml vs reference parity passed 16/16 while both diverged from the correct oracle.
Prior sessions kept "fixing" RoPE, GQA, and graph-node-budget (those were real but
secondary); the values-transpose bug — the one that actually broke generation — was
masked because the two implementations mirrored each other's mistake.

**Fix:** transpose `values` with `permute(0,2,1,3)` + `cont` before the `[hidden, seq]`
reshape (and mirror in the reference). Result: `full_logits_parity` maxdiff 0.026 -> 0.0002,
0 elements over 1e-2 (was 19645), argmax 16/16, generation coherent.

## Verified correct (locked in)

- ggml mul_mat conventions, RMSNorm arity, RoPE NEOX layout, grouped-GQA expand
  (`gqa_group_expand`), causal mask (`ggml_diag_mask_inf(scores, 0)`), tied LM head,
  `ggml_new_graph_custom(4096)` budget.
- Parity tests `mul_mat_orientation`, `repeat_head_ordering`, `layer0_parity`,
  `full_logits_parity` all pass; `gen_smoke_ggml` produces coherent text; the reference
  first-token matches burn (`first_token_matches_burn`).

## Verified on-device (kernel/QEMU) — DONE

The same ggml forward runs in the bare-metal kernel (`kernel/src/main.rs`, embedded via
513 MB `include_bytes!`) in a UART chat loop. Booted under QEMU:

```
[Kernel] MMU on; heap via Newlib malloc
[ggml] mul_mat c0 = 8.000 (status 0)
[Kernel] SmolLM-135M ready. Type a prompt, Ctrl-D to exit.
You> The capital of France is
Bot>  Paris. It is the largest city in France and the second largest in the world. It is also the capital of the country. ...
```

Run with `make run-interactive` in `llamacpp/` (requires `ARM_GNU_TOOLCHAIN` + QEMU,
see `llamacpp/Makefile`). Build: `cargo build -p kernel --target aarch64-unknown-none`.

## Full challenge log

Every challenge and how it was overcome is documented in
[`README.md` (Challenges and how we overcame them)](README.md) and [`bugs.md`](bugs.md)
— including the headline **"reshape is not a transpose"** bug, the circular-validation
trap, the causal mask, grouped-GQA, and the silent graph-node-budget failure.

## Lessons locked in

1. A plain ggml `reshape` is never a transpose. If an axis order changes, you must
   `permute` + `cont`.
2. Two implementations that share a code idea can both be wrong and "agree". Validate
   against an independent oracle (the committed burn kernel), not against a mirror.
3. Causal mask is required; "causal for free" was incorrect.
4. Keep all validation in Rust testcases; keep test binaries/scratches gitignored.