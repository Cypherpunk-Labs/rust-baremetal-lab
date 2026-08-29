# llamacpp — ggml compute engine in the bare-metal kernel

Implements the decision from `research/spike02/report.md`: **replace Burn's compute
layer with the freestanding `ggml` CPU backend** (Strategy A), driven from Rust over a
narrow FFI seam, so SmolLM-135M runs natively on `aarch64-unknown-none` under QEMU —
with the numbers validated against an independent oracle.

**Status: DONE and verified end-to-end.**

```
$ make run-interactive   # boots the kernel in QEMU, UART chat
[Kernel] SmolLM-135M ready. Type a prompt, Ctrl-D to exit.
You> The capital of France is
Bot>  Paris. It is the largest city in France and the second largest in the world. It is also the capital of the country. ...
```

No Python, no host OS. The exact same ggml forward runs on the host (parity harness) and
in the bare-metal kernel (embedded 513 MB F32 safetensors).

## Workspace layout

- `ggml-src/` — vendored, trimmed snapshot of llama.cpp's `ggml/` subtree (CPU backend
  only). Source of the static `libggmlcpu.a` built by `ggml-sys`.
- `ggml-sys/` — `#[no_std]`-capable `-sys` crate. `build.rs` compiles the ggml CPU
  backend with the host clang (for the `harness`) or the Arm GNU Toolchain
  (`aarch64-none-elf-gcc`/`g++`, Newlib + libstdc++) for `aarch64-unknown-none`.
  `src/lib.rs` holds hand-written FFI bindings (no bindgen, so it builds for the
  bare-metal target without host libclang or `cc`'s std-detection).
- `os-api/` — the shim layer providing the libc/libstdc++ symbols ggml expects on the
  bare-metal target (`malloc`/`free`/`posix_memalign`, pthread no-ops, clock/getenv
  stubs, `fprintf`/`printf` → UART) over the kernel heap. On the host it is empty.
- `model/` — `no_std` SmolLM architecture + the ggml forward graph
  (`src/forward.rs`): embedding → 30× (RMSNorm → QKV → RoPE → causal attention →
  output proj → residual → RMSNorm → SwiGLU → residual) → final norm → tied LM head.
- `model-data/` — embeds the 513 MB F32 `model.safetensors` (`include_bytes!`) + the
  tokenizer blob.
- `parity/` — Rust test harness (`tests/`) that validates the ggml forward against a
  pure-Rust reference **and** against the committed `burn/` kernel as the ground-truth
  oracle.
- `harness/` — host (`std`) binary exercising the FFI, model parse, and layer-0 diag.
- `kernel/` — the bare-metal kernel (boot/MMU/UART/2 GiB heap + tokenizer, ported from
  `burn/`), wired to `ggml-sys` + `os-api` + `model` + `model-data`. This is the
  SmolLM-135M chat loop that runs in QEMU.

## Quick start

```sh
# host validation (parity tests + generation smoke)
cargo test -p parity                              # layer0 + full_logits parity
cargo test -p parity --test parity -- --ignored   # gen_smoke_ggml, first_token_matches_burn, ...

# bare-metal kernel in QEMU (aarch64-unknown-none)
make run-interactive                              # requires ARM_GNU_TOOLCHAIN + QEMU (see Makefile)
```

Toolchain: Rust nightly with `rust-src` + `build-std` (`.cargo/config.toml` enables it),
and the Arm GNU Toolchain `aarch64-none-elf` (with Newlib + libstdc++, **not** brew's
`aarch64-elf-*`). `ARM_GNU_TOOLCHAIN` must point at it (the Makefile exports it).

## Status vs. the spike02 tickets

| Ticket | Scope | Status |
|---|---|---|
| T1 | Freestanding ggml feasibility build (`libggmlcpu.a` for `aarch64-none-elf`) | **DONE** — `make kernel` links & runs |
| T2 | os-api shim (`malloc`, pthread no-ops, clock, UART `fprintf`) | **DONE** — kernel boots and computes on-device |
| T3 | Load weights into RAM (safetensors embedded; zero-copy `ggml_get_rows`) | **DONE** — embedded F32 safetensors, no GGUF/mmap needed |
| T4 | Single-kernel parity harness (mul_mat orientation, repeat ordering) | **DONE** — `parity/tests/mulmat.rs` |
| T5 | Full ggml decode path replacing Burn's transformer | **DONE** — full 30-layer forward, causal, verified in QEMU |

## Challenges and how we overcame them

This is the war-story log. The summary: the forward math was implemented quickly and
"parity" against a hand-written reference passed 16/16 — but both implementations had the
*same* silent bug, so the easy checks said "correct" while the model produced garbage.
Only an **independent oracle** (the committed `burn/` kernel) exposed it.

### 1. The silent killer: `reshape` is not a transpose (the bug that broke everything)

**Challenge.** `layer0_parity` (every intermediate to ~1e-6) and `full_logits_parity`
(argmax 16/16) both passed, yet greedy decoding produced nonsense tokens. Two
implementations — ggml `model/src/forward.rs` and the pure-Rust `parity::Ref` — agreed
with each other but not with the real model.

**Root cause.** Attention `values` is produced by ggml in memory order
`[head_dim, seq, n_head]` (d fastest, then seq, then head). The forward converted it to
the `[hidden, seq]` the output projection expects with `ggml_reshape_2d(...)` — but
ggml's `reshape` **re-interprets the buffer without moving data**. What should have been a
transpose was just a re-tagging, so `o = Wo @ values` multiplied against a
head/token-scrambled array. Because the reference used the exact same reshape, the layer
diffs lined up at whatever garbage both produced.

**Overcome.** Compare against the **committed `burn/` kernel** (the trusted oracle), not
against a mirror of our own logic. Burn predicts token `7042` (" Paris") for the prompt;
a tiny test (`first_token_matches_burn`) pinned us to that token. Then traced backwards:
`xb` matched burn to 1e-8, so the bug was in attention; the values path was the only
place a layout-dependent op fed a matmul. Fix = true transpose:
`values = permute(values, 0, 2, 1, 3)` + `cont` before the `[hidden, seq]` reshape
(mirrored in the reference). Post-fix: maxdiff **0.026 → 0.0002**, 0 elements over
1e-2 (was 19645), first token = 7042, generation coherent.

**Lesson.** In ggml, `reshape`/`permute`/`view` never copy. If an axis *order* changes you
must `ggml_cont` a `ggml_permute`; a "reshape" is only valid when it's a pure releveling
of a contiguous block. And: two implementations that share a code assumption can both be
wrong — validate against something independent.

### 2. Two implementations agreeing ≠ correct (validation methodology)

**Challenge.** The `parity::Ref` was written to mirror the ggml graph *"to match the
graph under test"* (non-causal, grouped GQA). That made 16/16 parity look meaningful when
it was actually circular — both could share the same error.

**Overcome.** Established the committed `burn/` kernel (causal attention, grouped GQA,
Burn's tested tensor ops, running on the host) as the ground-truth oracle, and added tests
that compare ggml output to *burn's* tokens/intermediates, not to the mirror. `parity`
instructs: "validate against the oracle, never against a hand-written mirror alone."

### 3. Missing causal mask (and a wrong assumption)

**Challenge.** The original forward was non-causal (full attention over the whole
sequence). A hand-wave in the earlier report said generation re-feeding full history "is
causal for free" — wrong: earlier tokens attending to the last token pollutes every
position's representation.

**Overcome.** Added the mask on the scores tensor
`[ne0=key, ne1=query, ne2=head]` via `ggml_diag_mask_inf(ctx, scores, 0)` (masks
key > query — the same semantics as burn's `tril_mask`). Bound the FFI call in
`ggml-sys`, used it in both `forward` and `layer0_diag`, and mirrored the skip in the
reference. Maxdiff improved even before the transpose fix, confirming it was needed.

### 4. GQA: ggml's `repeat` is interleaved, SmolLM is grouped

**Challenge.** SmolLM maps query head `h` to kv head `h // (n_head/n_kv)` (grouped).
`ggml_repeat` assigns `h % n_kv` (interleaved) — semantically different.

**Overcome.** A tiny model-free test (`repeat_head_ordering`) proved ggml's interleave
(`[0,1,0,1]` for nh=4, nk=2). Implemented `gqa_group_expand`: reshape K/V to
`[hd*seq, n_kv]`, `ggml_get_rows` with index `idx[p] = p/rep` (a 1-D i32 tensor), reshape
back to `[hd, seq, n_head]`. Bolted into both `forward` and `layer0_diag`; the
`parity::Ref` uses the identical grouped mapping and agrees to 1e-6.

### 5. The graph node budget silently corrupts results

**Challenge.** With the default `ggml_new_graph` (2048 nodes) the full 30-layer graph far
exceeds capacity, and ggml **silently** produces all-zero logits — looks like a math bug.

**Overcome.** `ggml_new_graph_custom(ctx, 4096, false)`. Added a hard rule: always assert
logits are finite/nonzero after a forward. This is the single most dangerous
wrong-but-silent trap in the whole effort.

### 6. RoPE layout: apply in `[hd, n_head, seq]`, permute out

**Challenge.** RoPE is easiest applied with seq on axis 2 and equal across heads ggml's
NEOX (`mode 2`, `rotate_half` pairing `d` with `d+half`, `theta^-2k/dim`) is the
standard HF `neox` rope — but the surrounding layout gymnastics caused phantom mismatches
in earlier debugging.

**Overcome.** Reshape q/k to `[head_dim, n_head, seq]`, apply `ggml_rope` there, then
`permute(0,2,1,3)` + `cont` to the canonical `[head_dim, seq, n_head]`. `layer0_parity`
and burn cross-checks confirm the rotation (pos-0 identity + real rotation at pos 1).
This also closed the earlier false "rope is broken" spiral — the harness rope test
(identity at pos 0) had shown it was never rope.

### 7. Debug/tooling spiral (why earlier sessions burned hours)

**Challenge.** Every comparison needed a dump, and the dumps kept mixing layouts, spawning
recompiles of a 513 MB embedded model (`include_bytes!`), churning multi-GB temporary
binaries on a nearly-full disk.

**Overcome.**
- All validation lives in **Rust testcases** in `parity/tests/` (no Python, per project
  rule).
- Every flat dump carries an explicit `[axes]` layout annotation; comparisons use the
  *same* convention on both sides.
- Model-free ggml semantics probes (mul_mat orientation, repeat ordering) live
  separately in `parity/tests/mulmat.rs` so cheap tests don't need the 513 MB model.
- Generated artifacts never touch git; scratches go to a gitignored dir.

## Verified facts (locked in)

- `ggml_new_graph_custom(ctx, 4096, false)` required (node budget; default is too small).
- `ggml_mul_mat(a, b)` computes `aᵀ@b`; HF `[out, in]` weights stored verbatim with
  `ne0=in, ne1=out` feed `mul_mat(W, x)` correctly.
- `ggml_rms_norm` takes only `eps`; the affine weight is a separate `ggml_mul`.
- Scores `[ne0=key, ne1=query, ne2=head]`; `ggml_soft_max` softmaxes over `ne0` (keys) —
  exactly attention; `ggml_diag_mask_inf(0)` adds causality.
- `ggml_repeat` interleaves kv heads; SmolLM needs grouped (use `gqa_group_expand`).
- Tied LM head: `logits = embedᵀ @ x` via `mul_mat(embed, x)` → `[vocab, seq]`.

## Parity results (current)

```
layer0_parity: ALL LAYER-0 INTERMEDIATES MATCH (to ~1e-6)
full_logits_parity: seq=16 maxdiff=0.000238 elements_over_1e-2=0
argmax agree 16/16
first_token_matches_burn: 7042 == burn ground truth   OK
```

## FFI note

`ggml_graph_plan`/`ggml_graph_compute` return/pass `struct ggml_cplan` by value (56
bytes). Crossing that struct-by-value return from Rust is an sret ABI trap, so the
bindings intentionally expose only `ggml_graph_compute_with_ctx`, which plans and computes
internally and returns a plain `ggml_status`.