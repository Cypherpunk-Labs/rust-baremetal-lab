# rust-baremetal-lab
An experiment on running Rust on Baremetal using QEMU

---

# Building Baremetal Rust on MacOS (QEMU)

This guide walks through creating freestanding `#![no_std]` Rust kernels for both ARM64 and x86_64 architectures, compiled on MacOS and emulated via QEMU.

[Environment setup](environment.md)

[ARM64 Kernel](arm64.md)

[x86_64 Kernel](x86_64.md)

---

# Running a small language model 

Two complete bare-metal SmolLM-135M engines live in this repo:

- **[`llamacpp/` — ggml compute engine (recommended)](llamacpp/README.md)** — Burn's
  compute layer replaced with a freestanding `ggml` CPU backend driven from Rust over a
  narrow FFI seam (Strategy A from `research/spike02/report.md`). Verified end-to-end:
  host parity tests pass, and the `aarch64-unknown-none` kernel boots in QEMU and chats:
  `The capital of France is` → *" Paris. It is the largest city in France and the second
  largest in the world...."* No Python, no host OS.
- **[`burn/` — burn-flex engine (baseline)](burn/README.md)** — the original
  `no_std` kernel (KV cache, UART chat loop, GPT-2 BPE) plus a std `host` comparison
  binary and a QEMU/Linux-guest variant.

### Required toolchain

Arm GNU Toolchain (`aarch64-none-elf` + Newlib/libstdc++) for the kernel build, Rust
nightly with `rust-src` + `build-std`, and QEMU. See `llamacpp/Makefile` / `burn/`.

### Performance notes (burn-flex baseline; fp32)

3-way performance comparison (10-run averages):

| Path | Avg TPS |
|------|---------|
| std host, native macOS | 7.06 |
| no_std kernel, QEMU/HVF | 5.09 |
| std host in QEMU/Linux guest | 3.73 |
| Ollama M2 Pro | 338.44 |

ggml introduces the quantized-kernel path this baseline is moving toward; the immediate
goal was correctness parity, which is now proven.

### How the ggml path was validated

- `llamacpp/parity/tests/`: `layer0_parity` (intermediates to ~1e-6), `full_logits_parity`
  (maxdiff ~2e-4, argmax 16/16), `first_token_matches_burn` (matches the burn kernel's
  token `7042`/" Paris").
- Generation smoke test and a live QEMU chat run.
- The critical bug found & fixed: attention `values` was reshaped to `[hidden, seq]`
  without a true transpose (ggml reshape does not relayout memory), scrambling heads/tokens
  and silently corrupting logits. `permute(0,2,1,3)` + `cont` fixed it. Full story in
  `llamacpp/bugs.md`.


---

Gemini Lesson Buildout  https://gemini.google.com/app/41e5a753ce217c9e 

[Full Context File](baremetal_rust_QEMU_lesson_plan.md)
