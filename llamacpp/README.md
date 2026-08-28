# llamacpp — ggml compute engine in the bare-metal kernel

This crate workspace implements the tickets from `research/spike02/report.md`:
replace Burn's compute layer with the freestanding `ggml` CPU backend (Strategy A),
driven from Rust over a narrow FFI seam.

## Workspace layout

- `ggml-src/` — vendored, trimmed snapshot of llama.cpp's `ggml/` subtree
  (CPU backend only). Source of the static `libggmlcpu.a` built by `ggml-sys`.
- `ggml-sys/` — `#[no_std]`-capable `-sys` crate. `build.rs` compiles the ggml
  CPU backend:
  - **host** target: clang/clang++ + system SDK (used by `harness`);
  - `aarch64-unknown-none`: clang `--target=aarch64-none-elf -ffreestanding`
    with the bundled minimal C headers in `ggml-sys/freestanding-headers/`.
  `src/lib.rs` holds hand-written FFI bindings (no bindgen, so it builds for
  the bare-metal target without host libclang).
- `os-api/` — the T2 shim layer. On `aarch64-unknown-none` it provides the
  libc symbols ggml expects (`malloc`/`free`/`posix_memalign`, pthread no-ops,
  clock/getenv stubs, `fprintf`/`printf` -> UART sink) over the kernel's global
  allocator. On the host it is empty (system libc is used).
- `harness/` — host (`std`) validation binary. Proves T1 on the dev box and is
  the seed for the T4 parity harness.
- `kernel/` — the bare-metal kernel crate (boot platform ported from `burn/`:
  asm/UART/MMU/2 GiB heap + tokenizer), wired to `ggml-sys` + `os-api`. Intended
  T1-kernel / T5 target.

## Status

- **T1 (host): DONE + verified.** `cargo run -p harness` builds `libggmlcpu.a`
  and runs a `ggml_mul_mat` all-ones graph through the Rust FFI, printing
  `mul_mat c[0] = 8.000000` (correct).
- **T2 (os-api shim): written.** Compiles; becomes active only on the
  `aarch64-unknown-none` target.
- **T1-kernel / T3 / T4 / T5: scaffolded**, gated on the blocker below.

## Blocker (open question #1 from the report)

The current `ggml` CPU backend (`ggml-cpu.cpp`, `repack.cpp`, ...) includes C++
STL headers (`<cstdint>` then `<vector>`/`<string>`/...). A truly freestanding
build with `-nostdlib++` has no C++ standard library, so `build.rs`'s
`aarch64-unknown-none` branch currently fails at the first STL include. This
lives in the ggml CPU backend itself, not just the llama.cpp model layer.

**Fix path (not yet done):** vendor/compile a freestanding `libc++` (or a
minimal STL subset) for `aarch64-none-elf` and point the build at it. Until
then the kernel crate cannot link ggml; the host path above is the validated
route for T1/T4 work today.

## FFI note

`ggml_graph_plan`/`ggml_graph_compute` return/pass `struct ggml_cplan` by value
(56 bytes). Crossing that struct-by-value return from Rust is an sret ABI trap,
so the bindings intentionally expose only `ggml_graph_compute_with_ctx`, which
plans and computes internally and returns a plain `ggml_status`.
