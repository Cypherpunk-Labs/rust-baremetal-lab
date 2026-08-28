# SmolLM-135M on a Bare-Metal ARM64 Kernel (Burn + QEMU)

Run [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) inference in a
`no_std`, single-core ARM64 kernel built with [Burn](https://burn.dev) 0.21 +
`burn-flex`, booted as a raw ELF under QEMU `virt` on Apple Silicon.

The kernel has **no OS** — no std, no libc. It enables a minimal identity MMU,
maps RAM as Normal cacheable memory, drives the PL011 UART directly, allocates a
2 GiB heap, deserializes the embedded weights, and runs a greedy autoregressive
generation loop, printing tokens over the serial console.

**This project is a deliberate exercise in running Burn on bare metal.** Both TCG
and HVF work end-to-end; HVF requires the identity MMU (details below).

---

## Layout

```
burn/
├── Makefile              # all build/test/run commands (see below)
├── README.md
├── Cargo.toml            # workspace: kernel + model-builder
├── linker.ld             # kernel linker script (entry 0x40080000)
├── rust-toolchain.toml   # nightly + aarch64-unknown-none
├── download-weights.sh   # checksum-verified safetensors download
├── kernel/
│   ├── .cargo/config.toml    # build-std + -Tlinker.ld  (MUST build from kernel/)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs       # boot asm, UART, allocator, generation loop
│       ├── mmu.rs        # identity MMU (required for HVF)
│       ├── model.rs      # SmolLM-135M architecture (burn-nn)
│       ├── lib.rs        # shared with model-builder
│       └── smollm-135m.bin  # embedded weights (gitignored, generated)
└── model-builder/
    ├── Cargo.toml
    ├── model.safetensors    # gitignored, via download-weights.sh
    └── src/
        ├── main.rs          # safetensors -> kernel record (.bin)
        └── bin/validate.rs  # host-side reference run
```

Note the workspace target dir is `burn/target/` (root), so the kernel ELF lands at
`burn/target/aarch64-unknown-none/release/kernel`.

---

## Prerequisites

- Rust nightly (see `rust-toolchain.toml`; tested with `1.99.0-nightly`)
  with `rustup target add aarch64-unknown-none`
- `qemu-system-aarch64` (tested with **11.1.0** from Homebrew)
- A macOS host with Hypervisor.framework (for HVF; not required for TCG)

`aarch64-unknown-none` includes `+neon` and has no soft-float, so Burn/pulp/gemm
compile cleanly. `build-std = ["core", "alloc"]` + `compiler-builtins-mem` is
required (see `kernel/.cargo/config.toml`). That config is only picked up when
Cargo runs inside `kernel/` — hence the Makefile `kernel` target does
`cd kernel && cargo build ...`.

---

## Build & Test Commands (codified in `Makefile`)

| Command               | Purpose                                                                     |
|-----------------------|-----------------------------------------------------------------------------|
| `make weights`        | Download SmolLM-135M `model.safetensors` (SHA-256 verified) into `model-builder/` |
| `make model`          | Convert safetensors -> `kernel/src/smollm-135m.bin` (runs model-builder)    |
| `make tokenizer`      | Convert `model-builder/tokenizer.json` -> `kernel/src/tokenizer.bin`        |
| `make tokenizer-check`| Validate the tokenizer blob against known token ids (host)                  |
| `make validate`       | Host reference run: prints 10 greedy tokens (~2 s/step)                     |
| `make kernel`         | Build the no_std aarch64 kernel ELF                                         |
| `make host`           | Build the standalone **std** comparison binary (`host/`, release)          |
| `make host-run`       | Run the std binary (greedy benchmark + chat)                                |
| `make host-linux`     | Cross-compile the std binary to a **static aarch64 Linux ELF** (musl)       |
| `make linux-image`    | Download the Ubuntu 24.04 ARM64 cloud image (SHA-256 verified)              |
| `make seed`           | Regenerate the cloud-init NoCloud seed ISO (`model-builder/linux/seed.iso`) |
| `make run-linux`      | Boot the Ubuntu guest under QEMU/HVF; run the std host binary inside it     |
| `make run-tcg`        | Boot kernel under TCG emulation (**works**, ~5 s/step with KV cache)   |
| `make run-hvf`        | Boot kernel under HVF acceleration (**works**, ~0.19 s/step with KV cache) |
| `make chat`           | Boot under HVF into the interactive chat loop (type a prompt)            |
| `make bench N=5`      | Sequential perf comparison: run **host** and **kernel** N times, avg TPS  |
| `make bench-linux N=5`| Perf comparison: run the std host binary **inside the QEMU/Linux guest** N times |

Expected output (host `make validate` and kernel agree on tokens):

```
[step 0] token=30
[step 1] token=824
...
Final tokens: [15496, 11, 30, 824, 31, 34, 32, 33, 39, 31, 32, 33]
```

Both paths also print timing/throughput. The kernel reads the ARM generic timer
(`cntpct_el0` / `cntfrq_el0`) directly — no `std`, no interrupts:

```
# host (make validate)
avg 2023.55 ms/token, 0.494 tokens/sec

# kernel (make run-hvf)
[step 0] token=30 (259 ms)
...
[Kernel] avg 203.30 ms/token, 4.919 tokens/sec (cntfrq=24000000 Hz)

# kernel (make run-tcg)
[step 0] token=30 (15035 ms)
...
```

HVF (native execution) is ~46x faster than TCG and ~6x faster than the host
reference. With the KV cache, generation is ~O(1) per step: the prompt is
processed once and later steps feed a single token (~194 ms on HVF vs ~5.4 s on
TCG) instead of re-running full-causal attention over the whole prefix.

Manual QEMU invocation (what `make run-tcg` runs):

```sh
qemu-system-aarch64 -machine virt -cpu max -m 4G -serial stdio \
  -display none -kernel target/aarch64-unknown-none/release/kernel
```

The `-cpu max` flag is **required** — the default virt CPU rejects the ELF
("image is from incompatible architecture").

---

## Kernel Details

- **Entry point** `0x40080000` (linker.ld), stack top after BSS + 1 MiB.
- **Boot asm** (`_start`, in `kernel/src/main.rs`): enables FP/SIMD
  (`CPACR_EL1.FPEN = 3 << 20` + `isb`) **before** any Rust runs, sets SP, calls
  `kernel_main`. Without this, the very first NEON instruction (e.g. a `vec!`
  `ldr q0`) traps as Undefined Instruction (ESR EC=0x7) with no vector table →
  CPU loops at `0x200`. *This was the original "QEMU hangs during load" bug.*
- **UART**: PL011 at `0x0900_0000` (`DR` +0x000, `FR` +0x018 bit5 TXFF, `CR` +0x030 = 0x301).
- **Heap**: `0x7000_0000`, 2 GiB, `linked_list_allocator::Heap` wrapped in an
  `UnsafeCell` + custom `GlobalAlloc` (safe: single core, no interrupts).
- **MMU**: identity map of the first 16 GiB (`kernel/src/mmu.rs`), Normal WB,
  AP=`0b00`, TTBR1 mirrors TTBR0, enabled from `kernel_main` before the heap or
  model code runs. Required so Burn's exclusive atomics work under HVF.
- **Weights**: `include_bytes!("smollm-135m.bin")` (538,067,477 bytes).
- `embed_tokens.weight` is `[49152, 576]` and is loaded **without transpose**
  (Burn's `load_embedding`); the model-builder must not transpose it.
- **Model**: `SmolLmModel` implements real Llama-style inference — RoPE
  (`rope_theta=10000`), GQA (9 q-heads / 3 kv-heads, repeat-interleaved), scaled
  dot-product attention with a causal `tril_mask`, and a **tied LM head**
  (`tie_word_embeddings=true`: logits = `hidden @ embed_tokens.weight^T` →
  `[B, S, 49152]`). Greedy argmax is over the vocab dim. A **KV cache**
  (`KvCache` + `forward_step`) caches RoPE'd keys/values per layer; the prompt
  is processed once and each later step attends over the cached prefix with
  ~O(1) work. RoPE is applied at insertion time (keys) and at query time with
  absolute positions.
- **Tokenizer**: GPT-2 byte-level BPE (`kernel/src/tokenizer.rs`). A host-side
  `tokenizer-builder` converts `model-builder/tokenizer.json` into a compact
  binary blob (`kernel/src/tokenizer.bin`, ~1.1 MB) that the no_std kernel
  decodes at boot: `byte_to_id[256]`, vocab byte-strings, and merge triples
  (left, right, merged) in rank order. Encode maps input bytes to base byte
  tokens then repeatedly applies the lowest-rank adjacent merge; decode maps
  token ids back to bytes. Correct for all normal text.
- **Sampling / chat**: `main.rs` samples with temperature 0.8 + top-k 40 from
  a deterministic xorshift64* PRNG seeded from the ARM generic counter (no
  `std`, no `rand`). `make chat` reads a prompt over PL011 RX (with echo and
  backspace), tokenizes, generates up to 64 tokens, decodes and streams output,
  then loops.

---

## Host (std) comparison binary (`host/`)

A standalone `std` crate that reuses the **same** `kernel` library code
(`kernel::model`, `kernel::tokenizer`) and the **same** `burn_flex::Flex` CPU
backend to run the identical greedy benchmark and chat loop on the host. This
lets you compare the no_std kernel (under HVF) against a normal `std` binary
with zero changes to the kernel sources — the no_std-only parts (UART, custom
heap, ARM timer, boot asm) live only in `kernel/src/main.rs` and are replaced
with `std` equivalents (`std::time::Instant`, `stdin`).

- `make host` builds it in release; `make host-run` runs the greedy benchmark
  then the interactive chat loop (prompt from `stdin`).
- `./target/release/host --bench N` runs the greedy benchmark N times
  sequentially and reports the average TPS.
- `make bench N=5` runs **both** the host binary and the kernel (N QEMU/HVF
  boots) and prints the average TPS for each, e.g.:

```
=== Host (std) benchmark: 10 runs ===
[Host] avg TPS over 10 runs: 7.060
[Host] Final tokens: [15496, 11, 30, 824, 31, 34, 32, 33, 39, 31, 32, 33]

=== Kernel (HVF) benchmark: 10 runs ===
  run 1 TPS: 5.107
  ...
  run 10 TPS: 5.128
  avg TPS over 10 runs: 5.091
```

Both paths must produce the identical token sequence (correctness gate). The
sampling helpers (`Rng`, `expf`, `sample_token`) are duplicated in `host/src/
main.rs` (not shared) so the kernel source stays untouched.

---

## Linux guest (std host binary under QEMU/HVF)

To compare **std-under-QEMU** vs **no_std-kernel-under-QEMU** (isolating the
std-vs-no_std difference, both under HVF), the same `host` binary is
cross-compiled to a **statically linked aarch64 Linux ELF** and run inside a
minimal Ubuntu 24.04 guest that QEMU boots with HVF:

- `make host-linux` — `cargo build -p host --release --target
  aarch64-unknown-linux-musl`. Linking is done with the toolchain's bundled
  `rust-lld` (resolved by `scripts/aarch64-linux-musl-linker.sh`, so the config
  has no hardcoded path) — no external cross-gcc required. `.cargo/config.toml`
  at the workspace root scopes this to the musl target only.
- `make linux-image` — downloads the official Ubuntu 24.04 ARM64 cloud image
  (`model-builder/linux/ubuntu-24.04-server-cloudimg-arm64.img`, qcow2, checksum
  verified by `download-linux-image.sh`). It is UEFI-only, so the boot script
  attaches QEMU's `edk2-aarch64-code.fd` firmware (`-drive if=pflash`).
- `make seed` — regenerates the NoCloud cloud-init seed ISO from
  `model-builder/linux/user-data` + `meta-data`. The user-data runs the
  benchmark in **`bootcmd`** (runs on *every* boot; `runcmd` is per-instance and
  would only run once), mounting the 9p share and teeing the host binary's
  output to the share so the macOS side can read `guest.log`.
- `make run-linux` / `make bench-linux N=5` — boots the guest under HVF, mounts
  the 9p shared folder (`model-builder/linux/share/smollm/`, tag `host0`) with
  the static binary + `.bin` data files, runs `HOST_DATA_DIR=/host/smollm
  ./host --bench 1`, and powers off. `scripts/run-linux.sh` is the QEMU driver.

```
=== Host in QEMU/Linux guest benchmark: 10 runs ===
  run 1 TPS: 3.672
  ...
  run 10 TPS: 3.715
  avg TPS over 10 runs: 3.725
```

Both paths (native std host and in-guest std host) produce the identical token
sequence, and the host reads its `.bin` files through `HOST_DATA_DIR` (default
`kernel/src/`), so the same binary runs natively on macOS and inside the guest.

---

## 3-way performance comparison (10-run averages)

The greedy benchmark (`--bench`) was run 10 times on each of the three paths —
all on QEMU `virt`/HVF on the same Apple Silicon host, all producing the
identical token sequence `[15496, 11, 30, 824, 31, 34, 32, 33, 39, 31, 32, 33]`:

| Path | Description | Avg TPS |
|------|-------------|---------|
| `./target/release/host` | std binary, native macOS (no QEMU) | **7.06** |
| kernel ELF under HVF | no_std kernel, QEMU/HVF (`make bench`) | **5.09** |
| std host in QEMU/Linux | static musl ELF in Ubuntu guest under HVF (`make bench-linux`) | **3.73** |

Interpretation:
- **std vs no_std** (both bare QEMU/HVF, isolating the code path): the std host
  in the Linux guest (3.73 TPS) is ~1.4x slower than the no_std kernel
  (5.09 TPS) — the guest OS (page tables, virtio/9p, etc.) adds overhead vs a
  kernel that talks straight to the UART.
- **QEMU/Linux guest vs native** (both std): the same std binary drops from
  ~7.06 TPS native to ~3.73 TPS in the guest (~1.9x), i.e. roughly half the
  throughput once it runs under the guest OS rather than directly on macOS.

---

## The Journey (read this before re-treading)

This section documents what was already diagnosed so future sessions don't repeat
the investigation.

### 1. Why "QEMU hangs at 100% CPU with no RAM growth" (FIXED)

- QEMU pre-maps guest RAM, so RAM never growing is **normal**, not a symptom.
- The real hang was an **FP/SIMD access trap**: at reset `CPACR_EL1.FPEN = 0`, so
  the first NEON instruction traps as Undefined Instruction (ESR EC=0x7) at
  `ELR=0x40080554` (`ldr q0, [x8, #0x3f0]` from a `vec!` init). With no exception
  vector (`VBAR=0`) the CPU spins at `0x200`.
- Fix: enable FP/SIMD in `_start` (see above). After this, TCG boots and infers
  correctly.
- The heap/data path was never the problem.

### 2. HVF data-aborts on exclusive atomics (FIXED via identity MMU)

- Under TCG, everything works (this is why the puzzle was hard: *CPU vs accel
  changes the result*).
- Under HVF on Apple Silicon, the first exclusive-access atomic raised a **data
  abort**: `ESR_EL1=0x96000035` (EC=0x25 Data abort, same-EL; DFSC=0x35
  "unsupported exclusive or atomic access" — QEMU's term for a faulting
  exclusive-access instruction HVF has no emulation for), `FAR` pointing at the
  `ldaxrb` inside `spin::Once`/`cubecl_common::stub::RwLock` lazy init. This is
  QEMU gitlab **#1611** (HVF has no `ldxr/stxr` fault emulation; the KVM backend
  does). It was not config-dependent: `-cpu host`/`max`, `gic-version=3` all
  fault identically.
- Root cause: our kernel ran with the **MMU off**, so all memory was treated as
  **Device-attribute**, and Apple hardware does not support exclusive accesses
  to Device memory. Real OSes boot under HVF because they enable the MMU and run
  with **Normal cacheable** memory, where exclusive atomics execute natively.
- Fix: enable a minimal identity MMU (`kernel/src/mmu.rs`) mapping the first
  16 GiB as Normal Write-Back blocks **before** the allocator/heap/model code
  runs. Burn's exclusive atomics then work unmodified under HVF.
- **MMU gotcha (cost us a lot of time):** the block descriptor's AP field must
  be `0b00` (EL1 RW only). With AP=`0b01` (EL0+EL1 RW), the *first instruction
  fetch* after `SCTLR_EL1.M` is set aborts with a permission fault
  (`ESR=0x8600000d`, IFSC=0x0d) because a page writable at EL0 is execute-never
  at EL1. QEMU's `arm_fi_to_lfsc` maps IFSC `0x0d` to permission-fault level 1
  (not the "MTE tag check" the ARM table suggests).
- Note: the 1 GiB identity blocks map the UART/GIC MMIO window as Normal WB too
  — harmless under QEMU (MMIO is emulated), but real hardware would need
  Device attributes for MMIO.

### 3. Workarounds / decisions made

- **HVF is the supported fast path.** `make run-hvf` boots, loads weights,
  deserializes, and generates the correct token sequence at ~0.19 s/step (with
  KV cache).
- **TCG also works** (~5 s/step with KV cache) and is useful for debugging.
- The kernel still uses a hand-rolled `UnsafeCell<Heap>` `GlobalAlloc` (safe:
  single core, IRQs off). Burn's own exclusive atomics (via `spin`/`once_cell`/
  `ahash` in `burn-core`/`cubecl`) run fine now that memory is Normal.

### 4. Data pipeline

- `model.safetensors` (538,090,408 bytes) is too big for git → `download-weights.sh`
  fetches it and verifies SHA-256
  `c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093`.
- `model-builder` converts it to a Burn `BinBytesRecorder` record
  (`kernel/src/smollm-135m.bin`, gitignored). The kernel loads it with
  `NoStdInferenceRecorder` (no `std`).
- Host `validate.rs` confirms the expected tokens (see output above) — the
  reference for kernel correctness.

---

## Known Remaining Work

- **KV cache**: **done** — `KvCache` + `forward_step` cache RoPE'd K/V per
  layer; generation is ~O(1) per step (prompt processed once, later steps feed a
  single token). ~194 ms/step on HVF (was ~324 ms) and ~5.4 s/step on TCG (was
  ~15 s). Token sequence unchanged.
- **Chat loop**: **done** — `make chat` boots under HVF into an interactive
  loop: reads a prompt over UART RX, tokenizes it with a GPT-2 byte-level BPE
  tokenizer (`kernel/src/tokenizer.rs` + `tokenizer.bin`, vocab 49152 / merges
  48900), generates with temperature 0.8 + top-k 40 sampling, and decodes the
  output in real time. The hardcoded `[15496, 11]` prompt is still used by the
  greedy benchmark (`make run-hvf`) only.
- **Tokenizer edge case**: the byte-level encoder does not replicate the exact
  pre-tokenizer split for **consecutive leading spaces** (e.g. `"  x"` tokens
  differently than `transformers`, though decoded text is identical). All
  normal text (words, single spaces, punctuation, digits) matches the reference
  exactly. Fixing this requires reimplementing the GPT-2 pre-tokenizer regex.
- **Linux guest**: **done** — the std host binary cross-compiles to a static
  aarch64 Linux ELF and runs inside a Ubuntu 24.04 cloud-image guest booted by
  QEMU/HVF (`make run-linux`, `make bench-linux`). Guest boot takes ~30-60 s
  (cloud-init + 9p model load); 10-run avg TPS 3.73 vs 7.06 native std and 5.09
  no_std HVF kernel (see the 3-way comparison table).

### Performance: 3-7 TPS vs Ollama's 340 TPS (diagnosed, not fixed)

The same SmolLM-135M model under Ollama/llama.cpp reaches ~340 TPS while Burn
here gets 3-7 TPS. The gap is **~50-90x**, not a single bug, and QEMU is *not*
the cause (the native macOS std binary also only hits 7.06 TPS). Root causes,
in impact order, with the source evidence:

1. **Single-threaded GEMM on one core.** Both `kernel/` and `host/` build
   `burn-flex` with **no `rayon`** (`host/Cargo.toml` enables only `["std"]`;
   the kernel is `default-features = false`). `matmul_2d_strided`/
   `matmul_batched_gemm` then always take the single-threaded branch
   (`get_parallelism` returns `Parallelism::None`, `burn-flex/src/ops/matmul.rs`).
   llama.cpp uses Metal GPU + all CPU cores.
2. **The no_std kernel runs a *scalar* GEMM — no NEON.** `burn-flex` pulls
   `gemm` with `default-features = false, features = ["f16"]`. `gemm` dispatches
   SIMD via `feature_detected!("neon")`, which under `no_std` degrades to
   `cfg!(feature = "neon")` = **always false** (`gemm-common/src/lib.rs:70-75`),
   so the kernel silently falls back to the portable scalar microkernel. With
   `std` enabled (the host build) the same macro is a real runtime
   `is_aarch64_feature_detected!` check, so the host gets a NEON microkernel.
   This alone is the measured host-vs-kernel gap (7.06 vs 5.09 TPS, ~1.4x).
3. **fp32 weights vs Ollama's Q4 quantization.** The kernel streams ~513 MB of
   fp32 weights per token; a Q4_K_M GGUF is ~80 MB (~6.4x less memory traffic).
   Decode is memory-bound once parallelism is available, so this dominates at
   the top end but is not the binding constraint at single-thread speeds.
4. **Unfused per-token decode overhead.** `forward_step` (`kernel/src/model.rs`)
   clones the whole [49152, 576] embedding via `.val()` every step (~113 MB
   copy/token), recomputes RoPE `cos`/`sin` trig per layer per step, appends the
   KV cache with `Tensor::cat` (copies the full prefix per layer per step), and
   materializes `expand`/`swap_dims`/`reshape` temporaries plus a 49152-wide
   logits/softmax. ~256 MFLOPs/token (30 layers × ~6.6 MFLOPs + tied lm-head
   576×49152) at ~1.8 GFLOP/s effective ≈ 10% of one M-core's fp32 peak.

Planned optimization work (not started), in TPS-per-effort order:

- **Phase 1 — kernel quick wins (~2x, low risk).** (a) Patch `gemm-common` via
  `[patch.crates-io]` so the no_std `feature_detected!` returns
  `cfg!(target_arch = "aarch64")` (NEON is aarch64 baseline) → NEON microkernel,
  ~1.4x; (b) cache the transposed tied lm-head once instead of `.val()` every
  step; (c) precompute RoPE `cos`/`sin` tables for all positions once; (d) replace
  per-step KV `Tensor::cat` with preallocated `[b, max_pos, kv_heads, head_dim]`
  tensors + `slice_assign`; (e) enable `rayon` on the `host/` build so the
  lm-head GEMM (28.3M ops ≥ burn-flex's 7M parallel threshold) parallelizes.
- **Phase 2 — f16 weights (~1.5-2.5x).** Convert weights to f16 in
  `model-builder` and load as `F16` in the kernel; halves bytes/token and
  `burn-flex` already dispatches `matmul_gemm::<f16>`. Verify precision on the
  135M model. Main win is byte-halving (the scalar fp16 path may lack a NEON
  microkernel under no_std).
- **Phase 3 — block quantization (~3-6x).** Q8_0/Q4_K quantized GEMM in the
  kernel — effectively reimplementing llama.cpp's dequant-on-load quant matmul
  in no_std Rust. Largest effort, largest payoff.
- **Deferred:** a custom M=1 GEMV bypassing burn-flex matmul (biggest potential
  single-thread win — the current gemm leaves ~5-10x on the table for M=1 shapes)
  but it abandons Burn's `Linear`/backend abstraction for decode. Metal/GPU is
  impossible for the no_std kernel and only relevant to the host path.