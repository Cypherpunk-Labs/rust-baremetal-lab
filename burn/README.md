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
| `make run-tcg`        | Boot kernel under TCG emulation (**works**, ~5 s/step with KV cache)   |
| `make run-hvf`        | Boot kernel under HVF acceleration (**works**, ~0.19 s/step with KV cache) |
| `make chat`           | Boot under HVF into the interactive chat loop (type a prompt)            |

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
- `kernel/x86_64-unknown-none.json` is a stale unused custom target — safe to delete.