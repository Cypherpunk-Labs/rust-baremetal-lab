# SmolLM-135M on a Bare-Metal ARM64 Kernel (Burn + QEMU)

Run [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) inference in a
`no_std`, single-core ARM64 kernel built with [Burn](https://burn.dev) 0.21 +
`burn-flex`, booted as a raw ELF under QEMU `virt` on Apple Silicon.

The kernel has **no OS** — no std, no libc, no MMU. It maps RAM, drives the
PL011 UART directly, allocates a 2 GiB heap, deserializes the embedded weights,
and runs a greedy autoregressive generation loop, printing tokens over the serial
console.

**This project is a deliberate exercise in running Burn on bare metal.** TCG works
end-to-end; HVF is blocked by a QEMU bug (details below).

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
- `qemu-system-aarch64` (tested with **10.0.3** from Homebrew)
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
| `make validate`       | Host reference run: prints 10 greedy tokens (~1 s/step)                     |
| `make kernel`         | Build the no_std aarch64 kernel ELF                                         |
| `make run-tcg`        | Boot kernel under TCG emulation (**works end-to-end**, ~13 s/step)          |
| `make run-hvf`        | Boot kernel under HVF acceleration (**hangs** — known bug, see below)       |

Expected output (host `make validate` and kernel `make run-tcg` agree on tokens):

```
[step 0] token=521
[step 1] token=397
...
Final tokens: [15496, 11, 521, 397, 397, 397, 397, 397, 397, 397, 397, 397]
```

Both paths also print timing/throughput. The kernel reads the ARM generic timer
(`cntpct_el0` / `cntfrq_el0`) directly — no `std`, no interrupts:

```
# host (make validate)
avg 1284.41 ms/token, 0.779 tokens/sec

# kernel (make run-tcg)
[step 0] token=521 (5742 ms)
...
[Kernel] avg 13502.90 ms/token, 0.074 tokens/sec (cntfrq=1000000000 Hz)
```

TCG costs ~10.5x the native host — expected for pure interpreter-emulation.
Per-step time grows with `seq_len` (full-causal attention re-runs the whole
prefix each step).

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
  `UnsafeCell` + custom `GlobalAlloc`. **Deliberately no locks/atomics** (single
  core, no interrupts) — see the HVF story below.
- **Weights**: `include_bytes!("smollm-135m.bin")` (538,067,357 bytes).
- `embed_tokens.weight` is `[49152, 576]` and is loaded **without transpose**
  (Burn's `load_embedding`); the model-builder must not transpose it.

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

### 2. HVF hangs on atomics (NOT FIXED, a QEMU bug)

- Under TCG, everything works (this is why the puzzle was hard: *CPU vs accel
  changes the result*).
- Under HVF on Apple Silicon, **exclusive-access atomics hang silently**:
  `compare_exchange_weak` (`ldaxrb`/`stxrb`) and even a single `swap`
  (`ldxr`/`stxr`) spin forever at 100% CPU. Plain loads/stores are fine.
- All `-cpu` variants tried (`max`, `host`, `cortex-a72`) and
  `gic-version=max` hang; `-d int` shows **0 exceptions taken** — it is a spin,
  not a trap. (Compare QEMU work item #3444: `qatomic_xchg` silently fails under
  HVF.) No guest-side fix is possible.
- Consequence: Burn's dependency tree itself uses atomics — `spin`, `ahash`,
  `once_cell`, `portable-atomic` are all pulled in by `burn-core`/`cubecl`, so
  HVF hangs during weight deserialization regardless of our allocator. We audited
  `burn-core`/`burn-tensor`/`burn-nn`/`burn-flex` themselves (atomic-free); the
  atomics come from their transitive deps.
- We removed the one atomic we controlled (the allocator's `spinning_top`
  spinlock → raw `Heap` + `GlobalAlloc`), which got HVF from "no output" to
  "heap initialized + loading weights", but it still hangs in the load path.

### 3. Workarounds / decisions made

- **TCG is the supported path.** `make run-tcg` boots, loads weights,
  deserializes, and generates the correct token sequence. Slow (~13 s/step) but
  correct.
- **HVF is blocked upstream.** If it ever matters: retest with a newer QEMU
  (HVF atomics/FFI handling is actively churny — see QEMU release notes and the
  WFI-halting fixes in 11.0.x), or run a TCG-based path.
- We explicitly did **not** keep the allocator spinlock; the kernel uses a
  hand-rolled `UnsafeCell<Heap>` `GlobalAlloc` (safe: single core, IRQs off).

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

- **HVF boot**: blocked on the QEMU atomics bug (see §2). Re-test when a QEMU
  release fixes HVF exclusive-access handling.
- `kernel/x86_64-unknown-none.json` is a stale unused custom target — safe to delete.