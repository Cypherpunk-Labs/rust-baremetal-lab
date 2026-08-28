# Research Spike 02 — Replacing Burn with llama.cpp/ggml in a no_std Embedded Rust Kernel

**Search date:** 2026-08-28. **Scope:** objectives 1–5 from `objective.md`. **Method:**
primary sources — llama.cpp/ggml source files fetched from `master`, GitHub repos,
crates.io, docs.rs, HF Hub, the rustc platform-support reference. Every URL,
license, activity state, and capability claim was checked against its primary
source during this session; uncertain or unverifiable items are flagged inline.
**Legend — Effort: S/M/L** (Small/Medium/Large). **Recency:** **Active** = commit
within last 12 months; **Stable** = maintenance-only; **Archived/Dormant** =
read-only or cold.

---

## Contents

- **§1** Dependency / Shim Matrix — the explicit "what breaks, what we must implement" answer
- **§2** Port / Binding Viability Matrix
- **§3** Strategy Comparison (A: link C · B: port kernels · C: adopt pure-Rust engine)
- **§4** Recommendation + prioritized follow-up tickets
- **§5** Citations
- **§6** Further Research

---

## §1. Dependency / Shim Matrix

The llama.cpp/ggml **CPU-only** path decomposed into its external dependencies,
mapped to kernel counterparts. This is grounded in the actual build and source
(CMakeLists/config, `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml-backend.cpp`,
`ggml-alloc.c`, `ggml-feats.h`, `llama-mmap.cpp`, `src/models/*.cpp`), not from memory.

Two distinct layers must be separated:

- **ggml (the tensor/compute kernel layer)** — C, self-contained, compiles to
  freestanding C with a small shim surface. This is the performance-critical,
  quantized-GEMM layer we actually want.
- **llama.cpp (the model/graph layer)** — C++17, heavier STL/host usage. This is
  the *reference implementation* we mirror: we express the SmolLM transformer
  as a ggml compute graph we drive from Rust (llama.cpp's own `build_graph`)
  rather than linking its C++ model layer. The boot platform + tokenizer
  (standalone, in `burn/`) are kept.

| Dependency | Used where (file / subsystem) | Kernel counterpart / shim | Effort | Blocking? |
|---|---|---|---|---|
| `malloc`/`free`/`calloc`/`realloc` | `ggml-alloc.c` (graph allocator), `ggml-backend.cpp` (`ggml_aligned_malloc`) | Wrap as `GGML_MALLOC/CALLOC/FREE` macros over `linked_list_allocator`; GGML_aligned via reuse of existing 2 GiB heap | S | No |
| `posix_memalign` | `ggml_aligned_malloc` (ggml.c:331) | Shim to `memalign`-style aligned alloc on top of existing heap (already need 32-byte tensor alignment, `TENSOR_ALIGNMENT=32`) | S | No |
| `pthread_create/join/self`, `pthread_mutex_*`, `pthread_cond_*` | `ggml-cpu.c` threadpool | **Single-threaded**: set `n_threads==1` (ggml-cpu.c has an `n_threads==1` fast path that skips thread creation); stub pthread as no-op mutex/cond; disable OpenMP (`-DGGML_OPENMP=OFF`) | S | No (blocking only if we insist on multithreading) |
| `stdatomic` (`__atomic_*` / `ldaxr`/`stlxr`) | `ggml-cpu.c` barrier/dynamic-chunk atomics | **Already solved**: `burn/` kernel runs identity-MMU + HVF with native exclusive atomics (`ldaxrb`/`stxrb`) — the exact reason the identity MMU exists | S | No |
| `#include <pthread.h>`/`<sched.h>`/`<unistd.h>`/`<sys/types.h>` | `ggml-cpu.c`, `ggml-cpu.cpp` block | Replace with `-ffreestanding` config; provide minimal freestanding headers when compiling ggml | S–M | No (needs a freestanding C compile config, not runtime) |
| CPU feature detection (`getauxval(AT_HWCAP)`, `prctl(PR_SVE_GET_VL)`, `sysctlbyname` on Apple) | `ggml-feats.h` (compile-time kernel; runtime only via `ggml_backend_cpu_aarch64_score`) | **Not required**: kernel is compiled for a fixed CPU with NEON known present; hardcode `GGML_CPU_ARM_ARCH`/NEON at build time (`-DGGML_NATIVE=OFF`, feature flags set by `-march`/`-mcpu`); stub feature-fns to return constants | S | No |
| NEON intrinsics (`arm_neon.h`) | `ggml-cpu/arch/arm/quants.c` (all Q/K-quant dot products) | Compiler-provided intrinsic header for `aarch64-none-elf`; `-march=armv8.2-a+dotprod+i8mm` gives SDOT/SMMLA (`ggml_vdotq_s32`, `vsmmlaq`) — these are the quant-GEMM workhorses | S | No (compiler-supplied, not external) |
| `libm` subset: `fabsf, floorf, sqrtf, expf, logf, tanhf, fminf, fmaxf, powf(sel.)` | Reference quant paths, fp16/bf16 converters in `ggml-impl.h` | Implement as freestanding `-ffreestanding` math or link `compiler-builtins`/musl-freestanding subset; **NEON/SIMD hot paths avoid libm** so this is cold-path only | S | No |
| `mmap`/`munmap`, `mlock`, `posix_fadvise`, `sysconf`, `llama_file` (`fopen`/`fread`/`ftello`) | `llama-mmap.cpp`, `llama-file.cpp`, GGUF loader (model loader only) | **Replace with load-into-RAM**: kernel's `include_bytes!` gives ram-resident GGUF; use `gguf_init_from_buffer` + `ggml_backend_cpu_buffer_from_ptr` (zero-copy wrapper for externally-held memory). Graph execution never mmaps. `mmap` is exclusively a loader concern | M | No (and we skip it — our weights are embedded) |
| C++ runtime: `new`/`delete` | `ggml-backend.cpp` (`ggml_backend_buffer_init`), ggml C++ tail | Provide minimal `operator new/delete` over the kernel heap, or compile ggml as pure C for the CPU backend | S–M | No |
| C++ STL: `std::vector/string/map/unordered_map`, `std::unique_ptr`, `std::min/max/sort`, `std::to_string` | llama.cpp core (`llm.cpp`, `src/llama-vocab.cpp`, `src/llama-model.cpp`, `ggml-backend.cpp`) | **Bypass llama.cpp**: we do not need its C++ model/graph layer — we build the graph in Rust over ggml. If linked anyway, requires a minimal libc++ (no host pthreads) — large, avoided surface | L | No — avoided by scoping to **ggml only** |
| `std::mutex`, `std::thread`, `std::atomic` | llama.cpp scheduler (interactive/graph scheduling) | Single-threaded stub; not in ggml compute path | M | No (avoided by single-thread scope) |
| `<regex>`, `<filesystem>` | llama.cpp utility/grammar paths (not CPU kernel) | Excluded — not on the inference hot path | — | No |
| `clock_gettime(CLOCK_MONOTONIC)`, `clock()`/`CLOCKS_PER_SEC` | `ggml_time()`/`ggml_cycles_profiling` (ggml.c timing) | Use **ARM generic timer** (already driven by `burn/` kernel) as time source; shim wall/mono clock | S | No |
| `getenv`/environment tuning vars | ggml init logging | Stub out (`getenv → NULL`); set defaults at compile time | S | No |
| `fprintf`/`stdio` logging | ggml init, NUMA, device-desc printf paths | Route `GGML_LOG`/`fprintf` to UART via `putchar` shim; or `-D` disable | S | No |
| NUMA detection (`syscall(SYS_getcpu)`, `/sys/devices/system/node/node*`, `/proc/cpuinfo`) | `ggml_numa_init`, `ggml-cpu.cpp` device description | Stub: single node, no NUMA. Cosmetic/logging only | S | No |
| OpenMP / `libgomp` | optional `ggml_threadpool` alternative | Disable (`-DGGML_OPENMP=OFF`); single-thread path | S | No |
| `dlopen`/`dlsym` | plugin/backend loading (not CPU path) | Not used by CPU backend — disabled | — | No |
| `sentencepiece`/Unicode/tokenizer C++ | `src/llama-vocab.cpp` | Already replaced by `burn/` GPT-2 byte-level BPE; not needed | — | No |

**Bottom line (blockers):** there are **no hard blockers** for the ggml compute
layer on a single-core freestanding aarch64 target. Every dependency is
*implementable* (thin shim) or *replaceable* (loader/time/logging stubs). The
single most intrusive item is **C++ threading/runtime** — which is confined to the
llama.cpp model layer and eliminated by scoping to ggml + a ggml compute graph
driven from Rust (replacing Burn's transformer).

**Note on the official position:** llama.cpp has no first-class bare-metal build —
feature request #8820 ("Bare metal build") was closed as stale without resolution
[8]. Nothing in that thread disallows a downstream freestanding ggml compile; it
simply has not been done upstream. A freestanding `ggml-cpu` static library
(`-ffreestanding -fno-exceptions -fno-rtti`, no threading) is the smallest patch
surface worth prototyping (see §4 ticket T2).

---

## §2. Port / Binding Viability Matrix

Assessment of every significant Rust binding/port/engine against the
`aarch64-unknown-none` freestanding goal.

| Project | URL | License | Last commit / activity | Links native C? | no_std buildable? | Obj #s met | Effort / Risk | Verdict |
|---|---|---|---|---|---|---|---|---|
| **NIGHTRUN** | github.com/hardrave/NIGHTRUN | MIT | Active (2026-07) | No (pure Rust) | **Partially** — UEFI-resident `no_std` Rust on x86_64 + Pi5(ARM64); hand-written NEON Q8/Q4_K/Q6_K kernels, token-for-token gated vs llama.cpp | 4, 5 | L / Med | **Forkable / blueprint** — closest architecture to ours; ARM64 NEON quant kernels are directly reusable patterns; no OS (UEFI Boot Services). Sub-field-of-record closest to our goal [10][11] |
| **mistral.rs** | github.com/EricLBuehler/mistral.rs | MIT | Active (2026-08) | Optional (MKL/Accel/CUDA) | No — std; `tokio`/`rayon` async runtime | 4, 5 | — / — | **Reference only** for a kernel; too async/runtime-bound [13] |
| **candle** | github.com/huggingface/candle | MIT OR Apache-2.0 | Active (2026-08) | Optional (MKL/Accel/CUDA) | No — `candle-core` has no no_std gate; std unconditionally | 4, 5 | L / Med | **Forkable / reference** — GGUF + quantized ops (`examples/quantized`, `quantized-qwen3-moe`) are the Rust reference for quant kernels; core needs std→alloc carve-out [14] |
| **ferrox** | github.com/antonellof/ferrox | Apache-2.0 | Active (2026-08) | No (pure Rust; IQ codebook tables retained under MIT notices) | No — std; `rayon`/`mmap`/async server. **No llama.cpp/ggml link** | 4, 5 | L / Med | **Forkable — best current pure-Rust GGUF engine**; supports SmolLM2; SIMD CPU; mmap weights (dequant in-matmul). Needs rayon→alloc, mmap→RAM-load for `aarch64-unknown-none` [15] |
| **llama_gguf** | github.com/Lexmata/llama-gguf | MIT OR Apache-2.0 | Active (2026-08, v0.14.0) | No (pure Rust; `rayon`, `memmap2`, `serde`) | No — std | 4, 5 | L / Med | **Forkable** — clean pure-Rust GGUF v1-3 + all K-quants; std surface (rayon/mmap/serde) to carve out [16] |
| **llama-cpp-2 / llama-cpp-sys-2** (utilityai) | github.com/utilityai/llama-cpp-rs | MIT OR Apache-2.0 | Active (2026-08) | **Yes** — submodule + cmake + bindgen | **No** — `parse_target_os()` in `build.rs` returns `Err` for `*-unknown-none` and **panics** ("Failed to parse target os") | 1, 3 | — / — | **Reference only** — host/std linker wrapper; compiles C++ via cmake at build; cannot target `aarch64-unknown-none` as-is [2][3] |
| **llama_cpp-rs** (edgenai) | github.com/edgenai/llama_cpp-rs | MIT OR Apache-2.0 | Dormant (2024-04) | **Yes** — submodule + bindgen/cmake | No | 1, 3 | — / — | **Reference only** — stale C++ wrapper, no no_std path [4] |
| **rust-llama.cpp** (mdrokz) | github.com/mdrokz/rust-llama.cpp | MIT | Dormant (2024-06) | **Yes** — submodule + `cc` | No | 1, 3 | — / — | **Reference only** — old, **no GGUF support** (README TODO still lists "Support GGUF") [5] |
| **llama-rs** (maxbrunsfeld) | github.com/maxbrunsfeld/llama-rs | None declared | Dormant (2023-08) | **Yes** — llama.cpp + Metal | No | 1, 3 | — / — | **Reference only / excluded** — stale macOS Metal bindings, unlicensed [6] |
| **llm** (rustformers) | github.com/rustformers/llm | MIT OR Apache-2.0 | **Archived** (2024-06) | No (pure-Rust ggml re-port) | No — heavy std | 4 | — / — | **Reference only** — archived; cancelled re-port of llama.cpp; useful historical reference for a Rust ggml port [7] |
| **burn** (tracel-ai) | github.com/tracel-ai/burn | Apache-2.0 | Active | No (Rust; NEON gated by `gemm` crate) | **Yes (in place)** — `#![no_std]` kernel runs SmolLM-135M today | 4 | — | **What we're replacing** — the scalar fp32 GEMM, no-NEON-under-no_std compute layer; its boot code (UART, MMU, heap, timer) is retained but the compute layer (burn-flex, gemm, tensor ops, Burn-transformer) is discarded [1] |
| **Anima OS** (ProbeLogits) | arXiv 2604.11943 / 2607.04668 | n/a (research) | Active (papers) | n/a | Rust bare-metal (claimed, no repo found) | 2, 3, 4 | — | **Unverified lead** — no public repo located (anti-hallucination: excluded from ranking) [9] |

**Key finding:** **every existing Rust llama.cpp *binding* is a std/host linker
wrapper — none is `aarch64-unknown-none` buildable, and none can be made so by a
flag.** The only no_std-capable Rust *compute* paths in the entire surveyed set
are **NIGHTRUN** (UEFI-resident, ARM64 NEON quant kernels) and our own **burn**
kernel (unquantized). The pure-Rust *engines* (ferrox, llama_gguf, candle,
mistral.rs) are all std-bound but are the correct *kernel-math reference* material
for a port.

---

## §3. Strategy Comparison

Platform in all cases: the existing kernel **boot infrastructure** (from `burn/`)
— `aarch64-unknown-none`, 2 GiB heap, identity MMU + HVF atomics, FP/SIMD
enabled. **Burn's compute library (burn-flex, gemm, tensor ops, fp32 weights) is
being replaced** by ggml/llama.cpp. SmolLM-135M, ~5 TPS (kernel) / ~7 TPS (host)
fp32 baseline; Ollama/llama.cpp ≈ 338–340 TPS on the same M2 Pro [1].

### Strategy A — Link ggml/llama.cpp C as the compute engine (replacing Burn)
Compile `ggml` (and optionally the C++ llama.cpp model layer) as a freestanding
static library for `aarch64-none-elf`, call it from Rust via FFI, and express the
SmolLM transformer as a **ggml compute graph** exactly as llama.cpp does — i.e.
**strip out Burn's compute layer** (burn-flex, gemm, tensor ops, `model.rs`'s
Burn-transformer) rather than augment it.

- **Strengths:** Direct access to llama.cpp's **quantized GEMM/GEMV kernels**
  (Q4_0–Q8_0, K-quants, SDOT/MMLA NEON paths) — the exact memory-traffic fix the
  spike names as the top lever. Zero-copy `include_bytes!` GGUF + 
  `ggml_backend_cpu_buffer_from_ptr`. NEON intrinsics are compiler-supplied; no
  true hard blockers exist (see §1). Llama.cpp is the parity baseline, so 1:1
  correctness is guaranteed by construction.
- **Weaknesses:** Must compile + maintain C for `aarch64-none-elf` in the tree
  (a second toolchain alongside Rust). Freestanding compile config
  (`-ffreestanding -fno-exceptions -fno-rtti`, `n_threads==1`, no OpenMP, no mmap)
  is a fork/patch — upstream has no bare-metal build (#8820 closed stale). C++ STL
  (if the full model layer is used) drags in a minimal libc++ — avoided by scoping
  to ggml only, but then you are writing your own ggml graph driver in Rust.
  Brings C risk/safety surface into a Rust kernel (mitigated by a narrow C ABI
  boundary: init, load, `ggml_compute`, free).
- **Effort:** M (ggml-only) → L (full llama.cpp model layer).
- **Risk:** Med — upstream moves fast; a vendored freestanding patch drifts.
- **Expected TPS impact:** fp32(513 MB/stream) → **Q4_K (~80 MB/stream)** is a
  ~6.4× reduction in memory traffic; with NEON SDOT/MMLA vs today's scalar GEMM
  this is the single largest lever. Realistic expectation vs the 5–7 TPS fp32
  baseline is roughly **5–20×** toward llama.cpp parity on the same core — but
  single-processor (no thread tiling) caps absolute TPS vs Ollama's multi-core;
  expect tens of TPS, not the full 340.
- **Go / No-go:** **GO (replace Burn's compute layer with the ggml graph)**,
  using llama.cpp's C++ model layer as the *reference implementation* for the
  graph (llama.cpp's `build_graph` maps 1:1 onto ggml ops). We build the same
  graph in Rust over ggml, keeping the shim + tokenizer in Rust. Highest
  confidence to close the measured fp32-vs-Q4 gap fastest.

### Strategy B — Port llama.cpp kernels to no_std Rust
Reimplement llama.cpp's quantized GEMM/GEMV and GGUF loader in `#![no_std]` Rust
for `aarch64-unknown-none` (the NIGHTRUN / candle / ferrox approach — used in place,
no dequant, NEON kernels, token-gated against llama.cpp).

- **Strengths:** Rust-first (matches the repo's stance); single toolchain; no C;
  reuses the existing `no_std` heap/timer/MMU infrastructure directly; full control
  over the numerics at the HW level (this spike's "system innovation").
  NIGHTRUN proves it end-to-end on ARM64 [11]; candle/ferrox/llama_gguf supply the
  Rust reference kernels to port.
- **Weaknesses:** Largest effort — the quant kernels (Q4_K, Q6_K, IQ) are subtle
  (codebooks, repack, DOT/MMLA variants) and must be validated token-for-token
  against llama.cpp to avoid silent numeric drift. Nowhere near the current
  single-model, unquantized (and now-to-be-replaced) Burn scope.
- **Effort:** **XL** — GGUF parser + tensor format + ~8 quant dot-product families
  + NEON (SDOT/MMLA) + fused decode, each with a parity gate.
- **Risk:** Med–High — numeric-parity effort is the bulk of the schedule; a wrong
  codebook or unpacking order silently corrupts output.
- **Expected TPS impact:** Same memory-traffic win as Strategy A (quantity of data
  moved is format-driven, identical Q4_K). Kernel speed depends on our own NEON
  quality vs ggml's heavily-tuned kernels; realistically **0.5–1.0× of Strategy A**
  at parity-of-effort, i.e. tens of TPS.
- **Go / No-go:** **HOLD as primary; strong follow-on.** Too large to be the
  immediate next step, but the durable end-state (Rust-first, no C). Recommended
  as the *target* architecture the kernel should converge on after Strategy A
  de-risks the numerics.

### Strategy C — Adopt/adapt an existing pure-Rust engine
Embed an existing std pure-Rust GGUF engine (ferrox / llama_gguf / candle) after
removing its std surface, or adopt NIGHTRUN wholesale.

- **Strengths:** Reuses a maintained, llama.cpp-parity engine; ferrox already
  supports SmolLM2 and has NEON CPU kernels; llama_gguf is a clean crate layout.
  No C, no C++ toolchain (Rust-first).
- **Weaknesses:** **None of these are `#![no_std]`.** Each would require a
  std→alloc carve-out (rayon thread-pool, `mmap` loader, `serde`, and in mistral.rs
  a tokio runtime) — effectively becoming a Strategy-B port of *their* kernels,
  plus upstream-merge burden. NIGHTRUN is UEFI-resident (Boot Services + framebuffer
  keyboard/display) which overlaps but doesn't match our QEMU-virt PL011 kernel.
  Adopting NIGHTRUN wholesale means adopting its UEFI platform, not our kernel's
  boot platform.
- **Effort:** L–XL (de-std-ify an existing engine).
- **Risk:** Med — you inherit upstream API churn plus your fork drift.
- **Expected TPS impact:** Same format-driven traffic win; engine-specific kernel
  quality. Ferrox CPU decode measured at ~parity with llama.cpp on Metal (0.91–0.98×)
  but its *CPU* path is noted as "behind almost everywhere"; treat as ≈ Strategy B.
- **Go / No-go:** **NO as an adoption** (would scramble the platform); **YES as a
  kernel-math reference** — mine ferrox/llama_gguf/candle for the GGUF/quant code for
  Strategy B, and use NIGHTRUN as the architectural correctness-and-performance
  exemplar.

---

## §4. Recommendation

**Recommended path: Strategy A-scoped-to-ggml — strip Burn's compute layer out
of the kernel and replace it with the freestanding `ggml` C engine, carrying
Q4_K GGUF weights via `include_bytes!`, expressing the SmolLM transformer as a
ggml compute graph driven from Rust over a narrow C ABI (llama.cpp's own
`build_graph` being the reference), and keeping the kernel boot platform + the
standalone tokenizer in Rust.** Retain the pure-Rust kernel port (Strategy B) as
the long-term target, fed by the numerics work unlocked here.

**Why:** (1) It is the **only** path that directly and fastest attacks the
responsible measured bottleneck (fp32 513 MB → Q4_K ~80 MB per token, ~6.4×
less memory traffic, plus NEON SDOT/MMLA vs the current scalar GEMM) [1]. (2) The
ggml compute layer has **no hard blockers** on a single-core freestanding AArch64
target (§1) — the C++ STL/threading that *would* be costly is confined to the
llama.cpp model layer, which we use as reference rather than linking wholesale.
(3) It is correctness-guaranteed by
construction (llama.cpp is the parity baseline). (4) Because we replace Burn's
compute layer at a deliberate numeric boundary, the "C surface" stays a narrow
FFI seam over the boot platform, keeping the platform intact and giving a clean
seam to later swap in pure-Rust kernels (Strategy B). The closure of #8820 [8]
confirms no upstream assistance, so a vendored freestanding patch is required —
accept that maintenance cost.

### Prioritized, costed follow-up tickets (design → stub → GGUF load → single kernel → full engine)

| # | Ticket | Objective(s) | Scope | Est. effort |
|---|---|---|---|---|
| T1 | **Freestanding ggml feasibility build** — compile `ggml` (CPU backend only, no backend GPU, `-ffreestanding -fno-exceptions -fno-rtti`, no OpenMP, `n_threads=1`) as a static `libggml-cpu.a` for `aarch64-none-elf`; document the exact CMake/config bitmask; produce a standalone "hello ggml" mul_mat C test run on the bare-metal kernel boot platform. Proves the compile story end-to-end. | 1, 2 | De-risk the whole spike | M |
| T2 | **ggml shim layer (os-api crate)** — a small `no_std` crate providing: `GGML_MALLOC/FREE` over `linked_list_allocator`, `posix_memalign` shim, pthread/mutex/cond no-op single-thread stubs, ARM generic-timer wall clock for `ggml_time`, `fprintf`→UART logger, feature-detect stubs. | 1 | Enables every later C link | M |
| T3 | **GGUF load into RAM** — replace `mmap`-based GGUF loading with `gguf_init_from_buffer` over `include_bytes!` of a Q4_K SmolLM GGUF; build a host-side converter (safetensors → Q4_K gguf via llama-quantize) so the kernel consumes the same artifact Ollama does. | 2, 4, 5 | Ship the memory-traffic fix | M |
| T4 | **Single mul_mat kernel** — first ggml op integrated: load embedded Q4_K weights into a `ggml_backend_cpu_buffer_from_ptr`, run `ggml_mul_mat` for one layer, compare output to the llama.cpp host reference (tolerance-gated). Token-for-token parity harness established here (mirror NIGHTRUN's method [11]). | 2, 5 | First numeric proof | L |
| T5 | **Full ggml decode path** — express the whole per-token graph as ggml ops (feed-forward + attention as ggml ops, llama.cpp's `build_graph` as reference), replacing Burn's `model.rs` transformer; keep tokenizer in Rust; let ggml manage its KV cache internally. Measure TPS vs the 5–7 TPS fp32 baseline. | 1, 2, 5 | Quantized-waits unlock the ~5–20× target | L |
| T6 | **Assessment gate & Strategy-B seed** — decide whether the pending C surface is acceptable long-term; if not, fork the ggml ARM kernels' *numbers* into Rust (mine ferrox/llama_gguf/candle + NIGHTRUN NEON Q8/Q4_K kernels) toward the pure-Rust target. Parity harness from T4 gates every kernel. | 4, 5 | Long-term Rust-first convergence | XL (phased) |

**Go/No-go summary:** Strategy A (ggml-only) = **GO**; Strategy B = **HOLD-then-GO**
(durable end-state); Strategy C = **NO as adoption, YES as reference**.

---

## §5. Citations

Primary sources (all checked 2026-08-28 unless noted):

1. **burn kernel boot platform & the Burn-compute 3–7 vs ~340 TPS gap** — `burn/README.md` (3-way bench: host 7.06 TPS, kernel 5.09 TPS, Linux guest 3.73 TPS; Ollama M2 Pro 338.44 TPS; diagnosis §"Switching", incl. no-NEON scalar GEMM under no_std and fp32-vs-Q4 (~513 MB vs ~80 MB) memory traffic) — in this repo; `burn/kernel/Cargo.toml` (burn 0.21, `burn-flex`, `linked_list_allocator` 0.10.6), `burn/kernel/src/main.rs:24-56` (2 GiB heap, `include_bytes!`, HEAP 0x7000_0000).
2. **utilityai/llama-cpp-rs** — https://github.com/utilityai/llama-cpp-rs (MIT/Apache); `llama-cpp-sys-2/build.rs` `parse_target_os()` matches only windows/apple/android/linux and panics on others — https://github.com/utilityai/llama-cpp-rs/blob/main/llama-cpp-sys-2/build.rs ; bug confirmed in community report "Failed to parse target os x86_64-unknown-none" — https://community.libretranslate.com/t/ltengine-llm-powered-local-machine-translation/1862/5
3. **llama-cpp-2 / llama-cpp-sys-2** — crates.io https://crates.io/crates/llama-cpp-2 (MIT OR Apache-2.0, 0.1.154)
4. **edgenai/llama_cpp-rs** — https://github.com/edgenai/llama_cpp-rs (MIT/Apache; last commit 2024-04)
5. **mdrokz/rust-llama.cpp** — https://github.com/mdrokz/rust-llama.cpp (MIT; last commit 2024-06; README TODO "Support GGUF")
6. **maxbrunsfeld/llama-rs** — https://github.com/maxbrunsfeld/llama-rs (no license; last commit 2023-08)
7. **rustformers/llm** — https://github.com/rustformers/llm (MIT/Apache; **archived 2024-06-24**)
8. **llama.cpp bare-metal feature request, issue #8820** — https://github.com/ggml-org/llama.cpp/issues/8820 (closed stale; "Please use the new Discussions forum" note)
9. **Anima OS / ProbeLogits** — arXiv:2604.11943, arXiv:2607.04668 (no public repo located; anti-hallucination note)
10. **NIGHTRUN** — https://github.com/hardrave/NIGHTRUN (MIT; UEFI-resident, no_std Rust, x86_64 + Raspberry Pi 5, NEON Q8_0/Q4_K/Q6_K, token-for-token gated vs llama.cpp); https://nightrun.io/ repository, commit 64, created 2026-07-27
11. **NIGHTRUN correctness method** (batched prefill bit-identical to decode; token-for-token parity gates) — https://github.com/hardrave/NIGHTRUN README
12. **ggml CPU backend** — https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c (pthread/threadpool, `n_threads==1` fast path, `ggml_mul_mat` chunking, type-trait `vec_dot` dispatch, `__ARM_FEATURE_MATMUL_INT8` → mmla `nrows=2`); `ggml/src/ggml-cpu/arch/arm/quants.c` (NEON quant dot products); `ggml-backend.cpp` (`ggml_aligned_malloc`), `ggml-alloc.c`, `ggml-feats.h`, `llama-mmap.cpp` (mmap is loader-only)
13. **mistral.rs** — https://github.com/EricLBuehler/mistral.rs (MIT; std/tokio/rayon)
14. **candle** — https://github.com/huggingface/candle (MIT OR Apache-2.0; `candle-core::quantized::gguf_file`, `examples/quantized`, `examples/quantized-qwen3-moe`)
15. **ferrox** — https://github.com/antonellof/ferrox (Apache-2.0; pure-Rust GGUF/MoE, CPU/Metal/CUDA, no llama.cpp link, SmolLM2 supported; benchmarks/RESULTS.md shows CPU decode "behind almost everywhere"); author writeup https://www.fratepietro.com/2026/ferrox-rust-gguf-inference-engine/
16. **llama_gguf** — https://github.com/Lexmata/llama-gguf , https://docs.rs/llama_gguf (MIT/Apache; GGUF v1-3, K-quants, `memmap2`, `rayon`)
17. **rustc target** — `aarch64-unknown-none` is Tier 2, library support core+alloc, includes `FEAT_AdvSIMD` (NEON), supports C via `aarch64-none-elf`, needs a user linker script — https://doc.rust-lang.org/beta/rustc/platform-support/aarch64-unknown-none.html
18. **GGUF format spec** — https://github.com/ggml-org/ggml/blob/master/docs/gguf.md ; HF GGUF hub — https://huggingface.co/docs/hub/gguf ; SmolLM2-135M-Instruct GGUF artifacts (Q8_0 ≈ 0.14 GB, f16 ≈ 0.27 GB) — https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF

Repo licenses/activity verified against primary sources on 2026-08-28. Claims
about the ability of bindings to build for `*-unknown-none` rest on inspection of
their actual `build.rs`/dependency surface (cites 2, 3, 5, 6) — all confirmed to
pull `std` and/or compile C++.

---

## §6. Further Research

Prioritized open questions, each tied to an objective, with suggested search terms
and target sources.

1. **Does a `n_threads==1` ggml compile actually drop ALL pthread/STL linkage, and what exactly links?** *(Obj 1, 3.)* *Terms:* "ggml cpu n_threads 1 single thread pthread dependency", "ggml freestanding". *Source:* compile binaries of ggml under `-ffreestanding` and inspect `nm` for undefined pthread/STL symbols — the empirical blocker check beyond this report's static analysis.
2. **Exact quantified TPS model for single-core Q4_K 135M** *(Obj 5).* *Terms:* "smollm2-135m Q4_K bench single core", "llama.cpp tg128 135M". *Source:* run `llama-bench` with `-t 1` on M2/M3 for SmolLM2-135M Q4_K vs fp32 to get a same-core apples-to-apples ceiling; extrapolate the ~6.4× traffic win precisely.
3. **NEON SDOT vs SMMLA (dotprod vs i8mm) which does Q4_K GEMV win on for `-cpu max`/QEMU virt?** *(Obj 2.)* *Terms:* "ggml mmla smmla q4_K vdot speedup armv8.2", "Q4_0_X_X sdot". *Source:* llama.cpp PRs (#4966 mmla, #5780 Q4_0_X_X) + measure on the target.
4. **QEMU virt `-cpu max` NEON/SVE feature parity with real Apple cores** *(Obj 2, 5).* *Terms:* "qemu virt cpu max sdot i8mm feature", "hvftool tcg simd". *Source:* `ggml_cpu_has_*` output under QEMU/HVF vs host; confirms which ggml fast paths the validation harness can actually test.
5. **Can the Rust side drive ggml without copying (zero-copy decode)?** *(Obj 2.)* *Terms:* "ggml_backend_cpu_buffer_from_ptr", "ggml tensor data alignment include_bytes". *Source:* ggml-backend.cpp ABI; prototypes alignment of `include_bytes!` blobs (32-byte).
6. **Is there a maintained `aarch64-none-elf` freestanding C cross-toolchain / libm subset we can vendor?** *(Obj 1.)* *Terms:* "aarch64-none-elf freestanding libm compiler-builtins", "newlib freestanding". *Source:* Arm GNU toolchain + `compiler-builtins`/`cc` crate capabilities; cc crate already cross-compiles for `aarch64-unknown-none` host-side.
7. **How far is a pure-Rust Q4_K NEON kernel from contiguous dequant-in-matmul parity today (ferrox/Candle/NIGHTRUN)?** *(Obj 4.)* *Terms:* "ferrox cpu Q4_K NEON benchmark", "NIGHTRUN NEON tok/s Pi5". *Source:* ferrox `benchmarks/RESULTS.md` (CPU rows), candle `quantized`, NIGHTRUN PI5 logs — the Strategy-B seed set.
8. **Does SmolLM-135M (not 2) have an exact Q4_K GGUF matching the kernel's target architecture (tied head, GQA)?** *(Obj 2, 5.)* *Terms:* "SmolLM-135M GGUF Q4_K bartowski". *Source:* HF Hub `bartowski/SmolLM-135M-Instruct-GGUF` / `HuggingFaceTB/SmolLM-135M`; confirm architecture-equivalent quantization artifacts.
