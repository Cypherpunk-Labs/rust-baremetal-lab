# Research Spike 02 — Replacing Burn with llama.cpp/ggml in a no_std Embedded Rust Kernel

## Context

Spike 01 surveyed the Rust-native AI landscape and ranked the relevant reference
implementations (vLLM, BitNet, FlashAttention, **NIGHTRUN**, T-MAC, mistral.rs,
candle, **llama.cpp**, etc.). This spike narrows the focus to a concrete,
buildable decision: **can we replace the Burn compute library entirely with
llama.cpp / ggml in a `#![no_std]` bare-metal Rust kernel, and if so, how?**

The boot platform for this spike is the existing bare-metal kernel (the `burn/`
project in this repo) — its *platform* infrastructure we keep:

- A working `#![no_std]`, `#![no_main]` **aarch64-unknown-none** kernel,
  booted as a raw ELF under QEMU `virt` on Apple Silicon (TCG + HVF), driving
  the PL011 UART directly, no OS, no libc.
- A 2 GiB `linked_list_allocator` heap, an identity MMU (required for HVF
  exclusive atomics), FP/SIMD enabled in boot asm, an ARM generic timer,
  embedded weights via `include_bytes!`, a GPT-2 byte-level BPE tokenizer, a KV
  cache, and an interactive chat loop.
- It currently runs **SmolLM-135M** inference via Burn 0.21 + `burn-flex`.

**The intent of this spike is to strip Burn out entirely.** The boot platform
(asm, UART, MMU, heap, timer) and the standalone tokenizer are kept; the **Burn
compute layer** — `burn-flex`, the `gemm` crate, Burn's tensor ops, the
`model.rs` transformer expressed in Burn tensors, and its fp32 weights — is to
be replaced with llama.cpp/ggml's quantized compute-graph approach, used as
closely to llama.cpp's own usage of ggml as we can.

The known problem this spike is motivated by: **Burn's compute layer gets 3–7
TPS where Ollama / llama.cpp gets ~340 TPS on the same model.** The previous
work diagnosed the dominant causes:

1. Single-core, scalar GEMM (no NEON dispatch under `no_std`).
2. **fp32 weights vs llama.cpp's Q4/Q8 quantization** — the same model streams
   ~513 MB of fp32 per token vs ~80 MB of Q4_K GGUF (~6.4x less memory traffic).
   This is the single most direct path to the gap and is exactly what adopting
   GGUF + llama.cpp's quantized kernels would address.
3. Unfused per-token decode overhead.

llama.cpp is the de-facto local inference engine and the performance/format
parity baseline every Rust runtime gates against (spike01, §2.9). It is C/C++,
MIT-licensed, and its CPU backend is what we would need to run under a custom
kernel. **No first-class bare-metal/no_std build of llama.cpp exists** (public
feature request, not resolved) — the question of what it takes to link it into,
or port it for, a freestanding single-core kernel is genuinely open. Spike 01's
NIGHTRUN (#4) is the closest published architecture to ours and gates its bare-metal
Rust kernels token-for-token against llama.cpp, which is the method this spike
should adopt as its correctness reference.

The final report will be read by engineers deciding what to explore next.

## Objectives (prioritized)

1. **Direct C/C++ linking viability** — Whether (and how much of) llama.cpp /
   ggml can be compiled and **statically linked into the aarch64 `no_std`
   kernel**. Enumerate every libc / C++ STL / runtime dependency (malloc/free,
   `std::thread`/`pthread`, `std::mutex`/atomics, `std::vector/string/map`,
   `<regex>`, `mmap`, `dlopen`/`dlsym`, OpenMP/`libgomp`, `libm`,
   `clock_gettime`, file I/O) and classify each as **implementable /
   replaceable / blocker** on a single-core freestanding target. Identify the
   minimal shim layer (os-api crate, allocator bindings, time source, memory
   source in place of `mmap`, single-threaded mutex/thread stubs).

2. **GGUF + quantized-kernel path** — Assess adopting llama.cpp's **GGUF model
   format** and its **quantized GEMM/GEMV kernels** (Q2–Q8/K-quants) as the path
   that directly attacks the fp32-vs-Q4 memory-bandwidth gap measured in the
   burn kernel. Evaluate the default `mmap`-based weight loader and how to
   replace it with the kernel's load-into-RAM / `include_bytes!` pattern.

3. **Existing Rust bindings survey** — Identify and assess every existing Rust
   binding/wrapper of llama.cpp (e.g. `llama-cpp-2` / `llama-cpp-sys-2`
   (utilityai), `edgenai/llama_cpp-rs`, `mdrokz/rust-llama.cpp`,
   `maxbrunsfeld/llama-rs`, `rustformers/llm`, `llm_client`, `drama_llama`).
   For each: whether it builds for / can target `aarch64-unknown-none`, which of
   its transitive native dependencies it drags in, license, maintenance state,
   and whether it is viable as-is, forkable, or only a reference.

4. **Rust reimplementation paths (port rather than link)** — Compare the
   port-centric alternatives surfaced by spike01: **NIGHTRUN** (bare-metal Rust
   kernel reimplementing llama.cpp-style quant kernels, gated against llama.cpp),
   **mistral.rs** (pure-Rust GGUF/quant inference, no C toolchain), and
   **candle** (HF's Rust framework with GGUF/safetensors loaders). Determine
   which already run on freestanding targets or are closest, and what a
   port-of-kernels (rather than a C link) would cost.

5. **Integration strategy recommendation** — Synthesize the above into a
   **single recommended path** (direct C link vs port kernels vs adopt an
   existing pure-Rust engine) with a per-path **effort / risk / performance /
   maintainability** estimate and a clear go/no-go for each, given the existing
   kernel boot infrastructure (from `burn/`) and the goal of replacing Burn's
   compute layer with ggml.

## Scope & Definitions

- **System innovation** = changes to the runtime/platform layer (memory layout,
  allocation strategies, scheduler, drivers, kernel, numerics at the HW level).
- **Model innovation** = changes to the model/algorithm layer.
- **In scope**: llama.cpp / ggml (the C/C++ engine), its Rust bindings, and the
  Rust reimplementation engines (NIGHTRUN, mistral.rs, candle as relevant); the
  dependency/shims required to run CPU inference in a single-core
  `aarch64-unknown-none` kernel; the target = **SmolLM-135M** on **ARM64 QEMU
  `virt`/HVF** reusing the existing `burn/` kernel's boot infrastructure as the
  platform and **replacing Burn's compute layer** with ggml/llama.cpp.
- **Include**: active (updated ≤ 12 months) or historically significant
  bindings/ports; anything that demonstrates a no_std / freestanding path for
  ggml-class kernels, even if experimental.
- **Exclude**: GPU/Metal/CUDA/Vulkan backends (not usable in the single-core
  kernel — CPU backend only); pure obfuscation; SaaS-only; unverifiable
  projects (anti-hallucination — a port that cannot be confirmed against its
  primary source is excluded and noted).

## Methodology

- Sources: GitHub, crates.io, docs.rs, project docs/READMEs, llama.cpp issues
  (e.g. the bare-metal build feature request), OSS-bindings primary repos, NIGHTRUN /
  mistral.rs / candle repos. **Record the search date.**
- For llama.cpp/ggml: map its **external dependencies** from the actual build
  (`CMakeLists`, `ggml/src/ggml-cpu`, headers) — not from memory — and classify
  each against the no_std kernel's capabilities.
- For each Rust binding/port: record repo URL, license, last commit, activity,
  whether it links native C libs, and a concrete **no_std / freestanding
  assessment** (does its `build.rs` produce a bare-metal target? does it pull
  `std`?).
- For each candidate: one-paragraph summary + how it maps to objectives +
  viability verdict. Rank against the rubric below, not gut feel.

## Deliverables

Write a single markdown report to `research/spike02/report.md` (create if
missing) containing:

### 1. Dependency / Shim Matrix
The llama.cpp / ggml CPU path broken into its external dependencies. Columns:
Dependency | Used where (file/subsystem) | Kernel counterpart / shim | Effort (S/M/L) | Blocking? (yes/no). This is the explicit "what breaks, what must we implement" answer.

### 2. Port / Binding Viability Matrix
Columns: Project | URL | License | Last commit / activity | Links native C? | no_std buildable? | Obj #s met | Effort / Risk | Verdict.

### 3. Strategy Comparison
For each of the three strategies (A: link ggml/llama.cpp C into the kernel;
B: port llama.cpp kernels to no_std Rust; C: adopt/adapt an existing pure-Rust
engine such as NIGHTRUN/mistral.rs/candle): a section with strengths,
weaknesses, effort estimate, risk, expected TPS impact vs the current Burn
compute baseline, and a go/no-go recommendation.

### 4. Recommendation
A single recommended path with justification and a prioritized, costed list of
follow-up tickets (design → stub → GGUF load → single kernel → full engine),
each mapped to the chosen objective(s).

### 5. Citations
Every claim about capabilities/dependencies cites a primary source (repo,
docs.rs page, issue number, or paper). Prefer primary over secondary sources.

### 6. Further Research
Prioritized open questions, each tied to an objective, with suggested search
terms and target sources.

## Constraints

- **Deliverable is a feasibility + effort decision report — no working
  integration code** in this spike.
- Target platform is the existing aarch64 QEMU `virt`/HVF kernel **boot
  infrastructure** (from `burn/`) with SmolLM-135M; CPU backend only; **Burn's
  compute library is replaced by ggml/llama.cpp**.
- Rust-first / no-C-toolchain approaches get precedence where parity exists
  (per spike01's Rust-first stance), but the C-link path must be assessed fairly
  on merit.
- No hallucinated projects — a binding/port that cannot be verified is excluded
  or explicitly flagged. Note uncertainty where facts cannot be confirmed.

## Acceptance Criteria

- [ ] Every llama.cpp/ggml external dependency classified as implementable /
      replaceable / blocker, with a named kernel shim for each implementable one.
- [ ] ≥ 6 verifiable Rust bindings/ports assessed, each with a concrete no_std
      assessment and license/activity recorded.
- [ ] At least 3 integration strategies costed (link C / port kernels / adopt
      engine) with effort, risk, and expected TPS impact.
- [ ] A single recommended path with go/no-go and prioritized follow-up tickets.
- [ ] All key claims cited to primary sources.

## Questions to resolve during research (answer inline if found)

- Can `ggml`'s CPU backend be built freestanding for `aarch64-none` (no libc,
  no pthread, no `std::thread`)? What is the smallest patch surface?
- What is the minimal set of OS shims (time, memory-in-place-of-`mmap`,
  single-threaded mutex/thread/no-op OpenMP) required to link `ggml` into our
  kernel, and which are blockers?
- Do any existing Rust llama.cpp bindings build under `aarch64-unknown-none`,
  or is every one a `std`/host-only linker wrapper?
- Which existing pure-Rust engine (NIGHTRUN / mistral.rs / candle) is closest to
  running a GGUF SmolLM-135M model on a freestanding single-core ARM64 target,
  and how far are we from it?
- What is the expected TPS gain from replacing Burn's compute layer with GGUF
  Q4/Q8 ggml kernels, vs the 3–7 TPS burn baseline, before considering
  NEON/threading?
