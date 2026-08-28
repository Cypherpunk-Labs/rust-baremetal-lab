# Research Spike 03 — Network-Enabled Kernel (Dual-Boot: UEFI + Raw-ELF, Shared Core)

## Context

Spike 02 assessed integrating llama.cpp/ggml into the `#![no_std]` burn kernel and
recommended a strategy for closing the fp32-vs-Q4 memory-bandwidth performance gap.
This spike changes axis: it asks **how to make the existing bare-metal kernel
network-capable** so it can **serve inference as a remote API / webshell** and
**consume external APIs** (e.g. fetch prompts, remote model/config, tool calls).

The foundation is the existing **burn no_std kernel** (the `burn/` project in this
repo):

- A working `#![no_std]`, `#![no_main]` **aarch64-unknown-none** kernel, booted as a
  raw ELF under QEMU `virt` on Apple Silicon (TCG + HVF), driving the PL011 UART
  directly, no OS, no libc.
- A 2 GiB `linked_list_allocator` heap, identity MMU (required for HVF exclusive
  atomics), FP/SIMD enabled in boot asm, ARM generic timer, weights via
  `include_bytes!`, GPT-2 byte-level BPE tokenizer, KV cache, interactive chat loop.
- Runs **SmolLM-135M** via Burn 0.21 + `burn-flex`. `main.rs` currently mixes
  platform-specific boot code (`_start` asm, exception vectors, PL011 UART, identity
  MMU, heap init) with the platform-neutral model/tokenizer/generation logic.

**The core premise of this spike (decided with the sponsor):** build a
**shared-core, dual-adapter** architecture — one platform-neutral core
(model + tokenizer + HTTP + API handling) with **two boot/net adapters**:

1. **UEFI boot path** (`aarch64-unknown-uefi`, NIGHTRUN-style) using the UEFI
   firmware's **Simple Network Protocol (SNP)** for the NIC driver (no custom driver
   to write), and
2. **Raw-ELF boot path** (the existing `aarch64-unknown-none` kernel) with a
   **custom virtio-net device driver** (the large, hardware-specific piece).

Both adapters talk to a shared smoltcp (TCP/IP) + hand-rolled HTTP layer on top.

**This spike is DESIGN-ONLY (decided with the sponsor): no working integration
code.** It produces the feasibility study, architecture, dependency/risk matrix,
and a prioritized, costed step plan (S1–S8) for later implementation spikes.
The final report will be read by engineers deciding what to explore, build, and
in what order.

## Objectives (prioritized)

The objectives are ordered as the intended build-out progression (simple → complex),
matching the sponsor's request.

1. **Feasibility of networking in UEFI mode** — Establish with primary sources
   whether a `#![no_std]` Rust aarch64 environment can obtain working networking
   under UEFI (SNP via the `uefi` crate), and what the minimal dependency stack is
   (`uefi`, `smoltcp`, `smoltcp-uefi`). Confirm the `aarch64-unknown-uefi` target,
   the `edk2-aarch64-code.fd` firmware already available, and the QEMU hostfwd
   mechanism to make a served API reachable from the host.

2. **Feasibility of a custom raw virtio-net driver** — Establish what a raw
   virtio-net driver for the existing raw-ELF kernel entails (device discovery,
   virtqueues, receive/transmit over the existing heap), what crates exist
   (e.g. `virtio-drivers`), and classify its effort/risk — the only genuinely
   hardware-specific, large piece of the plan.

3. **Shared-core architecture design** — Define how to split the existing `main.rs`
   into a platform-neutral core (`kernel::model`, `kernel::tokenizer`, generation,
   HTTP, API/webshell handlers) and platform adapters (UEFI console/SNP vs raw-ELF
   UART/virtio-net), so both adapters reuse one model build with no duplicated logic.

4. **Serving inference remotely (API + webshell)** — Design the HTTP-over-TCP layer
   (smoltcp TCP sockets + minimal HTTP/1.1 framing) exposing `GET /health` then
   `POST /chat`, plus a streaming webshell/console, reachable from the host via
   `hostfwd` → `http://localhost:PORT`. Product objective: **serve an API / webshell**.

5. **Consuming external APIs (client)** — Design the outbound path: smoltcp TCP
   client + DNS; fetch from an external API over QEMU slirp user-mode net (host
   gateway `10.0.2.2`). Product objective: **consume APIs**.

6. **Integration & hardening** — Design how both adapters share state (per-connection
   KV cache), re-entrancy of generation, UART+network logging, hostfwd configuration,
   Makefile targets, and measuring network-served TPS vs the serial baseline.

## Scope & Definitions

- **System innovation** = runtime/platform changes (memory layout, scheduling,
  drivers, boot path, network stack). **Model innovation** = model/algorithm changes.
  This spike is almost entirely **system innovation**; the model is reused as-is.
- **In scope**: the UEFI boot path and its SNP networking; the raw-ELF boot path and
  its virtio-net driver; the shared smoltcp + HTTP core layer; the API/webshell server
  interface; the HTTP/DNS client interface; host-reachability via QEMU hostfwd;
  design/effort/risk assessment for the S1–S8 progression.
- **In scope for design**: full feasibility + architecture + step plan. **Out of
  scope for this spike: writing/committing any working integration code** (per the
  design-only decision). A *step-0 hello-world* is described as a future action, not
  built here.
- **Exclude**: GPU/Metal/CUDA backends (not usable); full TCP congestion control
  tuning (smoltcp defaults suffice); TPM/secure-boot signing; production TLS for the
  served API (noted as future hardening; slirp user-net is not for hostile networks).
- **Target platform**: QEMU `virt` on Apple Silicon (HVF/TCG), aarch64,
  **SmolLM-135M**, CPU only, reusing the existing `burn/` kernel as the raw-ELF
  platform and adding a UEFI platform. `edk2-aarch64-code.fd` is the available UEFI
  firmware (already used by the existing Linux-guest boot path).

## Methodology

- **Sources**: rustc platform-support (tier table), docs.rs/crates.io for `uefi`,
  `smoltcp`, `smoltcp-uefi`, `virtio-drivers`; QEMU networking docs; edk2/ArmVirtPkg;
  prior-art repos (`smoltcp-uefi`, `full-controll-os`, NIGHTRUN); the existing
  `burn/` kernel source. **Record the search date.**
- **Classify** each networking/boot dependency as **implementable / replaceable /
  blocker** on each of the two targets, mirroring spike02's dependency-shim method.
- For each candidate dependency/protocol: record repo URL, license, activity, and a
  concrete no_std / target assessment (does it build for `aarch64-unknown-uefi` /
  `aarch64-unknown-none`? does it pull `std`?).
- Deliver design artifacts (architecture diagram/text, dependency/risk matrix,
  S1–S8 step plan) — research + design only, no code.

## Deliverables

Write a single markdown report to `research/spike03/report.md` (create if missing)
and a companion `research/spike03/objective.md` containing:

### 1. Feasibility (UEFI networking)
Primary-source answer to "can networking be enabled in UEFI mode?", with the
confirmed dependency stack (`aarch64-unknown-uefi`, `uefi` SNP, `smoltcp`,
`smoltcp-uefi`), firmware availability (`edk2-aarch64-code.fd`), and host-reachability
mechanism (`hostfwd`), each cited.

### 2. Architecture (shared-core, dual-adapter)
Text/ascii dataflow showing: shared core (model, tokenizer, generation, HTTP, API/
webshell) ↔ smoltcp TCP ↔ two adapters (UEFI/SNP and raw-ELF/virtio-net). Explicitly
describe the seam where platform-specific boot code detaches from the core.

### 3. Boot / Networking Dependency & Risk Matrix
Columns: Component | Used where | Adapter | Implementable / Replaceable / Blocker |
Effort (S/M/L) | Risk | Kernel shim / dependency. Cover UEFI SNP path and raw
virtio-net path (device discovery, virtqueue, MMIO, PCIe on `virt`).

### 4. Step Plan (S1–S8)
Prioritized, costed progression simple→complex, each step mapped to objective(s):
S1 foundation/feasibility protos; S2 raw virtio-net driver; S3 UEFI SNP adapter;
S4 shared smoltcp net layer; S5 HTTP server + `POST /chat` (serve API); S6 webshell/
console (streaming); S7 HTTP/DNS client (consume APIs); S8 integration & hardening.

### 5. Citations
Every capability/dependency claim cites a primary source.

### 6. Further Research
Prioritized open questions tied to objectives (e.g. whether AAVMF exposes SNP for a
virtio-net device, verifiable only at build time; smoltcp `alloc` on nightly).

## Constraints

- **Deliverable is a design/feasibility report — no working integration code** in
  this spike (sponsor decision). No source files beyond `objective.md`/`report.md`
  are created.
- Target platform is aarch64 QEMU `virt` (HVF/TCG), CPU only, SmolLM-135M, reusing
  the existing `burn/` kernel for the raw-ELF path and adding a UEFI path.
- Rust-first / no-C-toolchain approaches get precedence where parity exists, but the
  raw virtio-net C-driver alternative (or crate reuse) is assessed fairly on merit.
- No hallucinated projects — any binding/port not verifiable against its primary
  source is excluded or explicitly flagged. Note uncertainty where facts cannot be
  confirmed.

## Acceptance Criteria

- [ ] Feasibility answer ("can networking be enabled in UEFI mode?") answered with
      primary-source citations (rustc target tier, uefi SNP, smoltcp, smoltcp-uefi,
      QEMU hostfwd, edk2 firmware).
- [ ] Explicit shared-core / dual-adapter architecture with the boot-platform seam
      described.
- [ ] Dependency & risk matrix for BOTH adapters (UEFI/SNP and raw virtio-net).
- [ ] Prioritized S1–S8 step plan with effort/risk per step mapped to objectives.
- [ ] Both sponsor product objectives addressed: serve API/webshell; consume APIs.
- [ ] All key claims cited to primary sources; uncertainties flagged.

## Questions to resolve during research (answer inline if found)

- Can `ggml`-class / burn model code compile unchanged for both `aarch64-unknown-none`
  and `aarch64-unknown-uefi` (both no_std), or must the model crate be kept
  platform-neutral with boot code split into adapters?
- Does AAVMF (`edk2-aarch64-code.fd`) expose UEFI SNP for a `virtio-net-*` device, or
  must that be verified empirically at build time? (Only verifiable by booting.)
- Does `smoltcp`'s `alloc` feature (required for owned sockets) work on our nightly +
  build-std setup, and is it compatible with `aarch64-unknown-uefi`?
- Is `virtio-drivers` usable on QEMU `virt` aarch64 for a raw-ELF kernel with our
  identity MMU / 2 GiB heap, or should virtqueues be hand-rolled?
- What is the cleanest way to make an HTTP server + webshell share one model (KV
  cache per connection) without diverging from the existing single-threaded loop?
