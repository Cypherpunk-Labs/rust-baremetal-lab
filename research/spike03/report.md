# Research Spike 03 — Network-Enabled Kernel (Dual-Boot: UEFI + Raw-ELF, Shared Core)

**Search date:** 2026-08-28. **Scope:** objectives 1–6 from `objective.md`.
**Method:** primary sources — rustc platform-support, docs.rs/crates.io for `uefi`,
`smoltcp`, `smoltcp-uefi`, `virtio-drivers`; QEMU networking docs; the existing
`burn/` kernel source in this repo; prior-art repos. Every URL, license, activity
state, and capability claim was checked against its primary source during this
session; uncertain or unverifiable items are flagged inline.

**This spike is DESIGN-ONLY.** It answers feasibility, defines the shared-core /
dual-adapter architecture, produces a dependency & risk matrix, and a prioritized
costed step plan (S1–S8). **No working integration code is written** in this spike.

**Legend — Effort: S/M/L/XL** (Small/Medium/Large/X-Large). **Recency:**
**Active** = commit within last 12 months; **Stable** = maintenance-only;
**Archived/Dormant** = read-only or cold.

---

## §1. Feasibility — Can networking be enabled in UEFI mode?

**Answer: Yes.** A `#![no_std]` Rust aarch64 environment can get working networking
under UEFI with a small, proven dependency stack. The key insight is that under UEFI
the **firmware already provides the NIC driver** via the Simple Network Protocol (SNP),
so no device driver needs to be written on the UEFI path.

### Confirmed dependency stack

| Component | Role | Target support | Source |
|---|---|---|---|
| `aarch64-unknown-uefi` | rustc target (Tier 2, no_std) | Yes — Tier 2, std-capable, no_std usable | rustc platform-support [1] |
| `uefi` crate 0.40.0 (2026-08-27, Active, MIT/Apache) | Safe UEFI bindings; `uefi::proto::network::{snp,http,ip4config2,pxe}` | aarch64-unknown-uefi | docs.rs [2], GitHub [3] |
| `smoltcp` 0.13/0.14 (Active, 0BSD) | no_std TCP/IP stack (TCP/UDP/ICMP sockets), no heap required; `alloc` feature on nightly | no_std | smoltcp GitHub/README [4], docs.rs [5] |
| `smoltcp-uefi` 0.2.0 (2025-03, MIT, experimental PoC) | `SnpDevice` implementing `smoltcp::phy::Device` over UEFI SNP + time utils | no_std UEFI | GitHub [6], docs.rs [7] |
| `edk2-aarch64-code.fd` (AAVMF) | UEFI firmware already present at `/opt/homebrew/share/qemu/edk2-aarch64-code.fd` (64 MB); already used by this repo's Linux-guest boot path | aarch64 QEMU `virt` | this repo (session logs) [8] |
| `-netdev user,hostfwd=...` | Host↔guest port forwarding so a served API is reachable from the host | QEMU | QEMU networking docs [9] |

### How it fits together on the UEFI path

1. Boot the kernel as a UEFI application (`cargo build --target aarch64-unknown-uefi`)
   under the existing `edk2-aarch64-code.fd` firmware, with a virtio-net NIC attached
   (`-device virtio-net-pci -netdev user,...`), the established AAVMF guest-net combo
   [10][11].
2. Get the `SimpleNetworkProtocol` from the boot services via the `uefi` crate
   (`uefi::proto::network::snp`), initialize it [2].
3. Wrap it in `smoltcp_uefi::SnpDevice` so smoltcp sees it as a `phy::Device` [6][7].
4. Build a TCP server on a smoltcp listen socket; implement minimal HTTP/1.1 framing
   ourselves (smoltcp is TCP/UDP/ICMP — **it does not implement HTTP**, see [4]).
5. Make it reachable from the host via `-netdev user,id=n0,hostfwd=tcp::HOSTPORT-:GUESTPORT`
   → `curl http://localhost:HOSTPORT/...` [9].

### Confirmed prior art
- **`smoltcp-uefi`** provides an `SnpDevice` + a QEMU test harness (OVMF + bridge NIC)
  proving the UEFI-SNP→smoltcp path works [6].
- **`full-controll-os`** (TNTTheLagger) is a UEFI-native OS using UEFI SNP + smoltcp
  for "early-boot networking … no firmware hand-holding, no PXE stack, no OS services"
  [12].

### Feasibility caveats (flagged)
- **HTTP is not in smoltcp** — we hand-roll minimal HTTP/1.1 framing over TCP. This is
  small for a `/health`, `/chat`, and webshell (SSE/chunked streaming), but must be
  scoped. TLS is out of scope now (clear-text on slirp user-net is fine for this
  non-hostile dev environment; noted as future hardening).
- **slirp user-mode net passes TCP/UDP but NOT ICMP** — so `ping` to the guest fails;
  this is irrelevant to HTTP/API over TCP [9][13].
- **SNP for virtio-net under AAVMF**: several sources show AAVMF guest networks with a
  `virtio-net-*` device and `-netdev user` booting successfully [10][11], but whether
  the specific `edk2-aarch64-code.fd` build installs an SNP driver for the virtio-net
  device is **only fully verifiable by booting** (see §6.1). If SNP is unavailable,
  fallbacks are: use a different NIC the firmware knows (e.g. `e1000e`, whose EFI
  option ROM `efi-e1000e.rom` is in this QEMU's share dir), or pull raw virtio-net
  handling into the UEFI adapter.
- The `smoltcp-uefi` crate is a **small, experimental PoC** (0 stars; "highly
  experimental") [6]. It is thin enough to either use as-is, vendor, or reimplement
  (~167 lines of relevant code). Treat as reference, not core dependency.

---

## §2. Architecture — Shared Core, Dual Adapter

The existing `burn/kernel/src/main.rs` (482 lines) mixes **boot/platform code** (the
`_start` asm stub, exception vectors, PL011 UART driver, identity MMU init, heap init,
`kernel_main`) with **platform-neutral logic** (the burn model `kernel::model`,
tokenizer `kernel::tokenizer`, KV cache, `sample_token`, `Rng`, the chat loop). The
design goal is to split these so **one model build** serves both boot paths.

```
                        ┌────────────────── SHARED CORE (platform-neutral) ──────────────────┐
                        │ kernel::net::http    HTTP/1.1 server (listen) + client (connect)    │
                        │ kernel::api          routes: GET /health, POST /chat, webshell      │
                        │ kernel::chat         generation loop, sample_token, Rng, KV cache   │
                        │ kernel::model        burn SmolLM-135M  (no_std, already)            │
                        │ kernel::tokenizer    GPT-2 BPE                                      │
                        └───────────────▲──────────────────────────────────▲────────────────┘
                         transport       │                                  │
                        ┌───────────────┴──────────────────┐   ┌───────────┴─────────────────┐
                        │ smoltcp TCP/IP (shared)          │   │ smoltcp TCP/IP (shared)     │
                        │   phy::Device trait              │   │   phy::Device trait         │
                        └───────────────▲──────────────────┘   └───────────▲─────────────────┘
             Adapter     ┌──────────────┴─────────────┐        ┌─────────────┴──────────────┐
                         │ UEFI adapter                │        │ Raw-ELF adapter             │
                         │ aarch64-unknown-uefi        │        │ aarch64-unknown-none        │
                         │ efi_main, uefi-services     │        │ _start asm, vectors, PL011  │
                         │ SNP (firmware NIC driver)   │        │ identity MMU, heap          │
                         │ uefi console / serial       │        │ UART                        │
                         │ SnpDevice (smoltcp-uefi)    │        │ VIRTIO-net driver (own)     │
                         └───────────────┬─────────────┘        └─────────────┬──────────────┘
                  QEMU    ┌──────────────┴──────────────────┐   ┌─────────────┴──────────────┐
                          │ edk2-aarch64-code.fd +          │   │ raw ELF, -kernel            │
                          │ virtio-net-pci                  │   │ + virtio-net-pci            │
                          └───────────────┬─────────────────┘   └─────────────┬──────────────┘
                  host     └──────── hostfwd tcp::HOST-:GUEST ────────────────┘
                                        http://localhost:HOST
```

### The boot-platform seam (critical)
To reuse one model across both paths, the platform-specific code must detach from the
core:

- **Move out of `main.rs` into the raw-ELF adapter:** `_start` asm, exception vectors,
  PL011 UART, identity MMU init, the `GlobalAlloc` heap setup that depends on the
  raw-ELF linker script, and `kernel_main`'s entry wiring.
- **The core crate** (`kernel::model`, `kernel::tokenizer`, `kernel::chat`,
  `kernel::net`, `kernel::api`) must compile for BOTH `aarch64-unknown-none` and
  `aarch64-unknown-uefi` (both are no_std; see §6.3 for the `smoltcp` `alloc`/nightly
  caveat and §6.4 for confirming burn-flex portability).
- **UEFI adapter** provides: `efi_main` entry, memory allocation via `uefi::allocator`
  + `uefi-services`, console output via UEFI std streams (or the same PL011 base, but
  UEFI console is the idiomatic choice), and the `SnpDevice` for smoltcp.
- **Raw-ELF adapter** provides what it already does (UART/MMU/heap) plus the raw
  virtio-net driver exposing the same `smoltcp::phy::Device` trait.

The two adapters never share boot code; they share only the core + smoltcp net layer.
This matches the burn repo's existing convention of keeping `kernel` a reusable crate
(already consumed by `host`, `model-builder` via `kernel = { path = "../kernel" }`).

---

## §3. Boot / Networking Dependency & Risk Matrix

### 3.1 UEFI path (aarch64-unknown-uefi + SNP)

| Component | Used where | Implementable / Replaceable / Blocker | Effort | Risk | Kernel counterpart / dependency |
|---|---|---|---|---|---|
| `aarch64-unknown-uefi` target | build target | Implementable (Tier 2, no_std) | S | Low | rustc [1] |
| `uefi` crate boot services (`snp`, `ip4config2`) | NIC + DHCP | Implementable (active, safe wrappers) | S | Low | uefi-rs [2][3] |
| `SimpleNetworkProtocol` init/start | NIC bring-up | Implementable; need the firmware to expose it | S | **Med (unverified SNP-virtio)** | uefi::proto::network::snp [2] |
| `smoltcp-uefi::SnpDevice` | phy::Device for smoltcp | **Replaceable** (tiny PoC; vendor or reimplement ~167 LoC) | S | Low–Med | [6][7] |
| `smoltcp` + `alloc` (nightly) | TCP/IP stack | Implementable (no_std, no-heap core) | S–M | Low–Med (nightly + alloc) | [4][5]; §6.3 |
| HTTP framing | serve API/webshell | Implementable (hand-rolled over TCP) | S–M | Low | — |
| `edk2-aarch64-code.fd` firmware | boot | Present on host (already used for Linux guest) | S | Low | this repo [8] |
| `hostfwd` | host-reachable serve | Implementable (`-netdev user,hostfwd=tcp::HOST-:GUEST`) | S | Low | QEMU [9] |

### 3.2 Raw-ELF path (aarch64-unknown-none + custom virtio-net)

| Component | Used where | Implementable / Replaceable / Blocker | Effort | Risk | Kernel counterpart / dependency |
|---|---|---|---|---|---|
| Existing boot asm/MMU/heap/UART | platform | In place (already works) | — | — | `main.rs` / `kernel::mmu` |
| virtio-net device discovery (PCIe/MMIO on `virt`) | NIC discovery | Implementable (PlatformAt/PCI device-tree walk) | M | Med | `virtio-drivers` crate or hand-walked FDT |
| virtqueue transport | NIC data path | Implementable (split ring vq, `alloc`-backed descriptors) | M–L | Med | `virtio-drivers` `VirtIOHeader`/`VirtQueue` or hand-rolled |
| descriptor rings / MMIO BARs | NIC data path | Implementable over 2 GiB heap + identity MMU | M | Med | confirm identity-MMU mapping of BARs |
| RX/TX smoltcp `phy::Device` | net layer | Implementable (glue ring↔phy) | M | Med | smoltcp unmanaged device pattern |
| Interrupts/eventfd | TX/RX notification | **Replaceable** — use MMIO polling (smoltcp poll loop) instead of GIC interrupts for v1 | M | Low–Med | existing exception vector infra; no GIC driver yet |
| `virtio-drivers` crate | driver skeleton | Forkable/Reference (not no_std-first? verify host/no_std) | M | Med | sea-level/virtio-drivers [15] (§6.5) |

---

## §4. Step Plan (S1–S8) — prioritized, costed, simple→complex

Each step maps to objectives and is costed; **no step is built in this spike** — this
is the plan an implementation spike follows.

| Step | Title | Objective(s) | Scope | Effort | Risk | Exit criteria |
|---|---|---|---|---|---|---|
| **S1** | Foundation & feasibility protos | 1, 2, 3 | Workspace/scaffolding; design-only. Future action: (a) hello-world `aarch64-unknown-uefi` app booted under existing `edk2-aarch64-code.fd`; (b) verify AAVMF SNP-virtio availability; (c) confirm burn model compiles for both targets. | M | Med | Demonstrates both targets compile+serial; SNP-virtio question answered |
| **S2** | Raw virtio-net driver | 2, 3 | Virtio-net device on raw-ELF kernel: discovery, virtqueue transport, RX/TX over 2 GiB heap, MMIO-polling (no GIC). Expose `smoltcp::phy::Device`. | L–XL | High | Raw-ELF kernel passes/receives frames (loopback or hostfwd echo) |
| **S3** | UEFI SNP adapter | 1, 3 | `efi_main` + uefi-services + SNP init + `SnpDevice` (use/vendor smoltcp-uefi); DHCP via ip4config2 or static. | M | Med | UEFI kernel reaches TCP loopback/echo |
| **S4** | Shared smoltcp net layer | 3, 4 | One smoltcp config, static IPv4 + event/poll loop, shared TCP read/write abstraction consumed by both adapters. | M | Med | Both adapters drive identical net layer; frames flow |
| **S5** | HTTP server + `POST /chat` | 4 | Minimal HTTP/1.1 framing on a listen socket; `GET /health`, `POST /chat` calling the model. Host-reachable via `hostfwd`. **Serve-API deliverable.** | L | Med | `curl http://localhost:PORT/chat` returns generation |
| **S6** | Webshell / streaming console | 4 | Streaming (chunked/SSE) + a raw-TCP REPL wrapping the chat loop (tokenizer→generate→stream). **Serve-webshell deliverable.** | M–L | Med | Interactive prompt/response over network socket |
| **S7** | HTTP/DNS client | 5 | smoltcp TCP client + DNS; fetch an external API over slirp (host gateway 10.0.2.2). **Consume-API deliverable.** | M | Med | Kernel performs an outbound HTTP fetch, UART/log result |
| **S8** | Integration & hardening | 4, 5, 6 | Shared KV cache per connection; re-entrancy; UART+network logging; hostfwd config; Makefile targets; measure network-served TPS vs serial baseline. | M–L | Med | Stable served API on both boot paths; TPS delta reported |

**Ordering rationale (simple→complex):** S1 de-risks the whole spine (does the model
port to UEFI? does SNP-virtio exist?) before any deep driver work. S2 (raw virtio-net)
is the largest, most uncertain piece, so it is front-loaded after feasibility and run
in parallel with S3 (UEFI SNP, which is small because firmware does the NIC work).
S4 unifies; S5/S6 deliver serving; S7 delivers outbound; S8 hardens.

---

## §5. Citations

Primary sources (all checked 2026-08-28 unless noted):

1. **`*-unknown-uefi` rustc targets (Tier 2)** — https://doc.rust-lang.org/rustc/platform-support/unknown-uefi.html (aarch64-unknown-uefi listed; efi_main no_std example). Tier-2 status per https://doc.rust-lang.org/rustc/platform-support.html ; maintained for aarch64 by @rust-lang/arm-maintainers.
2. **`uefi` crate 0.40.0** — https://docs.rs/uefi/latest/uefi — `proto::network` (modules `snp`, `http`, `ip4config2`, `pxe`); https://docs.rs/uefi/latest/uefi/proto/network/snp/index.html ("Simple Network Protocol … packets transmitted and received").
3. **rust-osdev/uefi-rs** — https://github.com/rust-osdev/uefi-rs (MIT OR Apache-2.0; active; 1,643★; uefi-test-runner `proto/network/snp.rs`).
4. **smoltcp** — https://github.com/smoltcp-rs/smoltcp (0BSD; "does not need heap allocation at all"; TCP/UDP/ICMP/raw sockets; `alloc` feature only on nightly; "HTTP is not provided" — it is a TCP/IP stack).
5. **smoltcp on docs.rs** — https://docs.rs/smoltcp (0.13/0.14; features `std` default vs `alloc` nightly, `medium-ethernet`, `proto-ipv4`, `socket-tcp`, etc.).
6. **smoltcp-uefi (ifd3f)** — https://github.com/ifd3f/smoltcp-uefi (MIT; `SnpDevice` as `smoltcp::phy::Device`; `scripts/test_on_qemu.sh` booting UEFI app + OVMF + bridge NIC; "highly experimental").
7. **smoltcp-uefi docs.rs** — https://docs.rs/smoltcp-uefi/latest/smoltcp_uefi/ ; deps `smoltcp 0.12` + `uefi 0.34.1`.
8. **`edk2-aarch64-code.fd` present locally** — `/opt/homebrew/share/qemu/edk2-aarch64-code.fd` (verified this session); the burn repo's Linux-guest path already attaches it (`burn/scripts/run-linux.sh` uses `-drive if=pflash,...,file=${FW}`) and session logs document AAVMF-only Ubuntu cloud image boot.
9. **QEMU networking / User Networking (slirp)** — https://wiki.qemu.org/Documentation/Networking : slirp "cannot ping (ICMP)" but "TCP and UDP will"; `-netdev user,id=n0,hostfwd=hostip:hostport-guestip:guestport` forwards host ports to the guest; guest gateway 10.0.2.2. Also https://qemu-project.gitlab.io/qemu/system/devices/net.html .
10. **AAVMF guest-net examples** — https://docs.anduinos.com/Virtualization/ARM64-In-QEMU.html and Fedora https://docs.fedoraproject.org/.../Booting_a_QEMU_image show `edk2-aarch64/AAVMF` + `virtio-net(-pci)` + `-netdev user` booting UEFI guests. (Caveat: does not alone prove the specific firmware build's SNP driver; §6.1.)
11. **buildroot aarch64-efi / other edk2-in-qemu references** — https://github.com/buildroot/buildroot/blob/master/board/aarch64-efi/readme.txt (`-device virtio-net-device -netdev user,id=eth0` with `QEMU_EFI.fd`); FreeBSD-on-QEMU gist uses `virtio-net-device` + `tap` under `edk2-aarch64-code.fd`.
12. **full-controll-os (TNTTheLagger)** — https://github.com/TNTTheLagger/full-controll-os ("UEFI-native … early-boot networking using UEFI SNP + smoltcp — no OS services").
13. **QEMU net docs (hostfwd redirects TCP/UDP)** — https://www.qemu.org/docs/master/system/devices/net.html ("TCP, UDP or UNIX connections can be redirected from the host to the guest"; ICMP generally does not work with user-mode net).
14. **The existing burn kernel** — this repo `burn/kernel/src/main.rs` (boot asm/vectors/PL011/MMU/heap interleaved with `kernel::model`, `kernel::tokenizer`, `kernel::chat`); `burn/kernel/Cargo.toml` (burn-core/burn-nn/burn-flex 0.21 `default-features=false`, linked_list_allocator 0.10); `burn/kernel/.cargo/config.toml` (`build-std=["core","alloc"]`).
15. **virtio-drivers crate** — https://github.com/rust-vmm/virtio-drivers (or sea-level/virtio-drivers variant; Rust virtio-capable driver crate; verify no_std/target fit at build; flagged §6.5 as unverified-for-this-target).
16. **NIGHTRUN** — https://github.com/hardrave/NIGHTRUN (UEFI-resident LLM, ARM64 NEON quant kernels; cited as the architectural reference for a UEFI bare-metal platform; no network component confirmed — networking is an on-addition this spike designs).
17. **smoltcp `alloc`+nightly requirement** — https://docs.rs/smoltcp feature list: "The `alloc` feature … only works on nightly rustc." (burn already uses nightly via `burn/rust-toolchain.toml`.)

### Verified availability on this host (2026-08-28)
- `qemu-system-aarch64` 11.1.0 with `edk2-aarch64-code.fd` (64 MB) and EFI option ROMs including `efi-virtio.rom`, `efi-e1000e.rom`.
- Rust nightly + `aarch64-unknown-none` target + `build-std=["core","alloc"]` already configured for the burn kernel.

---

## §6. Further Research

Prioritized open questions, each tied to an objective, with suggested search terms
and target sources.

1. **Does AAVMF (`edk2-aarch64-code.fd`) install an SNP driver for a `virtio-net-*` device?** *(Obj 1.)* *Terms:* "AAVMF virtio-net SNP", "edk2 ArmVirtQemu virtio-net SNP protocol". *Source:* boot a hello-provider app that enumerates protocols on the NIC handle under HVF; if absent, test `e1000e` (option ROM present) or hand virtio. Only empirically verifiable — flagged until then.
2. **Does the burn model (`burn-flex`, `burn-nn`, `burn-core`) compile unchanged for `aarch64-unknown-uefi`?** *(Obj 3.)* *Terms:* "burn-flex no_std aarch64-unknown-uefi", "burn default-features false target". *Source:* add the target and `cargo check`; if it fails, decide feature/config deltas (both are no_std, so likely fine).
3. **smoltcp `alloc` on our nightly + build-std setup, and on `aarch64-unknown-uefi`.** *(Obj 4/5.)* *Terms:* "smoltcp alloc feature nightly build-std", "smoltcp uefi target". *Source:* docs.rs feature list + a minimal compile. Flag: smoltcp 0.13/0.14 requires rustc ≥1.91; burn toolchain is already nightly.
4. **`virtio-drivers` (rust-vmm) usability on QEMU `virt` aarch64 raw-ELF kernel with identity MMU + 2 GiB heap.** *(Obj 2.)* *Terms:* "virtio-drivers no_std aarch64 QEMU virt", "virtio-drivers accepts drivers". *Source:* crate README/docs + a probe in the raw-ELF kernel; decide reuse vs hand-rolled virtqueue.
5. **GIC interrupts vs MMIO polling for virtio-net TX/RX on the raw-ELF path.** *(Obj 2.)* *Terms:* "QEMU virt GIC virtio-net eventfd", "virtio mmio polling no interrupt". *Source:* docs.rs/virtio spec; plan polling-first (fits the existing poll loop and no-GIC driver), GIC later.
6. **Cleanest concurrent-serving model for one SmolLM KV cache across connections (HTTP + webshell).** *(Obj 6.)* *Terms:* "burn KV cache concurrent", "smoltcp server per-connection". *Source:* burn `Cache` API + smoltcp `SocketSet`; design a per-connection cache or serialize generation behind a lock.
7. **HTTP framing library vs hand-rolled for no_std UEFI/raw-ELF.** *(Obj 4/5.)* *Terms:* "no_std http server crate aarch64", "rust uhttp no_std". *Source:* docs.rs/crates.io survey (`uhttp`, `httparse`, `tiny_http` std-only) — likely hand-roll for HTTPServe; verify before committing to a crate.
8. **TLS for the served API (future hardening).** *(Obj 4/6.)* *Terms:* "no_std TLS rust qemu", "mosquitto rustls no_std". *Source:* rustls (std) vs `tls-parser`/embed — out of current scope, noted as future.
