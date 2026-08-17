# Research Spike 01 — Rust Native AI on Bare Metal

**Search date:** 2026-08-17. **Scope:** all six objectives from `objective.md`.
**Method:** GitHub, crates.io, arXiv, Papers With Code, project docs. Every repo
URL, license, and star count was verified against its primary source during this
session; uncertain facts are flagged inline. Unverifiable projects were excluded
per the anti-hallucination constraint (see §8).

Legend — Recency: **Active** = commits within last 12 months; **Stable** = historical, maintenance-only; **Archived** = read-only. Maturity: 1 (research code) – 5 (production).

---

## 1. Project Matrix

75 verified projects across the six objectives. Rank is overall (rubric in §3). Follow-up sweeps (search dates 2026-08-17; DuckDuckGo SERP review + GitHub API) added **AXIOM**, **LLM OS**, **Zero**, **MicroFlow**, **Fable-OS**, **airox-os/core**, a dedicated **Rust no_std AI OS** pass (**alice-aegis**, **BraiNIXOS**, **gehirn-system**, **claudio-os**, **Nyx**, **gpu-compute-nostd**, **edgedl**, **TrustOS**), the driver-learning sweep (**arceos**, **nvme-nostd**, **rustrial-os**), and a ~100-result GitHub sweep (20 targeted queries, 164 unique hits, 2026-08-17) that surfaced the remaining no_std AI OS cluster (**Squirrel AIOS**, **Noesis-OS**, **MielinOS**), the BitNet b1.58 CPU-engine pair (**r3-engine**, **project-willamette**), and **dumbnet** (no_std NN library); agent-framework noise (IronClaw 12.6k★, Astrid 10.3k★, ~50 "agent OS" clones) was classified to §8 leads. A final arXiv paper sweep (2026-08-17) captured the **AI-native OS research lineage** (ProbeLogits/Elastic Gang/Governed MCP, NVMe-direct KV offload, context demand-paging, tiered memory, ternary CPU kernels) in §7. Scope widened to ~10 years.

### Objective 1 — Rust low-level drivers (PCIe / NVMe / net / GPU)

| # | Project | URL | License | Lang | Obj #s | S/M | Recency | Maturity | Rank |
|---|---------|-----|---------|------|--------|-----|---------|----------|------|
| 1 | virtio-drivers (rcore-os) | https://github.com/rcore-os/virtio-drivers | MIT | Rust | 1 | System | Active | 4 | 13 |
| 2 | Nova GPU driver (NVIDIA/Red Hat) | https://github.com/NVIDIA/nova · rust-for-linux.com/nova-gpu-driver | GPL-2.0 | Rust | 1 | System | Active | 3 | 31 |
| 3 | irqlevel/nos | https://github.com/irqlevel/nos | MIT | Rust (C++ kernel core) | 1 | System | Active | 3 | 23 |
| 4 | Rust-for-Linux NVMe (rnvme) | https://rust-for-linux.com/nvme-driver | GPL-2.0 | Rust | 1 | System | Active | 3 | 46 |
| 5 | vhost-device-gpu (rust-vmm) | https://github.com/rust-vmm/vhost-device | Apache-2.0 | Rust | 1 | System | Active | 4 | 47 |
| 6 | Apple AGX GPU driver (Asahi) | https://asahilinux.org/docs/hw/soc/agx/ | GPL-2.0 | Rust | 1 | System | Active | 4 | 48 |
| 7 | arceos | https://github.com/arceos-org/arceos | Apache-2.0 | Rust | 1, 2 | System | Active | 4 | 32 |
| 8 | nvme-nostd | https://github.com/suhteevah/nvme-nostd | Apache-2.0 | Rust no_std | 1 | System | New (2026) | 2 | 55 |
| 9 | gpu-compute-nostd | https://github.com/suhteevah/gpu-compute-nostd | MIT / Apache-2.0 | Rust no_std | 1 | System | New (2026) | 2 | 42 |
| 10 | rustrial-os | https://github.com/lazzerex/rustrial-os | MIT | Rust | 1 | System | Active (2026) | 3 | 62 |
| 11 | TrustOS | https://github.com/nathan237/TrustOS | Apache-2.0 | Rust | 1 | System | Active (2026) | 3 | 61 |
| 12 | e1000-driver | https://github.com/elliott10/e1000-driver | GPL-2.0 | Rust | 1 | System | Dormant in-repo | 2 | 66 |
| 13 | nvme_driver (rcore-os) | https://github.com/rcore-os/nvme_driver | none declared | Rust | 1 | System | Dormant (2023) | 2 | 63 |
| 14 | pci-driver (crate) | https://crates.io/crates/pci-driver | MIT OR Apache-2.0 | Rust | 1 | System | Low activity | 2 | 68 |
| 15 | pci-rs (rcore-os) | https://github.com/rcore-os/pci-rs | Apache-2.0 | Rust | 1 | System | Dormant (2023) | 2 | 69 |

### Objective 2 — AI under a custom Rust kernel

| # | Project | URL | License | Lang | Obj #s | S/M | Recency | Maturity | Rank |
|---|---------|-----|---------|------|--------|-----|---------|----------|------|
| 16 | AXIOM | https://github.com/Kanchisaw03/axiom | none declared | Rust no_std | 2, 3, 6 | System | New (2026) | 2 | 38 |
| 17 | alice-aegis (A.L.I.C.E.) | https://github.com/Aefinity-AI/alice-aegis | Apache-2.0 | Rust no_std | 2, 3 | System | New (2026) | 2 | 17 |
| 18 | BraiNIXOS | https://github.com/jbrahy/BraiNIXOS | MIT / Apache-2.0 | Rust no_std | 2, 3 | System | New (2026) | 2 | 33 |
| 19 | gehirn-system | https://github.com/dev-natebrito/gehirn-system | AGPL-3.0 | Rust | 2, 3 | System | Pre-alpha (2026) | 1 | 70 |
| 20 | claudio-os | https://github.com/suhteevah/claudio-os | AGPL-3.0 | Rust no_std | 2, 3 | System | New (2026) | 2 | 64 |
| 21 | Nyx | https://github.com/Asmodeus14/Nyx | Apache-2.0 | Rust | 2, 3 | System | Active (2026) | 3 | 53 |
| 22 | LLM OS | https://github.com/ranjan42/llm-os-rust | MIT | Rust | 2, 3 | System | New (2026) | 2 | 67 |
| 23 | coconutOS | https://github.com/coconut-os/coconutOS | ISC | Rust | 2, 3 | System | Active (2026) | 2 | 16 |
| 24 | MerlionOS (merlion-infer) | https://github.com/MerlionOS/merlion-infer | MIT | Rust | 2, 3 | System | Active (2026) | 2 | 15 |
| 25 | airox-os/core | https://github.com/airox-os/core | MIT | Rust no_std | 2, 3 | System | Dormant (2025) | 2 | 72 |
| 26 | Squirrel AIOS | https://github.com/tusharkhatriofficial/squirrel | none declared (README: MIT) | Rust no_std | 2, 3 | System | Active (2026) | 2 | 26 |
| 27 | Noesis-OS | https://github.com/beltromatti/Noesis-OS | MIT | Rust no_std | 2, 3 | System | New (2025-11) | 2 | 27 |
| 28 | MielinOS | https://github.com/cool-japan/mielin | Apache-2.0 | Rust | 2, 3 | System | Active (2026, v0.1.0-rc) | 3 | 51 |

### Objective 3 — Low-level OS for AI

| # | Project | URL | License | Lang | Obj #s | S/M | Recency | Maturity | Rank |
|---|---------|-----|---------|------|--------|-----|---------|----------|------|
| 29 | Anima OS (ProbeLogits) | https://arxiv.org/abs/2604.11943 · /abs/2607.04668 | n/a (research) | Rust no_std | 3 | System | Active | 3 | 7 |
| 30 | NIGHTRUN | https://github.com/hardrave/NIGHTRUN | MIT | Rust | 3, 4 | System | Active (2026) | 3 | 4 |
| 31 | vLLM (PagedAttention) | https://github.com/vllm-project/vllm | Apache-2.0 | Python/C++/Rust | 3, 5, 6 | System | Active | 5 | 1 |
| 32 | Zero | https://github.com/silicatetech/zero | AGPL-3.0 | Rust | 2, 3 | System | New (2026) | 2 | 19 |
| 33 | AIOS (LLM Agent OS) | https://github.com/agiresearch/AIOS | n/a | Python | 3 | System | Active | 3 | 44 |
| 34 | Fable-OS | https://github.com/robiot/fable-os | none detected | Rust + C (lwIP/mbedTLS) | 3 | System | Active (2026) | 2 | 74 |
| 35 | PunkGo | https://arxiv.org/abs/2602.20214 | n/a | Rust (paper) | 3 | System | Submitted 2026 | 2 | 75 |

### Objective 4 — Local hardware AI

| # | Project | URL | License | Lang | Obj #s | S/M | Recency | Maturity | Rank |
|---|---------|-----|---------|------|--------|-----|---------|----------|------|
| 36 | llama.cpp | https://github.com/ggml-org/llama.cpp | MIT | C/C++ | 4, 3 | System | Active | 5 | 9 |
| 37 | mistral.rs | https://github.com/EricLBuehler/mistral.rs | MIT | Rust | 4, 3, 6 | System | Active | 4 | 6 |
| 38 | candle | https://github.com/huggingface/candle | Apache-2.0/MIT | Rust | 4 | System | Active | 5 | 8 |
| 39 | lele | https://github.com/miuda-ai/lele | MIT | Rust | 4, 3 | System | Active | 3 | 20 |
| 40 | burn | https://github.com/tracel-ai/burn | Apache-2.0 | Rust | 4 | System | Active | 4 | 21 |
| 41 | RunIT_Mac | https://github.com/muhammed-eldabea/RunIT_Mac | MIT | Rust + Metal | 4 | System | New (2026) | 2 | 34 |
| 42 | Ollama | https://github.com/ollama/ollama | MIT | Go | 4 | System | Active | 5 | 49 |
| 43 | tinygrad | https://github.com/tinygrad/tinygrad | MIT | Python | 4, 3 | System | Active | 4 | 39 |
| 44 | MicroFlow | https://github.com/matteocarnelos/microflow-rs | MIT / Apache-2.0 | Rust | 4, 6 | System | Stable (2024) | 3 | 25 |
| 45 | edgedl | https://github.com/g1ibby/edgedl | MIT | Rust no_std | 4 | System | New (2026) | 2 | 65 |
| 46 | dumbnet | https://github.com/djugei/dumbnet | none detected | Rust no_std | 4 | System | Dormant (2019) | 2 | 73 |

### Objective 5 — Beyond quantization

| # | Project | URL | License | Lang | Obj #s | S/M | Recency | Maturity | Rank |
|---|---------|-----|---------|------|--------|-----|---------|----------|------|
| 47 | BitNet b1.58 / bitnet.cpp | https://github.com/microsoft/BitNet | MIT | C++/Python | 5 | System+Model | Active | 5 | 2 |
| 48 | T-MAC | https://github.com/microsoft/T-MAC | MIT | C++ | 5 | System | Active | 4 | 5 |
| 49 | MatMul-free LM | https://github.com/ridgerchu/matmulfreellm | Apache-2.0 | Python | 5 | Model | Active | 3 | 14 |
| 50 | pulp-transformer (FWSA) | https://github.com/pulp-platform/pulp-transformer | Apache-2.0 | C | 5 | System | Active | 2 | 36 |
| 51 | AWQ | https://github.com/mit-han-lab/llm-awq | MIT | Python/CUDA | 5 | Model | Active | 5 | 28 |
| 52 | MXFP8 / Microscaling (OCP MX) | https://github.com/microsoft/microxcaling | MIT | Python/CUDA | 5 | System | Active | 3 | 35 |
| 53 | Medusa | https://github.com/FasterDecoding/Medusa | Apache-2.0 | Python | 5 | System+Model | Active | 4 | 30 |
| 54 | DeepSeek-V3 | https://github.com/deepseek-ai/DeepSeek-V3 | MIT code / DL model lic | Python | 5 | Model | Active | 5 | 22 |
| 55 | SparseGPT | https://github.com/IST-DASLab/sparsegpt | Apache-2.0 | Python | 5 | Model | Stable | 3 | 50 |
| 56 | Speculative Decoding | https://arxiv.org/abs/2211.17192 | n/a (paper) | n/a | 5 | Model | Stable | 5 | 40 |
| 57 | Lookahead Decoding | https://github.com/hao-ai-lab/LookaheadDecoding | Apache-2.0 | Python | 5 | Model | Active | 3 | 43 |
| 58 | Wanda | https://github.com/locuslab/wanda | MIT | Python | 5 | Model | Active | 3 | 45 |
| 59 | KVQuant | https://github.com/SqueezeAILab/KVQuant | none detected | Python/CUDA | 5 | System+Model | Stable | 2 | 59 |
| 60 | FlexGen / FlexLLMGen | https://github.com/FMInference/FlexLLMGen | Apache-2.0 | Python | 5 | System | Archived 2024-12 | 4 | 71 |
| 61 | r3-engine | https://github.com/dhilipsiva/r3-engine | none declared | Rust (safe, AVX-512/Wasm) | 5, 4 | System | New (2026-01) | 3 | 54 |
| 62 | project-willamette | https://github.com/nangman-infra/project-willamette | Apache-2.0 | Rust | 5, 4 | System | Active (2026) | 3 | 56 |

### Objective 6 — Core innovation (memory / recall / reasoning / hallucination)

| # | Project | URL | License | Lang | Obj #s | S/M | Recency | Maturity | Rank |
|---|---------|-----|---------|------|--------|-----|---------|----------|------|
| 63 | FlashAttention | https://github.com/Dao-AILab/flash-attention | BSD-3-Clause | CUDA/C++ | 5, 6 | System | Active | 5 | 3 |
| 64 | Qdrant | https://github.com/qdrant/qdrant | Apache-2.0 | Rust | 6 | System | Active | 5 | 10 |
| 65 | RWKV | https://github.com/BlinkDL/RWKV-LM | Apache-2.0 | Python/C++ | 6 | Model | Active | 4 | 11 |
| 66 | Mamba | https://github.com/state-spaces/mamba | Apache-2.0 | Python/C++ | 6 | Model | Active | 4 | 12 |
| 67 | MemGPT / Letta | https://github.com/letta-ai/letta | Apache-2.0 | Python | 6 | Model | Active | 4 | 18 |
| 68 | RAG | https://arxiv.org/abs/2005.11401 | n/a (paper) | n/a | 6 | Model | Stable | 5 | 24 |
| 69 | ReAct | https://github.com/ysymyth/ReAct | MIT | Python | 6 | Model | Active | 4 | 29 |
| 70 | SelfCheckGPT | https://github.com/potsawee/selfcheckgpt | MIT | Python | 6 | Model | Active | 3 | 37 |
| 71 | Test-Time Compute (o1-style) | https://arxiv.org/abs/2408.03314 | n/a (paper) | n/a | 6 | Model | Stable | 5 | 41 |
| 72 | Tree of Thoughts | https://github.com/princeton-nlp/tree-of-thought-llm | MIT | Python | 6 | Model | Stable | 4 | 52 |
| 73 | Chain-of-Thought | https://arxiv.org/abs/2201.11903 | n/a (paper) | n/a | 6 | Model | Stable | 5 | 57 |
| 74 | Contrastive Decoding | https://github.com/XiangLi1999/ContrastiveDecoding | unspecified | Python | 6 | Model | Stable | 3 | 58 |
| 75 | Self-Consistency | https://arxiv.org/abs/2203.11171 | n/a (paper) | n/a | 6 | Model | Stable | 5 | 60 |

**Matrix notes:**
- vLLM ships first-class Rust code in-tree (`rust/` dir, `rust-toolchain.toml`, `build_rust.sh`) — a rare production Rust path into an LLM serving engine.
- **Rust no_std AI OS** is now a distinct, well-populated sub-field. The focused GitHub-API pass (2026-08-17, ~100-result sweep) surfaced new kernels beyond the original cluster, split by strategy: (a) *inference in the kernel* — **Zero** (Ring-0, 194.5 tok/s vendor-reported), **alice-aegis** (BitNet b1.58 ternary UEFI unikernel, ~2.80 tok/s measured), **gehirn-system** (pre-alpha), **Noesis-OS** (ARM64 no_std microkernel with on-device inference pipelines); (b) *secure inference serving* — **BraiNIXOS** (capability microkernel, W^X+KPTI); (c) *AI-as-architect kernels* — **Squirrel AIOS** (x86_64 no_std, Limine UEFI+BIOS, GPU-as-cognitive-substrate), **claudio-os** (294 KLOC agent OS), **Nyx** (quantum+AI), **MielinOS** (distributed-AI-agent microkernel, WASM sandbox, live migration); (d) *agent OSes (cloud)* — **Fable-OS**, **LLM OS**. Only Zero and alice-aegis actually execute CPU-only inference on bare metal. None is mature (>2 stars).
- Driver-learning sweep added pure-Rust OS/driver references: **arceos** (779-star modular unikernel — the closest "our platform" archetype), **nvme-nostd** (no_std NVMe 1.4 over PCI BAR0 MMIO), **gpu-compute-nostd** (no_std NVIDIA GPU compute driver with tensor ops — fills the no_std GPU gap), **rustrial-os** (RTL8139 NIC + TCP/IP stack), and **TrustOS** (48★, 264 KLOC bare-metal Rust, zero C, validated AMD GPU SDMA on real hardware).
- The author **suhteevah** maintains the most complete no_std AI-driver stack found: `nvme-nostd`, `gpu-compute-nostd`, and the `claudio-os` agent kernel.
- MicroFlow is the oldest inclusion (~2022–24): a compiler-based Rust TinyML engine with static + page-based memory allocation for 2 kB-RAM 8-bit MCUs.
- FlexGen renamed to FlexLLMGen, archived 2024-12 (read-only); kept for historical significance.
- The ~100-result GitHub sweep (20 queries, 164 unique hits, 2026-08-17) shows the query surface for "rust ai os" is dominated by **agent-framework noise** (~50 repos: IronClaw 12.6k★, Astrid 10.3k★, AgentOS/runtime clones like bbarit, rustyhand, feros, LoopForge, broomva's *arcan/lago/praxis* family, openfang-cn localizations, etc.) and **"AI-native Linux distros"** (ZethraOS, NuraOS, AuraOS, aulinx, HakOS, LivingOS, Jarvis-OS…) that wrap Linux/systemd with agents — none are bare-metal Rust kernels, so they are classified in §8, not the matrix. The true no_std AI OSes only surface via OSdev/arXiv/crate-specific queries.
- **No license detected** for KVQuant, AXIOM, Fable-OS, **Squirrel AIOS** (README badge says MIT, GitHub API returns none), **r3-engine**, and **dumbnet** — flag for licensing review before reuse. Zero is AGPL-3.0-or-later (dual-licensed; commercial license separate).
- DuckDuckGo (2026-08-17, main site, query "rust ai os", saved SERP reviewed): the top-10 results are **KnoxOS** (marketing site + docs-only repo), **OpenFang** (two marketing domains, "Agent OS, Rust-powered"), **Kernex** (Rust agent *runtime*, not an OS), **airox-os/core** (added to matrix), **Rig** (8290★ Rust LLM-agent framework), plus non-project hits (ZDNet article, rust4ai.com, Rust Foundation position statement). **Notable:** none of the top-10 are the bare-metal LLM kernels in this matrix (Zero, MerlionOS, NIGHTRUN…) — the phrase "rust ai os" is captured by marketing/agent-framework sites, which is why the hobby bare-metal cluster only surfaces via OSdev/arXiv/crate queries. DDG's `html.duckduckgo.com` endpoint (queried earlier) showed no exact-phrase hits and then CAPTCHA-blocked further automated queries.

---

## 2. SWOT Analysis (top 10)

### 2.1 vLLM / PagedAttention — Rank 1 [System]
**Strengths:** OS-paging-inspired KV-cache reduces fragmentation and enables ~2–4× serving throughput [1]; production-grade, 89.3k stars; 200+ model architectures; PagedAttention + continuous batching + prefix caching; Rust components in-tree.
**Weaknesses:** Massive monorepo; bare-metal reuse means cherry-picking the paging module, not the engine; most kernels assume CUDA/torch.
**Opportunities:** The block-table physical/virtual KV-page model is a blueprint for a Rust no_std memory manager.
**Threats:** Whole-project adoption would drag in torch/CUDA — out of scope for bare metal.
**Relevance:** Highest-value memory-management pattern for a Rust inference runtime, and proof Rust has a place in production LLM serving internals.

### 2.2 BitNet b1.58 / bitnet.cpp — Rank 2 [System+Model]
**Strengths:** Ternary {-1,0,1} weights give 2.37–6.17× CPU speedup and 55–82% energy reduction [3][4]; ~100B models on a single CPU at human-reading speed; 40.1k stars, MIT; LUT kernels on ARM + x86 + GPU; official 2B-4T model.
**Weaknesses:** Requires training in the 1.58-bit regime — not post-hoc quantization of existing models; embeddings kept higher precision.
**Opportunities:** Demonstrates sub-2-bit is viable on ubiquitous hardware; kernel techniques (grouped-bit partial sums, table lookup) are language-agnostic.
**Threats:** Ecosystem limited to specially-trained ternary models so far.
**Relevance:** Blueprint for a ternary-weights + LUT inference engine in Rust.

### 2.3 FlashAttention — Rank 3 [System]
**Strengths:** IO-aware tiling makes attention memory linear rather than quadratic [2]; 24.7k stars, BSD-3-Clause; FA-2/3/4 lineage; embedded in nearly every serving stack.
**Weaknesses:** Reference is CUDA/GPU-specific; CPU path via Triton is not no_std-friendly.
**Opportunities:** Online-softmax + block tiling is portable to Rust CPU/NEON and pays off at long context.
**Threats:** Re-implementation risks subtle numerical divergence without strong test vectors.
**Relevance:** Kernel-fusion + memory-layout template for our attention path on bare-metal CPU.

### 2.4 NIGHTRUN — Rank 4 [System]
**Strengths:** UEFI-resident LLM runtime with zero conventional OS; boots into a chat LLM on x86 PCs and Raspberry Pi 5; hand-written AVX2+FMA+F16C and NEON quantized kernels (Q8_0/Q4_K/Q6_K); token-for-token parity gated against llama.cpp; ~20 tok/s Llama 3.2 1B on 8-core QEMU x86; MIT; 222 stars, active.
**Weaknesses:** Prototype/experimental; single-author; QEMU-validated so far.
**Opportunities:** The closest thing to our target architecture — storage sealed after model load, no OS beneath; strong correctness-gate methodology to replicate.
**Threats:** Young project; may stall.
**Relevance:** Direct existence proof and reference implementation for a bare-metal Rust inference path.

### 2.5 T-MAC — Rank 5 [System]
**Strengths:** LUT-based mixed-precision GEMM (int1/2/3/4) with no dequantization; 4–5× token-gen speedup vs llama.cpp; linear FLOPs scaling in bit-width; EuroSys 2025; MIT [14].
**Weaknesses:** Kernel code is C++/TVM/llvm-codegen heavy; x86 support uneven on low-bandwidth parts.
**Opportunities:** Table-lookup + shift-accumulate maps cleanly to ARM/NEON `tbl`/`pshuf` — the single most relevant kernel technique for a Rust low-bit runtime.
**Threats:** Repo activity slowed after 2025, but the technique lives on inside bitnet.cpp.
**Relevance:** Direct kernel design reference; "easy/moderate" Rust port because LUTs are architecture-agnostic.

### 2.6 mistral.rs — Rank 6 [System]
**Strengths:** Full PagedAttention, FlashAttention V2/V3, continuous batching, prefix caching, GGUF/EXL2 quantization natively in Rust; CPU (MKL/Accelerate) + CUDA + Metal; pure-Rust build needs no C toolchain; 7.6k stars, MIT.
**Weaknesses:** Pre-1.0; CUDA paths rely on cuDNN/cuBLAS bindings.
**Opportunities:** Its `mistralrs-paged-attn` crate is direct precedent for Rust-managed KV-cache memory — allocator patterns to port to bare metal.
**Threats:** Fast-moving API surface.
**Relevance:** Strongest proof that the attention-memory algorithms are fully expressible in Rust.

### 2.7 Anima OS / ProbeLogits — Rank 7 [System]
**Strengths:** Bare-metal x86-64 kernel (~285k lines Rust) with a resident in-kernel LLM engine: GGUF loader, AVX-512 quantized kernels, work-stealing graph executor (~226 barriers/token); 1,666 tok/s on SmolLM2-135M (1.39× llama.cpp) and 15 tok/s on Qwen2.5-7B; kernel-level "ProbeLogits" safety guards; "elastic gang" co-scheduling (1.28–1.75× vs static core splits) [30].
**Weaknesses:** Research papers only — no public repo located; unverified beyond the papers.
**Opportunities:** Strongest published evidence of AI as a resident OS primitive; per-token cost data informs design.
**Threats:** Claims rest on preprints; no code to inspect.
**Relevance:** The most directly relevant published existence proof of "AI on Rust bare metal."

### 2.8 candle — Rank 8 [System]
**Strengths:** HF's minimalist Rust ML framework; CPU (pure-Rust + MKL), CUDA, Metal, WASM; GGUF/safetensors/ONNX loaders; 90+ example models; motivated by edge/serverless cold-start; 20.9k stars, dual-licensed.
**Weaknesses:** No_std not fully supported; focus is inference, not a kernel.
**Opportunities:** The natural foundation (or reference) for a Rust AI runtime; backends are pluggable.
**Threats:** None major; well-maintained by HF.
**Relevance:** Most-used Rust inference stack; its backend abstraction informs our hardware layer.

### 2.9 llama.cpp — Rank 9 [System]
**Strengths:** De-facto local inference engine: GGUF format, Q2–Q8/K-quants, CPU AVX/AVX-512/NEON + CUDA/Metal/ROCm/Vulkan/WebGPU backends; 124k stars; the parity baseline every Rust runtime gates against.
**Weaknesses:** C/C++; not embeddable in a no_std kernel as-is.
**Opportunities:** Its AVX-512 quantized matvec kernels and GGUF parser are the reference for our Rust ports; format and tests are reusable.
**Threats:** None — but porting kernels is non-trivial.
**Relevance:** Performance/format reference point for all bare-metal Rust inference work.

### 2.10 Qdrant — Rank 10 [System]
**Strengths:** Production Rust vector database: HNSW ANN, memory-mapped storage, SIMD distance, custom quantization; Apache-2.0; active.
**Weaknesses:** User-space server (std), not a no_std library.
**Opportunities:** Proves a production vector store is viable in safe Rust; mmap/HNSW patterns inform a bare-metal storage/recall layer.
**Threats:** None for reference purposes.
**Relevance:** Storage & recall layer for RAG-style memory under a Rust AI runtime.

---

## 3. Ranking

### 3.1 Rubric (weights sum to 100)

| Criterion | Weight |
|---|---|
| Relevance to objectives | 30 |
| Technical novelty | 20 |
| Code / license quality | 15 |
| Maturity & activity | 15 |
| Portability to our stack (Rust no_std) | 10 |
| Documentation / evidence | 10 |
| **Total** | **100** |

### 3.2 Score table (0–10 per criterion, weighted; top 20 shown)

| Rank | Project | Rel (×30) | Nov (×20) | C/Lic (×15) | Mat (×15) | Port (×10) | Docs (×10) | Total |
|------|---------|--------|--------|--------|--------|--------|--------|-------|
| 1 | vLLM / PagedAttention | 9.0 (27) | 9.0 (18) | 10.0 (15) | 10.0 (15) | 8.0 (8) | 9.0 (9) | **92.0** |
| 2 | BitNet b1.58 / bitnet.cpp | 10.0 (30) | 9.0 (18) | 9.0 (13.5) | 9.0 (13.5) | 7.0 (7) | 8.0 (8) | **90.0** |
| 3 | FlashAttention | 9.0 (27) | 9.0 (18) | 9.0 (13.5) | 10.0 (15) | 7.0 (7) | 9.0 (9) | **89.5** |
| 4 | NIGHTRUN | 10.0 (30) | 9.0 (18) | 9.0 (13.5) | 6.0 (9) | 10.0 (10) | 9.0 (9) | **89.5** |
| 5 | T-MAC | 10.0 (30) | 9.0 (18) | 8.0 (12) | 8.0 (12) | 9.0 (9) | 8.0 (8) | **89.0** |
| 6 | mistral.rs | 9.0 (27) | 8.0 (16) | 9.0 (13.5) | 9.0 (13.5) | 10.0 (10) | 8.0 (8) | **88.0** |
| 7 | Anima OS | 10.0 (30) | 10.0 (20) | 6.0 (9) | 6.0 (9) | 9.0 (9) | 9.0 (9) | **86.0** |
| 8 | candle | 9.0 (27) | 7.0 (14) | 9.0 (13.5) | 9.0 (13.5) | 10.0 (10) | 8.0 (8) | **86.0** |
| 9 | llama.cpp | 9.0 (27) | 7.0 (14) | 10.0 (15) | 10.0 (15) | 6.0 (6) | 9.0 (9) | **86.0** |
| 10 | Qdrant | 8.0 (24) | 7.0 (14) | 10.0 (15) | 10.0 (15) | 9.0 (9) | 9.0 (9) | **86.0** |
| 11 | RWKV | 9.0 (27) | 9.0 (18) | 8.0 (12) | 8.0 (12) | 8.0 (8) | 8.0 (8) | **85.0** |
| 12 | Mamba | 9.0 (27) | 9.0 (18) | 8.0 (12) | 8.0 (12) | 7.0 (7) | 8.0 (8) | **84.0** |
| 13 | virtio-drivers | 9.0 (27) | 6.0 (12) | 9.0 (13.5) | 9.0 (13.5) | 10.0 (10) | 8.0 (8) | **84.0** |
| 14 | MatMul-free LM | 10.0 (30) | 9.0 (18) | 7.0 (10.5) | 6.0 (9) | 9.0 (9) | 7.0 (7) | **83.5** |
| 15 | MerlionOS | 10.0 (30) | 9.0 (18) | 8.0 (12) | 4.0 (6) | 9.0 (9) | 8.0 (8) | **83.0** |
| 16 | coconutOS | 10.0 (30) | 9.0 (18) | 8.0 (12) | 4.0 (6) | 9.0 (9) | 8.0 (8) | **83.0** |
| 17 | alice-aegis | 10.0 (30) | 9.0 (18) | 9.0 (13.5) | 2.0 (3) | 10.0 (10) | 8.0 (8) | **82.5** |
| 18 | MemGPT / Letta | 9.0 (27) | 9.0 (18) | 8.0 (12) | 8.0 (12) | 5.0 (5) | 8.0 (8) | **82.0** |
| 19 | Zero | 10.0 (30) | 9.0 (18) | 7.0 (10.5) | 3.0 (4.5) | 10.0 (10) | 8.0 (8) | **81.0** |
| 20 | lele | 9.0 (27) | 8.0 (16) | 8.0 (12) | 6.0 (9) | 9.0 (9) | 7.0 (7) | **80.0** |

Full ranked list (ranks 21–75, total score): burn 79.5 · DeepSeek-V3 79.5 · irqlevel/nos 79.0 · RAG 79.0 · MicroFlow 78.5 · **Squirrel AIOS 78.0** · **Noesis-OS 78.0** · AWQ 78.0 · ReAct 78.0 · Medusa 77.5 · Nova 77.5 · arceos 77.5 · BraiNIXOS 77.5 · RunIT_Mac 77.0 · MXFP8 77.0 · pulp-transformer (FWSA) 77.0 · SelfCheckGPT 76.5 · AXIOM 76.0 · tinygrad 76.0 · Speculative Decoding 75.5 · Test-Time Compute 75.5 · gpu-compute-nostd 75.5 · Lookahead 75.0 · AIOS 74.5 · Wanda 74.0 · rnvme 74.0 · vhost-device-gpu 73.5 · Apple AGX 73.5 · Ollama 73.5 · SparseGPT 73.0 · **MielinOS 72.5** · ToT 72.5 · Nyx 71.5 · **r3-engine 71.5** · nvme-nostd 71.0 · **project-willamette 70.5** · CoT 70.5 · Contrastive Decoding 70.0 · KVQuant 69.0 · Self-Consistency 69.0 · TrustOS 69.0 · rustrial-os 68.5 · nvme_driver 68.0 · claudio-os 67.5 · edgedl 67.5 · e1000-driver 66.0 · LLM OS 65.5 · pci-driver 64.5 · pci-rs 63.5 · gehirn-system 63.5 · FlexGen 62.0 · airox-os/core 59.5 · **dumbnet 59.5** · Fable-OS 59.0 · PunkGo 57.0.

> **Zero (rank 19, 81.0)** and **alice-aegis (rank 17, 82.5)** remain the two most on-target projects for our spike: bare-metal Rust unikernels running CPU-only LLM inference (Zero: Ring-0 Qwen3-1.7B at 194.5 tok/s, vendor-reported; alice-aegis: BitNet b1.58 ternary, ~2.80 tok/s single-thread, per-claim hardware logs, bit-exact integer semantics). Both rank below the big engines only on maturity (0–1 stars, single authors). **Squirrel AIOS (rank 26, 78.0)** is the most explicit "AI-as-kernel-principal" design (no_std x86_64, Limine UEFI+BIOS, GPU-as-cognitive-substrate); **Noesis-OS (rank 27, 78.0)** is the strongest *no_std edge-AI microkernel* (ARM64, instant boot, lock-free IPC, virtio, microVM packaging). **MicroFlow (rank 25, 78.5)** is the strongest no_std AI-inference reference for 2 kB-RAM MCUs; **r3-engine (rank 54, 71.5)** and **project-willamette (rank 56, 70.5)** are the newest pure-Rust CPU-only BitNet b1.58 engines (AVX-512 zero-copy; NEON on Pentium-M-class hardware). **BraiNIXOS (rank 33, 77.5)** is the strongest security-first no_std microkernel for serving inference. **arceos (rank 32, 77.5)** is the most credible Rust OS platform. **gpu-compute-nostd (rank 42, 75.5)** fills the no_std GPU-compute gap. **nvme-nostd (rank 55, 71.0)** and **rustrial-os (rank 62, 68.5)** are the best pure-Rust driver references. **Fable-OS (rank 74, 59.0)** is a real 314-star agentic kernel but its inference is a remote Anthropic API — lowest practical relevance to bare-metal local AI; **airox-os/core (rank 72, 59.5)** is a real Rust kernel whose AI subsystem is only a stub.

### 3.3 Justification for top 3

1. **vLLM / PagedAttention (92.0).** Highest combined maturity (production, 89.3k stars) and relevance across objectives 3/5/6. Its paged KV-cache is a memory-management blueprint directly portable to a Rust no_std allocator, and the project already proves Rust has a place in production LLM serving internals.
2. **BitNet b1.58 / bitnet.cpp (90.0).** The strongest evidence "beyond 4-bit" is achievable today: ternary models run ~100B-scale on a single CPU at human-reading speed, MIT-licensed, with LUT kernels whose techniques transfer directly to a Rust engine.
3. **FlashAttention (89.5).** The canonical kernel-fusion/IO-awareness innovation with a proven, well-documented algorithm portable to Rust CPU targets, delivering order-of-magnitude attention memory savings — the template our attention path should follow. (NIGHTRUN ties at 89.5; FlashAttention edges it on maturity/evidence breadth, NIGHTRUN wins on directness to our architecture — see 2.4.)

---

## 4. Labels

Per `objective.md`: **System** = runtime/platform layer; **Model** = model/algorithm layer.

| Project | Label | Project | Label |
|---|---|---|---|
| vLLM / PagedAttention | [System] | DeepSeek-V3 | [Model] |
| BitNet b1.58 | [System+Model] | SparseGPT | [Model] |
| FlashAttention | [System] | Speculative Decoding | [Model] |
| NIGHTRUN | [System] | Lookahead Decoding | [Model] |
| T-MAC | [System] | Wanda | [Model] |
| mistral.rs | [System] | KVQuant | [System+Model] |
| Anima OS | [System] | FlexGen | [System] |
| candle | [System] | RWKV | [Model] |
| llama.cpp | [System] | Mamba | [Model] |
| Qdrant | [System] | MemGPT / Letta | [Model] |
| virtio-drivers | [System] | RAG | [Model] |
| MatMul-free LM | [Model] | ReAct | [Model] |
| MerlionOS | [System] | SelfCheckGPT | [Model] |
| coconutOS | [System] | Test-Time Compute | [Model] |
| Zero | [System] | Tree of Thoughts | [Model] |
| AXIOM | [System] | Chain-of-Thought | [Model] |
| MicroFlow | [System] | Contrastive Decoding | [Model] |
| Fable-OS | [System] | Self-Consistency | [Model] |
| LLM OS | [System] | — | — |
| lele | [System] | arceos | [System] |
| pulp-transformer (FWSA) | [System] | nvme-nostd | [System] |
| burn | [System] | rustrial-os | [System] |
| airox-os/core | [System] | — | — |
| alice-aegis | [System] | BraiNIXOS | [System] |
| gehirn-system | [System] | claudio-os | [System] |
| Nyx | [System] | gpu-compute-nostd | [System] |
| TrustOS | [System] | edgedl | [System] |
| Squirrel AIOS | [System] | Noesis-OS | [System] |
| MielinOS | [System] | r3-engine | [System] |
| project-willamette | [System] | dumbnet | [System] |
| All Obj-1 driver projects (Nova, rnvme, irqlevel/nos, AGX, vhost-device-gpu, e1000, nvme_driver, pci-driver, pci-rs) | [System] | — | — |

---

## 5. Citations

[1] Kwon, W., Li, Z., Zhuang, S., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." arXiv:2309.06180 (2023). https://arxiv.org/abs/2309.06180
[2] Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Ré, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." arXiv:2205.14135 (2022). https://arxiv.org/abs/2205.14135
[3] Wang, Ma, Chen, Xiao, et al. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764 (2024). https://arxiv.org/abs/2402.17764
[4] W.F. Chiang et al. "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs." arXiv:2502.11880 (2025). https://arxiv.org/abs/2502.11880
[5] Sun, M., Liu, Z., Bair, A., Kolter, J.Z. "A Simple and Effective Pruning Approach for Large Language Models." arXiv:2306.11695 (2023). https://arxiv.org/abs/2306.11695
[6] Cai, T., Li, Y., Geng, Z., et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." arXiv:2401.10774 (2024). https://arxiv.org/abs/2401.10774
[7] Zhu, R.-J., Zhang, Y., Sifferman, E., et al. "Scalable MatMul-free Language Modeling." arXiv:2406.02528 (2024). https://arxiv.org/abs/2406.02528
[8] Jung, V.J.B., Burrello, A., Scherer, M., Conti, F., Benini, L. "Optimizing the Deployment of Tiny Transformers on Low-Power MCUs." arXiv:2404.02945 (2024). https://arxiv.org/abs/2404.02945
[9] Darvish Rouhani, B., et al. "Microscaling Data Formats for Deep Learning." arXiv:2310.10537 (2023). https://arxiv.org/abs/2310.10537
[10] Kwon, W. et al. (PagedAttention) — see [1].
[11] Fu, Y., Bailis, P., Stoica, I., Zhang, H. "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding." arXiv:2402.02057 (2024). https://arxiv.org/abs/2402.02057
[12] Lin, J., Tang, J., Tang, H., et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv:2306.00978 (2023). https://arxiv.org/abs/2306.00978
[13] Hooper, C., Kim, S., Mohammadzadeh, H., et al. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." arXiv:2401.18079 (2024). https://arxiv.org/abs/2401.18079
[14] Wei, J., Cao, S., Cao, T., et al. "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge." arXiv:2407.00088 (2024). https://arxiv.org/abs/2407.00088
[15] DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437 (2024). https://arxiv.org/abs/2412.19437
[16] Sheng, Y., Zheng, L., Yuan, B., et al. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." arXiv:2303.06865 (2023). https://arxiv.org/abs/2303.06865
[17] Leviathan, Y., Kalman, M., Matias, Y. "Fast Inference from Transformers via Speculative Decoding." arXiv:2211.17192 (2022). https://arxiv.org/abs/2211.17192
[18] Lewis, P., Perez, E., Piktus, A., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." arXiv:2005.11401 (2020). https://arxiv.org/abs/2005.11401
[19] Packer, C., et al. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560 (2023). https://arxiv.org/abs/2310.08560
[20] Peng, B., et al. "RWKV: Reinventing RNNs for the Transformer Era." arXiv:2305.13048 (2023). https://arxiv.org/abs/2305.13048
[21] Gu, A., Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752 (2023). https://arxiv.org/abs/2312.00752
[22] Dao, T., Gu, A. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." arXiv:2405.21060 (2024). https://arxiv.org/abs/2405.21060
[23] Wei, J., Wang, X., Schuurmans, D., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." arXiv:2201.11903 (2022). https://arxiv.org/abs/2201.11903
[24] Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." arXiv:2210.03629 (2022). https://arxiv.org/abs/2210.03629
[25] Yao, S., et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." arXiv:2305.10601 (2023). https://arxiv.org/abs/2305.10601
[26] Snell, C., et al. "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters." arXiv:2408.03314 (2024). https://arxiv.org/abs/2408.03314
[27] Wang, X., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." arXiv:2203.11171 (2022). https://arxiv.org/abs/2203.11171
[28] Li, X., et al. "Contrastive Decoding: Open-ended Text Generation as Optimization." arXiv:2210.15097 (2022). https://arxiv.org/abs/2210.15097
[29] Manakul, P., Liusie, A., Gales, M. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." arXiv:2303.08896 (2023). https://arxiv.org/abs/2303.08896
[30] ProbeLogits / Anima OS papers. arXiv:2604.11943 and arXiv:2607.04668. https://arxiv.org/abs/2604.11943 · https://arxiv.org/abs/2607.04668
[31] PunkGo. "Sovereignty Kernel for AI Agent Execution." arXiv:2602.20214 (2026). https://arxiv.org/abs/2602.20214
[32] AIOS. "AIOS: LLM Agent Operating System." COLM 2025. https://openreview.net/forum?id=L4HHkCDz2x
[33] Frantar, E., Alistarh, D. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." arXiv:2301.00774 (2023). https://arxiv.org/abs/2301.00774
[34] Carnelos, M., et al. "MicroFlow: A Rust-Based Inference Engine for Tiny Transformers on Memory-Constrained MCUs." arXiv:2409.19432 (2024). https://arxiv.org/abs/2409.19432 · repo https://github.com/matteocarnelos/microflow-rs (Apache-2.0/MIT; paper in Internet of Things, 2025)
[35] Silicate Technology. "Zero: bare-metal AI/OS in Rust." Repo https://github.com/silicatetech/zero · site https://zerokernel.com (AGPL-3.0; vendor-reported Qwen3-1.7B @ 194.5 tok/s CPU-only, Ring-0; unverified independently)
[36] "Fable-OS: an agentic OS whose kernel is controlled directly by an LLM (remote Anthropic API)." Repo https://github.com/robiot/fable-os (no license detected; x86_64, QEMU)
[37] Arceos Team. "ArceOS: an experimental modular OS / unikernel written in Rust." https://github.com/arceos-org/arceos (Apache-2.0, 779 stars, active since 2023; driver module ecosystem)
[38] "nvme-nostd: a `#![no_std]` NVMe driver in Rust over PCI BAR0 MMIO (NVMe 1.4 init, admin/I/O commands, PRP lists, block-device abstraction)." https://github.com/suhteevah/nvme-nostd (Apache-2.0; verified 2026-08-17)
[39] "rustrial-os: hobby x86-64 bare-metal Rust OS with RTL8139 NIC DMA driver, TCP/IP stack (Ethernet/ARP/IPv4/ICMP/UDP/TCP), PCI enumeration." https://github.com/lazzerex/rustrial-os (MIT, 5 stars, 460 commits)
[40] "airox-os/core: AI-native Rust OS; blog_os-derived no_std kernel whose AI subsystem (`kernel/src/ai.rs`) is a functional stub." https://github.com/airox-os/core (MIT, 2 stars, dormant since 2025-05; AI claims README-only)
[41] "Rig: build modular and scalable LLM applications in Rust." https://github.com/0xPlaygrounds/rig (MIT, 8.3k stars, active). "Kernex: Rust runtime for AI agents with OS-level sandboxing (Seatbelt/Landlock)." https://github.com/kernex-dev/kernex (Apache-2.0/MIT) — both agent frameworks, not OSes; context for the "rust ai os" query surface.
[42] "A.L.I.C.E. / alice-aegis: from-scratch no_std Rust BitNet b1.58 ternary transformer as a UEFI unikernel + Linux twin; bit-exact integer semantics, measured results with hardware logs (~2.80 tok/s decode, i5-10210U)." https://github.com/Aefinity-AI/alice-aegis (Apache-2.0; weights: microsoft/bitnet-b1.58-2B-4T, MIT)
[43] "BraiNIXOS: capability-based, security-first no_std microkernel to serve LLM inference to remote clients; aarch64/x86_64, W^X + KPTI, measured boot, zero external deps." https://github.com/jbrahy/BraiNIXOS (MIT OR Apache-2.0)
[44] "gehirn-system: bare-metal Rust OS purpose-built for local LLM inference; pre-alpha Phase 01, module map incl. `puppet` (inference), `unit00` (GPU/VFIO), `rei` (drivers)." https://github.com/dev-natebrito/gehirn-system (AGPL-3.0)
[45] "claudio-os: bare-metal Rust UEFI OS (52 crates, 294 KLOC, 0 lines C) running multi-agent Claude coding sessions with tool use, TLS 1.3, PE-loader/Win32 compat." https://github.com/suhteevah/claudio-os (AGPL-3.0-or-later)
[46] "Nyx: Rust-first bare-metal OS unifying classical, quantum, and AI-driven computing; QCLang, AI kernel entity, SMP, ext4 on NVMe, Intel GPU acceleration." https://github.com/Asmodeus14/Nyx (Apache-2.0)
[47] "gpu-compute-nostd: bare-metal #![no_std] NVIDIA GPU compute driver with tensor ops for LLM inference (PCI enumeration, Falcon MC, GPFIFO dispatch, Kepler→Ada, MatMul)." https://github.com/suhteevah/gpu-compute-nostd (MIT/Apache-2.0)
[48] "TrustOS: 264 KLOC bare-metal Rust OS, zero C; x86_64/ARM64/RISC-V; drivers validated on real hardware (e.g. AMD RX 580X SDMA); TLS 1.3, PXE self-replication." https://github.com/nathan237/TrustOS (Apache-2.0, 48 stars)
[49] "edgedl: no_std INT8 neural-network inference engine for ESP32-S3; compile-time model embedding, SIMD (TIE728), arena allocator, zero heap." https://github.com/g1ibby/edgedl (MIT)
[50] "Squirrel AIOS: bare-metal no_std x86_64 OS designed from first principles for AI sovereignty; AI-as-kernel-principal, GPU exposed as cognitive substrate, Limine boot (UEFI+BIOS), no Linux/POSIX." https://github.com/tusharkhatriofficial/squirrel (2★; README badge MIT, GitHub API shows no license file — review before reuse)
[51] "Noesis-OS: bare-metal no_std ARM64 microkernel for on-device AI inference on edge boards/microVMs; instant boot (QEMU/HVF), lock-free SPSC/SPMC IPC, EDF+RR hybrid scheduler, virtio blk/net, microVM packaging." https://github.com/beltromatti/Noesis-OS (MIT)
[52] "MielinOS: Rust microkernel-based OS for distributed AI agents; capability IPC, WASM sandbox, Kademlia-DHT + QUIC neural mesh, live agent migration, tensor acceleration (SVE2/AVX-512/NPU), Arm/RISC-V/x86/Cortex-M." https://github.com/cool-japan/mielin (Apache-2.0, v0.1.0-rc.1)
[53] "r3-engine: 100%-safe-Rust 1.58-bit ternary (BitNet) inference engine; zero-copy paging, AVX-512, zero heap allocations in the execution loop, Wasm/SIMD128 cross-compile; vendor-reported 80–117 tok/s single-core (Ryzen 9950X3D)." https://github.com/dhilipsiva/r3-engine (no license declared)
[54] "project-willamette: two-piece Rust CPU-only LLM runtime (offline bake + execute) for humble hardware (older x86, ARM, retro); zero-copy mmap, ARM/x86_64/i686, validated on Pentium-M-class (verified floor ~100M params ≥5 tok/s); BitNet b1.58 GGUF start." https://github.com/nangman-infra/project-willamette (Apache-2.0)
[55] "dumbnet: no_std, no-allocation neural-network library (stack-only); compile-time dimension checking; aimed at embedded devices without an OS." https://github.com/djugei/dumbnet (23★, 2019, no license declared)
[56] "IronClaw: 'Agent OS' focused on privacy/security/extensibility (12.6k★) — userland agent assistant, not bare-metal; Astrid: portable capability-secure general OS (10.3k★), not AI-specific. Both classified as architectural leads, not matrix entries." https://github.com/nearai/ironclaw · https://github.com/astrid-runtime/astrid
[57] Son, D. "Governed MCP: Kernel-Level Tool Governance for AI Agents via Logit-Based Safety Primitives." arXiv:2604.16870 (2026). https://arxiv.org/abs/2604.16870
[58] Mason, T. "The Missing Memory Hierarchy: Demand Paging for LLM Context Windows (Pichay)." arXiv:2603.09023 (2026). https://arxiv.org/abs/2603.09023
[59] Li, Z., et al. "MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models." arXiv:2505.22101 (2025). https://arxiv.org/abs/2505.22101
[60] Yin, W., et al. "LLM as a System Service on Mobile Devices (LLMS)." arXiv:2403.11805 (2024). https://arxiv.org/abs/2403.11805
[61] Shang, J., et al. "AI-native Memory: A Pathway from LLMs Towards AGI." arXiv:2406.18312 (2024). https://arxiv.org/abs/2406.18312
[62] Xu, J., et al. "eLLM: Elastic Memory Management Framework for Efficient LLM Serving." arXiv:2506.15155 (2025). https://arxiv.org/abs/2506.15155
[63] Jeong, B., et al. "DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference." arXiv:2604.26557 (2026). https://arxiv.org/abs/2604.26557
[64] Qiu, S., et al. "Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving." arXiv:2605.03375 (2026). https://arxiv.org/abs/2605.03375
[65] Pan, X., et al. "InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference." arXiv:2409.04992 (2024). https://arxiv.org/abs/2409.04992
[66] Wang, T., et al. "Swarm: Co-Activation Aware KVCache Offloading Across Multiple SSDs." arXiv:2603.17803 (2026). https://arxiv.org/abs/2603.17803
[67] Deng, L., et al. "KVNAND: Efficient On-Device Large Language Model Inference Using DRAM-Free In-Flash Computing." arXiv:2512.03608 (2025). https://arxiv.org/abs/2512.03608
[68] Kim, B., et al. "FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices." arXiv:2601.17063 (2026). https://arxiv.org/abs/2601.17063
[69] Ma, Y., et al. "TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models." arXiv:2607.07388 (2026). https://arxiv.org/abs/2607.07388
[70] Kyung, K., et al. "SSD Offloading for LLM Mixture-of-Experts Weights Considered Harmful in Energy Efficiency." arXiv:2508.06978 (2025). https://arxiv.org/abs/2508.06978
[71] Peng, J., et al. "Harnessing Your DRAM and SSD for Sustainable and Accessible LLM Inference with Mixed-Precision and Multi-level Caching." arXiv:2410.14740 (2024). https://arxiv.org/abs/2410.14740
[72] Kilictas, B., et al. "Bare-Metal Tensor Virtualization: Overcoming the Memory Wall in Edge-AI Inference on ARM64." arXiv:2601.03324 (2026). https://arxiv.org/abs/2601.03324
[73] Jang, H., et al. "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories." arXiv:2606.12556 (2026). https://arxiv.org/abs/2606.12556
[74] Ganjihal, S.R., et al. "Predictive Multi-Tier Memory Management for KV Cache in Large-Scale GPU Inference." arXiv:2604.26968 (2026). https://arxiv.org/abs/2604.26968
[75] Hwang, S., et al. "Hardware-based Heterogeneous Memory Management for Large Language Model Inference." arXiv:2504.14893 (2025). https://arxiv.org/abs/2504.14893
[76] Zuo, F., et al. "FairyFuse: Multiplication-Free LLM Inference on CPUs via Fused Ternary Kernels." arXiv:2604.20913 (2026). https://arxiv.org/abs/2604.20913
[77] Oh, H., et al. "T-SAR: A Full-Stack Co-design for CPU-Only Ternary LLM Inference via In-Place SIMD ALU Reorganization." arXiv:2511.13676 (2025). https://arxiv.org/abs/2511.13676
[78] Zhang, H., et al. "Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference." arXiv:2505.06461 (2025). https://arxiv.org/abs/2505.06461

Repo URLs, licenses, and star counts were verified on 2026-08-17 against GitHub/crates.io; paper claims against arXiv abstracts. Primary sources preferred throughout.

---

## 6. Further Research (prioritized, mapped to objectives)

1. **Bare-metal driver build-out references — Obj 1.** *Terms:* "nvme nostd", "gpu compute nostd", "rtl8139 rust driver", "arceos driver modules". *Sources:* **nvme-nostd** (no_std NVMe 1.4 over PCI BAR0 MMIO), **gpu-compute-nostd** (no_std NVIDIA GPU compute + tensor ops), **rustrial-os** (RTL8139 NIC + TCP/IP), **TrustOS** (validated AMD GPU SDMA on real hw), **arceos** (modular unikernel ecosystem), plus rcore `virtio-drivers`/`pci-rs`. *Why:* these give the cleanest pure-Rust no_std templates for our own NVMe/NIC/GPU/PCI drivers; validate their patterns against the rcore stack before writing ours.
2. **Rust port of LUT-based low-bit GEMM (T-MAC / bitnet.cpp kernels) — Obj 5.** *Terms:* "T-MAC rust", "bitnet rust", "lookup table gemm rust". *Sources:* microsoft/T-MAC kernels, crate searches, and the two new pure-Rust CPU BitNet engines **r3-engine** (AVX-512 zero-copy, 100% safe, Wasm) and **project-willamette** (NEON, humble-hardware benchmarks) — validate their integer/GEMV patterns against T-MAC/bitnet.cpp. *Why:* highest-value kernel to validate on ARM/NEON (`tbl`/`pshuf`).
3. **Bare-metal Rust AI OS architecture comparison — Obj 2/3.** *Compare:* **Zero vs alice-aegis** (the only two actually executing CPU-only inference on bare metal: Ring-0 Qwen3 vs BitNet b1.58 UEFI unikernel — note alice-aegis' bit-exact integer semantics and per-claim hardware logs as the measurement standard), then **Squirrel AIOS** (AI-as-kernel-principal, GPU-as-cognitive-substrate), **Noesis-OS** (ARM64 no_std edge-AI microkernel), **BraiNIXOS** (secure-serving microkernel), **NIGHTRUN**, **MerlionOS**, **coconutOS**, **AXIOM**, **MielinOS** (distributed-agent microkernel), **gehirn-system**, **claudio-os**, **Nyx**, **LLM OS**, **airox-os/core**. *Terms:* each repo README + Anima OS arXiv 2607.04668 related-work. *Why:* no mature precedent exists; this cluster defines the current frontier of "AI as resident OS primitive." AXIOM's tensor-native allocator + LayerLock scheduler is the most explicit design for memory-constrained 7B inference; Zero, alice-aegis, Squirrel, and Noesis are the prime build-out candidates (verify claims independently).
4. **Paged KV-cache as a Rust no_std allocator pattern — Obj 3/6.** *Terms:* "paged attention rust", "block table allocator no_std". *Sources:* vllm `rust/` dir, mistral.rs `mistralrs-paged-attn`. *Why:* bridges system memory management with LLM serving.
5. **Sub-8-bit numerics in Rust (MXFP8/MXFP4, OCP MX) — Obj 5.** *Terms:* "microscaling formats rust", "OCP MX spec". *Sources:* microsoft/microxcaling, OCP MX spec v1.0. *Why:* self-contained spec-driven numerics; future-proofs narrow-format support.
6. **Constant-memory architectures for bare metal (RWKV, Mamba) — Obj 6.** *Terms:* "RWKV candle", "mamba rust", "SSM inference rust". *Why:* O(1)-per-token state gives predictable RAM budgets; ideal for no_std targets.
7. **Speculative decoding for a minimal runtime — Obj 5/6.** *Terms:* "medusa rust", "MTP speculative decoding". *Sources:* arXiv 2211.17192, FasterDecoding/Medusa, DeepSeek-V3 MTP. *Why:* latency win with no new hardware.
8. **Rust-native frameworks porting non-Rust techniques — Obj 1/5 cross-check.** *Terms:* "candle paged attention", "burn llm", "mistral.rs flash attention". *Why:* determines which techniques are already in Rust, avoiding re-invention.
9. **Distillation and model shrinking — Obj 5.** *Terms:* "knowledge distillation llm 2026", "model distillation". *Why:* objective explicitly lists it; only partially covered here.
10. **Offloading for memory-constrained hosts — Obj 5.** *Terms:* "llm offloading cpu ram disk". *Sources:* FlexLLMGen (archived — prefer active successors), llama.cpp offload paths. *Why:* enables models larger than physical RAM.
11. **Storage & recall for embedded RAG — Obj 6.** *Terms:* "rust vector database embedded", "hnsw no_std". *Sources:* Qdrant internals, MemGPT paging metaphor, skeg (lead, see §8). *Why:* retrieval index + embedding store is a natural Rust-embedded layer beside a small LM.
12. **AI-native memory/storage architecture for the no_std kernel — Obj 1/3/5/6.** *Sources:* the §7 paper lineage — DUAL-BLADE [63], Tutti [64], Pichay [58], LLMS [60], InstInfer [65], KVNAND [67], TF-Engram [69], FairyFuse [76], T-SAR [77]. *Why:* these papers supply the concrete design primitives (NVMe-direct KV DMA, context demand-paging, logit reads as syscalls, elastic inference gangs) that only a no_std kernel can implement without fighting a page-cache layer — the single highest-leverage gap between this report and a buildable "AI-native" kernel.

---

## 7. AI-Native OS Research Lineage (papers, 2026-08-17)

A focused arXiv sweep around the design question *"we control the hardware directly — how do we use NVMe and RAM in an AI-native way?"* The lineage is anchored by the Anima OS papers already in the matrix [30]; the literature splits into five themes, with the design synthesis for our no_std kernel in §7.6.

### 7.1 The Anima OS lineage — inference as an OS primitive

- **ProbeLogits** [30]: a kernel-level operation that reads a specific next-token logit as a zero-learned-parameter governance primitive; a safety gate becomes a single logit read (97–99% HarmBench block rate, 2.4–3.4× faster than Llama Guard 3), enforced below the WASM sandbox boundary because the model runs in-kernel.
- **Governed MCP** [57]: extends ProbeLogits to MCP tool calls — "the agent's syscalls." A 6-layer kernel gateway (schema → trust tier → rate limit → adversarial pre-filter → ProbeLogits semantic gate → constitutional policy) with a Blake3 audit chain, interposed on every tool call.
- **Elastic Gang** [30]: the hard-barriered inference gang as a first-class schedulable entity; ACK-latched epoch protocol changes core membership per token without barrier deadlock or logit corruption. Bit-exact under per-token membership change on AMD Zen 5; 1.75×/1.52×/1.28× general throughput at 25/50/75% inference duty vs a static 8-core split; core return costs 0.22 µs.

### 7.2 RAM & context as a paged memory system

- **Pichay — "The Missing Memory Hierarchy"** [58]: *the context window is not memory, it is L1 cache.* Measures 21.8% structural waste across 857 production sessions (4.45M effective tokens), then demand-pages context: evict stale content, serve page faults when the model re-requests it. For us this is implementable as *real* page faults inside the kernel, not a proxy.
- **MemOS** [59]: an operating system for memory-augmented generation — unified, lifecycle-managed memory tiers beyond parametric and activation memory.
- **LLM as a System Service** [60]: the LLM is a *stateful resident system service* whose KV cache persists across invocations; fine-grained chunk-wise KV compression + swap under tight memory budgets — the "inference is a service, not an app" model.
- **AI-native Memory** [61]: the theoretical framing — memory as a first-class AI-native subsystem rather than context stuffing.
- **eLLM** [62]: elastic memory management for LLM serving; adaptive KV/weight allocation across tiers.

### 7.3 NVMe/SSD as a first-class inference substrate

- **DUAL-BLADE** [63]: dual-path KV residency — assigns KV tensors to either the kernel page-cache path or an **NVMe-direct DMA path**; the page cache causes cache-thrashing and unpredictable latency. Our no_std kernel removes the page-cache layer entirely, so NVMe-direct is the default.
- **Tutti** [64]: makes SSD-backed KV practical by fixing fragmented-GPU-layout-induced tiny random I/Os (thousands of small requests per restore, CPU-bound even with GDS) — the fix is layout contiguity, which we control directly through our NVMe DMA descriptors.
- **InstInfer** [65]: in-storage attention offloading (attention compute inside SSD firmware) for cost-effective long-context inference; the software-pipelined offload pattern transfers even though NVMe commands can't invoke firmware compute.
- **Swarm** [66]: co-activation-aware KV-cache placement across multiple SSDs.
- **KVNAND** [67]: DRAM-free on-device LLM inference via in-flash computing — the storage-as-memory extreme.
- **FlashMoE** [68]: ML-based SSD cache replacement for MoE expert offload on edge devices.
- **TF-Engram** [69]: train-free SSD-backed semantic memory spanning a GPU–DRAM–SSD hierarchy.
- **SSD offloading considered harmful** [70]: SSD read-energy/bit is substantially higher than DRAM — for MoE weight offload during decode it can be a net loss; offload pays only past DRAM exhaustion. Design boundary, not a default.
- **Harnessing DRAM + SSD** [71]: mixed-precision weights + multi-level caching to keep low-HBM/old GPUs and CPU boxes viable.

### 7.4 Tiered memory & bare-metal tensors

- **Bare-Metal Tensor Virtualization** [72]: "virtual tensor core" in software on ARM64 (Apple Silicon) — direct mmap of weights + hand-tuned NEON SIMD kernels, "software-defined DMA," bypassing library containers. The closest published match to our architecture.
- **ITME** [73]: inference tiered memory expansion over disaggregated CXL-hybrid memories.
- **Predictive multi-tier KV management** [74]: unified KV sizing across HBM / CPU DRAM / CXL / NVMe (GPUDirect), addressing 57× memory over-provisioning in MLA.
- **Heterogeneous memory management** [75]: hardware-driven heterogeneous memory allocation for LLM inference.

### 7.5 CPU-only ternary inference (Zero-adjacent)

- **FairyFuse** [76]: multiplication-free ternary CPU inference — decode is memory-bandwidth-bound, so {-1,0,+1} weights replace FMAs with add/sub/no-op via fused kernels.
- **T-SAR** [77]: full-stack co-design for CPU-only ternary LLM inference via in-place SIMD register-file reorganization (dynamic in-register LUT generation) — removes the LUT memory bottleneck.
- **Challenging GPU Dominance** [78]: measured conditions under which CPUs outperform GPUs for on-device LLM inference.

### 7.6 Design synthesis for our no_std kernel

The consensus across all five themes: an AI-native OS manages **three address spaces with one mechanism** — model weights, the KV cache, and the context window — and the highest-leverage primitive is a paged block allocator with NVMe backing. Our no_std kernel is uniquely positioned because it removes the layers every paper above fights against:

1. **NVMe-direct KV pages** [63][64] — DMA straight into KV page frames with no page-cache double copy; batch I/Os by layout contiguity (Tutti) and place blocks by attention co-activation (Swarm).
2. **Context window as virtual memory** [58] — a real page-fault handler evicts/restores context from NVMe; LRU/ARC policies live in the kernel memory manager, not in a userspace proxy.
3. **Logit reads as syscalls** [30][57] — because the model runs in-kernel, safety and steering gates read a single logit with zero learned parameters, enforced below any sandbox.
4. **Gang-scheduled decode** [30] — the ACK-latched elastic gang cedes cores to general work between tokens (0.22 µs return) and auto-sizes once decode saturates at gang width.
5. **Ternary kernels to lower the bandwidth floor** [76][77] — keep decode memory-bound at minimum bytes/token; per [70], NVMe tiering only pays once DRAM is exhausted, so the weight format decides how often that happens.

---

## 8. Leads not yet verified / excluded

- **"NeuMIPS":** no verifiable project/paper with this name in the LLM-deployment context. Closest verified match for the intended idea (MCU/memory-efficient transformer deployment) is **pulp-transformer / FWSA** [8].
- **Folkering OS** (merknu/folkering-os): heavily indexed as an "AI-native bare-metal Rust OS" (Qwen3 + VirtIO GPU); a detailed Zing forum write-up (2026-04) corroborates the description, but the GitHub repo still returns 404 — existence of public code unconfirmed.
- **KnoxOS** (knoxos.com / github.com/knoxos/docs): marketed as an "AI-native OS in Rust" (cognition loop, Linux ABI emulation). Top hit on the DDG "rust ai os" SERP; now confirmed to have a **docs-only** repo (AGPL-3.0, 1 star, README + LICENSE only, pushed once 2026-04) — no code exists publicly; still excluded as vaporware.
- **OpenFang** (openfang.sh / openfang.cc): "Agent Operating System", "Production-Grade Agent OS | Rust-Powered, 24/7 Autonomous"; two marketing domains, no repo found — vaporware lead.
- **Kernex** (kernex.dev, kernex-dev/kernex): real Rust *agent runtime* (v0.10.0, Apache-2.0/MIT, OS-level sandboxing via Seatbelt/Landlock, SQLite memory) — but runs on macOS/Linux, not a custom kernel; architectural interest only for the agent layer.
- **Rig** (rig.rs, 0xPlaygrounds/rig): 8.3k-star Rust LLM-agent framework — not an OS; architectural interest only.
- **rust4ai.com** and the **Rust Foundation position statement** ("Rust and AI"): non-project context from the DDG SERP; no action.
- **AetherOS-Showcase** (danielforface): bare-metal AI OS whose core engine is private ("IP / stealth phase"); only architectural docs are public — partial.
- **Genasys** (stephendulaney.substack.com, 2026-01): "bare-metal OS with multi-agent orchestration"; described only in a Substack post, no repo found.
- **Helix OS** (helix-wiki.com): modular Rust kernel claiming an "NEXUS: 812K lines of intelligence" subsystem; marketing site, no repo — looks AI-generated; excluded.
- **xInfer** (guoqingbao.github.io/xinfer): "pure-Rust blazing-fast LLM inference"; repo not verified this session.
- **unillm** (cognisoc): modular Rust LLM runtime claiming 47 model architectures; only 2 stars, claims rest on the vendor's own blog — unverified.
- **JC-OS** (discussed on users.rust-lang.org): described as a kernel project embedding LLM primitives; no public repo located on GitHub search — unverified.
- **Ariel-ML** (arXiv:2512.09800): "AI-arithmetic-in-the-OS" instruction-level LLM arithmetic proposals; paper only, no code — unverified, flagged for citation completeness.
- **ntoskrnl-rs** (Claude Fable 5): a Windows kernel in Rust written BY an LLM — notable but not an AI OS; excluded.
- **PunkGo repo:** arXiv:2602.20214 says "open-source" but the URL is not resolvable; included on paper evidence only.
- **nvme-oxide:** indexed as a bare-metal no_std NVMe driver; crates.io and lib.rs now 404 (yanked).
- **paiml/pepita:** "tiny Rust Linux kernel for Sovereign AI" — README internally inconsistent (claims kernel yet uses Linux io_uring/ublk); excluded.
- **thesnmc/ZYO:** RL-optimized Linux CPU scheduler (sched_ext/eBPF) with a Rust LLM orchestrator — runs on Linux 6.12+, not a custom kernel; architectural interest only.
- **JARVIS OS:** Arch-Linux-based LLM-security research distro, not bare-metal; architectural interest only.
- **Theseus OS / Redox OS:** both actively checked — **no AI/ML features found**; don't conflate "Theseus AI" (blockchain) with Theseus OS.
- **rust-raspberrypi-OS-tutorials:** verified — no AI/ML demos exist.
- **MOROS / SafaOS / CharlotteOS / Maestro / DragonOS / octox / motor-os:** general-purpose Rust hobby/real OSes surfaced by the driver-learning sweep (OSHub roundup + comments); driver-relevant (PCI/net/storage) but **no AI/ML features** — excluded from the matrix; see §6 item 1 for the strongest driver references instead.
- **~100-result sweep (2026-08-17) — agent-framework noise bucket (~50 repos, mostly Rust, 0–159★):** IronClaw [56] (12.6k★, nearai, userland "secure personal AI assistant"), Astrid [56] (10.3k★, capability-secure general OS, not AI), bbarit-agent-oss, ginkida/rustyhand, ferosai/feros, GridWork-dev/gridwork, ghostapp-ai/ghost, hongmaple0820/maple-os, lucidos-dev/lucidos, rexleimo/LoopForge, Jokerautowrite/chuang-agent, k8nstantin/superx, broomva/* (arcan, lago, praxis, anima, autonom, spaces, haima, nous, vigil, arcan-os), dward1502/Arda*, WAHIB-EL-KHADIRI/AgentOS, o-kadam/bareclaw, Vivien83/captain, AA-Box/little-monkey, Tamang4607/rustyhand, etc. All are agent runtimes/orchestrators on top of a host OS — **architectural interest only, not matrix candidates.**
- **~100-result sweep — "AI-native Linux distro" bucket:** ZethraOS, NuraOS (YASSERRMD), AuraOS (venkatyarl), MohaMehrzad/aiOS, aulinx, Bantu-Os, AntonioBurgos91/aurum-os, nikhilkumarpanigrahi/nikhil-os, ajul8866/neuraos, TanujBairwa/NeuroOS, snowphn/HakOS, kluth/jarvis-os, RobertKodes/LivingOS, xolerc/xoleric, WolfurX/hearth-os, jboero/asterkube (Asterinas-based k8s). All wrap Linux/systemd/Arch with agents — **not bare-metal; excluded.**
- **SomaOS** (avsribhas-svg, 10★, MIT): "AI-native OS" with a real dual-interface agent desktop and a `soma-substrate` crate enforcing orientation-aligned AI-safety properties — but it is a **Buildroot minimal-Linux image**, not a custom kernel; architectural lead for the safety-substrate pattern.
- **Nexus agentic-core-os** (Nexus-Agentic, 0★, MIT) and **LahraCore-OS** (anabkl, 1★, MIT): bare-metal agent-microkernel visions ("Architectural Alpha" / "Foundational Boilerplate") with no implementation — vaporware leads.
- **5AM-OS** (Hari0701, 1★, MIT): x86_64 Rust **teaching** kernel whose shell explains live CPU state — self-narrating but **no AI/ML**; educational lead.
- **OxideLM** (MickyBalladelli, 0★, MIT/Apache-2.0): from-scratch pure-Rust Transformer stack (BPE, reverse-mode autograd, wgpu kernels, AdamW) — "bare-metal" here means *no PyTorch*, not no_std; engine/reference lead.
- **lmzuccarelli/rust-ai-unikernel-\*** (8 repos, 0★): simple unikernel *services* wrapping cloud APIs (grok/anthropic/openai/gemini/auth) + a Rust LLM-Council port — shows unikernel-As-a-cloud-client pattern, no local inference.
- **1.58-bit / ternary engine cluster (non-Rust or unverified, 2026):** BitMamba-2 + bitmamba.cpp (Zhayr1), Atomic-1Bit (guirguispierre), ternary-zero (skyhighbg22-jpg), vivekdixit3911 ESP32-S3 CNN, vitorengers/esp32-trm-bitnet, TernixEngine, edge-quantized-bitnet, tzervas/ternary-inference-rs (claimed Rust, repo is Python) — C++/Python/C engines supporting the BitNet thesis; Rust relevance limited; §6 item 2 covers the Rust angle.
- **skeg:** "RAM-frugal vector engine for Apple Silicon" for local RAG (r/LocalLLM, 2026-05); repo not verified this session — potential Obj-6 storage layer lead.
- **Bare-metal Rust BitNet engine (Reddit r/LocalLLM, 2026-05):** author-reported 66.8 tok/s BitNet 1.58b 4B on an RTX 3050 (4 GB VRAM), built from scratch to beat llama.cpp's abstraction overhead; no repo identified — methodology lead only (matches Zero's thesis: format-specific hand-optimized engines beat general frameworks).
- **GPTQ / RTN / EAGLE / MatFormer:** named in scope but not individually verified this session — flagged for citation completeness.

---

## 9. Answers to inline questions

- **Most mature Rust PCIe/NVMe driver set for custom kernels?** `virtio-drivers` + `pci-rs` + `nvme_driver` (rcore-os) form the most complete no_std driver set, with `irqlevel/nos` demonstrating a full Rust NVMe stack (PCI BAR mapping, queue pairs, MSI-X, WaitGroup sync I/O) on a hobby kernel. Rust-for-Linux's rnvme/Nova prove the abstraction pattern at production quality but are Linux-tree-bound. From the driver sweep: **nvme-nostd** (no_std NVMe 1.4 over PCI BAR0 MMIO), **gpu-compute-nostd** (no_std NVIDIA GPU compute + tensor ops — the no_std GPU gap), **rustrial-os** (RTL8139 NIC + TCP/IP), **TrustOS** (validated GPU drivers on real hardware), and **arceos** (779-star modular unikernel — best platform archetype).
- **AI inference engine inside a Rust hobby OS?** Yes, fifteen verifiable no_std/bare-metal examples, all 2025–26 prototypes: **NIGHTRUN** (UEFI-resident), **Anima OS** (papers only), **MerlionOS**, **coconutOS**, **Zero** (Ring-0 CPU-only inference, vendor-reported 194.5 tok/s — unverified), **alice-aegis** (BitNet b1.58 ternary, UEFI unikernel, measured ~2.80 tok/s + hardware logs), **BraiNIXOS** (secure-serving microkernel), **Squirrel AIOS** (AI-as-kernel-principal, x86_64 no_std), **Noesis-OS** (ARM64 no_std edge-AI microkernel), **MielinOS** (distributed-agent microkernel), **gehirn-system**, **claudio-os** (agent OS, cloud API), **Nyx**, **AXIOM** (tensor-native allocator + layer-boundary scheduling), **LLM OS** (inference still a placeholder). **Fable-OS** is a sixteenth, cloud-API inference. Only **Zero** and **alice-aegis** actually execute CPU-only inference on bare metal today. No mature precedent exists.
- **Which non-Rust techniques (GQA variants, paged attention) are ported to Rust so far?** Verified: **PagedAttention** is implemented in Rust inside vLLM's `rust/` dir and in mistral.rs (`mistralrs-paged-attn`); FlashAttention V2/V3 and continuous batching in mistral.rs. GQA variants and other ports in candle/burn were NOT verified this session (see §6 item 7).

---

*End of report. Compiled 2026-08-17 from four parallel research tracks; all facts verified against primary sources unless flagged.*