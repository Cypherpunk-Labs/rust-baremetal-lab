# Research Spike 01 — Rust Native AI on Bare Metal

## Context

This spike informs a decision on which projects to study, evaluate, or potentially
adopt as reference implementations / dependencies for building AI systems in pure
Rust on bare metal (no_std). The final report will be read by engineers deciding
what to explore next.

## Objectives (prioritized)

1. **Reference projects** — Rust projects (no_std or std) with drivers for
   PCIe, NVMe, network, and GPU. Priority on those usable in a custom kernel context.
2. **Custom-kernel AI** — projects running AI workloads under a Rust custom kernel.
3. **Low-level OS for AI** — projects building a minimal OS layer purpose-built for AI workloads.
4. **Local hardware AI** — projects running AI on local/consumer hardware.
5. **Beyond quantization** — novel optimizations to run frontier-class models on
   ubiquitous hardware (not just 4-bit/8-bit quant, but e.g. sparsity, distillation,
   kernel fusion, speculative execution, new numerics).
6. **Core innovation** — research/projects improving RAM access & allocation,
   memory storage & recall, reasoning & attention, and hallucination reduction.

## Scope & Definitions

- **System innovation** = changes to the runtime/platform layer (memory layout, allocation
  strategies, scheduler, drivers, kernel, numerics at the HW level).
- **Model innovation** = changes to the model/algorithm layer (architecture, attention
  mechanics, memory-for-recall mechanisms, training/inference techniques).
- **Include**: active (updated ≤ 12 months) or historically significant projects; academic
  papers implementing novel approaches; Rust-first projects, or non-Rust projects whose
  technique could be ported to Rust.
- **Exclude**: proprietary/closed-source with no public artifacts; vaporware with no code or
  paper; SaaS-only solutions.

## Methodology

- Sources: GitHub search, crates.io, arXiv, Papers With Code, Hacker News/Twitter/X leads,
  project websites. Note the search date.
- For each candidate record: project name, repo URL, language, license, last commit date,
  activity level, stars (if public), one-paragraph summary, how it maps to the objectives.
- Ranked against a scoring rubric (below), not gut feel.

## Deliverables

Write a single markdown report to `research/spike01/report.md` (create directory if missing) containing:

### 1. Project Matrix
Columns: Project | URL | License | Language(s) | Objective #s met | System/Model | Recency | Maturity | Rank

### 2. SWOT Analysis (per ranked project, top 8–10)
One section each: Strengths, Weaknesses, Opportunities, Threats, and a 1–2 sentence
"relevance to our goals" note.

### 3. Ranking
- Rubric (weights must sum to 100): Relevance to objectives (30), Technical novelty (20),
  Code/license quality (15), Maturity & activity (15), Portability to our stack (10),
  Documentation/evidence (10).
- Show the score breakdown table and a written justification for the top 3.

### 4. Labels
Tag every project `[System]`, `[Model]`, or `[System+Model]` using the definitions above.

### 5. Citations
- Every claim about capabilities/techniques cites a primary source (repo, paper with arXiv ID, or docs).
- Format: `[n] Author(s). Title. arXiv:ID (year). URL`
- Prefer primary over secondary sources.

### 6. Further Research
A prioritized list of follow-up questions, each tied to a specific objective, with
suggested search terms and target sources.

## Constraints

- Rust-first projects get precedence where parity exists.
- No hallucinated projects — if a name cannot be verified, exclude it.
- Note uncertainty explicitly when facts cannot be confirmed.

## Acceptance Criteria

- [ ] Matrix has ≥ 15 verifiable projects
- [ ] Every ranked project has SWOT
- [ ] Ranking has explicit rubric + weights + score table
- [ ] All projects labeled System / Model
- [ ] All key claims cited
- [ ] Further-research section maps to the 6 objectives

## Questions to resolve during research (answer inline if found)

- What is the most mature Rust PCIe/NVMe driver set for custom kernels?
- Has anyone shipped an AI inference engine inside a Rust hobby OS?
- Which non-Rust techniques (e.g. GQA variants, paged attention) are ported to Rust so far?