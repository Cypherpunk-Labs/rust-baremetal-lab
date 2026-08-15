# rust-baremetal-lab
An experiment on running Rust on Baremetal using QEMU

---

# Building Baremetal Rust on MacOS (QEMU)

This guide walks through creating freestanding `#![no_std]` Rust kernels for both ARM64 and x86_64 architectures, compiled on MacOS and emulated via QEMU.

[Environment setup](environment.md)

[ARM64 Kernel](arm64.md)

[x86_64 Kernel](x86_64.md)

---

# Running a small language model 

SmolLM-135M inference now runs on a bare-metal ARM64 kernel (burn-flex, `no_std`) with a KV cache, UART chat loop, and a GPT-2 BPE tokenizer, plus a std `host` comparison binary and a QEMU/Linux-guest variant.

[Current docs & usage](burn/README.md)

[Full Context Handover Plan](burn/rust_slm_full_context_plan.md) 

3-way performance comparison (10-run averages):

| Path | Avg TPS |
|------|---------|
| std host, native macOS | 7.06 |
| no_std kernel, QEMU/HVF | 5.09 |
| std host in QEMU/Linux guest | 3.73 | 
| Ollama M2 Pro | eval 338.44 |


---

Gemini Lesson Buildout  https://gemini.google.com/app/41e5a753ce217c9e 

[Full Context File](baremetal_rust_QEMU_lesson_plan.md)
