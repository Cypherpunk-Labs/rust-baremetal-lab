SHELL := /bin/bash

QEMU   := qemu-system-aarch64
ELF    := target/aarch64-unknown-none/release/kernel
MEM    := 4G
QEMU_COMMON := -machine virt -cpu max -m $(MEM) -serial stdio -display none -kernel $(ELF)

.PHONY: weights model tokenizer validate kernel run-tcg run-hvf chat clean all

all: weights model tokenizer kernel

## weights — download SmolLM-135M safetensors into model-builder/ (checksum-verified)
weights:
	./download-weights.sh

## model — convert safetensors -> kernel/src/smollm-135m.bin (model-builder)
model:
	cargo run -p model-builder --bin model-builder

## tokenizer — convert model-builder/tokenizer.json -> kernel/src/tokenizer.bin
tokenizer:
	cargo run -p model-builder --bin tokenizer-builder model-builder/tokenizer.json kernel/src/tokenizer.bin

## validate — run SmolLM-135M inference on the host (burn-flex + std) as a reference
validate:
	cargo run -p model-builder --bin validate

## tokenizer-check — validate the kernel tokenizer blob against known token ids
tokenizer-check:
	cargo run -p model-builder --bin tokenizer-check

## kernel — build the no_std aarch64 kernel (must run in kernel/ for build-std config)
kernel:
	cd kernel && cargo build --release --target aarch64-unknown-none

## run-tcg — boot kernel under TCG emulation (works, ~5s per step with KV cache)
run-tcg: kernel
	$(QEMU) $(QEMU_COMMON)

## run-hvf — boot kernel under HVF acceleration (works via identity MMU, ~0.19s per step with KV cache)
run-hvf: kernel
	$(QEMU) $(QEMU_COMMON) -cpu max -accel hvf

## chat — boot under HVF and enter the interactive chat loop (type a prompt)
chat: kernel
	$(QEMU) $(QEMU_COMMON) -cpu max -accel hvf

clean:
	cd kernel && cargo clean
	cargo clean