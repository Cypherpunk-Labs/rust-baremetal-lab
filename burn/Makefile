SHELL := /bin/bash

QEMU   := qemu-system-aarch64
ELF    := target/aarch64-unknown-none/release/kernel
MEM    := 4G
QEMU_COMMON := -machine virt -cpu max -m $(MEM) -serial stdio -display none -kernel $(ELF)

.PHONY: weights model validate kernel run-tcg run-hvf clean all

all: weights model kernel

## weights — download SmolLM-135M safetensors into model-builder/ (checksum-verified)
weights:
	./download-weights.sh

## model — convert safetensors -> kernel/src/smollm-135m.bin (model-builder)
model:
	cargo run -p model-builder --bin model-builder

## validate — run SmolLM-135M inference on the host (burn-flex + std) as a reference
validate:
	cargo run -p model-builder --bin validate

## kernel — build the no_std aarch64 kernel (must run in kernel/ for build-std config)
kernel:
	cd kernel && cargo build --release --target aarch64-unknown-none

## run-tcg — boot kernel under TCG emulation (fully working, ~13s per step)
run-tcg: kernel
	$(QEMU) $(QEMU_COMMON)

## run-hvf — boot kernel under HVF acceleration (currently hangs on atomics, see README)
run-hvf: kernel
	$(QEMU) $(QEMU_COMMON) -cpu max -accel hvf

clean:
	cd kernel && cargo clean
	cargo clean