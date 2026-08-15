SHELL := /bin/bash

QEMU   := qemu-system-aarch64
ELF    := target/aarch64-unknown-none/release/kernel
MEM    := 4G
N      := 5
QEMU_COMMON := -machine virt -cpu max -m $(MEM) -serial stdio -display none -kernel $(ELF)

.PHONY: weights model tokenizer validate kernel host run-tcg run-hvf chat bench clean all

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

## host — build the standalone std comparison binary (release)
host:
	cargo build -p host --release

## host-run — run the std comparison binary (greedy benchmark + chat)
host-run: host
	./target/release/host

## run-tcg — boot kernel under TCG emulation (works, ~5s per step with KV cache)
run-tcg: kernel
	$(QEMU) $(QEMU_COMMON)

## run-hvf — boot kernel under HVF acceleration (works via identity MMU, ~0.19s per step with KV cache)
run-hvf: kernel
	$(QEMU) $(QEMU_COMMON) -cpu max -accel hvf

## chat — boot under HVF and enter the interactive chat loop (type a prompt)
chat: kernel
	$(QEMU) $(QEMU_COMMON) -cpu max -accel hvf

## bench — sequential performance comparison: run host and kernel N times
## (default N=5, override with `make bench N=10`) and report average TPS each.
bench: host kernel
	@echo "=== Host (std) benchmark: $(N) runs ==="
	@./target/release/host --bench $(N) 2>&1 | grep -E "avg TPS|Final tokens"
	@echo ""
	@echo "=== Kernel (HVF) benchmark: $(N) runs ==="
	@sum=0; cnt=0; \
	for i in $$(seq 1 $(N)); do \
		line=$$(timeout 30 $(QEMU) $(QEMU_COMMON) -cpu max -accel hvf </dev/null 2>&1 | grep -m1 "avg .* tokens/sec"); \
		v=$$(echo "$$line" | awk '{print $$5}'); \
		echo "  run $$i TPS: $$v"; \
		sum=$$(echo "$$sum + $$v" | bc); cnt=$$((cnt+1)); \
	done; \
	avg=$$(echo "scale=3; $$sum / $$cnt" | bc); \
	echo "  avg TPS over $$cnt runs: $$avg"

clean:
	cd kernel && cargo clean
	cargo clean