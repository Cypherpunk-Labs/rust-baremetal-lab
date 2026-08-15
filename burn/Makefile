SHELL := /bin/bash

QEMU   := qemu-system-aarch64
ELF    := target/aarch64-unknown-none/release/kernel
MEM    := 4G
N      := 5
QEMU_COMMON := -machine virt -cpu max -m $(MEM) -serial stdio -display none -kernel $(ELF)

.PHONY: weights model tokenizer validate kernel host host-linux host-run linux-image seed run-tcg run-hvf run-linux chat bench bench-linux clean all

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

## host-linux — cross-compile the std comparison binary to a static aarch64 Linux ELF
host-linux:
	cargo build -p host --release --target aarch64-unknown-linux-musl

## host-run — run the std comparison binary (greedy benchmark + chat)
host-run: host
	./target/release/host

## linux-image — download the Ubuntu 24.04 ARM64 cloud image (checksum-verified)
linux-image:
	./download-linux-image.sh

## seed — regenerate the cloud-init NoCloud seed ISO from model-builder/linux/user-data
seed:
	@rm -rf model-builder/linux/cidata
	@mkdir -p model-builder/linux/cidata
	@cp model-builder/linux/user-data model-builder/linux/meta-data model-builder/linux/cidata/
	@rm -f model-builder/linux/seed.iso
	@hdiutil makehybrid -o model-builder/linux/seed.iso -iso -default-volume-name cidata model-builder/linux/cidata >/dev/null 2>&1
	@echo "seed.iso regenerated"

## run-linux — boot the Ubuntu guest under QEMU/HVF and run the std host benchmark
## inside it (via cloud-init bootcmd + 9p shared folder; result in .../share/smollm/guest.log)
run-linux: host-linux linux-image seed
	@cp target/aarch64-unknown-linux-musl/release/host model-builder/linux/share/smollm/host
	@chmod +x model-builder/linux/share/smollm/host
	@rm -f model-builder/linux/share/smollm/guest.log
	@scripts/run-linux.sh
	@echo ""
	@echo "=== Guest result (smollm/guest.log) ==="
	@cat model-builder/linux/share/smollm/guest.log

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

## bench-linux — run the std host binary inside the Ubuntu guest (QEMU/HVF) N times
## and average the TPS. Each boot auto-runs the benchmark via cloud-init bootcmd
## and writes the result back through the 9p share (guest.log).
bench-linux: host-linux linux-image seed
	@cp target/aarch64-unknown-linux-musl/release/host model-builder/linux/share/smollm/host
	@chmod +x model-builder/linux/share/smollm/host
	@echo "=== Host in QEMU/Linux guest benchmark: $(N) runs ==="
	@sum=0; cnt=0; \
	for i in $$(seq 1 $(N)); do \
		rm -f model-builder/linux/share/smollm/guest.log; \
		timeout 120 scripts/run-linux.sh </dev/null >/dev/null 2>&1; \
		v=$$(grep -m1 "avg TPS" model-builder/linux/share/smollm/guest.log | awk '{print $$NF}'); \
		echo "  run $$i TPS: $$v"; \
		sum=$$(echo "$$sum + $$v" | bc); cnt=$$((cnt+1)); \
	done; \
	avg=$$(echo "scale=3; $$sum / $$cnt" | bc); \
	echo "  avg TPS over $$cnt runs: $$avg"

clean:
	cd kernel && cargo clean
	cargo clean