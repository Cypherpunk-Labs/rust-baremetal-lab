# =============================================================================
# llamacpp bare-metal build & run (aarch64-unknown-none, QEMU `virt`)
#
# Prerequisites (captured here so nothing is lost in shell history):
#   1. Arm GNU Toolchain (aarch64-none-elf) with Newlib + libstdc++.
#      brew's `aarch64-elf-*` is NOT sufficient (no libstdc++/newlib).
#      Download from:
#        https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads
#      e.g. arm-gnu-toolchain-15.2.rel1-darwin-arm64-aarch64-none-elf.tar.xz
#      and extract to the path set in ARM_GNU_TOOLCHAIN below.
#   2. Rust nightly with `rust-src` and the `aarch64-unknown-none` target:
#        rustup toolchain install nightly
#        rustup component add rust-src --toolchain nightly
#        rustup target add aarch64-unknown-none
#      (build-std is required and enabled via .cargo/config.toml [unstable].)
#   3. QEMU:    brew install qemu
#   4. timeout: brew install coreutils   (provides /usr/local/bin/timeout)
#
# All generated artifacts (serial capture, logs) live under ./build so they
# are never scattered to /tmp. ./build and ./scratch are gitignored.
# =============================================================================

ROOT            := $(CURDIR)
BUILD_DIR       := $(ROOT)/build

# --- Toolchain locations -----------------------------------------------------
ARM_GNU_TOOLCHAIN ?= /Users/mkemp/tools/arm-gnu-toolchain-15.2.rel1-darwin-arm64-aarch64-none-elf
export ARM_GNU_TOOLCHAIN
export PATH := $(ARM_GNU_TOOLCHAIN)/bin:$(PATH)

TOOLCHAIN_CC := $(ARM_GNU_TOOLCHAIN)/bin/aarch64-none-elf-gcc
QEMU         := qemu-system-aarch64
TIMEOUT      := timeout

# --- Rust / target -----------------------------------------------------------
TARGET      := aarch64-unknown-none
KERNEL_PKG  := kernel
KERNEL_ELF  := target/$(TARGET)/debug/kernel

# --- Data files (embedded at compile time by the `model-data` crate) ---------
# The 513 MB SmolLM-135M F32 weights are cross-workspace pointers (see
# model-data/src/lib.rs). They must exist before any `kernel`/`parity` build.
MODEL_ST    := $(ROOT)/../burn/model-builder/model.safetensors
TOKENIZER_BIN := $(ROOT)/model-data/src/tokenizer.bin
TOKENIZER_SRC := $(ROOT)/../burn/kernel/src/tokenizer.bin
PARITY_PKG  := parity

# --- QEMU run configuration --------------------------------------------------
QEMU_MEM    := 4G
# The kernel runs a continuous chat loop, so non-interactive runs are stopped
# after QEMU_TIMEOUT seconds (overridable: make run QEMU_TIMEOUT=30).
QEMU_TIMEOUT ?= 20
SERIAL_LOG  := $(BUILD_DIR)/serial.log

# The 513 MB SmolLM-135M F32 weights are embedded into the kernel ELF at
# compile time (see the `model-data` crate, which points at the `burn`
# workspace's safetensors). No separate model file is loaded at runtime.

.PHONY: all check kernel kernel-release model-check tokenizer \
	harness test test-gen run run-interactive chat clean

all: kernel

# Verify every external tool this Makefile depends on exists.
check:
	@test -x "$(TOOLCHAIN_CC)" || { echo "ERROR: Arm GNU Toolchain missing at $(ARM_GNU_TOOLCHAIN)"; exit 1; }
	@command -v $(QEMU)   >/dev/null || { echo "ERROR: $(QEMU) not found (brew install qemu)"; exit 1; }
	@command -v $(TIMEOUT) >/dev/null || { echo "ERROR: $(TIMEOUT) not found (brew install coreutils)"; exit 1; }
	@rustup target list --installed 2>/dev/null | grep -q '$(TARGET)' || { echo "ERROR: rust target $(TARGET) missing (rustup target add $(TARGET))"; exit 1; }
	@rustup component list --installed 2>/dev/null | grep -q 'rust-src' || { echo "ERROR: rust-src missing (rustup component add rust-src)"; exit 1; }
	@echo "toolchain OK"

# Verify the embedded model + tokenizer data dependencies exist.
model-check:
	@test -f "$(MODEL_ST)" || { echo "ERROR: model.safetensors missing at $(MODEL_ST)"; exit 1; }
	@test -f "$(TOKENIZER_BIN)" || { echo "ERROR: tokenizer.bin missing at $(TOKENIZER_BIN)"; exit 1; }
	@echo "model data OK"

# Refresh the tokenizer blob from the burn kernel copy.
tokenizer:
	@test -f "$(TOKENIZER_SRC)" || { echo "ERROR: cannot find burn tokenizer at $(TOKENIZER_SRC)"; exit 1; }
	@mkdir -p "$(dir $(TOKENIZER_BIN))"
	@cp "$(TOKENIZER_SRC)" "$(TOKENIZER_BIN)"
	@cmp -s "$(TOKENIZER_SRC)" "$(TOKENIZER_BIN)" && echo "tokenizer.bin up to date"

# Build the bare-metal kernel (libggmlcpu.a is compiled by ggml-sys/build.rs).
kernel: check model-check
	cargo build -p $(KERNEL_PKG) --target $(TARGET)

# Release build of the bare-metal kernel.
kernel-release: check model-check
	cargo build -p $(KERNEL_PKG) --target $(TARGET) --release

# Build & run the host-side harness (verifies ggml FFI + model parse on host).
harness: model-check
	cargo run -p harness

# Run the parity test suite (mul_mat orientation, layer-0 diag parity, full
# logits parity vs the reference). The generation/ground-truth tests are
# #[ignore]d and run via `make test-gen`.
test: model-check
	cargo test -p $(PARITY_PKG) -- --nocapture

# Run the ignored generation + burn-ground-truth tests (proves the forward
# produces coherent text and matches the committed burn kernel's token 7042).
test-gen: model-check
	cargo test -p $(PARITY_PKG) -- --ignored --nocapture

# Boot the kernel under QEMU, pipe a demo prompt into the PL011 UART, and
# capture the full serial transcript (boot messages + generated reply) to
# build/serial.log. Stopped by $(TIMEOUT) after QEMU_TIMEOUT seconds.
run: kernel
	@mkdir -p $(BUILD_DIR)
	@printf 'The capital of France is\n' | $(TIMEOUT) $(QEMU_TIMEOUT) $(QEMU) \
		-M virt -m $(QEMU_MEM) -accel hvf -cpu host \
		-kernel $(KERNEL_ELF) -display none -monitor none -serial stdio \
		| tee $(SERIAL_LOG)

# Interactive boot (chat REPL): QEMU's PL011 becomes the terminal. Type into
# it; Ctrl-A then X quits.
run-interactive: kernel
	$(QEMU) -M virt -m $(QEMU_MEM) -accel hvf -cpu host \
		-kernel $(KERNEL_ELF) -nographic

# Alias for `run`: boots the kernel in QEMU, pipes a prompt, and prints the
# generated reply captured from the PL011 UART.
chat: run

clean:
	cargo clean
	rm -rf $(BUILD_DIR) scratch

