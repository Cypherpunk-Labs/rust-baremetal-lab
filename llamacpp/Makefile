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
# are tracked next to the source and never scattered to /tmp.
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

# --- QEMU run configuration --------------------------------------------------
QEMU_MEM    := 4G
QEMU_TIMEOUT ?= 8
SERIAL_LOG  := $(BUILD_DIR)/serial.log

# The 513 MB SmolLM-135M F32 weights are embedded into the kernel ELF at
# compile time (see the `model-data` crate, which points at the `burn`
# workspace's safetensors). No separate model file is loaded at runtime.

RUN_FLAGS   := -M virt -m $(QEMU_MEM) -accel hvf -cpu host \
	-kernel $(KERNEL_ELF) \
	-display none -serial file:$(SERIAL_LOG) -monitor none

.PHONY: all kernel harness check run run-interactive clean

all: kernel

# Verify every external tool this Makefile depends on exists.
check:
	@test -x "$(TOOLCHAIN_CC)" || { echo "ERROR: Arm GNU Toolchain missing at $(ARM_GNU_TOOLCHAIN)"; exit 1; }
	@command -v $(QEMU)   >/dev/null || { echo "ERROR: $(QEMU) not found (brew install qemu)"; exit 1; }
	@command -v $(TIMEOUT) >/dev/null || { echo "ERROR: $(TIMEOUT) not found (brew install coreutils)"; exit 1; }
	@rustup target list --installed 2>/dev/null | grep -q '$(TARGET)' || { echo "ERROR: rust target $(TARGET) missing (rustup target add $(TARGET))"; exit 1; }
	@rustup component list --installed 2>/dev/null | grep -q 'rust-src' || { echo "ERROR: rust-src missing (rustup component add rust-src)"; exit 1; }
	@echo "toolchain OK"

# Build the bare-metal kernel (libggmlcpu.a is compiled by ggml-sys/build.rs).
kernel: check
	cargo build -p $(KERNEL_PKG) --target $(TARGET)

# Build & run the host-side harness (verifies ggml FFI + model parse on host).
harness:
	cargo run -p harness

# Boot the kernel under QEMU and capture its UART output to build/serial.log.
# The kernel currently loops forever after booting, so QEMU_TIMEOUT stops it.
run: kernel
	@mkdir -p $(BUILD_DIR)
	-$(TIMEOUT) $(QEMU_TIMEOUT) $(QEMU) $(RUN_FLAGS)
	@echo "=== serial output ($(SERIAL_LOG)) ==="
	@cat $(SERIAL_LOG)

# Interactive boot (for the future chat REPL): QEMU's PL011 becomes the
# terminal. Type into it; Ctrl-A then X quits.
run-interactive: kernel
	$(QEMU) -M virt -m $(QEMU_MEM) -accel hvf -cpu host \
		-kernel $(KERNEL_ELF) -nographic

clean:
	cargo clean
	rm -rf $(BUILD_DIR)

