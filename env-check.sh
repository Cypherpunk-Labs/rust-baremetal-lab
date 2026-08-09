#!/bin/bash

echo "========================================="
echo "  Baremetal Rust Lab - Environment Info  "
echo "========================================="
echo "**Date:** $(date)"
echo ""

echo "**Operating System:**"
uname -sm
echo ""

echo "**Rust Toolchain:**"
rustc +nightly --version
cargo +nightly --version
echo ""

echo "**Emulators:**"
qemu-system-aarch64 --version | head -n 1
qemu-system-x86_64 --version | head -n 1
echo ""

echo "**Cargo Utilities:**"
bootimage --version
echo "========================================="