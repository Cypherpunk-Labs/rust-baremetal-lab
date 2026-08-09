## Phase 1: Environment Setup

Before writing code, install the required emulators and toolchains. 

*Note You can use env-check.sh and compare with versions.txt if you already have these installed to compare version that this lab was authored with.*

**1. Install QEMU**
```bash
brew install qemu

```

**2. Configure Rust Nightly**
Baremetal development requires unstable compiler features.

```bash
rustup toolchain install nightly
rustup default nightly
rustup target add aarch64-unknown-none
rustup component add rust-src llvm-tools-preview

```

**3. Install Bootimage (for x86_64)**

```bash
cargo install bootimage

```

[Back](README.md)