#!/bin/bash
# Resolve the Rust toolchain's bundled rust-lld and exec it with the args rustc
# passes. Used by .cargo/config.toml for the aarch64-unknown-linux-musl target
# so we don't hardcode a toolchain path (which changes on rustup update).
set -e
LLD="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/rust-lld"
exec "$LLD" "$@"