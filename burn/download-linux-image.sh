#!/usr/bin/env bash
set -euo pipefail

# Reproducible download of the Ubuntu 24.04 ARM64 cloud image used to run the
# std `host` binary inside QEMU/HVF (for the std-under-QEMU vs no_std-kernel
# comparison). Verified by SHA-256.
#
# Usage: ./download-linux-image.sh [output_path]
#   output_path defaults to model-builder/linux/ubuntu-24.04-server-cloudimg-arm64.img

URL="https://cloud-images.ubuntu.com/releases/24.04/release/ubuntu-24.04-server-cloudimg-arm64.img"
EXPECTED_SHA256="4a281a921b8d7db952895ab619736f10efe9f63e111fa5b5779ed18f023818aa"

OUT="${1:-$(dirname "$0")/model-builder/linux/ubuntu-24.04-server-cloudimg-arm64.img}"
TMP="${OUT}.part"

if [[ -n "${UBUNTU_SHA256:-}" ]]; then
    echo "note: overriding expected checksum with \$UBUNTU_SHA256" >&2
    EXPECTED_SHA256="${UBUNTU_SHA256}"
fi

mkdir -p "$(dirname "$OUT")"

if [[ -f "$OUT" ]]; then
    echo "note: $OUT already exists, skipping download" >&2
    echo "  (remove it to force a re-download)"
    exit 0
fi

echo "Downloading ${URL}"
curl -fL --progress-bar -o "${TMP}" "${URL}"

echo "Verifying checksum..."
ACTUAL_SHA256="$(shasum -a 256 "${TMP}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "error: checksum mismatch" >&2
    echo "  expected: ${EXPECTED_SHA256}" >&2
    echo "  actual:   ${ACTUAL_SHA256}" >&2
    rm -f "${TMP}"
    exit 1
fi

mv "${TMP}" "${OUT}"
echo "Saved ${OUT} ($(du -h "${OUT}" | cut -f1))"