#!/usr/bin/env bash
set -euo pipefail

# Reproducible download of SmolLM-135M weights.
#
# The 538 MB safetensors file is too large to commit to git (see ../.gitignore),
# so this script fetches it from HuggingFace and verifies its SHA-256 checksum.
#
# Usage: ./download-weights.sh [output_path]
#   output_path defaults to model-builder/model.safetensors

REPO="HuggingFaceTB/SmolLM-135M"
FILE="model.safetensors"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

# SHA-256 of the official SmolLM-135M model.safetensors (v1, 538,090,408 bytes).
# Recompute with: shasum -a 256 model.safetensors
EXPECTED_SHA256="c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093"

OUT="${1:-$(dirname "$0")/model-builder/model.safetensors}"
TMP="${OUT}.part"

if [[ -n "${SMOL_SHA256:-}" ]]; then
    echo "note: overriding expected checksum with \$SMOL_SHA256" >&2
    EXPECTED_SHA256="${SMOL_SHA256}"
fi

mkdir -p "$(dirname "$OUT")"

echo "Downloading ${URL}"
curl -fL --progress-bar -o "${TMP}" "${URL}"

if [[ -n "${EXPECTED_SHA256}" ]]; then
    echo "Verifying checksum..."
    ACTUAL_SHA256="$(shasum -a 256 "${TMP}" | awk '{print $1}')"
    if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
        echo "error: checksum mismatch" >&2
        echo "  expected: ${EXPECTED_SHA256}" >&2
        echo "  actual:   ${ACTUAL_SHA256}" >&2
        rm -f "${TMP}"
        exit 1
    fi
fi

mv "${TMP}" "${OUT}"
echo "Saved ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo "Convert to kernel record with: cargo run -p model-builder --bin model-builder"