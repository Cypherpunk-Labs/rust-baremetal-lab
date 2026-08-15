#!/usr/bin/env bash
# Boot the Ubuntu 24.04 ARM64 cloud image under QEMU/HVF and run the std `host`
# benchmark inside the guest (via cloud-init), using a 9p shared folder for the
# static binary + data files.
#
# Usage: ./run-linux.sh [--nographic] [-serial stdio|file]
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMG="${DIR}/model-builder/linux/ubuntu-24.04-server-cloudimg-arm64.img"
SEED="${DIR}/model-builder/linux/seed.iso"
SHARE="${DIR}/model-builder/linux/share"

QEMU="${QEMU:-qemu-system-aarch64}"
MEM="${MEM:-4G}"
CPU="${CPU:-max}"
FW="$(dirname "$(command -v qemu-system-aarch64)")/../share/qemu/edk2-aarch64-code.fd"

exec "${QEMU}" \
  -machine virt \
  -cpu "${CPU}" \
  -m "${MEM}" \
  -accel hvf \
  -display none \
  -serial stdio \
  -drive if=pflash,format=raw,readonly=on,file="${FW}" \
  -drive file="${IMG}",format=qcow2,if=virtio \
  -drive file="${SEED}",format=raw,if=virtio \
  -netdev user,id=n0 \
  -device virtio-net-device,netdev=n0 \
  -virtfs local,path="${SHARE}",mount_tag=host0,security_model=none,id=host0