#!/usr/bin/env bash
set -euo pipefail
image=${PAITON_NEO_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin@sha256:534287969135f581744ae481b578599468b0bf7ac9a4051b0941500e4c18da4d}
port=${PAITON_NEO_PORT:-8000}
cache=${PAITON_NEO_CACHE:-paiton-qwen38-neo-cache}
exec docker run --rm --name paiton-qwen38-neo \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p "127.0.0.1:${port}:8000" \
  --mount "type=bind,source=${cache},target=/models/cache" \
  "$@" "$image" --max-model-len 64000


#  --mount "type=volume,src=${cache},dst=/models/cache" \
