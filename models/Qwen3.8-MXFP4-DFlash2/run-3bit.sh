#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Require PAITON_W3ROT_DIR; the shared launcher validates it before starting Docker.
exec bash "$script_dir/run-rocm10.sh" --weights w3a4 "$@"
