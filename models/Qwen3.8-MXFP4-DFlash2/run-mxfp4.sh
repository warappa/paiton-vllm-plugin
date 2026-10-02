#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Select MXFP4 even when the optional 3-bit weights are present in the environment.
exec bash "$script_dir/run-rocm10.sh" --weights mxfp4 "$@"
