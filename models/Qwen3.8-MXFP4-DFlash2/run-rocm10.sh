#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The current image: the 65K mode by default, the long-context mode with --context above 65536. Options you pass
# come later and override these.
exec python3 "$script_dir/launch-rocm10.py" --release 65k --name paiton-qwen38 "$@"
