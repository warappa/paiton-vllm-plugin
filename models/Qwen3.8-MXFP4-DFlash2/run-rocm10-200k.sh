#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The current image in its long-context mode (--profile chat: 200,000 tokens, prefix caching). Options you pass
# come later and override these.
exec python3 "$script_dir/launch-rocm10.py" --release 65k --profile chat --name paiton-qwen38-200k "$@"
