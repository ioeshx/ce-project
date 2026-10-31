#!/usr/bin/env bash
# EraseAnything FLUX DiT experiment group: style.
set -euo pipefail
ERASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ERASE_ROOT/benchmark_dit.py" --cases style "$@"
