#!/usr/bin/env bash
# Native GEM DiT experiment group: style.
set -euo pipefail
GEM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$GEM_ROOT/benchmark_dit.py" --cases style "$@"
