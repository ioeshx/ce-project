#!/usr/bin/env bash
# EraseAnything SD3.5 Medium experiment group: i2p.
set -euo pipefail
ERASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ERASE_ROOT/benchmark_sd35.py" --cases i2p "$@"
