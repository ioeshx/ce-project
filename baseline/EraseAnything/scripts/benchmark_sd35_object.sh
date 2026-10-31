#!/usr/bin/env bash
# SD3.5 Object: church; church + golf ball + parachute + cassette player + banana.
set -euo pipefail
ERASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ERASE_ROOT/benchmark_sd35.py" --cases object "$@"
