#!/usr/bin/env bash
# FLUX Object: church; church + golf ball + parachute + cassette player + banana.
set -euo pipefail
ERASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ERASE_ROOT/benchmark_dit.py" --cases object "$@"
