#!/usr/bin/env bash
# RAZOR: native benchmark case group i2p; default SD3.5 Medium.
set -euo pipefail
METHOD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$METHOD_ROOT/benchmark_dit.py" --cases i2p "$@"
