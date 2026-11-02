#!/usr/bin/env bash
# RAZOR: native benchmark case group all; default SD3.5 Medium.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

METHOD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$METHOD_ROOT/benchmark_dit.py" --cases all "$@"
