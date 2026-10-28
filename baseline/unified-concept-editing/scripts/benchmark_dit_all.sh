#!/usr/bin/env bash
# Four DiT experiment groups, sequentially; SD3.5 Medium is the default model.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

UCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$UCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uce_benchmark.dit --cases all "$@"
