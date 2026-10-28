#!/usr/bin/env bash
# Four DiT experiment groups, sequentially; SD3.5 Medium is the default model.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark.dit --cases all "$@"
