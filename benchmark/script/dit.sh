#!/usr/bin/env bash
# One erased model/intervention, paired I2P and MS-COCO 1K evaluation.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

BENCHMARK_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$BENCHMARK_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m benchmark dit "$@"
