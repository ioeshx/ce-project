#!/usr/bin/env bash
# One erased model/intervention, paired I2P and MS-COCO 1K evaluation.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

export BENCHMARK_CALLER_DIR="$PWD"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m benchmark dit "$@"
