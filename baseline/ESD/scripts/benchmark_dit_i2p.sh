#!/usr/bin/env bash
# One nudity training run, evaluated on I2P and MS-COCO 1K with the same checkpoint.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark.dit --cases i2p "$@"
