#!/usr/bin/env bash
# One nudity edit, evaluated on I2P and MS-COCO 1K with the same checkpoint.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

UCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$UCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uce_benchmark.dit --cases i2p "$@"
