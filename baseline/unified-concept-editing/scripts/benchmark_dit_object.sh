#!/usr/bin/env bash
# ImageNet-10: church; church + golf ball + parachute + cassette player + banana.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

UCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$UCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uce_benchmark.dit --cases object "$@"
