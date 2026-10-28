#!/usr/bin/env bash
# ImageNet-10: church; church + golf ball + parachute + cassette player + gas pump.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark.dit --cases object "$@"
