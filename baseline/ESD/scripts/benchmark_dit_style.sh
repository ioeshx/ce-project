#!/usr/bin/env bash
# Van Gogh, Monet and Picasso are erased in three separate cases.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark.dit --cases style "$@"
