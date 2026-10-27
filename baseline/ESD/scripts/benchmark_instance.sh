#!/usr/bin/env bash
# Three independent instance target sets from benchmark/configs.
set -euo pipefail
ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark run --experiments instance "$@"
