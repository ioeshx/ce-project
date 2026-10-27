#!/usr/bin/env bash
# Three independent instance target sets from benchmark/configs.
set -euo pipefail
UCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$UCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uce_benchmark run --experiments instance "$@"
