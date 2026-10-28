#!/usr/bin/env bash
# Pikachu; Snoopy + Pikachu. Shared evaluation concepts come from dit_cases.json.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

UCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$UCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uce_benchmark.dit --cases ip "$@"
