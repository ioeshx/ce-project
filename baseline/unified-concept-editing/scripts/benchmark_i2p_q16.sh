#!/usr/bin/env bash
# Native joint moderation example; Q16 evaluates all 4,703 I2P prompts.
set -euo pipefail
UCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$UCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uce_benchmark run --experiments i2p_q16 "$@"
