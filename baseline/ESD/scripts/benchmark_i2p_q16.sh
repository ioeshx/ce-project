#!/usr/bin/env bash
# Joint moderation training; Q16 evaluates all 4,703 I2P prompts.
set -euo pipefail
ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark run --experiments i2p_q16 "$@"
