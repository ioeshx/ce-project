#!/usr/bin/env bash
# Erase nudity; NudeNet evaluates the 931 sexual-category I2P records.
set -euo pipefail
UCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$UCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uce_benchmark run --experiments i2p_nudenet "$@"
