#!/usr/bin/env bash
# Erase nudity; NudeNet evaluates the 931 sexual-category I2P records.
set -euo pipefail
ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark run --experiments i2p_nudenet "$@"
