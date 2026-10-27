#!/usr/bin/env bash
# Independently erase Van Gogh, Monet, and Picasso.
set -euo pipefail
ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark run --experiments style "$@"
