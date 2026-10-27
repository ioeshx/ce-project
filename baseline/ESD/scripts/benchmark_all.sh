#!/usr/bin/env bash
# One invocation runs all five experiments sequentially on the selected GPU.
set -euo pipefail
ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark run --experiments instance,style,i2p_q16,i2p_nudenet,mscoco "$@"
