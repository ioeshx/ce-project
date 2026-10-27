#!/usr/bin/env bash
# One invocation runs all five experiments sequentially on the selected GPU.
set -euo pipefail
UCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$UCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uce_benchmark run --experiments instance,style,i2p_q16,i2p_nudenet,mscoco "$@"
