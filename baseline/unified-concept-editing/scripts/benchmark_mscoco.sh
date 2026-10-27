#!/usr/bin/env bash
# Jointly erase all CIFAR-50 names, then evaluate MS-COCO's first 1K captions.
set -euo pipefail
UCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$UCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uce_benchmark run --experiments mscoco "$@"
