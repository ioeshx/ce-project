#!/usr/bin/env bash
# Jointly erase all CIFAR-50 names, then evaluate MS-COCO's first 1K captions.
set -euo pipefail
ESD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ESD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m esd_benchmark run --experiments mscoco "$@"
