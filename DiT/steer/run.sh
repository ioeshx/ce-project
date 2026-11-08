#!/usr/bin/env bash
set -euo pipefail
STEER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$STEER_DIR/run.py" "$@"
