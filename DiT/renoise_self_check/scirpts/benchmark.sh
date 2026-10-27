#!/usr/bin/env bash
# Run the four benchmark experiment groups, including sampling and evaluation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

model_name="sd35_medium" # "flux1_dev"
for experiment in i2p ip style object; do
    "${PYTHON:-python}" -u DiT/renoise_self_check/run.py \
        --model "$model_name" \
        --cases "$experiment" \
        --phase run \
        --mode both \
        --offline \
        --lambda-ref 0.01 \
        --rho 0.3 \
        --theta 0 \
        --margin 0 \
        --check-fraction 0.4 \
        --probe-seed 0 \
        "$@"
done
