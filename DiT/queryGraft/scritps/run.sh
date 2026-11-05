#!/usr/bin/env bash
set -e

export HF_ENDPOINT="https://hf-mirror.com"

querygraft_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec python "$querygraft_directory/run.py" benchmark \
  --model "${MODEL:-sd35_medium}" \
  --cases "${CASES:-all}" \
  --localization-layers "${LOCALIZATION_LAYERS:-all}" \
  --edit-layers "${EDIT_LAYERS:-last}" \
  --representation "${REPRESENTATION:-av}" \
  --eta "${ETA:-0.0}" \
  --rho "${RHO:-0.0}" \
  --update-rule "${UPDATE_RULE:-replace}" \
  --strength "${STRENGTH:-1}" \
  --source-mode "${SOURCE_MODE:-anchor}" \
  --source-user-weight "${SOURCE_USER_WEIGHT:-1}" \
  --source-anchor-weight "${SOURCE_ANCHOR_WEIGHT:-1}" \
  --source-scale "${SOURCE_SCALE:-1}" \
  "$@"
