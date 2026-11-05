#!/usr/bin/env bash
set -euo pipefail

# Override with space-separated lists, e.g. ETA_VALUES="0.2 0.4" RHO_VALUES="0.1 0.3".
read -r -a eta_values <<< "${ETA_VALUES:-0.1 0.2}"
read -r -a rho_values <<< "${RHO_VALUES:-0.1 0.2}"
querygraft_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
total=$((${#eta_values[@]} * ${#rho_values[@]}))
run_number=0

for eta in "${eta_values[@]}"; do
  for rho in "${rho_values[@]}"; do
    run_number=$((run_number + 1))
    printf '\n[%d/%d] eta=%s rho=%s\n' "$run_number" "$total" "$eta" "$rho"
    REPRESENTATION="${REPRESENTATION:-block_output}" CASES="${CASES:-i2p}" \
      bash "$querygraft_directory/run.sh" \
        --tasks "${TASKS:-i2p}" \
        --phase "${PHASE:-run}" \
        "$@" \
        --eta "$eta" --rho "$rho"
  done
done
