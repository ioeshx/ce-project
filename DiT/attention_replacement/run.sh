#!/usr/bin/env bash
set -e

attention_run_started_at=$(date +%s)
attention_run_model=${MODEL:-sd35_medium}
attention_run_cases=${CASES:-all}
attention_run_step_fraction=${STEP_FRACTION:-1}
attention_run_intervention=${INTERVENTION:-replace}
attention_run_strength=${STRENGTH:-0.1}

export HF_ENDPOINT="https://hf-mirror.com"

python DiT/attention_replacement/run.py collect \
  --model "$attention_run_model" \
  --cases "$attention_run_cases" \
  --step-fraction "$attention_run_step_fraction" \
  --intervention "$attention_run_intervention" \
  --strength "$attention_run_strength" \
  "$@"

python DiT/attention_replacement/run.py benchmark \
  --model "$attention_run_model" \
  --cases "$attention_run_cases" \
  --step-fraction "$attention_run_step_fraction" \
  --intervention "$attention_run_intervention" \
  --strength "$attention_run_strength" \
  "$@"

attention_run_exit_code=$?
attention_run_finished_at=$(date +%s)
attention_run_elapsed=$((attention_run_finished_at - attention_run_started_at))
printf '\nTotal elapsed: %02d:%02d:%02d (%d seconds)\n' \
  "$((attention_run_elapsed / 3600))" \
  "$((attention_run_elapsed / 60 % 60))" \
  "$((attention_run_elapsed % 60))" \
  "$attention_run_elapsed"

exit "$attention_run_exit_code"
