#!/usr/bin/env bash

attention_run_started_at=$(date +%s)

export HF_ENDPOINT="https://hf-mirror.com"

python DiT/attention_replacement/run.py collect \
  --model sd35_medium \
  --cases all \
  --step-fraction 1

python DiT/attention_replacement/run.py benchmark \
  --model sd35_medium \
  --cases all \
  --step-fraction 1

attention_run_exit_code=$?
attention_run_finished_at=$(date +%s)
attention_run_elapsed=$((attention_run_finished_at - attention_run_started_at))
printf '\nTotal elapsed: %02d:%02d:%02d (%d seconds)\n' \
  "$((attention_run_elapsed / 3600))" \
  "$((attention_run_elapsed / 60 % 60))" \
  "$((attention_run_elapsed % 60))" \
  "$attention_run_elapsed"

exit "$attention_run_exit_code"
