export HF_ENDPOINT="https://hf-mirror.com"

python DiT/attention_replacement/run.py collect \
  --model sd35_medium \
  --cases all \
  --step-fraction 0.5

python DiT/attention_replacement/run.py benchmark \
  --model sd35_medium \
  --cases all \
  --step-fraction 0.5