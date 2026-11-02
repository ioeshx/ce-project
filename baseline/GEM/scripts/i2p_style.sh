#!/usr/bin/env bash

export HF_ENDPOINT="https://hf-mirror.com"

baseline/GEM/scripts/benchmark_dit_all.sh \
  --cases i2p,style --device cuda:1 \
  --text-device cpu --gradient-checkpointing > baseline/GEM/logs/i2p_style.log 2>&1