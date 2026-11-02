#!/usr/bin/env bash

export HF_ENDPOINT="https://hf-mirror.com"

baseline/GEM/scripts/benchmark_dit_all.sh \
  --cases ip,object --device cuda:0 \
  --text-device cpu --gradient-checkpointing > baseline/GEM/logs/ip_object.log 2>&1
