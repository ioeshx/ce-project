#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec bash "$script_dir/benchmark_dit_all.sh" \
  --model flux1_dev --device cuda:0 --device-map balanced --offline \
  --max-memory '{"0":"12GiB","1":"12GiB"}' --gradient-checkpointing \
  --vlm-model "$repo_root/benchmark/weights/Qwen3-VL-2B-Instruct" \
  --resnet-weights-path "$repo_root/benchmark/weights/resnet50_v2.pth" "$@"
