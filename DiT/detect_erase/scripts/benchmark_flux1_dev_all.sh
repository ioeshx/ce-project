#!/usr/bin/env bash
set -euo pipefail

# 无参数时训练并评测。传入 --run-dir 可直接用 FLUX.1-dev 检查点采样和评测。
method_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if (( $# == 0 )); then
  run_time=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
  run_dir="$method_root/results/flux1_dev-benchmark-${run_time}"
  "$python_bin" "$method_root/train.py" \
    --model flux1_dev --run-dir "$run_dir" --cases all --stage all \
    --detect-loss ti-flow-supcon --erase-dir "$method_root/dataset/safe-describe" \
    --erase-target cfg --erase-state trajectory --erase-trajectory safe --true-cfg-scale 2.0 \
    --epochs 5 --max-steps 2000 --device cuda:0 --device-map balanced --offline --track
  set -- --run-dir "$run_dir" --offline --track
fi

# sample.py 按训练记录使用 true CFG，并为该采样协议生成匹配的 FLUX 原图。
exec "$python_bin" "$method_root/sample.py" \
  --cases all --device cuda:0 --device-map balanced --metric-device cuda:0 "$@"
