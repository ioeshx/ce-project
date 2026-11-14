#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

# 只训练 erase，采样复用 DETECT_RUN 中已有的 detect checkpoint。
MODEL="sd35_medium"
DETECT_RUN="$PWD/DiT/detect_erase/results/sd35_medium-dual-gpu-20261114-165237"
ERASE_TRAJECTORY="safe"
ERASE_DIR="DiT/detect_erase/dataset/safe-describe"
GPU=0
TRACK_MODE="local"
SAMPLE_PHASE="all" # all：采样并评测；sample：仅采样
LEARNING_RATE=1e-3
EPOCHS=5
BATCH_SIZE=1
SAMPLE_BATCH_SIZE=1
RUN_TIME=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
RUN_ROOT="$PWD/DiT/detect_erase/results/${MODEL}-erase-${ERASE_TRAJECTORY}-${RUN_TIME}"

export CUDA_VISIBLE_DEVICES="$GPU"
echo "Run: $RUN_ROOT"

# 不传 max-steps，使用训练入口的默认值 0，完成全部 epochs。
python DiT/detect_erase/train.py \
  --model "$MODEL" --run-dir "$RUN_ROOT" --stage erase --cases all \
  --prompt-space hidden --erase-dir "$ERASE_DIR" \
  --erase-target cfg --erase-state trajectory --erase-trajectory "$ERASE_TRAJECTORY" \
  --learning-rate "$LEARNING_RATE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --device cuda:0 --track --swanlab-mode "$TRACK_MODE" \
  --swanlab-name "erase-${ERASE_TRAJECTORY}-${RUN_TIME}"

python DiT/detect_erase/sample.py \
  --run-dir "$RUN_ROOT" --detect-run-dir "$DETECT_RUN" --cases all --phase "$SAMPLE_PHASE" \
  --no-zero-erase-clip-padding --batch-size "$SAMPLE_BATCH_SIZE" \
  --device cuda:0 --metric-device cuda:0 --track --swanlab-mode "$TRACK_MODE" \
  --swanlab-name "sample-${ERASE_TRAJECTORY}-${RUN_TIME}"

echo "Results: $RUN_ROOT"
