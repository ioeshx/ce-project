#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

# 直接修改下面的配置，然后运行本脚本。
MODEL="sd35_medium"
DETECT_LOSS="ti-flow-supcon" # clip / ti-flow / ti-flow-bce / ti-flow-supcon
ERASE_TARGET="cfg"          # cfg / conditional（无 CFG）
ERASE_STATE="trajectory"    # trajectory / image-noise
ERASE_TRAJECTORY="safe"     # unsafe / safe / student，仅 trajectory 使用
GPU=0
TRACK_MODE="local"         # local / offline / online
SAMPLE_PHASE="all"         # all：采样并评测；sample：仅采样
LEARNING_RATE=1e-3
EPOCHS=5
MAX_STEPS=2000
BATCH_SIZE=1
SAMPLE_BATCH_SIZE=1
ERASE_IMAGE_COUNT=10
ERASE_REUSE_IMAGES=true
RUN_TIME=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
RUN_ROOT="$PWD/DiT/detect_erase/results/${MODEL}-single-gpu-${RUN_TIME}"

# detect、erase、采样及评测依次执行，全部只使用这一张物理 GPU。
export CUDA_VISIBLE_DEVICES="$GPU"

TRAIN_ARGS=(
  --model "$MODEL" --run-dir "$RUN_ROOT" --cases all
  --learning-rate "$LEARNING_RATE"
  --epochs "$EPOCHS" --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE" --device cuda:0
  --track --swanlab-mode "$TRACK_MODE"
)
ERASE_ARGS=(
  --prompt-space hidden --erase-target "$ERASE_TARGET" --erase-state "$ERASE_STATE"
)
if [[ "$ERASE_STATE" == trajectory ]]; then
  ERASE_ARGS+=(--erase-trajectory "$ERASE_TRAJECTORY")
else
  ERASE_ARGS+=(--erase-image-count "$ERASE_IMAGE_COUNT")
  if [[ "$ERASE_REUSE_IMAGES" == false ]]; then
    ERASE_ARGS+=(--no-erase-reuse-images)
  else
    ERASE_ARGS+=(--erase-reuse-images)
  fi
fi

echo "Run: $RUN_ROOT"

python DiT/detect_erase/train.py \
  "${TRAIN_ARGS[@]}" --stage detect --prompt-space condition \
  --detect-loss "$DETECT_LOSS" --swanlab-name "detect-${RUN_TIME}"

python DiT/detect_erase/train.py \
  "${TRAIN_ARGS[@]}" --stage erase "${ERASE_ARGS[@]}" \
  --swanlab-name "erase-${RUN_TIME}"

python DiT/detect_erase/sample.py \
  --run-dir "$RUN_ROOT" --cases all --phase "$SAMPLE_PHASE" \
  --batch-size "$SAMPLE_BATCH_SIZE" --device cuda:0 --metric-device cuda:0 \
  --track --swanlab-mode "$TRACK_MODE" --swanlab-name "sample-all-${RUN_TIME}"

echo "Results: $RUN_ROOT"
