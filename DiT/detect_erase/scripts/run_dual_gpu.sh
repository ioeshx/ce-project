#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

# 直接修改下面的配置，然后运行本脚本。
MODEL="sd35_medium"
DETECT_LOSS="ti-flow-supcon" # clip / ti-flow / ti-flow-bce / ti-flow-supcon
ERASE_TARGET="cfg"          # cfg / conditional（无 CFG）
ERASE_STATE="trajectory"    # trajectory / image-noise
ERASE_TRAJECTORY="safe"     # unsafe / safe / student，仅 trajectory 使用
DETECT_GPU=0
ERASE_GPU=1
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
RUN_ROOT="$PWD/DiT/detect_erase/results/${MODEL}-dual-gpu-${RUN_TIME}"

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

# CUDA_VISIBLE_DEVICES 把各进程选中的物理 GPU 映射为进程内 cuda:0。
CUDA_VISIBLE_DEVICES="$DETECT_GPU" python DiT/detect_erase/train.py \
  "${TRAIN_ARGS[@]}" --stage detect --prompt-space condition \
  --detect-loss "$DETECT_LOSS" --swanlab-name "detect-${RUN_TIME}" &
DETECT_PID=$!

CUDA_VISIBLE_DEVICES="$ERASE_GPU" python DiT/detect_erase/train.py \
  "${TRAIN_ARGS[@]}" --stage erase "${ERASE_ARGS[@]}" \
  --swanlab-name "erase-${RUN_TIME}" &
ERASE_PID=$!

# 两项训练均成功后再采样；任一失败都会使脚本以非零状态退出。
TRAIN_STATUS=0
wait "$DETECT_PID" || TRAIN_STATUS=$?
wait "$ERASE_PID" || TRAIN_STATUS=$?
if (( TRAIN_STATUS != 0 )); then
  exit "$TRAIN_STATUS"
fi

SAMPLE_ARGS=(
  --run-dir "$RUN_ROOT" --phase "$SAMPLE_PHASE"
  --batch-size "$SAMPLE_BATCH_SIZE" --device cuda:0 --metric-device cuda:0
  --track --swanlab-mode "$TRACK_MODE"
)

CUDA_VISIBLE_DEVICES="$DETECT_GPU" python DiT/detect_erase/sample.py \
  "${SAMPLE_ARGS[@]}" --cases i2p,mscoco,style \
  --swanlab-name "sample-i2p-mscoco-style-${RUN_TIME}" &
SAMPLE_A_PID=$!

CUDA_VISIBLE_DEVICES="$ERASE_GPU" python DiT/detect_erase/sample.py \
  "${SAMPLE_ARGS[@]}" --cases ip,object \
  --swanlab-name "sample-ip-object-${RUN_TIME}" &
SAMPLE_B_PID=$!

SAMPLE_STATUS=0
wait "$SAMPLE_A_PID" || SAMPLE_STATUS=$?
wait "$SAMPLE_B_PID" || SAMPLE_STATUS=$?
if (( SAMPLE_STATUS != 0 )); then
  exit "$SAMPLE_STATUS"
fi

echo "Results: $RUN_ROOT"
