#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

# 单卡训练 erase，完成后双卡分组采样；复用已有 detect checkpoint。
MODEL="sd35_medium"
DETECT_RUN="$PWD/DiT/detect_erase/results/sd35_medium-dual-gpu-20261114-165237"
ERASE_TRAJECTORY="student"
ERASE_DIR="DiT/detect_erase/dataset/safe-describe"
TRAIN_GPU=0
SAMPLE_GPU_A=0
SAMPLE_GPU_B=1
TRACK_MODE="local"
SAMPLE_PHASE="all" # all：采样并评测；sample：仅采样
LEARNING_RATE=1e-3
EPOCHS=5
BATCH_SIZE=1
SAMPLE_BATCH_SIZE=1
RUN_TIME=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
RUN_ROOT="$PWD/DiT/detect_erase/results/${MODEL}-erase-${ERASE_TRAJECTORY}-dual-sample-${RUN_TIME}"

echo "Run: $RUN_ROOT"

# 不传 max-steps，使用训练入口的默认值 0，完成全部 epochs。
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python DiT/detect_erase/train.py \
  --model "$MODEL" --run-dir "$RUN_ROOT" --stage erase --cases all \
  --prompt-space hidden --erase-dir "$ERASE_DIR" \
  --erase-target cfg --erase-state trajectory --erase-trajectory "$ERASE_TRAJECTORY" \
  --learning-rate "$LEARNING_RATE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --device cuda:0 --track --swanlab-mode "$TRACK_MODE" \
  --swanlab-name "erase-${ERASE_TRAJECTORY}-${RUN_TIME}"

SAMPLE_ARGS=(
  --run-dir "$RUN_ROOT" --detect-run-dir "$DETECT_RUN" --phase "$SAMPLE_PHASE"
  --no-zero-erase-clip-padding --batch-size "$SAMPLE_BATCH_SIZE"
  --device cuda:0 --metric-device cuda:0 --track --swanlab-mode "$TRACK_MODE"
)

# 每个进程只看到分配给自己的物理 GPU，进程内均使用 cuda:0。
CUDA_VISIBLE_DEVICES="$SAMPLE_GPU_A" python DiT/detect_erase/sample.py \
  "${SAMPLE_ARGS[@]}" --cases i2p,mscoco,style \
  --swanlab-name "sample-i2p-mscoco-style-${ERASE_TRAJECTORY}-${RUN_TIME}" &
SAMPLE_A_PID=$!

CUDA_VISIBLE_DEVICES="$SAMPLE_GPU_B" python DiT/detect_erase/sample.py \
  "${SAMPLE_ARGS[@]}" --cases ip,object \
  --swanlab-name "sample-ip-object-${ERASE_TRAJECTORY}-${RUN_TIME}" &
SAMPLE_B_PID=$!

# 等待两组采样及评测结束，保留任一任务的失败状态。
SAMPLE_STATUS=0
wait "$SAMPLE_A_PID" || SAMPLE_STATUS=$?
wait "$SAMPLE_B_PID" || SAMPLE_STATUS=$?
if (( SAMPLE_STATUS != 0 )); then
  exit "$SAMPLE_STATUS"
fi

echo "Results: $RUN_ROOT"
