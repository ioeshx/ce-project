#!/usr/bin/env bash
set -euo pipefail
cd /home/shx/code/ce-new

MODEL=${MODEL:-sd35_medium}
TRACK_MODE=${TRACK_MODE:-local}
RUN_TIME=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
RUN_ROOT="/home/shx/code/ce-new/DiT/detect_erase/results/${MODEL}-${RUN_TIME}"
mkdir -p "$RUN_ROOT"

TRAIN_ARGS=(
  --model "$MODEL"
  --run-dir "$RUN_ROOT"
  --cases all
  --learning-rate 1e-3
  --epochs 5
  --max-steps 2000
  --batch-size 1
  --device cuda:0
  --track
  --swanlab-mode "$TRACK_MODE"
)

# GPU 0：训练所有 detect prompt
CUDA_VISIBLE_DEVICES=0 python DiT/detect_erase/train.py \
  "${TRAIN_ARGS[@]}" \
  --stage detect \
  --detect-loss clip \
  --swanlab-name detect &
DETECT_PID=$!

# GPU 1：独立训练所有 erase prompt
CUDA_VISIBLE_DEVICES=1 python DiT/detect_erase/train.py \
  "${TRAIN_ARGS[@]}" \
  --stage erase \
  --swanlab-name erase &
ERASE_PID=$!

# 等待两项训练成功完成
wait "$DETECT_PID"
wait "$ERASE_PID"

SAMPLE_ARGS=(
  --run-dir "$RUN_ROOT"
  --phase all
  --device cuda:0
  --metric-device cuda:0
  --batch-size 1
  --track
  --swanlab-mode "$TRACK_MODE"
)

# GPU 0：i2p、style 采样及指标计算
CUDA_VISIBLE_DEVICES=0 python DiT/detect_erase/sample.py \
  "${SAMPLE_ARGS[@]}" \
  --cases i2p,style \
  --swanlab-name i2p-style &
BENCHMARK_A_PID=$!

# GPU 1：ip、object 采样及指标计算
CUDA_VISIBLE_DEVICES=1 python DiT/detect_erase/sample.py \
  "${SAMPLE_ARGS[@]}" \
  --cases ip,object \
  --swanlab-name ip-object &
BENCHMARK_B_PID=$!

wait "$BENCHMARK_A_PID"
wait "$BENCHMARK_B_PID"

echo "Results: $RUN_ROOT"
