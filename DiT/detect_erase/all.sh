set -euo pipefail
cd /home/shx/code/ce-new

MODEL=sd35_medium
TRACK_MODE=local
RUN_TIME=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
RUN_ROOT="DiT/detect_erase/results/parallel-${MODEL}-${RUN_TIME}"
mkdir -p "$RUN_ROOT"

TRAIN_ARGS=(
  --model "$MODEL"
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
  --output-dir "$RUN_ROOT/train-detect" \
  --swanlab-name "detect-${RUN_TIME}" \
  > "$RUN_ROOT/detect.log" 2>&1 &
DETECT_PID=$!

# GPU 1：独立训练所有 erase prompt
CUDA_VISIBLE_DEVICES=1 python DiT/detect_erase/train.py \
  "${TRAIN_ARGS[@]}" \
  --stage erase \
  --output-dir "$RUN_ROOT/train-erase" \
  --swanlab-name "erase-${RUN_TIME}" \
  > "$RUN_ROOT/erase.log" 2>&1 &
ERASE_PID=$!

# 等待两项训练成功完成
wait "$DETECT_PID"
wait "$ERASE_PID"

DETECT_CHECKPOINTS=("$RUN_ROOT/train-detect/$MODEL"/*/prompts.pt)
ERASE_CHECKPOINTS=("$RUN_ROOT/train-erase/$MODEL"/*/prompts.pt)

# 合并真实训练得到的两种 prompt
python -m DiT.detect_erase.prompts \
  --detect-checkpoint "${DETECT_CHECKPOINTS[0]}" \
  --erase-checkpoint "${ERASE_CHECKPOINTS[0]}" \
  --output "$RUN_ROOT/prompts.pt"

SAMPLE_ARGS=(
  --checkpoint "$RUN_ROOT/prompts.pt"
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
  --output-dir "$RUN_ROOT/benchmark-i2p-style" \
  --swanlab-name "i2p-style-${RUN_TIME}" \
  > "$RUN_ROOT/i2p-style.log" 2>&1 &
BENCHMARK_A_PID=$!

# GPU 1：ip、object 采样及指标计算
CUDA_VISIBLE_DEVICES=1 python DiT/detect_erase/sample.py \
  "${SAMPLE_ARGS[@]}" \
  --cases ip,object \
  --output-dir "$RUN_ROOT/benchmark-ip-object" \
  --swanlab-name "ip-object-${RUN_TIME}" \
  > "$RUN_ROOT/ip-object.log" 2>&1 &
BENCHMARK_B_PID=$!

wait "$BENCHMARK_A_PID"
wait "$BENCHMARK_B_PID"