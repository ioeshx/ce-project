cd /home/shx/code/ce-new
set -e

time=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
RUN_DIR="./DiT/detect_erase/results/sd35_medium-detect-$time"
mkdir -p "$RUN_DIR"

# 训练全部概念的 detect soft token
CUDA_VISIBLE_DEVICES=0 python DiT/detect_erase/train.py \
  --run-dir "$RUN_DIR" \
  --model sd35_medium \
  --cases all \
  --stage detect \
  --prompt-space soft \
  --detect-loss clip \
  --learning-rate 1e-3 \
  --epochs 5 \
  --max-steps 2000 \
  --batch-size 1 \
  --tau 0.2 \
  --temperature 0.1 \
  --device cuda:0 \
  --track \
  --swanlab-mode local \
  --swanlab-name detect-train > "$RUN_DIR/train_detect.log" 2>&1

# 在全部 benchmark case 上评测检测能力，不生成图片
CUDA_VISIBLE_DEVICES=0 python DiT/detect_erase/sample.py \
  --run-dir "$RUN_DIR" \
  --cases all \
  --detect-only \
  --similarity cosine \
  --tau 0.2 \
  --multi-concept top1 \
  --batch-size 16 \
  --device cuda:0 \
  --track \
  --swanlab-mode local \
  --swanlab-name detect-benchmark > "$RUN_DIR/test_sample_detect.log" 2>&1

echo "实验目录：$RUN_DIR"