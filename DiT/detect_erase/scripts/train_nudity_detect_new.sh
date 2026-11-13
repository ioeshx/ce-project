cd /home/shx/code/ce-new
RUN_DIR="$PWD/DiT/detect_erase/results/sd35_medium-nudity-detect-$(date +%Y%m%d-%H%M%S)"

python DiT/detect_erase/train.py \
  --model sd35_medium \
  --stage detect --cases i2p --targets nudity \
  --dataset-dir DiT/detect_erase/dataset/i2p_generate \
  --run-dir "$RUN_DIR" \
  --prompt-space soft --detect-loss clip \
  --learning-rate 1e-3 --epochs 5 --batch-size 1 \
  --max-steps 0 --tau 0.2 --temperature 0.1 \
  --device cuda:0 \
  --track --swanlab-mode offline \
&& python DiT/detect_erase/sample.py \
  --run-dir "$RUN_DIR" \
  --cases i2p,mscoco --detect-only \
  --no-all-detect-prompts --similarity cosine --tau 0.2 \
  --batch-size 32 --device cuda:0 \
  --track --swanlab-mode offline