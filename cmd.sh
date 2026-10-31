git config --local http.proxy http://127.0.0.1:7890
git config --local https.proxy http://127.0.0.1:7890

git config --local --unset http.proxy 
git config --local --unset https.proxy

python DiT/early_semantic/run.py --percent 10

export HF_ENDPOINT="https://hf-mirror.com"
baseline/unified-concept-editing/scripts/benchmark_dit_all.sh --device cuda:0 > benchmark/logs/uce_dit.log 2>&1&

baseline/ESD/scripts/benchmark_dit_all.sh --device cuda:1 > benchmark/logs/esd_dit.log 2>&1&

# I2P：sexual 931条 NudeNet检测 + MS-COCO 1K质量评测
bash benchmark/script/dit.sh \
  --model sd35_medium --method original --cases i2p --offline

# IP：分别擦除 Pikachu、Snoopy+Pikachu 的两组评测
bash benchmark/script/dit.sh \
  --model sd35_medium --method original --cases ip --offline

# style：Van Gogh、Monet、Picasso 各自擦除的三组评测
bash benchmark/script/dit.sh \
  --model sd35_medium --method original --cases style --offline

# Object：church单概念、五概念联合擦除的两组评测
bash benchmark/script/dit.sh \
  --model sd35_medium --method original --cases object --offline

CUDA_VISIBLE_DEVICES=0,1 bash benchmark/script/dit.sh \
  --model flux1_dev --method original --cases all \
  --device-map balanced --device cuda:0 --offline
# --model flux1_schnell sd30_medium sd35_medium

# FLUX.1-dev：双卡
CUDA_VISIBLE_DEVICES=0,1 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model flux1_dev --method original \
  --device-map balanced --device cuda:0

# FLUX.1-schnell：双卡
CUDA_VISIBLE_DEVICES=0,1 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model  --method original \
  --device-map balanced --device cuda:0

# SD3.0 Medium：单卡
CUDA_VISIBLE_DEVICES=0 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model sd30_medium --method original --device cuda:0

# SD3.5 Medium：单卡
CUDA_VISIBLE_DEVICES=0 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model sd35_medium --method original --device cuda:0 > benchmark/logs/sd35_i2p.log


# clsoe form method
bash /home/shx/code/ce-new/retention_repair/script/cfrr_mscoco_sd14.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/retention_repair/script/cfrr_rerun_sd14.sh --device cuda:0 --offline

bash /home/shx/code/ce-new/output_response/script/output_response_mscoco_sd14.sh --device cuda:1 --offline > logs/output_response_sd14.log 2>&1 &
bash /home/shx/code/ce-new/output_response/script/output_response_rerun_sd14.sh --device cuda:1 --offline


# sd14 baseline
bash /home/shx/code/ce-new/baseline/unified-concept-editing/scripts/benchmark_all.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/ESD/scripts/benchmark_all.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/RECE/scripts/benchmark_all.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/AdaVD/scripts/benchmark_all.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/SPEED/scripts/benchmark_all.sh --device cuda:0 --offline

