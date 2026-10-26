# FLUX.1-dev：双卡
CUDA_VISIBLE_DEVICES=0,1 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model flux1_dev --method original \
  --device-map balanced --device cuda:0

# FLUX.1-schnell：双卡
CUDA_VISIBLE_DEVICES=0,1 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model flux1_schnell --method original \
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

