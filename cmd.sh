python /home/shx/code/ce-new/DiT/detectFix/run.py benchmark \
  --cases all --phase run --trace \
  --rho 0.2 --detection-start-step 5 --detection-mode once > DiT/detectFix/logs/detect_once.log 2>&1

python /home/shx/code/ce-new/DiT/detectFix/run.py benchmark \
  --cases i2p --phase run --trace \
  --device cuda:1 \
  --rho 0.2 --detection-start-step 5 --detection-mode every_step > DiT/detectFix/logs/detect_every_step.log 2>&1

# python /home/shx/code/ce-new/DiT/queryGraft/run.py benchmark \
#   --cases i2p --tasks i2p \
#   --device cuda:0 --phase run \
#   --update-rule zero --trace

python /home/shx/code/ce-new/DiT/queryGraft/run.py benchmark \
  --cases i2p \
  --tasks i2p \
  --device cuda:1 \
  --phase run \
  --trace > DiT/queryGraft/logs/new_decision.log 2>&1

python /home/shx/code/ce-new/DiT/queryGraft/run.py benchmark \
  --model sd35_medium --cases i2p --tasks i2p \
  --mask-rule anchor_min \
  --device cuda:1 --phase run --trace > DiT/queryGraft/logs/anchor_min.log 2>&1

python DiT/queryGraft/run.py benchmark \
  --model sd35_medium --cases i2p --tasks i2p,mscoco \
  --representation av --localization-layers all --edit-layers last \
  --source-mode anchor --update-rule replace --eta 0 --rho 0 \
  --device cuda:0 --phase run --trace > DiT/queryGraft/logs/trace_i2p.log 2>&1

DiT/queryGraft/scritps/source_activation.sh > DiT/queryGraft/logs/source_activation.log 2>&1
DiT/queryGraft/scritps/source_activation-2.sh > DiT/queryGraft/logs/source_activation-2.log 2>&1

REPRESENTATION=block_output bash DiT/queryGraft/run.sh \
  --device cuda:0 --phase run --device cuda:0 > DiT/queryGraft/logs/block_output.log 2>&1

bash DiT/queryGraft/sweep_eta_rho.sh --device cuda:0 > DiT/queryGraft/logs/sweep_eta_rho.log 2>&1
bash DiT/queryGraft/sweep_eta_rho-2.sh --device cuda:1 > DiT/queryGraft/logs/sweep_eta_rho-2.log 2>&1


REPRESENTATION=residual bash DiT/queryGraft/run.sh --device cuda:0 > DiT/queryGraft/logs/replace_residual.log

CASES=i2p bash DiT/queryGraft/run.sh \
  --device cuda:1 --tasks i2p --phase sample --trace> DiT/queryGraft/logs/i2p_record.log

MASK_MODE=majority_all bash DiT/attention_replacement/run.sh --device cuda:0 > DiT/attention_replacement/logs/majority_all.log

MASK_MODE=majority_last bash DiT/attention_replacement/run.sh --device cuda:1 > DiT/attention_replacement/logs/majority_last.log

# add：使用 CUDA 0
STEP_FRACTION=0.5 INTERVENTION=add STRENGTH=0.1 \
  bash DiT/attention_replacement/run.sh --device cuda:0 > DiT/attention_replacement/logs/add.log

# blend：使用 CUDA 1
STEP_FRACTION=0.5 INTERVENTION=blend STRENGTH=0.1 \
  bash DiT/attention_replacement/run.sh --device cuda:1 > DiT/attention_replacement/logs/blend.log  

baseline/RAZOR-cvpr2026/scripts/benchmark_dit_all.sh --device cuda:0 > baseline/RAZOR-cvpr2026/dit_all.log 2>&1
baseline/CASteer/scripts/benchmark_dit_all.sh --device cuda:1 > baseline/CASteer/dit_all.log 2>&1

HF_ENDPOINT="https://hf-mirror.com" baseline/EraseAnything/scripts/benchmark_sd35_all.sh \
  --cases i2p,object --device cuda:0 > benchmark/logs/eraseAnything_sd35_i2p_object.log 2>&1

HF_ENDPOINT="https://hf-mirror.com" baseline/EraseAnything/scripts/benchmark_sd35_all.sh \
  --cases style,ip --device cuda:1 > benchmark/logs/eraseAnything_sd35_style_ip.log 2>&1

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

