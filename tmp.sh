CUDA_VISIBLE_DEVICES=0,1 bash /home/shx/code/ce-new/baseline/ESD/scripts/benchmark_dit_all.sh \
  --model flux1_dev --device cuda:0 \
  --train-max-memory '{"0":"11776MiB","1":"13GiB"}' \
  --offline --gradient-checkpointing --cpu-offload \
  --vlm-model /home/shx/code/ce-new/benchmark/weights/Qwen3-VL-2B-Instruct \
  --resnet-weights-path /home/shx/code/ce-new/benchmark/weights/resnet50_v2.pth

CUDA_VISIBLE_DEVICES=1 TAU=0.2 bash /home/shx/code/ce-new/DiT/detect_erase/scripts/run_i2p_detection_ablation.sh

baseline/ESD/scripts/benchmark_dit_all.sh --model flux1_dev  > baseline/ESD/flux_all.log 2>&1

latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

latexmk -pdf -pvc -view=none -interaction=nonstopmode \
  -halt-on-error -file-line-error main.tex