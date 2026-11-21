#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

python_bin="${PYTHON:-python}"
model="${MODEL:-sd35_medium}"
tau="${TAU:-0.55}"
run_time=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
run_root="${RUN_ROOT:-DiT/detect_erase/results/${model}-pikachu-detection-ablation-${run_time}}"
dataset_dir="$run_root/training_data"

# Use the supplementary scene templates for training and the IP benchmark templates for evaluation.
"$python_bin" - "$dataset_dir" <<'PY'
from pathlib import Path
import sys

from benchmark.common import write_json
from cvpr.code.dataset.detect import expand_detection_prompts

target = "Pikachu"
write_json(Path(sys.argv[1]) / "ip.json", {
    "group": "ip",
    "template_source": "cvpr/code/dataset/detect.py",
    "datasets": [{"target": target, "samples": expand_detection_prompts(target)}],
})
PY

case_args=(--cases ip_pikachu --device "${DEVICE:-cuda:0}" --tau "$tau")
train_args=(
  "${case_args[@]}" --model "$model" --stage detect --targets Pikachu
  --dataset-dir "$dataset_dir" --prompt-space condition
  --learning-rate 1e-3 --epochs "${EPOCHS:-5}" --max-steps "${MAX_STEPS:-2000}"
  --batch-size 1 --classification-batch-size 16
  --classification-weight 1 --contrastive-temperature 0.1 --seed "${SEED:-0}"
  --ti-image-source concept-name --ti-image-count "${TI_IMAGE_COUNT:-100}"
)
eval_args=(
  "${case_args[@]}" --batch-size "${EVAL_BATCH_SIZE:-32}" --detect-only
  --ip-template-count 80 --seeds 0 --no-all-detect-prompts --similarity cosine
)

for loss in supcon ti-flow-supcon; do
  "$python_bin" DiT/detect_erase/train.py \
    "${train_args[@]}" --detect-loss "$loss" --run-dir "$run_root/$loss"
  "$python_bin" DiT/detect_erase/sample.py \
    "${eval_args[@]}" --run-dir "$run_root/$loss"
done

# Reuse the full variant's frozen CLIP configuration for the raw concept-name baseline.
"$python_bin" DiT/detect_erase/sample.py \
  "${eval_args[@]}" --base-clip --run-dir "$run_root/ti-flow-supcon" \
  --output-dir "$run_root/base-clip"

"$python_bin" - "$run_root" "$tau" <<'PY'
from pathlib import Path
import sys

from benchmark.common import read_json, write_csv

root = Path(sys.argv[1])
metrics = {
    "AUROC (%)": "auroc",
    "R@5% FPR (%)": "recall_at_5pct_fpr",
    "Acc (%)": "binary_accuracy",
    "Recall (%)": "recall",
    "FPR (%)": "false_positive_rate",
}
rows = []
for variant, folder in (("Base CLIP", "base-clip"), ("SupCon only", "supcon"),
                        ("SupCon + TI-flow", "ti-flow-supcon")):
    report = read_json(root / folder / "exps/ip_pikachu/detect_summary.json")["summary"]
    row = {"Variant": variant, "Records": report["recorded_count"], "tau": float(sys.argv[2])}
    row.update({name: 100 * report[key] for name, key in metrics.items()})
    rows.append(row)
write_csv(root / "ablation_summary.csv", rows)
print(f"Results: {root / 'ablation_summary.csv'}")
PY
