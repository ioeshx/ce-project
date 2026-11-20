#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

python_bin="${PYTHON:-python}"
model="${MODEL:-sd35_medium}"
device="${DEVICE:-cuda:0}"
tau="${TAU:-0.2}"
run_time=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
run_root="${RUN_ROOT:-DiT/detect_erase/results/${model}-i2p-detection-ablation-${run_time}}"

train_args=(
  --model "$model" --device "$device" --stage detect --cases i2p --targets nudity
  --dataset-dir "${DATASET_DIR:-DiT/detect_erase/dataset/synth_nudity_data}"
  --prompt-space condition --learning-rate 1e-3 --epochs "${EPOCHS:-5}"
  --max-steps "${MAX_STEPS:-2000}" --batch-size 1 --classification-batch-size 16
  --classification-weight 1 --contrastive-temperature 0.1 --seed "${SEED:-0}" --tau "$tau"
)
if [[ -n "${TI_IMAGE_COUNT:-}" ]]; then
  train_args+=(--ti-image-count "$TI_IMAGE_COUNT")
fi

eval_args=(
  --device "$device" --batch-size "${EVAL_BATCH_SIZE:-32}" --detect-only
  --cases i2p --i2p-subset sexual --i2p-prompt-count 931
  --no-all-detect-prompts --similarity cosine --tau "$tau"
)

for loss in supcon ti-flow-supcon; do
  "$python_bin" DiT/detect_erase/train.py \
    "${train_args[@]}" --detect-loss "$loss" --run-dir "$run_root/$loss"
  "$python_bin" DiT/detect_erase/sample.py \
    "${eval_args[@]}" --run-dir "$run_root/$loss"
done

# Reuse the full variant's text encoder configuration.
"$python_bin" DiT/detect_erase/sample.py \
  "${eval_args[@]}" --base-clip --run-dir "$run_root/ti-flow-supcon" \
  --output-dir "$run_root/base-clip"

"$python_bin" - "$run_root" "$tau" <<'PY'
from pathlib import Path
import sys

from benchmark.common import read_json, write_csv

root = Path(sys.argv[1])
metrics = {
    "Acc (%)": "binary_accuracy",
    "Recall (%)": "recall",
    "FPR (%)": "false_positive_rate",
    "AUROC (%)": "auroc",
    "R@5% FPR (%)": "recall_at_5pct_fpr",
}
rows = []
for variant, folder in (("Base CLIP", "base-clip"), ("SupCon only", "supcon"),
                        ("SupCon + TI-flow", "ti-flow-supcon")):
    report = read_json(root / folder / "exps/i2p_nudity/detect_summary.json")["summary"]
    row = {"Variant": variant, "Records": report["recorded_count"], "tau": sys.argv[2]}
    row.update({name: 100 * report[key] for name, key in metrics.items()})
    rows.append(row)
write_csv(root / "ablation_summary.csv", rows)
print(f"Results: {root / 'ablation_summary.csv'}")
PY
