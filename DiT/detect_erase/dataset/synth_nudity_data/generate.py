"""Keep all 960 positives and sample 480 negatives from the original train split."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


directory = Path(__file__).resolve().parent
source = directory.parent / "i2p_generate/train.jsonl"
source_bytes = source.read_bytes()
rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()]
rng = random.Random(42)
selected = [row for row in rows if row["label"] == 1]

# Each stratum retains one third of its negatives; 72/480 negatives are long.
# Values are (total, long). The original negative long-text share is 15%.
negative_quotas = {"N1": (160, 25), "N2": (128, 18), "N3": (96, 15),
                   "N4": (64, 10), "N5": (32, 4)}
for stratum, (total, long_count) in negative_quotas.items():
    for bucket, count in (("short", total - long_count), ("long", long_count)):
        candidates = [row for row in rows if row["label"] == 0
                      and row["stratum"] == stratum and row["length_bucket"] == bucket]
        selected.extend(rng.sample(candidates, count))
rng.shuffle(selected)

payload = {
    "group": "i2p",
    "template_source": "Subset of i2p_generate/train.jsonl; original text and labels unchanged",
    "templates": [],
    "selection": {
        "seed": 42,
        "source": "../i2p_generate/train.jsonl",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "positives": "all 960",
        "negative_sampling": "without replacement, stratified by stratum and length_bucket",
        "negative_quotas": {key: {"total": total, "long": long_count}
                            for key, (total, long_count) in negative_quotas.items()},
    },
    "datasets": [{"target": "nudity", "samples": selected}],
}
output = directory / "i2p.json"
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Saved {output}: {dict(Counter(row['label'] for row in selected))}")
