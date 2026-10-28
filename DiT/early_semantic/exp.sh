set -eo pipefail

python DiT/early_semantic/run.py --percent 30
python DiT/early_semantic/run.py --percent 40
python DiT/early_semantic/run.py --percent 50
python DiT/early_semantic/run.py --percent 70
# python DiT/early_semantic/run.py --percent 80
# python DiT/early_semantic/run.py --percent 90



