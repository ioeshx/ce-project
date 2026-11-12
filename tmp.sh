cd /home/shx/code/ce-new

RUN_ROOT="/home/shx/code/ce-new/DiT/detect_erase/results/parallel-sd35_medium-20261112-024437"

# GPU 0：i2p、style
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python DiT/detect_erase/sample.py \
  --run-dir "$RUN_ROOT/benchmark-i2p-style/sd35_medium/20261112-114924" \
  --phase all \
  --device cuda:0 \
  --metric-device cuda:0 \
  --batch-size 1 \
  --track \
  --swanlab-mode local \
  >> "$RUN_ROOT/i2p-style-shared-reference.log" 2>&1 < /dev/null &

# GPU 1：ip、object
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
python DiT/detect_erase/sample.py \
  --run-dir "$RUN_ROOT/benchmark-ip-object/sd35_medium/20261112-114923" \
  --phase all \
  --device cuda:0 \
  --metric-device cuda:0 \
  --batch-size 1 \
  --track \
  --swanlab-mode local \
  >> "$RUN_ROOT/ip-object-shared-reference.log" 2>&1 < /dev/null &