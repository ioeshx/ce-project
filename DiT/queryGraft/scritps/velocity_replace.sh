# cd /home/shx/code/ce-new
export HF_ENDPOINT="https://hf-mirror.com"

# cd /home/shx/code/ce-new

common=(
  --model sd35_medium
  --cases i2p --tasks i2p
  --representation velocity
  --edit-layers last --update-rule replace
  --eta 0 --rho 0
  --device cuda:0 --phase run
)

# 1. user prompt + anchor 联合条件
python DiT/queryGraft/run.py benchmark "${common[@]}" \
  --source-mode joint

# 2. source attention logits 加权
python DiT/queryGraft/run.py benchmark "${common[@]}" \
  --source-mode logits \
  --source-user-weight 0.5 --source-anchor-weight 2

# # 3. source Value 贡献加权
# python DiT/queryGraft/run.py benchmark "${common[@]}" \
#   --source-mode values \
#   --source-user-weight 0.5 --source-anchor-weight 2

# 4. 在最终 velocity 上放大联合条件带来的变化
python DiT/queryGraft/run.py benchmark "${common[@]}" \
  --source-mode extrapolate --source-scale 2