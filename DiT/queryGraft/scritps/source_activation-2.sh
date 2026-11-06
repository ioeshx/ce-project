# cd /home/shx/code/ce-new
export HF_ENDPOINT="https://hf-mirror.com"

# 共用设置；可统一修改 eta、rho
common=(
  --model sd35_medium
  --cases ip,style,object 
  # --tasks i2p
  --representation block_output
  --edit-layers last --update-rule replace
  --eta 0 --rho 0
  --device cuda:1 --phase run
  # --anchor-overrides '{"nudity":"clothed"}'
)

# 1. 联合条件，不加权
# python DiT/queryGraft/run.py benchmark "${common[@]}" \
#   --source-mode joint

# # 2. Attention logits 加权
# python DiT/queryGraft/run.py benchmark "${common[@]}" \
#   --source-mode logits \
#   --source-user-weight 0.5 --source-anchor-weight 2

# 3. Value 贡献加权
python DiT/queryGraft/run.py benchmark "${common[@]}" \
  --source-mode values \
  --source-user-weight 0.5 --source-anchor-weight 2

# 4. 联合条件激活变化放大至 2 倍
python DiT/queryGraft/run.py benchmark "${common[@]}" \
  --source-mode extrapolate --source-scale 2