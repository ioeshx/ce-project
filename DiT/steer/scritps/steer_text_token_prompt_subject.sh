# prompt-mode bare
# check steer-pool
# attention_residual:	attention 更新与残差相加后
# mlp_output:	  modulation、MLP 之后，尚未乘 MLP gate、加回残差
# block_output:	MLP gate 和最终残差相加后的完整输出

DiT/steer/run.sh \
  --cases all \
  --device cuda:0 \
  --steer-target mlp_output \
  --prompt-mode subject \
  --steer-token text \
  --subset-eval \
  --offline > DiT/steer/logs/steer_text_mlp_output.log 2>&1

  # --i2p-test-num 10 --mscoco-test-num 10 \
  # --ip-test-num 3 --style-test-num 3 --object-test-num 3 \

# DiT/steer/run.sh \
#   --cases all \
#   --device cuda:0 \
#   --prompt-mode bare \
#   --steer-pool \
#   --subset-eval \
#   --i2p-test-num 10 --mscoco-test-num 10 \
#   --ip-test-num 3 --style-test-num 3 --object-test-num 3 \
#   --offline > DiT/steer/logs/steer_pool_prompt_bare.log 2>&1

# DiT/steer/run.sh \
#   --cases all \
#   --device cuda:0 \
#   --prompt-mode bare \
#   --steer-token text \
#   --steer-target attention_residual \
#   --subset-eval \
#   --i2p-test-num 10 --mscoco-test-num 10 \
#   --ip-test-num 3 --style-test-num 3 --object-test-num 3 \
#   --offline > DiT/steer/logs/steer_text_attention_residual.log 2>&1


# DiT/steer/run.sh \
#   --cases all \
#   --device cuda:1 \
#   --prompt-mode bare \
#   --steer-target block_output \
#   --steer-token text \
#   --subset-eval \
#   --i2p-test-num 10 --mscoco-test-num 10 \
#   --ip-test-num 3 --style-test-num 3 --object-test-num 3 \
#   --offline > DiT/steer/logs/steer_text_block_output.log 2>&1

