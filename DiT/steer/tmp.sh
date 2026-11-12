DiT/steer/run.sh \
  --cases all \
  --device cuda:0 \
  --steer-target text_token_output \
  --prompt-mode bare \
  --subset-eval \
  --i2p-test-num 10 --mscoco-test-num 10 \
  --ip-test-num 3 --style-test-num 3 --object-test-num 3 \
  --offline > DiT/steer/logs/steer_text_token_output.log 2>&1

DiT/steer/run.sh \
  --cases all \
  --device cuda:1 \
  --steer-target text_token \
  --prompt-mode bare \
  --subset-eval \
  --i2p-test-num 10 --mscoco-test-num 10 \
  --ip-test-num 3 --style-test-num 3 --object-test-num 3 \
  --offline > DiT/steer/logs/steer_text_token.log 2>&1

# prompt-mode bare,subject,matched
DiT/steer/run.sh \
  --cases all \
  --device cuda:1 \
  --prompt-mode subject \
  --subset-eval \
  --i2p-test-num 10 --mscoco-test-num 10 \
  --ip-test-num 2 --style-test-num 2 --object-test-num 2 \
  --offline > DiT/steer/logs/prompt_mode_subject.log 2>&1

DiT/steer/run.sh \
  --cases all \
  --device cuda:0 \
  --prompt-mode bare \
  --subset-eval \
  --i2p-test-num 10 --mscoco-test-num 10 \
  --ip-test-num 2 --style-test-num 2 --object-test-num 2 \
  --offline > DiT/steer/logs/prompt_mode_bare.log 2>&1

DiT/steer/run.sh \
  --cases ip,style,object \
  --device cuda:0 \
  --no-score-gate \
  --subset-eval \
  --i2p-test-num 10 --mscoco-test-num 10 \
  --ip-test-num 2 --style-test-num 2 --object-test-num 2 \
  --offline > DiT/steer/logs/no_score_gate.log 2>&1

DiT/steer/run.sh \
  --cases all \
  --device cuda:1 \
  --no-clipping \
  --subset-eval \
  --i2p-test-num 10 --mscoco-test-num 10 \
  --ip-test-num 3 --style-test-num 3 --object-test-num 3 \
  --offline > DiT/steer/logs/no_clipping.log 2>&1

DiT/steer/run.sh \
  --cases all \
  --device cuda:0 \
  --vector-method pca \
  --subset-eval \
  --i2p-test-num 100 --mscoco-test-num 100 \
  --ip-test-num 10 --style-test-num 10 --object-test-num 10 \
  --offline > DiT/steer/logs/pca.log 2>&1

DiT/steer/run.sh \
  --cases all \
  --device cuda:1 \
  --vector-method pca-filter --pca-rank 4 \
  --subset-eval \
  --i2p-test-num 100 --mscoco-test-num 100 \
  --ip-test-num 10 --style-test-num 10 --object-test-num 10 \
  --offline > DiT/steer/logs/pca_filter_rank_4.log 2>&1

DiT/steer/run.sh \
  --cases all \
  --device cuda:0 \
  --vector-method mean --normalize-differences \
  --subset-eval \
  --i2p-test-num 100 --mscoco-test-num 100 \
  --ip-test-num 10 --style-test-num 10 --object-test-num 10 \
  --offline > DiT/steer/logs/normalize_diff.log 2>&1

DiT/steer/run.sh \
  --cases all \
  --device cuda:0 \
  --vector-method robust --robust-groups 5 \
  --subset-eval \
  --i2p-test-num 10 --mscoco-test-num 100 \
  --ip-test-num 10 --style-test-num 10 --object-test-num 10 \
  --offline > DiT/steer/logs/robust_group_5.log 2>&1


DiT/steer/run.sh \
  --cases all \
  --device cuda:0 \
  --vector-method robust --robust-groups 5 \
  --subset-eval \
  --i2p-test-num 10 --mscoco-test-num 10 \
  --ip-test-num 3 --style-test-num 3 --object-test-num 3 \
  --offline > DiT/steer/logs/robust_group_5.log 2>&1


bash /home/shx/code/ce-new/DiT/steer/run.sh \
  --cases ip,style,object \
  --target-anchor-config ce-new/DiT/steer/conf.json \
  --offline

# cuda:0：I2P、style
# --offline
bash /home/shx/code/ce-new/DiT/steer/run.sh \
  --cases i2p,style --device cuda:0 > DiT/steer/i2p_style_base.log 2>&1

# cuda:1：IP、Object
bash /home/shx/code/ce-new/DiT/steer/run.sh \
  --cases ip,object --device cuda:1 > DiT/steer/ip_object_base.log 2>&1


bash /home/shx/code/ce-new/DiT/steer/run.sh \
  --cases i2p,style --device cuda:0 --offline \
  --steer-target text_value --average-patches > DiT/steer/logs/i2p_style_target_textValue.log 2>&1

bash /home/shx/code/ce-new/DiT/steer/run.sh \
  --cases ip,object --device cuda:1 --offline \
  --steer-target text_value --average-patches > DiT/steer/logs/ip_object_target_textValue.log 2>&1
