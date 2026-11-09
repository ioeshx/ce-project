# cuda:0：I2P、style
# --offline
bash /home/shx/code/ce-new/DiT/steer/run.sh \
  --cases i2p,style --device cuda:0 > DiT/steer/i2p_style_base.log 2>&1

# cuda:1：IP、Object
bash /home/shx/code/ce-new/DiT/steer/run.sh \
  --cases ip,object --device cuda:1 > DiT/steer/ip_object_base.log 2>&1


bash /home/shx/code/ce-new/DiT/steer/run.sh \
  --cases all --device cuda:0 --offline \
  --steer-target text_value --average-patches > DiT/steer/logs/all_target_text_value.log 2>&1
