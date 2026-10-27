# DiT/renoise_self_check/scirpts/i2p_style.sh > DiT/renoise_self_check/logs/i2p_style.log  2>&1&

# I2P：概念擦除检测 + COCO 保留评价
CUDA_VISIBLE_DEVICES=0 python DiT/renoise_self_check/run.py \
    --model sd35_medium \
    --cases i2p \
    --phase run \
    --mode both \
    --offline

# Style：三种艺术家风格 case
CUDA_VISIBLE_DEVICES=0 python DiT/renoise_self_check/run.py \
    --model sd35_medium \
    --cases style \
    --phase run \
    --mode both \
    --offline