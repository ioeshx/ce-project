# DiT/renoise_self_check/scirpts/ip_object.sh > DiT/renoise_self_check/logs/ip_object.log 2>&1&

# IP：单目标和多目标 case
CUDA_VISIBLE_DEVICES=1 python DiT/renoise_self_check/run.py \
    --model sd35_medium \
    --cases ip \
    --phase run \
    --mode both \
    --offline


# Object：单目标和多目标 case
CUDA_VISIBLE_DEVICES=1 python DiT/renoise_self_check/run.py \
    --model sd35_medium \
    --cases object \
    --phase run \
    --mode both \
    --offline