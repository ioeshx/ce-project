python DiT/queryGraft/run.py benchmark --cases style,ip,object \
  --representation text_output --edit-layers all --phase run \
  --device cuda:0 > DiT/queryGraft/logs/text_output_style_instance_object.log 2>&1

python DiT/queryGraft/run.py benchmark --cases all \
  --representation text_output --edit-layers last --phase run \
  --device cuda:1 > DiT/queryGraft/logs/text_output_editLast_all.log 2>&1