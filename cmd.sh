bash /home/shx/code/ce-new/retention_repair/script/cfrr_mscoco_sd14.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/retention_repair/script/cfrr_rerun_sd14.sh --device cuda:0 --offline

bash /home/shx/code/ce-new/output_response/script/output_response_mscoco_sd14.sh --device cuda:1 --offline > logs/output_response_sd14.log 2>&1 &
bash /home/shx/code/ce-new/output_response/script/output_response_rerun_sd14.sh --device cuda:1 --offline


# baseline
bash /home/shx/code/ce-new/baseline/unified-concept-editing/scripts/benchmark_all.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/ESD/scripts/benchmark_all.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/RECE/scripts/benchmark_all.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/AdaVD/scripts/benchmark_all.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/SPEED/scripts/benchmark_all.sh --device cuda:0 --offline

