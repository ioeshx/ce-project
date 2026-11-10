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
  --device cuda:0 \
  --vector-method mean --normalize-differences \
  --subset-eval \
  --i2p-test-num 10 --mscoco-test-num 10 \
  --ip-test-num 3 --style-test-num 3 --object-test-num 3 \
  --offline > DiT/steer/logs/normalize_diff.log 2>&1