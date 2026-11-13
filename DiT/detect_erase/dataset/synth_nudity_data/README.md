# Nudity detect：960 正例 + 480 负例

从 `../i2p_generate/train.jsonl` 取子集：保留全部 960 条正例，用 seed=42 无放回分层抽取 480 条负例，合计 1440 条。原始文本、标签、ID 和其他字段均保留，标签仍覆盖 NudeNet 八类暴露。

| 负例类型 | 原数量 | 保留数量 |
|---|---:|---:|
| N1 安全穿衣人物 | 480 | 160 |
| N2 相近身体/动作但被遮盖 | 384 | 128 |
| N3 相关无人环境/物体 | 288 | 96 |
| N4 关键词干扰、遮挡与裁剪 | 192 | 64 |
| N5 普通无关场景 | 96 | 32 |

同时按文本长度分层，负例中长文本占比保持 15%（72 条）。`i2p.json` 可直接由现有训练入口读取，并记录抽样配置和源文件校验值。验证集继续使用原目录的 `val_id.jsonl`、`val_ood.jsonl`，不参与抽样或训练；当前训练入口不会自动评估它们。子集继承原训练集对验证集和 benchmark 的重复排查结果。

从仓库根目录重新抽样：

```bash
python DiT/detect_erase/dataset/synth_nudity_data/generate.py
```

仅训练 nudity detect，然后测试 I2P、MSCOCO，均开启追踪：

```bash
cd /home/shx/code/ce-new
RUN_DIR="$PWD/DiT/detect_erase/results/sd35_medium-nudity-detect-960pos-480neg-$(date +%Y%m%d-%H%M%S)"

python DiT/detect_erase/train.py \
  --model sd35_medium \
  --stage detect --cases i2p --targets nudity \
  --dataset-dir DiT/detect_erase/dataset/synth_nudity_data \
  --run-dir "$RUN_DIR" \
  --prompt-space soft --detect-loss clip \
  --learning-rate 1e-3 --epochs 5 --batch-size 1 \
  --max-steps 0 --tau 0.2 --temperature 0.1 --seed 0 \
  --device cuda:0 --track --swanlab-mode offline \
&& python DiT/detect_erase/sample.py \
  --run-dir "$RUN_DIR" \
  --cases i2p,mscoco --detect-only \
  --no-all-detect-prompts --similarity cosine --tau 0.2 \
  --batch-size 32 --device cuda:0 \
  --track --swanlab-mode offline
```

与上一轮保持相同的初始化、学习率、5 个 epoch 和阈值，不恢复旧 checkpoint。batch_size=1 时共 7200 次更新；正例仍累计学习 4800 次，负例从 7200 次降为 2400 次。
