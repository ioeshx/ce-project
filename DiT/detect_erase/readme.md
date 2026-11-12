## 一、算法与 loss

冻结所有模型权重，每个概念、每个阶段独立训练和保存。默认 `--prompt-space soft`：在 CLIP 的 token lookup 之后、position embedding 和 encoder 之前学习 **一个输入 token**。SD3.5 的两个 CLIP 各学习一个；FLUX 学习一个。BOS、EOS 和其余 token 固定，默认文本结构为 `[BOS] [soft] [EOS] [PAD]…`，可用 `--soft-template 'a photo of {}'` 添加固定上下文。

Detect 初始化为目标概念所有子 token 的输入 embedding 均值。Erase soft token 初始化为 EOS 的输入 embedding，T5 固定为空文本编码；它的初始 CLIP 输出不要求与空文本完全相同。`--prompt-space hidden` 保留原实现：detect 学习 CLIP pooled 输出、序列固定为概念编码；erase 学习完整 CLIP/T5 序列和 pooled 输出，并由空文本编码初始化。两个阶段可独立选择空间。

**默认 detect loss（`--detect-loss clip`）**：取每个 CLIP 的 EOS 表征经过 `text_projection` 后的特征。用户文本与 soft prompt 使用相同编码器，SD3.5 的两个余弦分数取算术平均：

$$
r_i(q)=\frac1K\sum_{k=1}^K\cos(f_k(q),f_k(p_i^d)),\qquad
\mathcal L_d=\operatorname{BCEWithLogits}\big((r_i(q)-\tau)/T,y_i\big).
$$

使用 detect 集全部正负样本。只加载 CLIP，梯度经过冻结 encoder 回到 soft token，不加载 T5、DiT 或 VAE。FLUX 原生 CLIP 没有 contrastive projection，余弦检测使用原版 `openai/clip-vit-large-patch14` 的预训练 projection；FLUX 的模型条件仍使用原生未投影 pooled。可用 `--clip-projection-model` 指定该 CLIP 模型的本地目录。

**备选 TI-flow（`--detect-loss ti-flow`）**：仅使用 detect 正样本，每条文本用冻结基础模型生成一张固定训练图，默认每概念 30 张。图片保存后复用，不做外部分类器筛选。VAE posterior mode 编码并应用原模型的 shift/scaling 得到 $z_0$：

$$
\sigma\sim U(0,1),\quad\epsilon\sim\mathcal N(0,I),\quad
z_t=(1-\sigma)z_0+\sigma\epsilon,\quad t=1000\sigma,
$$
$$
\mathcal L_{d,\mathrm{TI}}=\mathbb E\|v_\theta(z_t,t,p_i^d)-(\epsilon-z_0)\|_2^2.
$$

每个 batch 重新抽噪声和时间，仅优化 detect 的 CLIP 参数；T5 固定为目标概念编码。FLUX 将 noisy latent 和 velocity target 同样 pack，并使用其原生时间归一化。TI-flow 不再使用旧的正样本文本 velocity 蒸馏目标，也没有分类 loss；其检测效果通过独立评测确认。

**Erase loss** 保持 CFG 拟合：

$$
v_\mathrm{CFG}=v_\theta(x_t,t,p_i^e)+s[v_\theta(x_t,t,q^+)-v_\theta(x_t,t,p_i^e)],\qquad
\mathcal L_e=\mathbb E\|v_\mathrm{CFG}-v_\theta(x_t,t,q^a)\|_2^2.
$$

基础模型的空负条件 CFG 推进去噪轨迹，每一步共享同一个 $x_t,t$ 计算各分支，更新一次 erase prompt；teacher 和轨迹停止梯度。

**检测与采样**：默认开启 `--all-detect-prompts`，扫描训练运行目录 `train_detect/*.pt`，所有已保存的 detect prompt 共同参与比较。`cosine` 选最高分，`l2` 选完整 CLIP/T5 序列与 pooled 的 RMS 距离最小值。全量模式直接取 top1，**不使用 tau 或 L2 阈值**；每条输入均有一个预测概念。

候选集合与实验擦除目标分开：`predicted` 是全量 top1，`selected` 是 `predicted ∩ case.targets`。只有选中的概念属于当前实验的擦除目标时，才使用对应 erase prompt；否则使用空文本负条件。例如 `ip_pikachu` 的输入被预测为 Snoopy 时，记录预测 Snoopy，但不触发 Pikachu 擦除。

使用 `--no-all-detect-prompts` 恢复 case.targets 内的阈值检测：余弦达到 tau／RMS 距离不超过阈值才触发。`top1` 选有效候选的最佳项，`mean` 对所有有效项的 erase 条件取平均。全量无阈值模式仅支持 top1。

## 二、超参数

| 参数 | 默认值 / 含义 |
|---|---|
| `--stage` | `all`；当前仅训练检测时显式指定 `detect` |
| `--prompt-space` | `soft`；可选 `hidden`，每个阶段独立配置 |
| `--soft-template` | `{}`；恰好一个占位符 |
| `--detect-loss` | `clip`；可选 `ti-flow` |
| `--learning-rate` | `1e-3`，AdamW，weight decay=0 |
| `--epochs` | 5，每个概念、阶段分别计数 |
| `--max-steps` / `--steps`（训练） | 每概念、阶段最多 2000 次优化更新；0 不设上限 |
| `--batch-size` | 1 |
| `--tau`、`--temperature` | 0.2、0.1，用于训练分类 loss；tau 仅在关闭全量检测时用于采样门控 |
| `--seed` | 0，训练噪声、时间和 shuffle 的随机种子 |
| `--ti-image-count` | 全部正样本，当前每概念 30 张；可限制前 N 条 |
| `--ti-generation-steps` | 继承 `--sampling-steps`，用于固定训练图的生成 |
| `--ti-seed` | 继承 `--seed`；第 j 张图使用 seed+j |
| `--sampling-steps`（训练） | benchmark 默认步数，用于 erase 轨迹和 TI 训练图 |
| `--steps`（采样） | 继承训练采样配置；仅检测模式不使用 |
| `--mscoco-prompt-count`（采样） | 1000，MSCOCO caption 数量 |
| `--all-detect-prompts`（采样） | 默认开启：全部已保存 detector 参与 top1，无阈值；`--no-all-detect-prompts` 关闭 |
| `--output-dir`（采样） | 独立结果根目录；不填时写回 `--run-dir` |
| `--similarity`（采样） | `cosine`；可选 `l2` |
| `--l2-threshold` | 0.2，仅关闭全量模式时用于门控 |
| `--multi-concept` | `top1`；关闭全量模式后可选 `mean` |
| CFG | SD3.5 继承 benchmark guidance；FLUX true CFG=2，embedded guidance 继承 benchmark |
| `--dtype` | 继承模型配置，通常 BF16；参数及优化器状态为 FP32 |
| `--gradient-checkpointing` | 默认开启，可用 `--no-gradient-checkpointing` 关闭 |

Epoch 和更新数上限先到者结束。CLIP 分类 50 条、batch=1、5 epochs 是 250 次更新；TI-flow 30 张是 150 次。Erase 每个去噪步骤更新一次，50 对 × 40 步已达 2000 次；完整训练用 `--max-steps 0`。

## 三、实现与运行配置

当前只训练 detect 并评测文本检测能力：

```bash
RUN_DIR="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
python DiT/detect_erase/train.py --run-dir "$RUN_DIR" --model sd35_medium \
  --stage detect --prompt-space soft --detect-loss clip --track
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --cases all \
  --detect-only --batch-size 16 --track
```

`--detect-only` 不需要 erase checkpoint，不加载 DiT/VAE、不生成参考图或实验图、不计算图像指标。余弦只加载 CLIP，L2 额外加载 T5。它每次重算所选文本；指定新的 `--output-dir` 可独立保存此次全部结果，原有训练与实验文件保持原样。即使关闭 `--track` 仍保存检测统计和图；`--track` 额外同步 SwanLab。

快速检查及 L2 检测：

```bash
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --detect-only \
  --i2p-prompt-count 20 --ip-template-count 2 --style-template-count 2 --object-template-count 2
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --detect-only \
  --similarity l2
```

TI-flow 请另建实验根目录，避免覆盖 CLIP loss 的结果：

```bash
TI_RUN="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-ti-flow-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
python DiT/detect_erase/train.py --run-dir "$TI_RUN" --stage detect \
  --prompt-space soft --detect-loss ti-flow --ti-generation-steps 40 --ti-seed 0 --track
python DiT/detect_erase/sample.py --run-dir "$TI_RUN" --detect-only --track
```

`train_detect/ti_images/<概念>/positive_000.png` 等保存训练图，`images.json` 记录文本、种子和生成协议。`--resume` 复用完整图片，只补缺失／损坏的图片；缓存协议改变时另建 run。续训重新创建优化器：

```bash
python DiT/detect_erase/train.py --run-dir "$RUN_DIR" --stage detect \
  --targets nudity --resume --epochs 5 --track
```

Hidden 输出方案使用 `--prompt-space hidden`，从新 run 开始。`--cases`、`--targets`、`--limit-prompts` 可限制训练范围；`--model flux1_dev` 切换模型。默认余弦路径的 CLIP 放在 `--device`，不使用生成模型的 `--device-map`。

后续训练 erase 并生成／计算 benchmark 指标：

```bash
python DiT/detect_erase/train.py --run-dir "$RUN_DIR" --stage erase --prompt-space soft --track
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --cases all --track
```

`--phase sample` 仅生成，`--phase evaluate` 仅评测已生成图片。I2P 使用 NudeNet（Q16 可选），IP/style 使用 CS、FID、VLM，object 使用 ResNet Top1/Top3，MSCOCO 使用 CS、reference CS 和 FID。参考图复用 `benchmark/original/<模型>/` 中完整协议匹配的图片。完整双卡训练和 benchmark 仍可运行 `bash DiT/detect_erase/all.sh`；当前仅检测实验使用上面的独立命令即可。

`--cases all` 包含 MSCOCO；也可单独指定 `--cases mscoco`（case ID 为 `mscoco_nudity`）。它复用 benchmark 中 nudity 的保留能力协议，默认取 `benchmark/dataset/mscoco.csv` 前 1000 条 caption，擦除目标仍为 nudity；默认与全部已训练 detector 比较，无需新增训练数据。仅检测时把这些 caption 作为无关文本对照，统计误触发率；完整采样时同样经过门控，计算 CS/FID，并复用协议一致的参考图。

```bash
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --cases mscoco \
  --detect-only --mscoco-prompt-count 1000 --track
```

`--mscoco-prompt-count` 可缩小实验；生成并计算 FID 时至少取 2 条。双卡脚本将 MSCOCO 与 i2p/style 放在同一张卡。

重新测试已有 prompt，结果单独保存（包含 MSCOCO）：

```bash
cd /home/shx/code/ce-new
TRAIN_RUN="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-detect-20261112-202026"
TEST_RUN="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-detect-all-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python DiT/detect_erase/sample.py \
  --run-dir "$TRAIN_RUN" --output-dir "$TEST_RUN" \
  --cases all --detect-only --all-detect-prompts \
  --similarity cosine --multi-concept top1 --batch-size 16 --device cuda:0 \
  --track --swanlab-mode local --swanlab-name detect-all-benchmark
```

所有结果位于 `$TEST_RUN/exps/<case>/`，汇总表、日志和 SwanLab 在 `$TEST_RUN` 下。无需重新训练或复制 checkpoint，`--run-dir` 始终指向原训练目录。完整图片采样的续跑／单独评测也需保留同一组 `--run-dir` 和 `--output-dir`。

**保存结构**：模型名和开始时间仅在根目录出现一次。训练目录和各总汇表留在根目录，各 case 的检测、图片、指标和耗时统一放在 `exps/`。每完成一个概念立即单独保存 checkpoint。

```text
results/sd35_medium-<开始时间>/
├── run.json                       # 配置与逐概念 checkpoint 索引
├── train_detect/
│   ├── nudity.pt, pikachu.pt, …    # 每概念独立 CPU FP32 参数，不保存模型权重
│   ├── training.json
│   ├── loss.csv, epochs.csv, training_summary.json  # --track 时
│   └── ti_images/nudity/           # TI-flow 才创建
├── train_erase/                    # 训练 erase 后才创建
├── detect_summary.csv/json         # 全部 case 的检测汇总
├── exps/
│   ├── i2p_nudity/
│   │   ├── detect_manifest.json
│   │   ├── detect_gates.jsonl      # scores、matched、selected
│   │   ├── detect_records.csv     # 每条文本及其真实标签、检测结果
│   │   ├── detect_summary.json
│   │   ├── detect_by_prediction.csv  # 全量 top1 分布
│   │   ├── detect_by_concept.csv
│   │   ├── detect_by_detect_prompt.csv
│   │   ├── detect_plots/           # 混淆矩阵和比率图，各有 PNG/PDF
│   │   └── images/, manifest.json, metrics.json, sampling_time.json, …
│   ├── mscoco_nudity/             # MSCOCO 保留能力及误触发对照
│   ├── ip_pikachu/
│   └── …                          # 其他 case 同级
├── logs/
└── swanlog/
```

Checkpoint 格式为 `detect-erase-concept-v2`。Soft 文件保存 `<stage>_soft_tokens`（每个 CLIP 一个 `[1, hidden_size]` 张量）、空间、固定文本和模板。Hidden detect 保存 `detect_pooled`，完整序列在需要 L2/TI 时由固定概念生成；hidden erase 保存 `erase_sequence` 和 `erase_pooled`。同时保存概念、阶段、loss、阈值、更新数等信息。Detect 与 erase 可采用不同空间，运行时分别编码。

**检测统计**以实际触发的 selected 为准。日志和逐条 CSV 同时记录全部候选分数、`predicted`（全量 top1）和 `selected`（实际触发）；阈值模式下还保留 matched 列表。

- `detect_by_prediction.csv` 统计各候选被预测的次数和比率；该表覆盖全部已加载概念。JSON 同时记录 candidate_targets 和 threshold_applied。
- 每个 case 擦除目标一对其余的 TP/TN/FP/FN，以及 precision、recall、F1、FPR、FNR、binary accuracy。判成其他目标同时产生正确目标的 FN 和错误目标的 FP；汇总计数的分母是“已记录文本数 × case 目标数”。预测为 case 外概念不触发擦除：目标输入计漏检，无关输入计正确不触发。
- 保留正确触发率、其他目标触发率、完全未触发率、无关文本误触发率和 exact-set accuracy。分母为 0 记录 null；缺少日志的输入单独计为 missing，不充当 TN。
- I2P 按 `benchmark/dataset/real_nude_mapping.csv` 的 sample_id＋seed 对齐，125 条 `is_real_nude=1` 为正，其余为无关文本；标签只用于评测。IP/style/object 使用 benchmark concept 标签；MSCOCO 按保留能力对照记为非目标输入。
- 完整采样加 `--track` 也保存同样统计及 `trigger_plots/`，文件前缀为 `trigger_`，原始门控日志为 `gates.jsonl`。缓存图片对应日志缺失时显示 coverage；重复日志取最后一次。

**训练追踪**由 `--track` 开启：loss、学习率、梯度／参数范数、batch 大小、更新数；CLIP 分类额外记录训练批次准确率和正／负平均余弦；flow 记录时间步。每轮 epoch 同步 GPU 后计时，记录完整／部分 epoch 和总时间，TI 图生成、模型加载、预编码及保存计入总时间的 outside-epochs 部分。

SwanLab 默认 local，需要 `pip install 'swanlab[dashboard]'`；可选 `--swanlab-mode offline/online`。看板命令：`swanlab watch "$RUN_DIR/swanlog"`。未开启追踪时不导入 SwanLab。

仍需实验选择的配置：学习率、温度、训练时长、阈值、soft template、TI 图数量和生成步数，以及 top1/mean。当前采用上表默认值，不通过最终图像指标自动调参。

### 训练数据集来源与构造

目标取自 `benchmark/configs/dit_cases.json`，共 11 个。数据分别在 `dataset/detect/`、`dataset/erase/`，每个目录按 i2p、ip、style、object 分四个 JSON。

| 类别 | 目标 | Detect 每概念 | Erase 每概念 |
|---|---|---:|---:|
| i2p | nudity | 50 条 | 50 对 |
| ip | Pikachu、Snoopy | 50 条 | 80 对 |
| style | Van Gogh、Monet、Picasso | 50 条 | 30 对 |
| object | church、golf ball、parachute、cassette player、banana | 50 条 | 80 对 |

Detect 每概念 30 正样本＋10 相近负样本＋10 普通负样本。I2P 由 AI 直接编写成人裸体、穿衣成人和无关场景；IP/object 使用 Textual Inversion 对象模板 `dataset/personalized.py`，style 使用 `dataset/personalized_style.py`，填入目标／负概念。去重后不足则循环，style 17 种模板因此有重复；负概念保存在 JSON 的 `similar_concepts`、`ordinary_concepts`。CLIP loss 使用全部 50 条，TI-flow 只取 30 条正样本生成图像。

Erase 的 I2P 为 50 对共享上下文的成人场景，仅将 nude 换成 clothed。IP/object 使用 benchmark 全部 80 个 ImageNet 模板，style 使用全部 30 个 painting 模板；分别填入 target 与 `anchor=""`，空 anchor 保留模板上下文。按约定直接复用测试模板。

共 550 条 detect 文本、700 对 erase 文本。固定场景和负概念均保存在生成脚本，复现：`python DiT/detect_erase/dataset/generate.py`。
