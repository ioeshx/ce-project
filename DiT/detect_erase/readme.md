## 一、算法与 loss

冻结 text encoder、DiT 和 VAE，仅学习 text encoder 输出位置的条件向量。每个概念分别保存一对 detect / erase prompt。

**Detect**：学习 CLIP pooled 向量 $p_i^d$，初始化为目标概念的 pooled 编码。SD3.5 拼接两个 CLIP 的 pooled 输出，FLUX 使用 CLIP pooled 输出。

$$
r_i(q)=\cos(f_{\mathrm{CLIP}}(q),p_i^d),\quad
\hat y_i=\sigma((r_i(q)-\tau)/T),\quad
\mathcal L_d=\operatorname{BCE}(\hat y_i,y_i).
$$

默认使用 `dataset/` 中每个概念的全部 50 条正负样本，分类训练只使用 CLIP，不计算 velocity。初始化时额外保存概念的完整 CLIP/T5 编码，供 L2 检测使用，该序列不参与优化。备选 `--detect-loss velocity` 只使用正样本：序列条件固定为概念编码，仅优化 pooled，拟合完整正样本文本 $q^+$ 的输出：

$$
\mathcal L_{d,v}=\mathbb E\|v_\theta(x_t,t,H(c_i),p_i^d)-v_\theta(x_t,t,q^+)\|_2^2.
$$

**Erase**：学习完整 `prompt_embeds` 和 `pooled_prompt_embeds`，初始化为原模型空文本编码，包含 CLIP/T5 对应条件。使用 `erase/` 中的完整配对描述：

$$
v_{\mathrm{CFG}}=v_\theta(x_t,t,p_i^e)+s\big[v_\theta(x_t,t,q^+)-v_\theta(x_t,t,p_i^e)\big],\qquad
\mathcal L_e=\mathbb E\|v_{\mathrm{CFG}}-v_\theta(x_t,t,q^a)\|_2^2.
$$

从随机噪声出发，用冻结基础模型的空负条件 CFG 推进轨迹。每个去噪步骤在**同一个** $x_t,t$ 上计算各分支，更新一次 prompt；停止 teacher 和轨迹的梯度，只通过当前学生分支反传。所有文本、种子和时间步共享同一个概念 prompt，不生成图像反馈。

**采样**：余弦分数达到阈值后，将对应 erase embedding 传入 Diffusers 的 `negative_prompt_embeds`、`negative_pooled_prompt_embeds`；未触发时使用原模型空文本条件。各 benchmark case 只检测该 case 的目标概念，保留概念的输入也经过同一检测逻辑。

## 二、超参数

| 参数 | 默认值 / 含义 |
|---|---|
| `--model` | `sd35_medium`；支持 benchmark 中的 SD3 / FLUX 模型 |
| `--detect-loss` | `clip`；可选 `velocity` |
| `--learning-rate` | `1e-3`，AdamW，weight decay=0 |
| `--epochs` | 5，每个概念、每个阶段分别计数 |
| `--max-steps` / `--steps`（训练） | 每个概念、每个阶段最多 2000 次优化更新；0 表示不设上限 |
| `--batch-size` | 1；velocity 训练表示同时推进的轨迹数量 |
| `--tau` | 0.2，训练和采样的余弦阈值 |
| `--temperature` | 0.1，检测 BCE 的温度 |
| `--seed` | 0；训练中持续抽取新的噪声种子 |
| `--sampling-steps`（训练） | 模型 benchmark 默认去噪步数；区别于优化更新数 |
| `--steps`（采样） | 默认继承训练配置 |
| CFG 系数 | SD3 继承 benchmark `guidance_scale`；FLUX `true_cfg_scale=2`，embedded guidance 继承 benchmark |
| `--dtype` | 继承模型配置，默认 BF16；可选 FP32；可学习参数和优化器状态使用 FP32 |
| `--gradient-checkpointing` | 默认开启，减少训练激活占用；可用 `--no-gradient-checkpointing` 关闭 |

epoch 和更新次数上限先到者结束训练。CLIP 检测有 50 条文本，batch=1、5 epochs 对应 250 次更新。Erase 每个去噪步骤更新一次，50 对文本 × 40 步已经对应 2000 次更新；完整训练 5 epochs 使用 `--max-steps 0`。

## 三、实现与可选配置

- **数据**：直接使用已有 detect / erase JSON。非 i2p 的数据 anchor 是 `template.format("")`，不等于整条空文本；erase prompt 自身仍由整条空文本编码初始化。测试模板按本次要求直接用于训练。
- **相似度**：默认 `--similarity cosine`，只比较 CLIP pooled；`--similarity l2` 比较完整序列和 pooled，计算全部元素的 RMS 距离，距离不超过 `--l2-threshold 0.2` 时触发。完整 detect 条件的序列固定为概念编码，只有 pooled 被学习。
- **多概念**：默认 `--multi-concept top1`，在达到阈值的概念中选择最高余弦／最低距离的一项；`mean` 对所有触发概念的 erase 条件取平均。
- **训练配置**：`--stage all/detect/erase`、`--cases`、`--targets`、`--limit-prompts`；`--resume --run-dir ...` 读取选中概念各自的文件继续训练，重新创建优化器；尚无文件的概念从初始化开始。Detect 和 erase 分别独立训练，采样直接按需加载。
- **保存**：一次实验共用 `results/<模型>-<开始时间>/`。`--run-dir` 指定该根目录；不传时训练自动创建新目录。重新训练新建根目录，续跑复用原目录。每完成一个概念立即保存其独立 checkpoint，不重写其他概念文件。
- **Benchmark**：默认生成后计算指标；`--phase sample` 只生成，`--phase evaluate` 只评测已有实验，不加载生成模型或 prompt 权重。I2P 使用 NudeNet（Q16 可选），IP/style 使用 CS、FID 和 VLM 分类，object 使用 ResNet Top1/Top3。
- **参考图**：直接复用 `benchmark/original/<模型>/` 中 prompt、种子和完整采样配置一致的图片，仅补齐缺失或损坏的图片。
- **并发**：两张卡共用同一 `--run-dir`，分别写各自训练阶段和实验；根目录配置、指标和触发汇总使用文件锁合并。
- **仍需实验选择**：学习率、温度、训练时长、检测阈值和平均组合的效果；代码不利用最终图像指标自动调整它们。

目录结构（所有 benchmark case 均与 `i2p_nudity/` 同级）：

```text
results/sd35_medium-20261112-024437/
├── run.json                         # 各阶段模型配置、逐概念 checkpoint 索引、实验列表
├── metrics.csv/json                 # 所有 case 的最新评测汇总
├── trigger_summary.csv/json         # 所有 case 的触发汇总
├── sampling_time.csv/json           # 各 case 最近一次采样时间
├── train_detect/
│   ├── nudity.pt
│   ├── pikachu.pt
│   ├── snoopy.pt
│   ├── van_gogh.pt
│   ├── ...                          # 每个概念一个文件，空格转为下划线
│   ├── training.json
│   ├── loss.csv                     # --track 时保存
│   ├── epochs.csv
│   └── training_summary.json
├── train_erase/                      # 与 train_detect 相同
├── i2p_nudity/
│   ├── images/
│   ├── manifest.json
│   ├── metrics.json                 # 完整指标和评测配置
│   ├── sampling_time.json
│   ├── gates.jsonl                  # --track 时保存
│   ├── trigger_summary.json
│   ├── trigger_records.csv
│   ├── trigger_by_concept.csv
│   ├── trigger_by_erase_prompt.csv
│   └── plots/                       # 指标图；逐图评测明细也保存在 case 内
├── ip_pikachu/
├── ip_snoopy_pikachu/
├── style_van_gogh/
├── style_monet/
├── style_picasso/
├── object_church/
├── object_multi5/
├── logs/                            # 自动追加控制台日志
│   ├── train_detect.log
│   ├── train_erase.log
│   ├── sample_i2p_style.log
│   └── sample_ip_object.log
└── swanlog/                         # 所有阶段的 SwanLab 记录
```

完整双卡实验：GPU 0 训练 detect，GPU 1 训练 erase；两边完成后，GPU 0 跑 i2p/style，GPU 1 跑 ip/object。脚本创建一次实验根目录，开启全部追踪：

```bash
bash DiT/detect_erase/all.sh
# 可选：MODEL=flux1_dev TRACK_MODE=offline bash DiT/detect_erase/all.sh
```

单独训练全部阶段并采样：

```bash
RUN_DIR="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
python DiT/detect_erase/train.py --run-dir "$RUN_DIR" --model sd35_medium --stage all --track
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --cases all --track
```

`--stage all` 分别写入两个训练目录，每个概念各一个 `.pt` 文件。目录内不保存打包的 `prompts.pt`，无需执行合并命令。

每个文件只含所属概念、所属阶段的参数和训练信息，张量保存为 CPU FP32：

- Detect：`target`、`stage`、`group`、`detect_sequence`（固定概念编码）、`detect_pooled`（学习向量）、`detect_loss`、`tau`、`temperature`、更新次数和样本数。
- Erase：`target`、`stage`、`group`、`erase_sequence`、`erase_pooled`、`cfg_scale`、更新次数和样本数。
- 两者均带 `schema=detect-erase-concept-v1`，不保存另一阶段的向量或完整模型权重。共享模型及采样配置集中保存在 `run.json` 的 `config.detect`、`config.erase`。

只继续训练一个概念，例如 nudity 的 detect（其他概念 checkpoint 保持不变）：

```bash
python DiT/detect_erase/train.py --run-dir "$RUN_DIR" \
  --stage detect --targets nudity --resume --epochs 5 --track
```

双卡采样或续跑（仅加载各 case 的 targets 对应的 detect/erase 文件，自动跳过完整图片）：

```bash
CUDA_VISIBLE_DEVICES=0 python DiT/detect_erase/sample.py \
  --run-dir "$RUN_DIR" --cases i2p,style --device cuda:0 --metric-device cuda:0 --track &
CUDA_VISIBLE_DEVICES=1 python DiT/detect_erase/sample.py \
  --run-dir "$RUN_DIR" --cases ip,object --device cuda:0 --metric-device cuda:0 --track &
wait
```

`CUDA_VISIBLE_DEVICES=1` 后，`cuda:0` 表示物理第 1 张显卡。`--phase evaluate --run-dir "$RUN_DIR"` 仅评测已有 case。

首次准备 case 时可配置 `--i2p-prompt-count`、`--ip-template-count`、`--style-template-count`、`--object-template-count`、`--seeds`、相似度和阈值。已保存 case 沿用其配置；重新选择这些生成配置需新建实验。`--batch-size`、设备和评测参数可在续跑时调整。

训练还可用 `--targets`、`--limit-prompts`、`--resume`；采样使用 `--run-dir` 定位训练结果，可用 `--dry-run`、`--skip-vlm`、`--with-q16` 和评测权重路径。每个 case 的采样缓存只记录实际使用的概念文件，其他概念的重新训练不影响它。

**可选追踪**：训练和采样均添加 `--track` 开启 SwanLab；默认关闭，不导入 SwanLab、不写追踪文件。默认 `--swanlab-mode local`，所有阶段的数据保存在实验根目录的 `swanlog/`（内部运行目录由 SwanLab 管理）。可配置 `--swanlab-project`、`--swanlab-name`；云端记录使用 `--swanlab-mode online`，提前执行 `swanlab login`。

Local 模式需要安装看板扩展；只有 SwanLab 核心包时，可用 `--swanlab-mode offline` 保存离线记录，之后用 `swanlab sync` 同步到云端。

```bash
pip install 'swanlab[dashboard]'
python DiT/detect_erase/train.py --cases i2p --track
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --cases all --track
swanlab watch /path/to/run/swanlog
```

本地看板用法见 [SwanLab 文档](https://docs.swanlab.cn/guide_cloud/offline-board.html)。

训练追踪保存 `loss.csv`、`epochs.csv`、`training_summary.json`，并同步到 SwanLab：

- 每次更新的 loss、学习率、梯度 L2 范数、参数 L2 范数、epoch、更新次数和 velocity 时间步。
- CLIP 检测的更新前训练批次准确率、正／负样本平均余弦分数；epoch loss 按批次样本数加权。
- 每个概念、阶段的每轮 epoch 秒数。因 `--max-steps` 提前结束时标记 `partial=true`；velocity 的 `examples` 表示轨迹状态数量。
- `total_seconds` 从模型加载前开始，到最终 checkpoint 保存和模型释放结束，包含文本预编码和追踪开销；`epoch_seconds` 为各 epoch 之和，`outside_epochs_seconds` 为模型加载、预编码、保存等其余时间。Epoch 边界同步参与训练的 GPU 后计时。

采样逐批记录实际 `selected` 的 erase prompt，各 case 结束后汇总到 SwanLab 和以下文件：

- 运行根目录：`trigger_summary.csv/json`，每个 case 的总数和比率。
- Case 目录：`trigger_summary.json`、`trigger_records.csv`、`trigger_by_concept.csv`、`trigger_by_erase_prompt.csv`，分别保存汇总、逐条判定、按输入概念和按 erase prompt 的统计。

| 指标 | 定义 / 分母 |
|---|---|
| `trigger_rate` | 使用任一 erase prompt 的数量 / 已记录输入数 |
| `target_correct_rate` | 选中了对应 erase prompt / 目标输入数 |
| `target_other_rate` | 选中了其他 erase prompt / 目标输入数 |
| `target_wrong_only_rate` | 只选中错误项、未选中对应项 / 目标输入数 |
| `target_no_trigger_rate` | 没选中任何项 / 目标输入数 |
| `non_target_false_rate` | 无关输入选中了任一项 / 无关输入数 |
| `overall_accuracy` | 目标输入仅选中对应项、无关输入未触发 / 已记录输入数 |

所有比率同时保存对应 `*_count`；分母为 0 时记录 `null`。Mean 模式可以同时选中正确项和其他项，故 `target_correct_rate` 与 `target_other_rate` 可重叠；correct、wrong-only、no-trigger 三项互斥。

IP/style/object 按输入概念与该 case 的 targets 判断。I2P 按 `real_nude_mapping.csv` 的 **sample_id＋seed** 对齐，仅 `is_real_nude=1` 的 125 条算 nudity，其余输入均算无关；`--real-nude-mapping` 可指定映射文件。此标签仅用于统计，不参与门控，也不重新运行 NudeNet。

继续采样时按 sample_id＋seed 保留最后一次门控记录，不重复计数。`--phase evaluate --run-dir ... --track` 可仅汇总已有门控记录；缺少记录的缓存样本显示 `missing_count` 和 `coverage`，不当作未触发，相关比率仅统计已记录样本。

## 4. 训练数据集来源与构造

目标概念取自 `benchmark/configs/dit_cases.json`，共 11 个。Detect 保存于 `DiT/detect_erase/dataset/detect/`，erase 保存于 `DiT/detect_erase/dataset/erase/`，各自按 i2p、ip、style、object 分为四个 JSON。训练默认读取这两个目录，可通过 `--dataset-dir`、`--erase-dir` 指定其他位置。

| 类别 | 目标概念 | Detect：每个概念 | Erase：每个概念 |
|---|---|---:|---:|
| i2p | nudity | 50 条 | 50 对 |
| ip | Pikachu、Snoopy | 50 条 | 80 对 |
| style | Van Gogh、Monet、Picasso | 50 条 | 30 对 |
| object | church、golf ball、parachute、cassette player、banana | 50 条 | 80 对 |

**Detect**：每个概念由 30 条正样本、10 条相近负样本、10 条普通负样本组成。
1. I2P 由 AI 直接编写独立场景：正样本描述成人裸体，相近负样本描述穿衣成人，普通负样本描述无关场景，不使用模板。
2. IP/object 使用 Textual Inversion 对象模板 `dataset/personalized.py`，style 使用风格模板 `dataset/personalized_style.py`，分别填入目标、相近和普通负概念。模板去重后按顺序使用，不足时循环；style 仅有 17 条不同模板，因此 30 条正样本包含重复。负概念保存在 `similar_concepts`、`ordinary_concepts` 字段中。默认 CLIP 分类使用全部 50 条，velocity loss 仅使用 30 条正样本。

**Erase**：
1. I2P 由 AI 编写 50 个独立成人场景，每对共享上下文，仅将 `nude` 替换为 `clothed`。
2. IP/object 使用 `benchmark/dataset/prompt_templates.py` 的全部 80 条 `imagenet_templates`，style 使用全部 30 条 `painting_templates`。按本次约定直接复用 benchmark 模板。每条模板分别填入 target 和空 anchor 得到配对文本；空 anchor 指 `anchor=""`，仍保留模板上下文，不是整句空文本。

总计 550 条 detect 样本、700 对 erase 文本。场景和负概念均已固定在生成脚本中，从仓库根目录执行以下命令可复现：

```bash
python DiT/detect_erase/dataset/generate.py
```
