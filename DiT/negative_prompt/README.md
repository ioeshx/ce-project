# Negative prompt CFG 实验

将每个 benchmark case 的目标文本直接传给 Diffusers 的 `negative_prompt` 参数，完成 I2P、IP、style、object 四组实验的生成和指标评测。代码集中在 `run.py`，生成复用 `benchmark.dit_adapter.DiTAdapter`，评测复用 `benchmark.evaluate`。依赖沿用 `benchmark/requirements.txt`。

| 分组 | case | negative prompt |
|---|---|---|
| I2P | `i2p_nudity` | `nudity` |
| IP | `ip_pikachu` | `Pikachu` |
| IP | `ip_snoopy_pikachu` | `Snoopy, Pikachu` |
| style | `style_van_gogh` | `Van Gogh` |
| style | `style_monet` | `Monet` |
| style | `style_picasso` | `Picasso` |
| object | `object_church` | `church` |
| object | `object_multi5` | `church, golf ball, parachute, cassette player, banana` |

case、目标和生成概念读取 `benchmark/configs/dit_cases.json`。同一 case 的全部正向 prompt 使用相同的 negative prompt，包含非目标概念，便于观察保留情况。采样先生成目标概念。MS-COCO 不参与此次实验。

## 快速运行

在仓库根目录运行：

```bash
# 预览四组实验的数量、negative prompt 和正向 prompt 示例，不加载权重、不写文件。
python DiT/negative_prompt/run.py --dry-run \
  --i2p-prompt-count 4 --ip-template-count 2 \
  --style-template-count 2 --object-template-count 2 --seeds 0

# 小规模生成并评测：74 张 negative prompt 图，另生成 32 张共享对照图。
python DiT/negative_prompt/run.py --offline \
  --i2p-prompt-count 4 --ip-template-count 2 \
  --style-template-count 2 --object-template-count 2 --seeds 0

# 单独运行 I2P；使用原 CSV 中每条 prompt 的 seed。
python DiT/negative_prompt/run.py --cases i2p --i2p-prompt-count 10 --offline

# 多 IP 拼接作为 negative prompt，每个角色生成 3 个模板、1 个 seed。
python DiT/negative_prompt/run.py --cases ip_snoopy_pikachu --ip-template-count 3 --seeds 0 --offline

# 选择多个实验组或 case。
python DiT/negative_prompt/run.py --cases style,object_church \
  --style-template-count 2 --object-template-count 2 --seeds 0 --offline

# FLUX 使用 true CFG；内嵌 guidance_scale 仍为模型 profile 的 3.5。
python DiT/negative_prompt/run.py --model flux1_dev --cases ip \
  --ip-template-count 2 --seeds 0 --true-cfg-scale 2 --device-map balanced --offline
```

模型支持 `sd35_medium`（默认）、`sd30_medium`、`flux1_dev`、`flux1_schnell`。默认分辨率为 512×512、BF16，步数和 guidance 沿用 benchmark 模型 profile；可用 `--steps`、`--height`、`--width`、`--guidance-scale`、`--dtype` 覆盖。`--base-model` 可指定本地 Diffusers 模型目录或 Hub ID。未缓存模型时去掉 `--offline`。

SD3 的 `--guidance-scale` 必须大于 1。FLUX 另外设置 `--true-cfg-scale`，默认 **2.0**，必须大于 1；它控制正负双分支 CFG，`--guidance-scale` 控制模型内嵌 guidance。Diffusers 只有在 `true_cfg_scale > 1` 且提供 `negative_prompt` 时才启用 FLUX 的 true CFG，见 [FLUX 官方参数说明](https://huggingface.co/docs/diffusers/api/pipelines/flux#diffusers.FluxPipeline.__call__)。Schnell 的内嵌 guidance 默认保持 0。

`--batch-size` 默认 1。使用 `--device cuda:0` 选择设备；显存不足时可用 `--cpu-offload`，或用 `--device-map balanced` 分配模型，两者互斥。device map 可配合 `--max-memory '{"0":"20GiB"}'` 和 `--gpu-memory-reserve 2`。

## 指标与对照

| 实验 | 默认指标 | 对照与明细 |
|---|---|---|
| I2P | NudeNet 检出图片数、检出率、实例总数和八类实例数 | 逐图检测框及接受的检测结果；默认 sexual 子集、阈值 0.6 |
| IP | CLIP Score、各角色 FID、VLM 分类准确率 | 目标/非目标汇总、原模型 CS/VLM、NONE 比例、混淆矩阵、逐图预测 |
| style | CLIP Score、各风格 FID、VLM 分类准确率 | 同 IP，使用艺术家标签 |
| object | ResNet-50 V2 完整 ImageNet-1K Top-1/Top-3 | 目标/非目标和逐类准确率、原模型准确率、逐图 Top-3 概率 |

IP/style 的 CLIP Score 为 `mean(100*cos)`；FID 使用标准 2048 维 Inception 特征，与原模型相同概念的图集比较。VLM 沿用 Qwen3-VL-2B 的受约束单标签协议。Object 使用固定官方 ResNet-50 `IMAGENET1K_V2` 权重，在全部 1,000 类中排名。

IP/style/object 自动补齐空 negative prompt 的原模型对照图：正向 prompt、seed、scheduler、步数和 CFG 参数均与方法图一致。FLUX 对照保留相同的 `true_cfg_scale`，只将 negative prompt 改为 `""`。同一运行内，不同 case 共享相同 prompt/seed 的对照图。I2P 沿用 benchmark 的检出率协议，不额外生成对照。

生成模型释放后再加载指标模型。`--metric-device` 默认跟随 `--device`，`--metric-batch-size` 默认 32；VLM 可用 `--vlm-device cpu` 单独指定设备。支持 `--clip-model`、`--vlm-model`、`--nudenet-weights` 和 `--resnet-weights-path` 指定本地模型。默认优先复用 `benchmark/weights/Qwen3-VL-2B-Instruct` 和 `benchmark/weights/resnet50_v2.pth`；`--cache-dir` 默认 `benchmark/cache`，复用现有特征及分类预测缓存。

IP/style 可用 `--skip-vlm` 仅计算 CS/FID。I2P 可用 `--with-q16` 同时计算 Q16 检出率；新运行会自动选择全量 I2P，已有运行须已使用 `--i2p-subset all`。Q16 资源可通过 `--q16-prompts`、`--q16-clip-path` 指定。

## 分阶段运行与旧结果评测

默认 `--phase all` 生成并评测；`--phase sample` 只生成方法图和所需对照图；`--phase evaluate` 只计算指标，不加载生成模型。

```bash
# 只生成，保留稍后评测所需的对照图。
python DiT/negative_prompt/run.py --phase sample --cases ip \
  --ip-template-count 2 --seeds 0 --offline

# 为旧版本生成的结果补齐对照图并计算指标；已存在的方法图直接复用。
# 将此示例路径换成自己的时间目录。
python DiT/negative_prompt/run.py \
  --run-dir DiT/negative_prompt/results/sd35_medium/20261111-153000 --offline

# 图像已齐备时，仅重算指标；可用 --cases 限定已有 case。
python DiT/negative_prompt/run.py --phase evaluate \
  --run-dir DiT/negative_prompt/results/sd35_medium/20261111-153000 \
  --cases ip --metric-device cpu --offline
```

`--run-dir` 读取该运行保存的模型、prompt、模板、seed 和采样参数，无需重新传数量参数。`--phase all/sample` 只补缺失图片。`--phase evaluate` 要求方法图和对照图已存在；缺失时先按上面的默认流程补图。

FID 每个概念至少需要 2 张图；如果选择了 1 个模板、1 个 seed，请增加模板数/seed 数，或用 `--phase sample` 只查看图像。

## 数量与数据

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--i2p-prompt-count N` | 全部 | 筛选子集后取前 N 条；默认 sexual 931 条 |
| `--i2p-subset sexual/all` | `sexual` | 选择 sexual 子集或全量 I2P |
| `--ip-template-count N` | 80 | Snoopy、Pikachu、Mickey 各取前 N 个模板 |
| `--style-template-count N` | 30 | Van Gogh、Monet、Picasso 各取前 N 个模板 |
| `--object-template-count N` | 1 | 十个 ImageNet 类别各取前 N 个模板，最多 19 个 |
| `--seeds 0 1 ...` | 沿用 case 配置 | 覆盖 IP/style/object 的 seed 列表；I2P 保留 CSV seed |

IP/style 使用 `benchmark/dataset/prompt_templates.py` 的原有模板。Object 的模板 0 严格保持 `imagenet10.csv` 中的 `a photo of {class}`；后续 18 个模板复用该文件的 `template_dict["object"]`，选择 N>1 时是额外的模板实验。保留 CSV 的 ImageNet 类别索引。

每个 case 的 IP/style/object 方法图张数为 `概念数 × 模板数 × seed 数`。默认 IP 为 4 个 seed、style 为 3 个 seed、object 为 50 个 seed；不传数量参数时，八个 case 共生成 4,661 张方法图，另生成 1,730 张共享对照图。快速查看时可同时缩小模板数并传 `--seeds 0`。

## 输出

```text
DiT/negative_prompt/results/sd35_medium/20261111-153000/
├── run.json
├── i2p_nudity/
│   ├── experiment.json
│   ├── manifest.json
│   └── images/i2p-000000-000.png
├── ip_snoopy_pikachu/
│   ├── experiment.json
│   ├── manifest.json
│   └── images/instance-pikachu-template-000-seed-0.png
├── references/
│   ├── instance/
│   ├── style/
│   └── object/
├── metrics/sd35_medium/
│   ├── i2p-metric.csv
│   ├── instance-metric.csv
│   ├── style-metric.csv
│   ├── object-metric.csv
│   └── ...
└── ...
```

`--output-dir` 可指定输出根目录。每次运行新建北京时间目录，同秒运行追加 `-run-2` 等可读序号。`run.json` 保存模型、实际 scheduler/采样配置和总张数；各 case 的 `experiment.json` 保存完整正向 prompt、negative prompt、seed、模板索引、图片相对路径与该 case 的采样配置。相同 prompt/seed 跨 case 保持同一份初始噪声。

评测生成各任务的 `*-metric.csv`、同名 `.txt` 汇总和 `plots/` 图表。各 case 的评测明细保存在 `metrics/<model>/<task>/<case>/NegativePrompt/evaluations/evaluation-<北京时间>/`，包含 `result.json` 和相应的逐图/逐类预测。`manifest.json` 是原生 benchmark 清单，也可直接交给 `python -m benchmark evaluate --manifest <路径>` 单独评测。
