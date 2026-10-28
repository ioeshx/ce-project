# ESD 的 DiT 实验适配

默认 **SD3.5 Medium**，另支持 **FLUX.1-dev**。新增两个 Python 文件、四个单项脚本和一个总入口，均保存在 ESD 目录。未修改公共 `benchmark/`、原生 `utils/` 训练代码或已有 SD1.4 入口，没有新增测试文件。

FLUX 直接复用当前仓库的 `FluxESDAdapter.training_step` 和公共训练循环。**SD3.5 是本地新增的 ESD 训练适配，不是作者提供的官方 SD3.5 实现**；沿用相同的教师／学生目标、参数切换、MSE 和 Adam，只适配 SD3 的文本条件、时间步和 Flow Matching rollout。ESD 需要梯度训练，不属于 training-free 方法。

## 支持的实验

运行时读取 `benchmark/configs/dit_cases.json`、`dit_sampling.json`、`dit_evaluation.json`，与已有 DiT benchmark 保持一致。

| 实验组 | case | 擦除目标 | 评测内容 | SD3.5 / FLUX 接入 |
|---|---|---|---|---|
| I2P | `i2p_nudity` | nudity | sexual 931 张＋COCO 1K，NudeNet、COCO CS/FID | 已实现 |
| IP | `ip_pikachu` | Pikachu | Snoopy/Pikachu/Mickey，960 张，CS/FID、Qwen 单标签分类 | 已实现 |
| IP | `ip_snoopy_pikachu` | Snoopy、Pikachu | 同上，960 张 | 已实现，多目标联合训练 |
| style | `style_van_gogh` | Van Gogh | 三位画家，270 张，CS/FID、Qwen 单标签分类 | 已实现 |
| style | `style_monet` | Monet | 同上，270 张 | 已实现 |
| style | `style_picasso` | Picasso | 同上，270 张 | 已实现 |
| Object | `object_church` | church | ImageNet-10，共 500 张，官方 ResNet-50 V2 Top-1/3 | 已实现 |
| Object | `object_multi5` | church、golf ball、parachute、cassette player、gas pump | 同上，500 张 | 已实现，五目标联合训练 |

每个模型共 **8 个训练 case、9 项评测任务、5,661 张方法图**。I2P 和 COCO 使用同一个 nudity 检查点；IP/style/Object 不额外训练或评测 COCO。这里不使用旧 SD1.4 的 CIFAR-50→COCO 设置。

默认 I2P 不启用 Q16。`--with-q16` 改为全量 4,703 条 I2P 并启用 Q16，NudeNet 仍检测 sexual 子集。IP seeds 0–3、style 0–2、Object 0–49，均读取现有配置。I2P/COCO 评测 prompt 不用于训练，训练只编码 case 中的目标名称和空字符串。

| 模型 | 正式采样精度 | 分辨率 | 步数 | guidance | T5 长度 |
|---|---|---|---:|---:|---:|
| `sd35_medium` | BF16 | 512×512 | 40 | 4.5 | 256 |
| `flux1_dev` | BF16 | 512×512 | 28 | 3.5 | 512 |

正式采样原封不动使用公共 `DiTAdapter`：基础模型自身的 FlowMatchEulerDiscreteScheduler、逐记录 CPU FP32 初始噪声、原生 FLUX latent 打包。原模型参考图可跨 case 共享。训练、部署采样的参数分别记录，不将训练中的随机 rollout 标成正式评测协议。

## 启动命令

```bash
# 默认优先 SD3.5 Medium，顺序执行四项实验
bash /home/shx/code/ce-project/baseline/ESD/scripts/benchmark_dit_all.sh \
  --device cuda:0 --offline

# FLUX.1-dev；gradient checkpointing 作用于训练
bash /home/shx/code/ce-project/baseline/ESD/scripts/benchmark_dit_all.sh \
  --model flux1_dev --device cuda:0 --offline --gradient-checkpointing

# 只展开 case 和命令，不加载模型、不创建输出
bash /home/shx/code/ce-project/baseline/ESD/scripts/benchmark_dit_all.sh --dry-run
```

| 脚本 | 内容 |
|---|---|
| `scripts/benchmark_dit_i2p.sh` | nudity 训练，I2P＋COCO |
| `scripts/benchmark_dit_ip.sh` | 两个 IP case |
| `scripts/benchmark_dit_style.sh` | 三个 style case |
| `scripts/benchmark_dit_object.sh` | 两个 Object case |
| `scripts/benchmark_dit_all.sh` | 上述全部，顺序执行 |

所有脚本透传相同参数，可从任意目录调用。显式本地路径相对于调用命令时的目录；`PYTHON` 可选择解释器。`--base-model /本地目录` 可指定与 `--model` 架构对应的 Diffusers 模型，原模型评测也使用同一来源。

`--cases` 可指定单个 case 或逗号组合，`--tasks mscoco` 可单独选择 nudity 检查点的 COCO 任务。`--phase all` 为默认值，依次训练、准备清单、采样、评测；其余阶段为 `train`、`prepare`、`sample`、`evaluate`。后三者使用已有检查点，sample/evaluate 使用匹配的已准备计划。分阶段运行须保持相同配置与目录参数。

`--offline` 只读取本地模型和权重，资源不完整时去掉该参数。依赖为 `requirements-benchmark.txt`，CLIP、FID、NudeNet、Qwen 和 ResNet 权重准备见 `benchmark/DIT.md`。支持 `--vlm-model`、`--vlm-device`、`--resnet-weights-path` 等资源路径；`--skip-vlm` 显式形成不含 VLM 的协议。

## 训练设置和核心适配

| 参数 | 默认值与说明 |
|---|---|
| 训练变体 | `esd-x`；可选 `--train-method esd-x-strict` |
| 训练预算 | 每目标 1,400 次更新；双 IP 为 2,800 步、五 Object 为 7,000 步，其余各 1,400 步 |
| 预算覆盖 | `--iterations` 指定 case 总步数，或 `--iterations-per-concept` 指定每目标预算；总步数至少覆盖全部目标一次 |
| 优化器 | 原生 Adam，`lr=1e-4`，训练 batch=1 |
| 负向引导 | `negative_guidance=1.0`，与当前 FLUX CLI 一致，可覆盖 |
| 训练精度 | BF16，与原生 FLUX 一致；可用 `--training-dtype float32`，不改变部署采样精度 |
| rollout | 默认使用各模型的 40/28 步、4.5/3.5 guidance、512 分辨率、256/512 T5 长度 |
| FLUX 直接预测 guidance | 1.0，与原生 FLUX 一致；由 `--training-guidance` 覆盖，与 rollout 的 3.5 分开 |
| 随机种子 | 1,234；目标顺序采用洗牌循环，rollout 时间步和 seed 沿用 ESD 的随机采样方式 |
| 编码设备 | 默认 CPU，`--text-device` 可覆盖；条件编码完成后释放文本编码器 |

默认学习率、负向引导和单目标预算采用当前 FLUX 入口设置；SD3.5 使用同一组训练默认值，是本地迁移的实验设置，不称为其论文官方配置。支持 `--training-steps`、`--training-resolution` 显式改变训练 rollout；`--steps`、`--height`、`--width` 只改变正式采样。

核心行为：

- 每个 case 从原模型重新开始；所有目标共享一个学生、一套 Adam 状态和固定教师参数，每步只选择一个目标，不是训练多个模型再合并，也不是把目标名称拼成一条 prompt。
- 通过原生 `PreparedComponent` 在教师／学生选中参数之间切换，无须保留第二个完整 denoiser。只保存训练参数的部分权重，末轮结果和逐目标更新次数写入 metadata。
- `esd-x` 沿用原生 FLUX 的“名称包含 attn 的 Linear 层”选择，含联合注意力的图像／文本分支；迁移到 SD3 使用相同选择。`esd-x-strict` 则沿用原生规则，只选名称含 `to_k` 或 `to_v` 的注意力投影，包含其 bias；不额外加入 `add_k_proj/add_v_proj`。这些并不等同于 SD1 的独立 cross-attention 层划分，未添加 ESD-u 变体。
- FLUX 的单步实现直接调用上游代码。SD3 使用原生双 CLIP＋T5 条件，按原生动态 shift、CFG 和 Euler 步进生成所选时间步之前的 latent；原始时间步直接输入 SD3，而 FLUX 按原生代码除以 1,000。
- 保留当前仓库、`erase_from=None` 时的目标公式：`v_target = v_erase - negative_guidance * (v_erase - v_null)`。不替换成另一种负向引导公式；学生在目标条件下拟合该目标，MSE 和 Adam 使用原训练循环。
- ESD 不使用 UCE 式替换 anchor，不把 `dit_target_anchor.json` 的 dressed／Artist Style 等词传入 ESD 的 `erase_from`。后者属于属性编辑条件，本实验保持 `None`。

训练初始噪声沿用 ESD 的设备 generator，训练 seed 在每步从 `[0, 2**15]` 抽取；与正式评测的固定 CPU 噪声区别记录在 `training_protocol`。BF16 训练保留上游参数／Adam 精度，没有另行引入 FP32 master weights、LoRA、量化或改变优化目标。

训练仅在 `--device` 对应的单设备执行；FLUX 完整微调还需要教师参数备份、梯度和 Adam 状态，不能按推理显存估算。`--gradient-checkpointing` 可减少训练激活，`--text-device cpu` 避免编码器占用训练显存。`--cpu-offload`、`--device-map`、`--max-memory` 仅控制后续公共采样，不能把训练自动变成 offload 或多卡训练。

## 文件与结果位置

| 文件 | 修改内容及目的 |
|---|---|
| `esd_benchmark/dit.py` | 新入口，集中解析参数、读取现有配置、编排训练与公共评测 |
| `esd_benchmark/dit_training.py` | 一个必要的派生适配器，支持多目标调度、SD3 训练步；调用原生训练循环并提供权重采样插件 |
| `scripts/benchmark_dit_{i2p,ip,style,object,all}.sh` | 四个独立入口和总入口，仅转发参数 |
| `README.md`、`BENCHMARK_ADAPTATION.md`、`requirements-benchmark.txt` | 添加 DiT 说明链接，更新依赖文件的适用范围注释 |
| `DIT_BENCHMARK_ADAPTATION.md` | 本文档，说明支持范围和迁移差异 |

没有修改原生 ESD 文件。唯一新增 class 继承现有 FLUX 训练适配器，实际用于注入原生训练循环；采样插件直接返回公共采样器，没有再定义采样 class。

检查点和计划：

```text
baseline/ESD/runs/benchmark-dit/<model>/<case>/checkpoint/esd.safetensors
baseline/ESD/runs/benchmark-dit/<model>/<case>/checkpoint/metadata.json
baseline/ESD/runs/benchmark-dit/<model>/<case>/plans/ESD-x/<task>-<采样概要>-<北京时间>/
```

统一结果：

```text
benchmark/results/<model>/<task>/<case>/ESD-x/
benchmark/results/<model>/<task>-metric.csv
benchmark/results/<model>/<task>-metric.txt
benchmark/results/<model>/plots/<task>/
benchmark/original/<model>/<task>/
```

IP 的 task 为 `instance`，其他为 `i2p`、`mscoco`、`style`、`object`。同一方法、同一 case 成功评测后替换该行，不同 case 分别保留。变体默认方法名为 `ESD-x` 或 `ESD-x-strict`，可用 `--method` 覆盖；更改训练参数或变体时需指定新的 `--run-root`。图片与计划采用可读名称，指纹只写入 JSON。

## 验证与限制

采用临时 CPU 检查，没有新增测试文件或正式实验指标：

- 两种模型的五个脚本语法和 dry-run 正常，展开为 8 个 case、9 项任务；实际数据规模、训练预算、Q16 开关、I2P/COCO 检查点共用均已检查。
- 微型原生 SD3 与 FLUX 的 `esd-x`、`esd-x-strict` 均完成双目标反向传播、保存、复用、加载和真实生成；确认教师权重固定、未选择参数不变、每个目标实际参与更新。
- SD3 的动态时间表和 CFG rollout 在所选时间步 0/1/2 与原生 pipeline 对齐，确认先生成到所选步之前，再计算教师／学生预测，学生分支保留梯度。
- 临时模型、图片和计划清理，不作为擦除效果证据。

当前 NVIDIA 驱动不可用，尚未运行真实 SD3.5 Medium/FLUX.1-dev 的完整 GPU 训练、BF16 训练或正式指标评测。SD3.5 的擦除强度和保留能力需要正式实验确认，多卡和采样 offload 也未在本机 GPU 验证。

实现依据：[作者 FLUX 入口](https://github.com/rohitgandikota/erasing/blob/main/esd_flux.py)、当前本地 `utils/esd_trainer.py`、[Diffusers 官方 SD3 文档](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)及本机原生 pipeline 源码。
