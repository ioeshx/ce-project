# ESD统一实验适配记录

更新：2026-10-24。原始提交：`2ff19a10d6084fababbdcb75e10deb56bd21fddb`。

## 1. 支持的实验

五类实验均已实现，共9个独立模型。ESD需要梯度微调，不是training-free方法。每个case从原始SD1.4开始；多目标在同一个学生模型、同一个优化器和固定原模型教师上训练。

| 独立脚本 | 擦除配置 | 正式评测 |
|---|---|---|
| `scripts/benchmark_instance.sh` | Snoopy；Snoopy+Mickey；Snoopy+Mickey+SpongeBob，3个模型，默认ESD-x | 每模型5个角色共400条提示，seed 0–9，共4,000张；逐概念及目标/非目标CS、macro FID |
| `scripts/benchmark_style.sh` | Van Gogh、Monet、Picasso分别擦除，3个模型，默认ESD-x | 每模型3个画家共90条提示，seed 0–9，共900张；逐概念及目标/非目标CS、macro FID |
| `scripts/benchmark_i2p_q16.sh` | 默认联合擦除`violence;nudity;harm`，默认ESD-u | 4,703条原始I2P提示/seed，Q16检测全部图片，保存不当图片数和比例 |
| `scripts/benchmark_i2p_nudenet.sh` | 默认擦除`nudity`，默认ESD-u | 生成4,703张图，NudeNet检测sexual子集931张；阈值0.6、benchmark默认8个EXPOSED类别，保存图片数/比例、框数和类别明细 |
| `scripts/benchmark_mscoco.sh` | 一个ESD-x模型联合训练CIFAR-50全部目标 | 原始MS-COCO前1,000条caption，seed 0–999；CS和相对原模型生成图的FID |
| `scripts/benchmark_all.sh` | 依次执行上述五类、全部9个case | 训练、采样、指标、文本表格和PNG/PDF |

instance/style/MS-COCO使用`benchmark.experiment_cases.CASE_CONFIGS`指向的实际JSON，目标组合不在Shell中重复定义。CIFAR-50由`mscoco.json`中的`targets_from`读取`benchmark/dataset/prompt_templates.py`的`CIFAR50_classes`；不使用CIFAR图片，不将50个名称拼成一条长prompt。instance/style不再逐组附带COCO实验。

`i2p.json`仅指定评测协议，没有擦除目标。这里选择与已接入UCE相同的Q16目标`violence;nudity;harm`和NudeNet目标`nudity`，便于比较；不宣称这些词覆盖或成功擦除了I2P全部七类内容。可用`--q16-targets`、`--nudenet-targets`覆盖，多个概念用分号分隔。两项默认目标不同，独立保存模型、图片和CSV行。

## 2. 多目标支持与核心改动

### 原仓库的情况

新版`esd_sd.py`/`utils/esd_trainer.py`默认只准备一个`erase_concept`字符串的上下文。原仓库保留的`alt_esd_diffusers.py`已经有“解析多个概念、每步采样一个目标”的训练方式。因此，多目标不需要另设计损失或改用顺序合并模型。

### 唯一的原有源码修改

`utils/esd_trainer.py`的`run_esd_training`增加可选`adapter=None`参数，并在未提供时仍调用原来的`get_adapter(config.family)`。仅这两行发生变化；原有SD/SDXL/FLUX入口调用方式保持兼容。目的：让新入口注入SD训练适配器，不复制或替换原生训练循环。

### 新适配层中的训练行为扩展

- `JointSDAdapter.prepare_context`分别为每个目标调用原生SD适配器准备文本上下文，保留其原有token编码和教师/学生结构。
- 每步仅选一个目标上下文，再调用原生`training_step`。按洗牌循环遍历全部目标；这是均衡的多目标调度，区别于旧版独立均匀随机抽样，记录在metadata中。
- 所有目标共享一个学生参数集和Adam优化器。**不是对每个目标分别训练再合并，也不是把名称逗号连接当成一个概念。**
- 保留原生模块选择、MSE、Adam、随机扩散时间步、教师参数切换和学生梯度步骤。当前clone在`erase_from=None`时的目标为`epsilon_erase - negative_guidance * (epsilon_erase - epsilon_null)`，本适配未改写这一公式。
- 不将benchmark的`erasure_anchor`映射成ESD的`erase_from`：后者表示属性编辑的条件，并不是UCE式替换anchor。ESD这里保持原生`erase_from=None`及空negative prompt。

以上目标调度属于必要的训练适配扩展，已明确记录；没有新增损失、正则项或改变网络结构。五类实验没有因需要大幅算法修改而跳过，但接口支持不保证实际擦除强度。

## 3. 训练预算及采样配置差异

默认学习率`5e-5`、negative guidance为2、训练batch=1，采用原生SD默认值。每目标200次更新，因此三个instance模型分别为200/400/600步，各style为200步，Q16默认600步，NudeNet为200步，CIFAR-50为10,000步。使用`--iterations`可指定总步数，或用`--iterations-per-concept`改变每目标预算；总步数必须至少覆盖全部目标一次。训练seed默认为1234，保存每个目标实际被采样的次数。

`--train-method`可覆盖默认变体，支持`esd-x`、`esd-u`、`esd-all`、`esd-x-strict`和`selfattn`。方法名包含变体，ESD-x和ESD-u的汇总行不会同名覆盖。

| 参数 | 当前clone原生SD CLI默认 | 本适配默认及原因 |
|---|---|---|
| 优化中生成latent的scheduler | 继承基础模型scheduler | 使用共享profile的DPM-Solver++，匹配本工作区要求 |
| 优化rollout步数/CFG | 50步/3 | 共享profile的20步/7.5，可显式`--training-steps`、`--training-guidance-scale`覆盖 |
| 优化模型精度 | bfloat16 | 默认float32，避免Adam直接更新float16参数的精度问题；也可显式`--training-dtype bfloat16` |
| 优化初始噪声 | 原生设备generator/模型dtype | 每个record从CPU float32噪声生成，再cast到优化模型dtype/设备 |
| 正式部署采样 | 上游独立生成脚本的设置 | 固定使用benchmark共享DPM20、CFG7.5、512×512、CUDA float16、空negative prompt、attention slicing |

优化精度作为明确例外保存在检查点`settings.training_protocol`/`training_exception`、原生safetensors metadata及benchmark清单中；不影响部署采样精度。优化rollout复用原生`esd_sd_call`，只在派生pipeline中重载初始latent准备函数，调用`benchmark.sampling_profile.initial_noise`。scheduler也由共享profile构造，不改变原有SD采样辅助文件。

本文的默认训练配置是当前工作区统一协议下的ESD比较配置，不将其称为原论文所有超参数的逐项复现。优化配置与部署配置分别保存，避免把训练float32标成部署float16。

## 4. 新增/修改的文件

| 文件 | 修改内容/目的 |
|---|---|
| `utils/esd_trainer.py` | 为原生训练入口增加可选适配器参数，其他训练代码不变 |
| `esd_benchmark/training.py` | 多目标上下文调度、共享噪声/采样配置、调用原生训练、目标覆盖统计、原子检查点与完成性检查 |
| `esd_benchmark/protocol.py` | 读取benchmark实际目标和协议，组织训练预算、变体及来源记录 |
| `esd_benchmark/adapter.py` | 检查并加载原生选中参数的部分权重，再使用benchmark原有采样器 |
| `esd_benchmark/runner.py` | CLI参数、五类实验和阶段编排，隔离训练/采样/指标进程，保存不可变计划 |
| `esd_benchmark/i2p.py` | 独立调用benchmark的Q16或NudeNet；复用其CSV/表格/绘图，不需要加载另一检测器 |
| `esd_benchmark/__init__.py`、`__main__.py` | 定位本工作区及模块入口 |
| `scripts/benchmark_{instance,style,i2p_q16,i2p_nudenet,mscoco}.sh` | 五个独立启动脚本，透传CLI参数 |
| `scripts/benchmark_all.sh` | 一次依次运行所有五类实验 |
| `requirements-benchmark.txt` | SD实验和共享评测所需依赖，无需torch_xla或其他架构权重 |
| `tests/test_benchmark_integration.py` | 实际微型Diffusers网络的50目标训练、检查点加载、九组清单、脚本及独立检测测试 |
| `README.md` | 增加本适配说明链接 |

上述新代码、脚本、本文档均保存在ESD目录；没有修改`benchmark/`、UCE或其他baseline。通用评测与I2P结果发布接口复用benchmark；I2P独立评测包装沿用UCE接入时的实现，但在本目录内独立保存，没有运行时的跨baseline依赖。

## 5. 运行命令

```bash
cd /home/shx/code/ce-new/baseline/ESD
pip install -r requirements-benchmark.txt

# 只检查命令，不训练或产生实验结果
bash scripts/benchmark_all.sh --device cuda:0 --offline --dry-run

# 完整运行五类实验，共9个模型
bash scripts/benchmark_all.sh --device cuda:0 --offline

# 独立启动
bash scripts/benchmark_instance.sh --device cuda:0 --offline
bash scripts/benchmark_style.sh --device cuda:0 --offline
bash scripts/benchmark_i2p_q16.sh --device cuda:0 --offline
bash scripts/benchmark_i2p_nudenet.sh --device cuda:0 --offline
bash scripts/benchmark_mscoco.sh --device cuda:0 --offline
```

可从其他工作目录使用绝对路径调用；支持`PYTHON=/path/to/python`。`--offline`要求SD1.4、对应CLIP及FID/检测器资源已在本地；省略它允许库下载缺失模型资源。Q16官方学习参数需单独准备，缺失时直接报错：

```bash
cd /home/shx/code/ce-new
python -m benchmark download --resource q16 --include-backbone
python -m benchmark download --resource clip
python -m benchmark download --resource fid
python -m benchmark download --resource nudenet
```

检测器本地路径参数与UCE相同：`--q16-prompts`、`--q16-clip-path`、`--nudenet-weights`；NudeNet还支持`--nudenet-threshold`。Q16使用官方ViT-L/14及匹配学习参数；NudeNet使用当前benchmark的3.4.2/8类/0.6协议，不声称复现ESD README链接的历史NudeNet版本。

```bash
cd /home/shx/code/ce-new/baseline/ESD
# 自定义预算、实验名和保存目录
bash scripts/benchmark_mscoco.sh --device cuda:0 --offline \
  --iterations-per-concept 200 --run-root runs/esd-cifar50-oct24 \
  --method-prefix ESD-cifar50-oct24

# 已准备并完成采样后只重新计算指标，其他参数保持一致
bash scripts/benchmark_mscoco.sh --device cuda:0 --offline \
  --iterations-per-concept 200 --run-root runs/esd-cifar50-oct24 \
  --method-prefix ESD-cifar50-oct24 --phase evaluate
```

支持`--phase all|erase|benchmark|prepare|sample|evaluate`；`erase`在ESD中表示微调阶段，`benchmark`表示已有检查点的prepare+sample+evaluate。默认每组串行，出错会停止，可修复后以同一配置重跑。

## 6. 结果位置、续跑与协议保护

```text
baseline/ESD/runs/benchmark-sd14/<case>/checkpoint/
  esd.safetensors         # 原生erasing-esd-v2部分参数和metadata
  metadata.json           # 完整目标、预算、变体、训练协议、各目标次数、来源校验和
baseline/ESD/runs/benchmark-sd14/<case>/plans/
  <method>/<task-采样概要-时间>/<task>.json
  <method>/<task-采样概要-时间>/<task>-manifest.json

benchmark/results/sd14/{instance,style,i2p,mscoco}-metric.csv
benchmark/results/sd14/{instance,style,i2p,mscoco}-metric.txt
benchmark/results/sd14/<task>/<method>/images/<可读采样名-时间>/
benchmark/results/sd14/<task>/<method>/evaluations/evaluation-<时间>/
benchmark/results/sd14/plots/<task>/*.png 和 *.pdf
benchmark/original/<task>/<可读采样名-时间>/
```

默认检查点位于`runs/benchmark-sd14`。同配置会复用完整检查点；训练中断后不续接Adam状态，重跑该未完成模型时从基础模型重新训练。修改训练参数/目标/源码须选择新的`--run-root`，已有检查点不被覆盖。`--method-prefix`用于并存多轮CSV结果。采样只补齐缺图/坏图，评测不需要重复训练或生成。

`--output-dir`改变指标和方法生成图的根目录，`--original-dir`/`--cache-dir`改变共享参考图和FID特征缓存目录；相对路径按调用时的工作目录解析。所有命名可读，校验和只存入JSON/CSV，不作为文件或目录名。

正式生成完全由benchmark采样器完成，包括每条record独立CPU float32初始噪声再cast、DPM20、CFG7.5、512×512、CUDA float16、空negative prompt和attention slicing；训练模型和原始参考模型使用同一部署协议。CLIP为默认base-patch16，FID对比同提示/seed的原模型生成图。相同协议的既有COCO参考缓存可以复用。

显式CPU小规模检查使用`--device cpu --sampling-dtype float32`，训练dtype也必须为float32。优化覆盖使用`--training-steps`、`--training-resolution`、`--training-guidance-scale`；部署覆盖使用`--steps`、`--height`、`--width`、`--sampling-dtype`；缩小概念评测使用`--prompts-per-concept`/`--seeds`。这些设置均记录在清单，应使用独立实验名和结果目录。

I2P的两个脚本只加载选定检测器，另一项显示N/A。保存逐图预测/概率、NudeNet原始框和接受框以及类别框数。缺少资源、图像不完整、预测不完整或训练权重非有限值时，不写入成功分数；同名方法仅在评测成功后更新CSV，其他方法保留。历史清单不会因新参数而重新指向另一协议的图像。

## 7. 验证范围

2026-10-24结果：本目录6项集成检查和共享sampling profile的5项检查全部通过；六个Shell入口的语法、跨目录dry-run及参数解析已覆盖。原有源码改动经git diff核对仅为训练入口的可选适配器注入。

```bash
cd /home/shx/code/ce-new/baseline/ESD
python -m unittest discover -s tests -v
```

本目录集成检查覆盖：九组完整提示/seed、默认变体和预算、CIFAR-50完整目标列表、I2P的4703/931筛选、六个Shell入口跨工作目录dry-run/参数解析、共享噪声与batch独立性、部分权重加载、历史计划隔离、独立Q16/NudeNet结果发布和绘图，以及失败保留已有CSV。

另使用实际微型CLIP、VAE、UNet及DPM scheduler，在CPU上通过原生`run_esd_training`、Adam和ESD损失完成50个目标的一次ESD-x联合训练循环，并验证nudity的ESD-u优化路径，核对目标覆盖、参数确实更新、权重有限及检查点续用。该网络不是预训练SD1.4，测试用检测预测也不是实测指标；所有测试图像、权重和CSV均在临时目录并自动清理。

未启动正式SD1.4 GPU微调、数千张全量生成或正式基线指标计算。本文证明接入与数值链路可运行，实际擦除效果和保持能力需运行脚本后评估。
