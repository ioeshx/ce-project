# RAZOR的SD3.5 Medium / FLUX.1-dev benchmark适配

更新：2026-11-02。实际目录名为`baseline/RAZOR-cvpr2026/`。
此前适配评估基于“不允许大幅修改核心算法”而跳过；本次允许核心修改后，已实现两种模型的四组实验。

实现保留原扩散脚本的**第一个CLIP文本编码器编辑、forget/retain梯度比值评分、稀疏层选择及逐层扩展**，
并修复损失和停止条件。它是可运行的RAZOR DiT适配变体，不是未经修改的原脚本结果。
原`Razor-SD.py`保留作对照，新实现不导入其模块顶层加载模型/生成图的代码。

## 核心算法修改记录

核心集中在新增的`dit_editing.py:fit`，入口和部署adapter位于`benchmark_dit.py`。

### 1. 编辑对象及两种架构

仍只编辑第一个CLIP文本编码器，不改transformer、T5、第二个CLIP或VAE。
SD3.5第一编码器为带projection的CLIP，FLUX第一编码器为不带projection的CLIP：

- SD3.5用编辑器的`text_embeds`作为文本对齐特征。
- FLUX用编辑器的pooled_output通过冻结的匹配CLIP投影得到对齐特征。
- 视觉验证器改为匹配的HF CLIP-L/14（`openai/clip-vit-large-patch14`），使用预训练视觉/文本projection。
  **移除了原脚本将文本维度随机Xavier投影到另一个CLIP空间的做法**，避免随机特征对齐。
- CLIP仍为77-token上下文；T5长度按各模型公共配置。FLUX编辑影响其CLIP pooled conditioning，
  不是伪称修改了FLUX的T5序列或主denoiser。

编辑使用独立float32文本编码器副本，默认在CPU上计算梯度和视觉特征。
图像生成仍使用公共BF16 pipeline和CPU offload；每轮将当前编辑状态复制到pipeline的第一CLIP进行校准生成。
这样不会在具有Accelerate offload hook的部署模块上直接训练。

### 2. 修正AdamW最小化下的损失方向

原`RazorTextLoss`的GA为-cosine、GD为+cosine，实际最小化会加强forget对齐、减弱retain对齐，
与其注释“push apart / pull together”相反。新损失为：

```text
L_forget = + mean(cos(text_forget, image_forget))
L_retain = - mean(cos(text_anchor, image_anchor))
L_mismatch = + mean(cos(text_forget, roll(image_forget, 1)))
L = w_forget*L_forget + w_retain*L_retain + w_mismatch*L_mismatch
```

错配项使用循环位移，避免原随机排列在小batch中产生自配。
图像embedding来自校准生成图并冻结，梯度仅流向编辑文本编码器。
默认权重1/1/0.5，AdamW学习率1e-4，每层阶段4步，梯度裁剪1。
这些是核心目标的明确修正，不把它描述为只改模型名的无损迁移。

### 3. 梯度评分、选择及扩层

保留原`Razor-SD.py`实际执行的评分形式：

```text
importance = ||g_forget|| / (||weight|| + epsilon)
alignment = cosine(flatten(g_forget), flatten(g_retain))
score = importance / (1 + max(0, alignment))
tau = 0.5 * (mean(importance) + std(importance))
```

梯度按整个参数向量flatten后计算单一cosine，修正原脚本对二维权重沿dim=0求出一组cosine再平均的问题。
这里**没有改用README中另一种平方比值/(1-cos)^alpha公式**，也没有添加原实际代码并不存在的二分搜索。
按score排序，保留importance≥tau者，默认最多4个Linear/LayerNorm层；标准差用总体定义。
从第一层开始逐层扩展，只有active层的weight/bias进入优化器。
修复原脚本第一轮优化器提前包含全部selected层、之后才切换active层的问题。

### 4. 有辨别力的停止指标及多概念

原`clip_presence_score`把所有argmax合法索引都计为命中，单标签时presence永远为1，因此UA永远为0。
新停止条件只用于独立的方法校准数据：对每个目标，用冻结CLIP在该目标与其anchor两个标签间判别，
UA=1-目标命中率。**全部目标UA达到阈值**或active层耗尽时结束。
anchor生成图的anchor准确率同时记录；它不是最终benchmark指标。

各目标使用相同数量的模板，forget/anchor损失在一个batch上联合平均，共同决定一套层和参数。
style、IP/Object、nudity各用明确的语义模板，而非把所有词填入“in X style”。
默认每目标2个提示；模板、种子、每轮UA、anchor准确率及层评分写入metadata。
校准提示是代码中的固定方法模板，不读取待报告的评测图片/识别分数来挑选checkpoint。

默认阈值UA=.7；每目标仅2个校准提示时UA粒度为.5，因此达到阈值实际要求该轮两图均未判成目标。
这是原小批量校准思想的实现，不代表完整评测集已经达到同等擦除率。

### 5. 保存和部署

只导出实际active层的第一CLIP参数，使用`checkpoint/text_encoder.safetensors`及`metadata.json`。
采样adapter先加载原模型，再将这些部分参数加载到第一CLIP，之后完全委托公共采样。
这修复了原脚本不保存任何编辑结果，以及只保存transformer会丢失全部实际改动的问题。
不会声称导出了transformer LoRA。

### 默认参数和覆盖

| 参数 | 默认值 |
|---|---|
| `--edit-device` | cpu（可选择和生成相同的CUDA卡） |
| `--clip-model` | openai/clip-vit-large-patch14，需本地缓存或允许下载 |
| `--prompts-per-concept` | 2 |
| `--max-layers` | 4 |
| `--inner-steps` | 4 |
| `--learning-rate` | 1e-4 |
| `--forget-weight / --retain-weight / --mismatch-weight` | 1 / 1 / .5 |
| `--target-ua` | .7 |
| `--calibration-steps` | 公共40/28步，可显式覆盖并记录，正式采样仍按公共协议 |

编辑seed默认0。修改核心定义、匹配CLIP、模板数或优化设置时，用新的run-root/method区分实验。
它仍然包含梯度计算和AdamW更新，不应作为“没有参数优化的training-free”结果归类。

### 文件清单

| 文件 | 内容 |
|---|---|
| `dit_editing.py`（新增） | **核心迁移**：语义模板、多目标损失、梯度评分、扩层优化、正确停止及部分权重导出 |
| `benchmark_dit.py`（新增） | CLI、case配置、独立worker、第一CLIP权重adapter及公共采样/评测编排 |
| `Razor-SD.py` | 保留原样，不作为新benchmark入口执行 |
| `scripts/benchmark_dit_{all,i2p,ip,style,object}.sh`（新增） | 总入口及四组启动脚本 |
| `requirements-benchmark.txt`（新增） | 复用公共依赖，不需要OpenCLIP或其他原demo实验环境 |
| `.gitignore`、`Readme.md`、本文 | 忽略运行目录、说明入口及核心改动记录 |

新增工程Python文件仅2个，没有额外class层级、注册框架或test/目录；只有部署adapter一个必要的新class。
参数在入口一次校验，核心训练代码直接展示损失、评分、优化和保存逻辑。

## 实验与公共配置

默认 SD3.5 Medium，`--model flux1_dev` 切换到 FLUX.1-dev。四组实验直接读取
`benchmark/configs/dit_cases.json`，目标对应的 anchor 读取 `dit_target_anchor.json`，
采样与指标分别读取 `dit_sampling.json`、`dit_evaluation.json`。没有另外复制一份目标列表。

| 分组 | case | 目标 | 评测 |
|---|---|---|---|
| I2P | i2p_nudity | nudity → dressed | sexual 子集 931 图、同一方法设置下 COCO-1k |
| IP | ip_pikachu | Pikachu | Snoopy/Pikachu/Mickey，960 图 |
| IP | ip_snoopy_pikachu | Snoopy、Pikachu 联合擦除 | 同上 |
| style | style_van_gogh | Van Gogh | Van Gogh/Monet/Picasso，270 图 |
| style | style_monet | Monet | 同上 |
| style | style_picasso | Picasso | 同上 |
| Object | object_church | church | 十类各50图，共500图 |
| Object | object_multi5 | church、golf ball、parachute、cassette player、banana 联合擦除 | 同上 |

每模型8个方法适配case、9项评测、5,661张方法图。联合case使用一个方法产物，
不会分别评测单目标再冒充联合结果。I2P与COCO复用同一个nudity产物。
IP使用每提示seed 0–3，style使用0–2，Object使用0–49，COCO为行号seed，I2P为数据自带seed。

正式采样默认512×512、BF16，SD3.5为40步/CFG4.5/T5长度256，FLUX为28步/guidance3.5/T5长度512。
两者均使用原生FlowMatchEulerDiscreteScheduler。采样直接委托公共`DiTAdapter`，
逐记录CPU float32噪声再转模型dtype，FLUX使用原生2×2 latent packing。
生成图、原图、指标和采样耗时统一由benchmark管理，没有重写一套采样或评测算法。

IP/style计算CS、FID及Qwen单标签识别；Object使用ResNet-50 V2完整千类Top-1/Top-3。
默认I2P仅NudeNet；`--with-q16`显式改用I2P全量4,703提示并增加Q16。
当前`dit_evaluation.json`关闭全局VLM，IP/style case会显式开启；`--skip-vlm`可以覆盖。
VLM沿用公共受约束解码与独立版本缓存。

## 安装和启动

```bash
cd /home/shx/code/ce-new/baseline/RAZOR-cvpr2026
python -m pip install -r requirements-benchmark.txt
```

新入口只需要公共benchmark依赖，不要求安装原demo的其他模型/量化/VLM实验依赖。
需事先准备模型访问权限及缓存；`--offline`只从本地加载，缺资源会直接显示依赖或文件错误。
`--base-model`可传相同架构的本地Diffusers目录。

优先运行SD3.5，默认一条命令依次完成四组：

```bash
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_all.sh --device cuda:0 --offline
```

四个单独入口：

```bash
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_i2p.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_ip.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_style.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_object.sh --device cuda:0 --offline
```

FLUX使用相同脚本：

```bash
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_all.sh --model flux1_dev --device cuda:0 --offline
```

`--cases`接受分组或case ID的逗号组合；`--tasks mscoco`只跑nudity设置下的COCO评测。
`--method`可命名不同方法变体。脚本从任意工作目录启动，只转发参数，不做参数检查。
入口只解析、校验一次参数；总控隐藏GPU，将适配、采样、评测放入独立worker进程。
CUDA任务仅使用`--device`选择的卡，其他卡不被模型初始化占用。

默认启用CUDA的model CPU offload，适配和推理时依次将组件放入GPU；
`--no-cpu-offload`可将完整pipeline放入选中的GPU。CPU offload仍需要足够CPU内存和单个活动组件的显存，
不等于FLUX denoiser的分层多卡放置。本入口未扩展新的分布式训练或量化方法。
`--metric-device cpu`可以将指标放到CPU，`--vlm-device`可单独设置Qwen设备。

查看计划，不加载模型、不创建目录：

```bash
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_all.sh --model sd35_medium --dry-run
```

分阶段和重算：

```bash
# 仅生成本方法产物
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_all.sh --offline --phase fit
# 产物已存在时准备清单
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_all.sh --offline --phase prepare
# 按清单补齐缺失/损坏图片
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_all.sh --offline --phase sample
# 仅重算指标
bash /home/shx/code/ce-new/baseline/RAZOR-cvpr2026/scripts/benchmark_dit_all.sh --offline --phase evaluate
```

默认`all`自动执行fit→prepare→sample→evaluate。保持相同参数时复用已完成的产物和图片。
编辑/估计设置、源码或采样配置变化后，使用新的可读`--run-root`，例如`runs/sd35-20261102`；
要在CSV并存多个变体，同时设置不同`--method`。相同method+case再次成功评测更新该行，其他行保留。
`--max-records`、`--steps`、`--height`、`--width`、`--max-sequence-length`用于显式小规模协议；
这些覆盖会写入配置，不能把缩小数据集的结果当完整实验。CPU验证使用`--device cpu --dtype float32 --no-cpu-offload`。

## 输出位置

```text
baseline/RAZOR-cvpr2026/runs/benchmark-dit/<model>/<case>/checkpoint/
baseline/RAZOR-cvpr2026/runs/benchmark-dit/<model>/<case>/plans/RAZOR/<实验-采样参数-开始时间>/
    <task>.json
    <task>-manifest.json
    run.json
benchmark/results/<model>/<task>/<case>/RAZOR/images/<采样参数-开始时间>/
benchmark/results/<model>/<task>/<case>/RAZOR/evaluations/evaluation-<时间>/
benchmark/results/<model>/i2p-metric.csv
benchmark/results/<model>/mscoco-metric.csv
benchmark/results/<model>/instance-metric.csv
benchmark/results/<model>/style-metric.csv
benchmark/results/<model>/object-metric.csv
benchmark/results/<model>/<task>-metric.txt
benchmark/results/<model>/plots/<task>/*.png、*.pdf
benchmark/results/<model>/time/<task>-sampling-time.csv
benchmark/original/<model>/<task>/
```

目录使用实验名、模型名、采样参数和时间，不以hash命名；校验摘要只写入JSON。
`--output-dir`、`--original-dir`、`--cache-dir`可覆盖公共目录，`--run-root`只改变方法产物及计划位置。
实际图像位置以manifest中的image/reference路径为准。原模型图跨方法/case共享，历史清单和结果不改写。

## 本次验证

没有新增`test/`或工程测试文件。使用`/tmp`临时随机小模型验证，结束后清理：

- 两种模型的五个Shell入口均通过语法检查，从其他工作目录运行dry-run，正确展开8个case/9项评测。
- 直接用公共数据配置核对每模型5,661张方法图，目标、anchor、分组、seed和采样profile来自现有JSON。
- 随机初始化的原生SD3模型含Q/K norm、额外图像self-attention及动态scheduler；FLUX含double/single block和rotary embeddings。
- 两种模型分别完成I2P、style、双目标IP、五目标Object的fit、产物保存、adapter加载、真实pipeline生成和公共manifest/sample。
- 公共采样实际生成小规模方法图/参考图，并写入临时公共结果结构及耗时CSV；复用原图路径正常。
- 主denoiser参数在方法适配和采样后保持不变；本次没有将文本编辑或activation steering误导出为denoiser LoRA。

现有公共回归命令：

```bash
cd /home/shx/code/ce-new
python -m unittest benchmark.tests.test_dit_cases benchmark.tests.test_plugin_plans benchmark.tests.test_vlm
```

19项中17项通过，2项因工作区缺少旧retention_repair/output_response入口，按既有skip条件跳过。
临时fixture只验证计算和接入，不作为正式benchmark分数。
本次未启动完整SD3.5 Medium/FLUX.1-dev GPU实验，也未生成正式质量/擦除指标。

RAZOR额外验证：导出的active参数相对原第一CLIP确有非零变化，部分权重可加载，
AdamW更新、逐层扩展和目标/anchor校准判别均可执行。校准使用真实小型CLIP文本/视觉模块，
不以随机常数替代训练损失。

离线运行前还需缓存匹配CLIP-L/14视觉验证器（仅有SD3/FLUX的文本子目录还不够）：

```bash
python -c 'from transformers import CLIPModel, CLIPImageProcessor; CLIPModel.from_pretrained("openai/clip-vit-large-patch14"); CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")'
```

也可将该模型保存为本地目录，并通过`--clip-model /absolute/path/to/clip-vit-large-patch14`指定。
