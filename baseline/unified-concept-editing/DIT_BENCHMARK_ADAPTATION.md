# UCE 的 DiT 实验适配

默认模型为 **SD3.5 Medium**，另支持 **FLUX.1-dev**。新增两个 Python 文件和五个 Shell 入口，全部位于 UCE 目录；未修改 `benchmark/`、上游 `trainscripts/` 或已有 SD1.4 代码，没有新增测试文件。

FLUX 分支依据本仓库 `trainscripts/uce_flux_edit.py`。**SD3.5 是将这一文本投影编辑方案迁移到 SD3 架构的本地实现，并非作者提供的 SD3.5 官方实现。** 两者均采用 UCE 联合闭式求解，不引入梯度训练。当前验证的是计算及接口可运行，真实擦除效果需要正式模型实验确认。

## 可行实验与实际配置

运行时直接读取 `benchmark/configs/dit_cases.json`、`dit_target_anchor.json`、`dit_sampling.json`、`dit_evaluation.json`，不复制一套实验目标。

| 实验组 | case | 擦除目标 | 数据量和评测 | 两种模型的接入情况 |
|---|---|---|---|---|
| I2P | `i2p_nudity` | nudity | sexual 931 张，NudeNet；同一权重额外评测 COCO 1K，CS/FID | 已实现 |
| IP | `ip_pikachu` | Pikachu | Snoopy/Pikachu/Mickey，960 张，CS/FID、Qwen 单标签分类 | 已实现 |
| IP | `ip_snoopy_pikachu` | Snoopy、Pikachu | 同上，960 张 | 已实现，联合求解 |
| style | `style_van_gogh` | Van Gogh | 三位画家，270 张，CS/FID、Qwen 单标签分类 | 已实现 |
| style | `style_monet` | Monet | 同上，270 张 | 已实现 |
| style | `style_picasso` | Picasso | 同上，270 张 | 已实现 |
| Object | `object_church` | church | ImageNet-10，每类 50 张，共 500 张；ResNet-50 V2 Top-1/3 | 已实现 |
| Object | `object_multi5` | church、golf ball、parachute、cassette player、gas pump | 同上，500 张 | 已实现，五概念联合求解 |

每个模型共 **8 次擦除、9 项评测任务、5,661 张方法图**。原模型参考图按完整采样协议共享。COCO 只随 nudity case 进行一次，不使用旧 SD1.4 的 CIFAR-50→COCO 设置。

默认 I2P 不启用 Q16；`--with-q16` 将 I2P 扩为 4,703 张并启用 Q16，NudeNet 仍只检测 sexual 子集。IP seeds 为 0–3，style 为 0–2，Object 为 0–49。

| 模型 | 精度 | 分辨率 | 步数 | guidance | T5 长度 |
|---|---|---|---:|---:|---:|
| `sd35_medium` | BF16 | 512×512 | 40 | 4.5 | 256 |
| `flux1_dev` | BF16 | 512×512 | 28 | 3.5 | 512 |

保留各基础模型自己的 FlowMatchEulerDiscreteScheduler 配置；原图、方法图使用公共实现的逐记录 CPU FP32 初始噪声，FLUX 使用原生 latent 打包。生成、评测在独立子进程中执行，生成模型释放后才加载指标模型。

## 启动脚本和命令

| 脚本 | 执行内容 |
|---|---|
| `scripts/benchmark_dit_i2p.sh` | nudity 擦除，I2P＋COCO |
| `scripts/benchmark_dit_ip.sh` | 两个 IP case |
| `scripts/benchmark_dit_style.sh` | 三个 style case |
| `scripts/benchmark_dit_object.sh` | 两个 Object case |
| `scripts/benchmark_dit_all.sh` | 以上全部，顺序执行 |

在任意目录执行均可。显式本地路径相对于调用命令时的目录；`PYTHON` 环境变量可选择解释器。

```bash
# 默认优先使用 SD3.5 Medium，运行全部四项实验
bash /home/shx/code/ce-project/baseline/unified-concept-editing/scripts/benchmark_dit_all.sh \
  --device cuda:0 --offline

# FLUX.1-dev，采样使用 CPU offload 降低显存需求
bash /home/shx/code/ce-project/baseline/unified-concept-editing/scripts/benchmark_dit_all.sh \
  --model flux1_dev --device cuda:0 --offline --cpu-offload

# 只展开配置和命令，不加载权重、不创建输出
bash /home/shx/code/ce-project/baseline/unified-concept-editing/scripts/benchmark_dit_all.sh --dry-run

# 单独运行多 IP 擦除
bash /home/shx/code/ce-project/baseline/unified-concept-editing/scripts/benchmark_dit_ip.sh \
  --cases ip_snoopy_pikachu --device cuda:0 --offline
```

五个脚本接受同样的参数，`--cases` 也可以指定逗号组合的 case。`--tasks mscoco` 可单独选择 nudity case 的 COCO 评测。

`--phase all` 为默认值，包含擦除、准备、采样和评测。其他可选阶段：

- `erase`：只编辑并保存权重，相同配置复用已完成的检查点。
- `prepare`：使用已有检查点准备提示、图片清单，不生成图片。
- `sample`：读取匹配的已准备计划，只补缺图或坏图。
- `evaluate`：读取匹配的已准备计划，更新 CSV、TXT、PNG/PDF。

分阶段运行时保持同样的模型、case、编辑参数、采样参数和目录参数。已有图片也可直接用公共入口 `python -m benchmark evaluate --manifest <清单路径>` 重算指标。

`--offline` 只读取本地或缓存中的资源，资源不完整时去掉该参数。`--base-model /本地模型目录` 可覆盖基础模型路径，目录必须与 `--model` 的架构匹配；同一目录也用于原模型对照。

依赖复用 `requirements-benchmark.txt`；评测权重准备见 `benchmark/DIT.md`。IP/style 默认需要 Qwen3-VL-2B-Instruct，Object 需要官方 ResNet-50 V2，I2P 需要 NudeNet，CS/FID 需要 CLIP 和 Inception 权重。支持 `--vlm-model`、`--resnet-weights-path`、`--nudenet-weights` 等本地路径。`--skip-vlm` 显式形成不含 VLM 的评测协议。

采样可选 `--cpu-offload` 或 `--device-map balanced`，二者互斥。FLUX 另支持 `auto`、`--max-memory`、`--gpu-memory-reserve`，均传给公共采样器。当前执行层仅暴露 `--device` 指定的一张 GPU，自动放置也限制在该卡内；其他 GPU 的辅助设备或内存预算在入口报错。编辑默认在同一卡执行，可用 `--edit-device cpu` 改为 CPU；两层闭式求解始终在 CPU 以 FP32 执行。

编辑、采样和评测均在独立子进程执行，总控进程隐藏全部 GPU。编辑结束后子进程退出，CUDA 上下文随之释放，不再让总控保留数百 MB 显存。公共实现为 `benchmark/execution.py`，`dit_editing.py` 只新增运行设备映射，未改闭式公式。执行层升级对旧源码摘要采用明确的 before/after 兼容映射，继续复用原检查点和计划，不改写历史记录；其他参数或算法源码发生变化仍要求新实验。

## 编辑算法和迁移差异

保留上游 FLUX 选择的两层：

1. `context_embedder.weight`：文本序列输入投影。
2. `time_text_embed.text_embedder.linear_1.weight`：pooled CLIP 条件输入投影。

FLUX 使用 T5 代表向量和 pooled CLIP 向量。SD3.5 使用拼接序列的 **T5 段**代表向量和两个 CLIP 的 pooled 拼接向量，保留其原生 CLIP 序列输入。SD3.5 的 pooled 输入不是 FLUX 的 768 维，所以按层名选择表示，不沿用上游按宽度 768 判断的写法。这是接口和层映射的迁移，没有新增另一套 attention 编辑算法。

每层联合求解以下上游公式，`C_E/C_G/C_P` 分别为目标、guide、保留概念的列矩阵：

```text
A = λW + erase_scale · P(C_G) C_Eᵀ + preserve_scale · P(C_P) C_Pᵀ
B = λI + erase_scale · C_E C_Eᵀ    + preserve_scale · C_P C_Pᵀ
W_new = A B⁻¹
P(C) = W C + b
```

保留上游 guide/retain 输出包含原始 bias 的行为；只编辑 weight，bias 不变。使用 `torch.linalg.solve(B.T, A.T).T` 替代显式求逆，数学目标相同。求解使用 FP32，采样加载时转换为模型精度。默认 `erase_scale=10`、`preserve_scale=1`、`lamb=0.1`，与上游 FLUX 入口一致，与旧 SD1.4 入口默认值不同。

保留上游 `attention_mask.sum()-2` 的 T5 内容 token 选取规则；SD3.5 加上前置 CLIP 序列的偏移。空字符串沿用上游的负索引语义，指向 T5 段的末尾 padding。编辑时的编码精度跟随 `--dtype`，默认 BF16；上游命令行默认 FP32，这一差异记录在 metadata 中。

默认不扩展概念提示，不自动加入保留概念；可用 `--preserve-concepts '概念A;概念B'` 显式添加。anchor 读取已有映射：nudity→dressed，IP→Cartoon Character，style→Artist Style，Object 按目标使用 Building、ball、cloth 等。`--anchor ' '` 可显式改为空格。多目标支持不同 guide，对应关系保存在 `erasure_parameters.guides`。

编辑器从单文件或分片 safetensors 中只读取这两层的 weight/bias，不加载完整 denoiser；编码完成后释放文本编码器，再求解。只保存两个 FP32 weight。采样插件加载公共 `DiTAdapter` 后覆盖这两层，不另写采样循环或生成完整模型副本。

## 文件修改记录

| 新增或修改文件 | 内容及目的 |
|---|---|
| `uce_benchmark/dit.py` | 统一入口、参数解析、读取现有配置、阶段编排 |
| `uce_benchmark/dit_editing.py` | 两层联合闭式求解、SD3.5 表示映射、部分权重读写和采样插件 |
| `scripts/benchmark_dit_{i2p,ip,style,object,all}.sh` | 四个单项脚本和总入口，只转发参数 |
| `README.md`、`BENCHMARK_ADAPTATION.md` | 添加本说明的链接 |
| `DIT_BENCHMARK_ADAPTATION.md` | 本文档 |

没有新增 class；入口统一处理参数，少量函数组织配置、运行流程和核心公式。上游算法文件与公共 benchmark 保持原样。

## 保存路径

检查点与计划保存在本方法目录：

```text
baseline/unified-concept-editing/runs/benchmark-dit/<model>/<case>/
  checkpoint/uce.safetensors
  checkpoint/metadata.json
  plans/UCE/<task>-<采样概要>-<北京时间>/
```

图片和指标继续汇入公共目录：

```text
benchmark/results/<model>/<task>/<case>/UCE/
benchmark/results/<model>/<task>-metric.csv
benchmark/results/<model>/<task>-metric.txt
benchmark/results/<model>/plots/<task>/
benchmark/original/<model>/<task>/
```

`task` 为 `i2p`、`mscoco`、`instance`、`style`、`object`；IP 对应 `instance`。同一方法、同一 case 成功评测后替换该行，不同 case 分别保留。`--method` 可设置不同版本名；修改编辑参数时使用新的 `--run-root`，避免覆盖已有检查点。

目录名称可读，指纹仅写 JSON。CLI 可覆盖 `--output-dir`、`--original-dir`、`--cache-dir`；结果路径仍由公共 benchmark 决定，本次未向其中添加工程代码或文档。

## 验证结果与实际限制

使用临时运行检查，没有新增 `test/`、`tests/` 或正式实验结果：

- 两种模型的五个脚本通过 Shell 语法、dry-run 和八个 case／九项任务计数检查；dry-run 不创建目录。
- 实际数据展开数量正确；Q16 全量切换、多目标 anchor 覆盖正确。
- 1/2/5/50 目标、有/无保留向量的小矩阵求解与上游逐项累积、显式求逆公式一致。
- 随机初始化的微型原生 SD3/FLUX 模型完成联合编辑、检查点复用、准备清单、部分权重加载、真实 CPU 采样；只有指定两层 weight 改变。覆盖单文件与分片权重。临时图片、模型和计划已清理。

当前环境 NVIDIA 驱动不可用，尚未运行真实 SD3.5 Medium/FLUX.1-dev 的完整 GPU 实验，没有正式擦除效果指标。GPU BF16、多卡和 CPU offload 的实际执行依赖公共采样器，本次未在 GPU 上验证。

依据：[作者 FLUX 实现](https://github.com/rohitgandikota/unified-concept-editing/blob/main/trainscripts/uce_flux_edit.py)、[Diffusers 官方 SD3 文档](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)，并核对本机 Diffusers 的 `StableDiffusion3Pipeline.encode_prompt` 与 `SD3Transformer2DModel` 源码。
