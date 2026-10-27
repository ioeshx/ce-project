# UCE统一实验适配记录

更新：2026-10-24。适配的上游提交：`7c724d9d7d19190e9bd33fe62fb656f1df87cd8e`。

## 1. 支持范围与实际实验配置

五类实验均可用现有UCE算法实现，**没有修改`trainscripts/uce_sd_erase.py`或其他上游算法文件**。新入口直接调用该文件的`UCE()`。多个目标进入同一个闭式问题，不逐目标累积修改模型；各case从原始SD1.4独立开始。

| 启动脚本 | 实际case与擦除目标 | 正式评测 |
|---|---|---|
| `scripts/benchmark_instance.sh` | Snoopy；Snoopy+Mickey；Snoopy+Mickey+SpongeBob，3个独立模型 | 每个模型评测5个角色共400条提示，seed 0–9，共4,000张；逐概念及目标/非目标CS和macro FID |
| `scripts/benchmark_style.sh` | Van Gogh、Monet、Picasso各自擦除，3个独立模型 | 每个模型评测3个画家共90条提示，seed 0–9，共900张；逐概念及目标/非目标CS和macro FID |
| `scripts/benchmark_i2p_q16.sh` | 默认联合擦除`violence;nudity;harm`，1个模型 | 全部4,703条I2P提示，每条使用原记录seed；Q16不当图片数及比例 |
| `scripts/benchmark_i2p_nudenet.sh` | 默认擦除`nudity`，1个模型 | 生成全部4,703张；仅检测categories包含sexual的931张，阈值0.6、benchmark默认8个EXPOSED类别；图片数/比例及检测框数 |
| `scripts/benchmark_mscoco.sh` | 联合擦除`CIFAR50_classes`全部50个概念，1个模型 | 原始MS-COCO前1,000条caption，不按目标过滤，seed为0–999；CS及相对原模型生成图的FID |
| `scripts/benchmark_all.sh` | 一次依次执行上述五类，合计9个独立模型 | 完成擦除、生成、指标及图表；顺序使用所选GPU |

instance/style/MS-COCO的目标与anchor读取`benchmark/configs/`中的实际JSON，沿用`benchmark.experiment_cases.CASE_CONFIGS`映射。CIFAR-50由`mscoco.json`的`targets_from`解析自`benchmark/dataset/prompt_templates.py`。不加载CIFAR图片。instance/style不额外附带COCO。

`i2p.json`仅定义评测协议，没有指定擦除概念，因此这里明确采用两种可配置设置：Q16沿用UCE README中moderation示例的三个概念；NudeNet采用nudity擦除。这不是对I2P七类内容全部成功擦除的保证。两项默认目标不同，分别保存检查点、图片和结果。可通过`--q16-targets`、`--nudenet-targets`覆盖，用分号分隔多个词，并在跨论文比较时统一目标。

## 2. 算法设置和改动边界

- 保留原生最后一个subject token选择、同时编辑所有`attn2.to_k/to_v`、Equation 7、float32矩阵累加及原生求逆。没有换成其他UCE实现，也没有添加训练、attention hook或选择层机制。
- 保持原生默认`erase_scale=1`、`preserve_scale=1`、`lamb=0.5`；默认不展开提示、不提供额外preserve概念。可通过同名CLI参数调整；`--preserve-concepts`使用分号分隔。CIFAR-50默认也不自动增加CIFAR-100保留项。
- guide默认使用benchmark的`erasure_anchor`，恰为一个字面空格`" "`，并广播到全部目标；不调用上游CLI对guide做`strip()`，因此保留该空格。style的原生CLI默认guide是`art`，此处为匹配当前benchmark而覆盖为空格，**这一配置差别已明确记录**。单独style脚本可传`--guide art`复现该guide设置。
- 使用原生函数的必要包装：设置该函数依赖的模块级`device`、`torch_dtype`；模型加载加入`local_files_only`/revision，并使用共享scheduler构造器和attention slicing。编辑阶段只编码文本，没有生成图，float32为闭式编辑精度。
- 上游README的内容moderation例子传`concept_type=unsafe`，但当前原生CLI的choices只有`art/object`。新入口直接向不依赖concept_type的`UCE()`传概念文本列表，避开这一参数解析缺口，不改公式或扩充核心算法。
- 原生输出只包含K/V权重，不能作为完整UNet用`strict=True`加载。因此新增采样插件，严格核对全部K/V键、形状、有限值、模型/目标/revision和校验和，然后应用这些权重；未编辑的参数仍来自基础模型。
- **原始核心算法修改：无。** 没有因不支持而跳过的实验；支持多目标接口不等于保证实际擦除效果，仍需查看正式评测。

## 3. 新增/修改文件

| 文件 | 内容和目的 |
|---|---|
| `scripts/benchmark_{instance,style,i2p_q16,i2p_nudenet,mscoco}.sh` | 五个独立入口，透传设备、离线、阶段、路径等参数；可从任意工作目录调用 |
| `scripts/benchmark_all.sh` | 一个总入口，依次运行全部9个case |
| `uce_benchmark/protocol.py` | 读取benchmark配置、组织目标/guide/preserve项，记录原生源码和包装器校验和 |
| `uce_benchmark/editing.py` | 调用未经修改的原生`UCE()`，加入原子检查点保存、完成性检查和同配置续跑 |
| `uce_benchmark/adapter.py` | 将原生部分K/V检查点接入benchmark的SD1采样器 |
| `uce_benchmark/runner.py` | CLI解析及阶段编排；隔离编辑/采样/评测进程，生成不可变的可读实验计划 |
| `uce_benchmark/i2p.py` | 分别调用benchmark现有Q16或NudeNet，复用benchmark的CSV、文本表格和绘图接口；未请求的检测器写N/A |
| `uce_benchmark/__init__.py`、`__main__.py` | 定位当前工作区和模块入口 |
| `requirements-benchmark.txt` | 引用工作区benchmark依赖及原生UCE所需diffusers版本 |
| `tests/test_benchmark_integration.py` | CPU数值检查、完整配置/脚本路由、部分权重加载、离线/续跑、独立检测与结果持久化检查 |
| `README.md` | 增加本说明链接 |

没有在`benchmark/`新增或修改代码。新增代码、脚本和本文档均在本UCE目录中；运行时指标依旧汇入benchmark的统一结果目录。

## 4. 运行方式

在本目录准备依赖，使用现有与CUDA匹配的PyTorch环境：

```bash
cd /home/shx/code/ce-new/baseline/unified-concept-editing
pip install -r requirements-benchmark.txt

# 查看全部命令；不加载模型、不生成检查点/图片/分数
bash scripts/benchmark_all.sh --device cuda:0 --offline --dry-run

# 全部五类实验，依次执行
bash scripts/benchmark_all.sh --device cuda:0 --offline

# 分别启动其中一类
bash scripts/benchmark_instance.sh --device cuda:0 --offline
bash scripts/benchmark_style.sh --device cuda:0 --offline
bash scripts/benchmark_i2p_q16.sh --device cuda:0 --offline
bash scripts/benchmark_i2p_nudenet.sh --device cuda:0 --offline
bash scripts/benchmark_mscoco.sh --device cuda:0 --offline
```

所有脚本支持`PYTHON=/path/to/python`。`--offline`只使用本地模型/检测资源；尚未下载时省略此参数。Q16缺少学习参数时须先准备官方权重，不会退回临时文本分类器：

```bash
cd /home/shx/code/ce-new
python -m benchmark download --resource q16 --include-backbone
python -m benchmark download --resource clip
python -m benchmark download --resource fid
python -m benchmark download --resource nudenet
```

SD1.4首次加载需要可用的`CompVis/stable-diffusion-v1-4`模型缓存。Q16使用匹配的官方ViT-L/14及学习参数；NudeNet使用nudenet包内权重或显式本地路径。独立脚本只加载其所请求的检测器，Q16不会要求NudeNet资源，NudeNet也不会要求Q16资源。

```bash
# 自定义I2P擦除目标或本地检测资源
cd /home/shx/code/ce-new/baseline/unified-concept-editing
bash scripts/benchmark_i2p_q16.sh --device cuda:0 --offline \
  --run-root runs/uce-q16-custom --method-prefix UCE-q16-custom \
  --q16-targets 'violence;nudity;harm' \
  --q16-prompts /path/to/prompts.p --q16-clip-path /path/to/ViT-L-14.pt

bash scripts/benchmark_i2p_nudenet.sh --device cuda:0 --offline \
  --nudenet-weights /path/to/320n.onnx --nudenet-threshold 0.6
```

## 5. 续跑、目录与共享采样协议

默认检查点根目录为`runs/benchmark-sd14/`。相同配置再次执行会复用已完成的编辑检查点，采样只补齐缺失或损坏图片。改变编辑参数、目标、模型revision或源码版本后，应使用新的`--run-root`；检查失败会报错，不覆盖原检查点。需要并存多轮指标时也改`--method-prefix`。

```bash
bash scripts/benchmark_all.sh --device cuda:0 --offline \
  --run-root runs/uce-paper-oct24 --method-prefix UCE-paper-oct24
# 后续只评测：保持此前全部参数，仅替换阶段
bash scripts/benchmark_all.sh --device cuda:0 --offline \
  --run-root runs/uce-paper-oct24 --method-prefix UCE-paper-oct24 --phase evaluate
```

支持`--phase all|erase|benchmark|prepare|sample|evaluate`。`benchmark`执行prepare、sample、evaluate，要求已有匹配的擦除检查点。单独sample/evaluate要求先prepare，并使用完全一致的设置。各组顺序执行，一组报错则停止，便于修复后续跑。

```text
baseline/unified-concept-editing/runs/benchmark-sd14/<case>/checkpoint/
  uce.safetensors          # 上游UCE原生K/V部分权重
  metadata.json           # 原生编辑设置、guide、来源、校验和、时间
baseline/unified-concept-editing/runs/benchmark-sd14/<case>/plans/
  <method>/<task-采样概要-时间>/<task>.json
  <method>/<task-采样概要-时间>/<task>-manifest.json

benchmark/results/sd14/{instance,style,i2p,mscoco}-metric.csv
benchmark/results/sd14/{instance,style,i2p,mscoco}-metric.txt
benchmark/results/sd14/<task>/<method>/images/<可读采样名-时间>/
benchmark/results/sd14/<task>/<method>/evaluations/evaluation-<时间>/
benchmark/results/sd14/plots/<task>/*.png 和 *.pdf
benchmark/original/<task>/<可读采样名-时间>/
```

`--output-dir`可指定指标和方法图的根目录，例如`--output-dir results`将它们放回本UCE目录；`--original-dir`/`--cache-dir`可指定共享参考图及FID特征缓存。路径按调用时工作目录解析。同名method成功评测覆盖主CSV旧行，历史图片和逐次评测明细保留。I2P的Q16与NudeNet使用不同method名，单独运行其中一项不会覆盖另一项。

正式生成严格使用`benchmark/configs/sd14_sampling.json`：SD1.4、DPM-Solver++ 20步、CFG7.5、512×512、CUDA float16、空negative prompt、attention slicing。每条记录用独立CPU generator生成float32噪声再转换；采样batch划分不改变初始噪声。原模型、擦除模型使用同一协议；COCO参考图可以复用现有匹配缓存。CLIP为benchmark默认base-patch16，FID相对于原模型生成图，不是真实COCO照片。

显式小规模CPU检查必须传`--device cpu --sampling-dtype float32`。`--steps`、`--height`、`--width`、`--seeds`、`--prompts-per-concept`、`--max-records`均属于协议覆盖，保存在清单中。使用独立run-root、output-dir和method-prefix隔离这类结果，不作为正式论文分数。

I2P未选择的检测器为N/A，不填零；检测器缺失、预测不完整、图像缺失或损坏时不会发布成功指标。逐图Q16概率/标签、NudeNet原始框与接受框、类别检测框数均保存在对应evaluation目录。

## 6. 验证与限制

2026-10-24检查结果：本目录6项集成检查、共享sampling profile的5项检查全部通过；六个Shell入口的语法、dry-run和参数解析已覆盖。原生UCE脚本的git diff为空。

```bash
cd /home/shx/code/ce-new/baseline/unified-concept-editing
python -m unittest discover -s tests -v
```

验证范围：实际原生UCE函数在微型矩阵上联合编辑50个目标，与独立Equation 7计算对照；仅K/V部分权重应用；原子检查点与同配置续跑；六个Shell入口从其他工作目录启动、所有阶段参数解析；9组完整提示清单、3组instance/3组style、I2P的4703/931筛选及COCO的50目标/1000caption/逐行seed；共享采样协议及COCO参考身份；独立检测器加载、未选项N/A、CSV更新和PNG/PDF重建、失败保留旧行。

单元测试中的微型模型、检测器和图像只验证数值与接口，不是实验分数，全部写入临时目录并清理。未启动数千张正式SD1.4采样，也未将模拟检测分数写入正式结果目录。完整GPU实验的擦除效果、图像质量和耗时需要实际运行上述脚本测量。
