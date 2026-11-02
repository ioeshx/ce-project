# CASteer的SD3.5 Medium / FLUX.1-dev benchmark适配

更新：2026-11-02。此前在“不得大幅修改核心算法”的要求下跳过迁移；本次明确允许核心修改，
现已实现两种模型的I2P、IP、style、Object四组实验。原SD/SDXL/SANA入口保留。

这是联合注意力上的CASteer移植：保留配对均值差和投影消除公式，明确新增干预张量的定义，
不将整个joint-attention输出直接当成原论文的独立cross-attention输出。

## 核心计算和修改记录

### 1. 联合注意力中的cross-modal分量

新增逻辑位于`core/diffusion_steering.py:HookManager._register_dit_hooks`。
设图像query为Q_i，联合key为K=[K_text,K_image]（SD3按其原生相反顺序），则收集：

```text
A = softmax(Q_i K^T / sqrt(d))
cross = A [V_text, 0_image]
```

使用第二次memory-efficient SDPA实现这个分量，不显式展开完整attention矩阵。
**分母仍是原来的联合softmax，不是只在文本token上重新归一化的cross-attention。**
使用原层的Q/K投影、Q/K norm和FLUX rotary embeddings，保留原生图像/文本拼接顺序。

将cross分量通过原层输出投影的线性部分（不加bias）映射到该层输出空间后，应用原CASteer公式：

```text
s = normalize(mean(cross_positive) - mean(cross_anchor))
alpha = beta * max(dot(cross, s), 0)
output_new = output_native + (cross - alpha*s - cross)
```

代码先计算steered_cross - cross，再加到原输出；统计收集或beta=0时delta严格为0。
基准模型的原attention processor没有被替换，原softmax、图像value分量和输出bias继续由原生模型计算。
只有作用于图像token的文本value贡献被直接steer。

- **SD3.5**：每个joint block只处理`block.attn`的图像输出投影；额外图像self-attention `attn2`完全不注册。
- **FLUX double blocks**：处理joint attention的图像分支，不直接修改文本输出投影。
- **FLUX single blocks**：联合序列中的前T个文本token不直接修改。对融合`proj_out`，仅使用attention对应的weight列
  投影cross分量，MLP对应列不参与delta；随后保留原来的gate、residual和其余运算。

这一定义与原独立CA在注意力分解方式上不同，实验写作应注明“CASteer的joint-attention移植版本”。
后续层的文本状态仍可能通过模型原有图像-文本交互间接受到影响。

### 2. 向量、CFG、batch和精度

`core/controller.py`：

- 新增显式`classifier_free_guidance`标记。SD3.5只steer conditional半个batch；FLUX原生guidance并非双分支CFG，
  因而对batch内全部图片steer，不再把batch>1误认为需要跳过前一半。
- 新入口`precompute_matrix=False`，直接使用原有rank-one投影公式，省去从未用于clipping路径的C×C矩阵。
  例如FLUX有57个block、28个step，预存每层每步的大矩阵会造成不必要的内存占用。
- 向量可保存在CPU，仅小向量随当前层输出搬到对应设备；不再把整幅activation搬回CPU。
- 点积/归一化用float32，输出恢复输入dtype，支持BF16；不再强制half。零范数方向使用EPS下界，成为无操作方向。
- 多概念继续按配置顺序逐个应用原生CASteer投影；没有合并成一个字符串或把多个实验得分拼接。

`core/vector_dump.py`新增相同CFG标记；其按step/layer累加均值的计算不变。
`core/construct_prompts.py`将ImageNet类文件改为相对源码定位，使脚本可从任意目录启动。

### 3. 向量估计与默认参数

`benchmark_dit.py:fit`直接使用原concrete/style/human-related配对提示构造函数。
positive和anchor使用同seed生成，按原`normalize(mean_pos-mean_neg)`形成每目标、每步、每层方向。
anchor来自公共JSON；相比原demo的可选空anchor，这是当前benchmark预先规定的实验设置。

- beta/strength=2，开启intermediate clipping。
- concrete/style默认每目标50对提示。
- human-related使用原函数的实际15×14=210对提示；原README写104对与源码不一致，这里按实际210对执行。
- `--calibration-prompts`可显式缩小配对数量，metadata保存实际提示和数量。
- 使用完整模型本身估计方向，没有从其他架构的turbo模型迁移向量。
- 每步向量对应当前采样步数；`--calibration-steps`必须和正式`--steps`一致。
- 没有优化器、反向传播或权重更新，仍为training-free推理干预。

产物为`checkpoint/steering.pt`及`metadata.json`，并非修改过的transformer权重。

### 文件清单

| 文件 | 内容 |
|---|---|
| `benchmark_dit.py`（新增，唯一新增Python文件） | CLI、case配置、向量估计、方法adapter及公共benchmark编排 |
| `core/diffusion_steering.py` | **核心新增**：SD3/FLUX分类、joint/single文本value分量提取及图像分支hook |
| `core/controller.py` | **核心修改**：显式CFG、紧凑投影、BF16/FP32计算和设备传递 |
| `core/vector_dump.py` | **核心修改**：collector的CFG/batch语义 |
| `core/construct_prompts.py` | 类名文件相对定位；模板内容不变 |
| `scripts/benchmark_dit_{all,i2p,ip,style,object}.sh`（新增） | 总入口及四个分组入口 |
| `requirements-benchmark.txt`（新增） | 复用公共依赖 |
| `.gitignore`、`README.md`、本文 | 运行目录忽略、文档入口和核心改动记录 |

只有一个新增Python文件；没有新建backend/registry/测试工程。只在命令入口校验参数，
复用现有controller/collector/HookManager类，不新增一组重复的控制器类。

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
cd /home/shx/code/ce-new/baseline/CASteer
python -m pip install -r requirements-benchmark.txt
```

新入口只需要公共benchmark依赖，不要求安装原demo的其他模型/量化/VLM实验依赖。
需事先准备模型访问权限及缓存；`--offline`只从本地加载，缺资源会直接显示依赖或文件错误。
`--base-model`可传相同架构的本地Diffusers目录。

优先运行SD3.5，默认一条命令依次完成四组：

```bash
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_all.sh --device cuda:0 --offline
```

四个单独入口：

```bash
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_i2p.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_ip.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_style.sh --device cuda:0 --offline
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_object.sh --device cuda:0 --offline
```

FLUX使用相同脚本：

```bash
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_all.sh --model flux1_dev --device cuda:0 --offline
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
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_all.sh --model sd35_medium --dry-run
```

分阶段和重算：

```bash
# 仅生成本方法产物
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_all.sh --offline --phase fit
# 产物已存在时准备清单
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_all.sh --offline --phase prepare
# 按清单补齐缺失/损坏图片
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_all.sh --offline --phase sample
# 仅重算指标
bash /home/shx/code/ce-new/baseline/CASteer/scripts/benchmark_dit_all.sh --offline --phase evaluate
```

默认`all`自动执行fit→prepare→sample→evaluate。保持相同参数时复用已完成的产物和图片。
编辑/估计设置、源码或采样配置变化后，使用新的可读`--run-root`，例如`runs/sd35-20261102`；
要在CSV并存多个变体，同时设置不同`--method`。相同method+case再次成功评测更新该行，其他行保留。
`--max-records`、`--steps`、`--height`、`--width`、`--max-sequence-length`用于显式小规模协议；
这些覆盖会写入配置，不能把缩小数据集的结果当完整实验。CPU验证使用`--device cpu --dtype float32 --no-cpu-offload`。

## 输出位置

```text
baseline/CASteer/runs/benchmark-dit/<model>/<case>/checkpoint/
baseline/CASteer/runs/benchmark-dit/<model>/<case>/plans/CASteer/<实验-采样参数-开始时间>/
    <task>.json
    <task>-manifest.json
    run.json
benchmark/results/<model>/<task>/<case>/CASteer/images/<采样参数-开始时间>/
benchmark/results/<model>/<task>/<case>/CASteer/evaluations/evaluation-<时间>/
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

CASteer额外验证：收集统计和strength=0时输出逐元素相同；strength=2确实改变图像输出；
SD3.5 unconditional输出不受直接steering影响，FLUX batch中的全部图片都受干预。
BF16 rank-one clipping结果与显式公式一致，零方向无NaN，hook均可移除。
