# Steer：SD3.5 概念擦除基础实现

从本仓库 CASteer 的 SD3.5 迁移版本中独立出基础算法，作为后续实验起点。
默认 SD3.5 Medium，也可选择同架构的 SD3 Medium；当前不包含 FLUX。
运行时只依赖公共 `benchmark` 和本目录，不导入 `baseline/CASteer`。

## 文件与算法入口

| 文件 | 内容 |
|---|---|
| `method.py` | 向量构造、投影干预、SD3 hooks、离线统计、benchmark adapter |
| `run.py` | 参数入口与 fit → prepare → sample → evaluate 编排 |
| `run.sh` | 转发命令行参数 |
| `imagenet_classes.txt` | 从 CASteer 复制的 ImageNet 类名，保持原配对模板 |
| `README.md` | 算法定义与运行方法 |

后续修改构造方法，直接编辑 `construct_vectors()`；修改向量使用方式，编辑
`steer_output()`。收集位置与统计维度在 `SD3Steering` 中，提示词在 `prompt_pairs()` 中。
目前实现均值差，不预置其他候选算法。

## 可选配置

两个开关相互独立，默认保持原来的计算位置和位置平均方式：

| 参数 | 含义 |
|---|---|
| `--steer-target text_to_image`（默认） | 在 image token 上修改经过输出投影的 `A_image,text V_text` |
| `--steer-target text_value` | 直接修改 `add_v_proj` 得到的 `V_text`，然后交给原生 attention 使用 |
| `--average-patches`（默认） | 对位置求均值，每层每步使用一个共享方向 |
| `--no-average-patches` | 保留位置维度，每个位置使用自己的方向 |

`text_to_image` 的位置是图像 patch；`text_value` 的位置是文本 token。
不做位置平均时，仍然跨 prompt 求均值，最后分别对每个位置的 channel 方向归一化。
推理按同一位置对应使用向量；分辨率、文本序列长度沿用离线统计设置。
不平均时，统计量和产物大小会随位置数增加。

```bash
# 原有位置：投影后的文本→图像贡献，位置平均
bash /home/shx/code/ce-new/DiT/steer/run.sh --steer-target text_to_image --average-patches --dry-run
# 同一位置，保留所有 image patch
bash /home/shx/code/ce-new/DiT/steer/run.sh --steer-target text_to_image --no-average-patches --dry-run
# 直接修改 text Value，位置平均
bash /home/shx/code/ce-new/DiT/steer/run.sh --steer-target text_value --average-patches --dry-run
# 直接修改 text Value，保留所有 text token
bash /home/shx/code/ce-new/DiT/steer/run.sh --steer-target text_value --no-average-patches --dry-run
```

以上命令只展示计划；实际运行时移除 `--dry-run`，按需添加 `--cases`、`--device`、`--offline`。

## 1. 离线统计

沿用迁移版定义：提取 joint attention 中**文本 Value 对图像 token 的贡献**：

```text
cross = W_out · Attention(Q_image, [K_image, K_text], [0_image, V_text])
positive_mean = mean(cross_positive, over prompts and image patches)
negative_mean = mean(cross_negative, over prompts and image patches)
vector = normalize(positive_mean - negative_mean, over channels)
```

保留图像 keys，因此 softmax 分母仍覆盖图像和文本；输出投影不加 bias。
每层、每个去噪步分别计算。点积、均值和归一化使用 FP32。
这是 `text_to_image` 模式。`text_value` 模式在相同层、时间步直接收集
`V_text = add_v_proj(text_hidden_states)`。两种模式都在同一位置构造、应用方向。

提示词模板与 CASteer 一致：IP/Object 为 `ImageNet 类名 with concept`，
style 为 `ImageNet 类名, concept style`，默认各 50 对；I2P 为 15 类人物 × 14 类场景，
共 210 对。负侧概念读取公共 `benchmark/configs/dit_target_anchor.json`。
正负两侧使用相同 seed、相同初始噪声、相同原生调度器；统计时保留未干预轨迹。
执行所有去噪步，但 `output_type="latent"` 跳过 VAE 图像解码；不调用外部检测器或评价指标。

离线向量按配置缓存。每次运行先在当前 `--run-root` 的对应 model/case 下查找完整匹配的
`metadata.json` 和 `steering.pt`，命中后直接复用；配置变化时自动另存，切回旧配置会再次命中。
新配置保存到可读目录：

```text
runs/<model>/<case>/<steer_target>-<mean 或 per-position>/offline/
  flow-euler40-512x512-bfloat16-pairs50-seed0-<开始时间>/checkpoint/
  steering.pt    # vectors、positive_means、negative_means，均为 CPU FP32
  metadata.json  # 参数、实际 prompt pairs、timesteps、采样配置
```

平均模式的三个张量维度均为 `[concept, step, layer, channel]`；
不平均模式为 `[concept, step, layer, position, channel]`。
metadata 记录目标位置、是否平均以及位置轴对应 text token 还是 image patch。
因此后续可以从已保存均值重新构造方向。缓存匹配包括离线算法版本、模型及 revision、
目标和 anchor、实际正负提示词、seed、目标位置、是否平均、dtype、步数、分辨率、
文本长度、CFG、负提示词和原生 scheduler 配置。省略提示词数量与显式指定默认数量等价，
空 revision 与未指定 revision 等价。

beta、clipping、benchmark 任务/样本数、batch size、GPU 编号、评测参数和输出目录
不参与离线匹配。仅改变这些参数会复用向量，无需手动修改 `--run-root`。
控制台的 `[fit:reuse]` 表示复用，`[fit:collect]` 表示计算新配置。
同一配置被多个进程同时请求时，后来的进程等待已有计算完成再复用。

旧版 `<case>/checkpoint` 和 `<case>/<variant>/checkpoint` 也会被搜索。
原始 v1 产物对应 `text_to_image + average-patches`，参数匹配时直接使用，保留原文件。
源码摘要用于记录来源，离线算法由 `settings["implementation"]` 标识；以后改变收集或构造
算法时，应同步更新该版本字段。调整在线干预或注释不会使离线缓存失效。

## 2. 在线干预

采用加法形式，**默认 beta=-2 表示擦除**：

```text
features = cross                      # text_to_image；text_value 使用 V_text
score = dot(features, vector)          # 仅沿 channel 点积
score = max(score, 0)                  # --clipping，默认开启
features_new = features + beta * score * vector
native_output_new = native_output + (features_new - features)  # text_to_image
V_text_new = features_new                                  # text_value
```

因此 beta=-2 对应原 CASteer 的正 strength=2；beta=0 为无操作。
`--no-clipping` 使用有符号点积；正 beta 沿概念方向增强正投影。
多概念按 benchmark 的目标顺序依次执行。

hooks 保留原生 attention processor 和模型权重。
`text_to_image` 只将 cross 分量的变化加到图像输出，不直接修改文本输出或 `attn2`；
`text_value` 只在文本 Value 投影处干预，原生 attention 将其变化传递给图像和文本输出。
存在 CFG 时，只直接修改 conditional 半个 batch；guidance≤1 时处理全部样本。

## 安装与运行

复用公共依赖：

```bash
python -m pip install -r /home/shx/code/ce-new/benchmark/requirements.txt

# 查看全部计划，不加载模型、不创建运行目录
bash /home/shx/code/ce-new/DiT/steer/run.sh --dry-run

# 默认：SD3.5 Medium，四组实验及 I2P 对应的 COCO 保留评测
bash /home/shx/code/ce-new/DiT/steer/run.sh --device cuda:0 --offline
```

默认继承公共采样设置：512×512、BF16、40 步、CFG 4.5、T5 长度 256。
CUDA 默认使用 model CPU offload；`--no-cpu-offload` 将 pipeline 放入指定 GPU。
`--base-model /absolute/local/pipeline` 指定本地 Diffusers 模型。

四组实验使用同一个入口：

```bash
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases i2p --offline
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases ip --offline
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases style --offline
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases object --offline
```

| 分组 | cases / 目标 | 公共评测 |
|---|---|---|
| I2P | nudity → clothed；与 COCO 共用同一向量 | sexual 931 图的 NudeNet；COCO-1k 保留能力 |
| IP | Pikachu；Snoopy + Pikachu | 每 case 960 图，CS/FID 与 Qwen 分类 |
| style | Van Gogh、Monet、Picasso 分别擦除 | 每 case 270 图，CS/FID 与 Qwen 分类 |
| Object | church；五目标联合擦除 | 每 case 500 图，ResNet-50 V2 Top-1/Top-3 |

共 8 个 case、9 项评测、5,661 张方法图。联合 case 使用同一个多概念向量文件。
图像、参考图、指标和耗时统一由公共 benchmark 管理。
`--skip-vlm` 关闭 IP/style 的 VLM，`--with-q16` 将 I2P 改为全量 4,703 条并增加 Q16。
`--tasks mscoco` 可以只运行 nudity 产物的 COCO 保留评测。

分阶段与调整强度：

```bash
# 离线统计
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases i2p --phase fit --offline
# 准备、采样、评测：与 fit 保持同一采样配置
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases i2p --phase prepare --offline
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases i2p --phase sample --offline
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases i2p --phase evaluate --offline
# 复用统计，使用不同 beta 与有符号点积
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases i2p --beta -1 --no-clipping --offline
# 小规模接入检查；不作为完整 benchmark
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases ip_pikachu --calibration-prompts 2 \
  --max-records 3 --steps 2 --height 64 --width 64 --max-sequence-length 16 \
  --skip-vlm --run-root /home/shx/code/ce-new/DiT/steer/runs/smoke-20261109 --offline
```

默认结果名包含目标位置、平均方式、beta 与 clipping，例如
`Steer-text_to_image-mean-erase-2-clip`、`Steer-text_value-per-position-erase-2-clip`；
`--method` 可自定义实验名。这样四种配置的指标和图片可以分别保存。
计划目录由公共 benchmark 使用可读参数与开始时间命名，不使用哈希目录。
方法产物在 `--run-root`；新运行的生成图片默认在
`DiT/steer/results/<model>/<task>/<case>/<method>/images/`，可用 `--image-output-dir` 覆盖。
指标汇总、评测明细和对比图继续统一写入 `benchmark/results/<model>/`，由 `--output-dir` 控制。
参考图默认在 `benchmark/original/<model>/`。已有实验清单继续引用其原来的图片路径。

空 `model_revision`（缺失、null、空字符串）统一表示默认版本，不单独拆分比较组。
要复用现有指标刷新分组、汇总表和图，无需生成图片或重新评测：

```bash
for task in i2p mscoco instance style object; do
  python -m benchmark refresh --model sd35_medium --task "$task"
done
```

参考：[CASteer 论文](https://arxiv.org/abs/2503.09630)、
本仓库 `baseline/CASteer/DIT_BENCHMARK_ADAPTATION.md`。

## 已完成的验证

新增两种目标 × 两种平均方式均通过小型原生 SD3 验证：
离线统计、位置维度、正负均值保存、向量加载、双目标与 batch=2 采样。
`text_value` 的结果与手工 joint-attention 计算一致；默认 `text_to_image + mean`
的模型输出与旧实现逐元素一致。仅文本嵌入使用本地小型 fixture，未启动正式实验。

原生小型随机 SD3 模型验证了 joint softmax 分量、Q/K norm、额外 self-attention、
CFG 条件分支、beta=0、FP32/BF16、多概念顺序及 hook 移除。
本地小型完整 SD3 pipeline 跑通四组代表 case 的离线统计、保存/加载、公共清单与采样，
覆盖双目标 IP、五目标 Object、batch=2 和 COCO 保留采样。
现有 `benchmark.tests.test_dit_cases`、`benchmark.tests.test_plugin_plans` 共 16 项，
14 项通过，2 项因旧方法入口不存在按既有条件跳过。
当前环境没有可用 CUDA，尚未运行完整 SD3.5 Medium 生成与质量指标评测。
