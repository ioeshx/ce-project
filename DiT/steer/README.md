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
默认使用均值差；可显式选择稳健聚合、差分归一化及两种 PCA 构造。

## 可选配置

目标位置和位置平均方式独立于后面的向量构造选项：

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

## 向量构造：四种可选改进

所有公式均针对一个 concept、layer、timestep，以及不平均时的一个位置独立计算。
第 $j$ 对 prompt 的特征为 $h_j^+,h_j^-\in\mathbb R^C$，差分为
$d_j=h_j^+-h_j^-$。定义 $\operatorname{unit}(x)=x/\max(\|x\|_2,10^{-8})$。
位置平均模式先对每个 prompt 的位置求均值，再做上述差分及后续操作。
默认向量仍是 $\operatorname{unit}(\mu)$，其中
$\mu=\operatorname{mean}(h^+)-\operatorname{mean}(h^-)$。

| 参数 | 默认 | 作用 |
|---|---|---|
| `--vector-method mean` | 开启 | 对差分求均值 |
| `--vector-method robust` | 关闭 | 分组均值的几何中位数 |
| `--normalize-differences` | 关闭 | 每对差分先按 channel 归一化，可与所有构造方式组合 |
| `--vector-method pca` | 关闭 | 未中心化差分的第一主方向 |
| `--vector-method pca-filter` | 关闭 | 用前 k 个主方向过滤均值差 |

四种 `vector-method` 互斥；归一化开关独立。以下用 $x_j$ 表示参与聚合的差分：
默认 $x_j=d_j$；启用归一化后 $x_j=\operatorname{unit}(d_j)$。
记 $\bar x=N^{-1}\sum_j x_j$。

### 1. 稳健统计：median of means

使用 `--seed` 生成一个固定的 prompt 对随机排列，再分成 G 组 $I_g$，各组大小相差至多 1。
同一划分用于所有 layer、timestep 和位置。先求组均值，再求它们的几何中位数：

$$
m_g=\frac{1}{|I_g|}\sum_{j\in I_g}x_j,\qquad
q^*=\arg\min_q\sum_{g=1}^{G}\|q-m_g\|_2,\qquad s=\operatorname{unit}(q^*).
$$

代码用 Weiszfeld 迭代近似求解。初始化 $q^{(0)}=G^{-1}\sum_g m_g$，随后：

$$
w_g^{(r)}=\frac{1}{\max(\|q^{(r)}-m_g\|_2,10^{-8})},\qquad
q^{(r+1)}=\frac{\sum_g w_g^{(r)}m_g}{\sum_g w_g^{(r)}}.
$$

当更新距离不大于 `tol × max(1, ||q||)` 时停止，否则达到迭代上限后返回。
`--robust-groups` 默认 `min(5, N)`，不能大于 N；`--robust-max-iter` 默认 100，
`--robust-tol` 默认 `1e-5`。G=1 退化为均值；G=N 对逐对差分直接求几何中位数。
少数异常组的影响会减弱，多数组共同带有的偏差仍会保留。

### 2. 先归一化差分再聚合

单独添加 `--normalize-differences`，保留默认 `mean`：

$$
s=\operatorname{unit}\left(\frac1N\sum_{j=1}^N
\frac{d_j}{\max(\|d_j\|_2,10^{-8})}\right).
$$

它让差分主要按方向参与平均，减少大幅值样本的主导作用。
开启该开关后，robust、PCA 的输入也使用 $x_j$。

### 3. 从多对差分提取第一主方向

把每个 $x_j^\top$ 放在矩阵 X 的一行，直接做薄 SVD：

$$
X=U\Sigma V^\top,\qquad
s=\begin{cases}v_1,&v_1^\top\bar x\ge0,\\-v_1,&v_1^\top\bar x<0.\end{cases}
$$

`--vector-method pca` 使用这个方向。**不减去差分均值**，因为共同概念偏移正包含在均值中。
符号与均值差对齐，使 CASteer 的正点积门控始终沿同一概念朝向工作。
没有非零差分信号时返回零向量。

### 4. 用前 k 个主方向过滤均值差

仍使用上面的未中心化 SVD，保留 $V_k=[v_1,\ldots,v_k]$，计算：

$$
\tilde\mu=V_kV_k^\top\bar x,\qquad s=\operatorname{unit}(\tilde\mu).
$$

`--vector-method pca-filter --pca-rank 4` 使用这个构造，默认 k=4，不能大于 N。
未启用差分归一化时，过滤的就是原始均值差；与归一化组合时，过滤的是归一化差分的均值。
通道数少于 k 时使用全部可用方向。k=1 且均值投影非零时，方向与第三种方法一致；
保留全部差分方向会基本恢复原均值差。
最终仍输出一个向量，线上投影机制不变。

```bash
# 1. 稳健聚合
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases ip_pikachu --steer-target text_value \
  --vector-method robust --robust-groups 5 --offline
# 2. 每对差分归一化后求均值
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases ip_pikachu --steer-target text_value \
  --normalize-differences --offline
# 3. 第一主方向
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases ip_pikachu --steer-target text_value \
  --vector-method pca --offline
# 4. 前四个主方向过滤均值差
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases ip_pikachu --steer-target text_value \
  --vector-method pca-filter --pca-rank 4 --offline
```

添加 `--phase fit` 只构造向量；添加 `--dry-run` 只查看计划。
上述配置同样支持两个 steer target、两种位置平均方式及全部四组 benchmark。
不平均位置时，在每个位置分别应用同一公式；不会把位置和通道拼成一个向量。

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

### 自定义 target / anchor 对照

`--target-anchor-config /path/to/config.json` 只覆盖离线构造负样本时的 anchor。
不指定时仍读取公共 `benchmark/configs/dit_target_anchor.json`。
支持公共配置的平铺 `{ "target": "anchor" }` 格式，以及本目录 `conf.json` 的
`{ "concept_pair": { "target": "anchor" } }` 格式；文件必须包含所选 case 的全部 target，
anchor 必须是字符串，允许空字符串。擦除 target 仍由 benchmark case 定义。

本目录 `conf.json` 设置 `nudity → clothed`，其余 target 的 anchor 均为空字符串。
空 anchor 表示负 prompt 只保留共享上下文，同时去掉 `with` 或 `, … style` 后缀：

| 分组 | 正 prompt 示例 | 使用 conf.json 的负 prompt |
|---|---|---|
| I2P | `a woman on a beach, nudity` | `a woman on a beach, clothed` |
| IP | `tench with Pikachu` | `tench` |
| style | `tench, Van Gogh style` | `tench` |
| Object | `tench with church` | `tench` |

空的是 anchor，负 prompt 仍保留 ImageNet 类名；不会将整条负 prompt 设为空。
该设置不修改 CFG 的 `negative_prompt`，也不修改 benchmark 的评测 prompt、seed、
采样协议、擦除/保留划分或指标计算。方法输出随 steering 向量变化。

```bash
# 自定义 anchor；可加 --phase fit 只统计向量，或 --dry-run 查看计划
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all \
  --target-anchor-config /home/shx/code/ce-new/DiT/steer/conf.json --offline
```

自定义文件的默认结果名追加 `-anchors-<文件名去扩展名>`，例如 `-anchors-conf`，
便于与默认 anchor 的指标并列比较；`--method` 仍可显式指定结果名。
修改同一个文件继续做其他对照时，使用不同 `--method` 保存各组指标。
缓存按实际 anchor 和正负 prompt 匹配，不同文件路径但相同有效配置可复用；
IP/style/Object 的空 anchor 不会命中原 anchor 的向量或逐对差分缓存，
I2P 的 anchor 未改变时仍可复用已有向量。

离线向量按配置缓存。每次运行先在当前 `--run-root` 的对应 model/case 下查找完整匹配的
`metadata.json` 和 `steering.pt`，命中后直接复用；配置变化时自动另存，切回旧配置会再次命中。
新配置保存到可读目录：

```text
runs/<model>/<case>/<steer_target>-<mean 或 per-position>/offline/
  flow-euler40-512x512-bfloat16-pairs50-seed0-<开始时间>/checkpoint/
  steering.pt    # vectors、positive_means、negative_means，均为 CPU FP32
  metadata.json  # 参数、实际 prompt pairs、timesteps、采样配置
```

新构造配置会在目录标签、默认结果名和缓存配置中标明，例如 `pca-filter-k4`、
`robust-g5-i100-tol1e-05`、`unit-diff`。不传新参数时保持原结果名，并继续复用旧均值向量。

启用任一改进后，首次统计另外保存
`checkpoint/paired_differences/<concept>/pair-0001.pt`，每个文件是一个 prompt 对的
`[step, layer, channel]` 或 `[step, layer, position, channel]` 差分。
后续改变构造方法、归一化开关、分组参数或 PCA rank 时，会查找相同采样配置的这些差分，
打印 `[fit:reconstruct]`，直接重算向量，不加载 DiT 或重新采样。
匹配的最终向量仍打印 `[fit:reuse]`。
旧均值缓存没有逐对差分，因此第一次启用这些方法需要补做一次离线统计。
逐对差分使用 CPU FP32 保存，重构时按块读取；不平均位置时磁盘占用会显著增加。
例如 50 对、40 步、24 层、1536 通道的差分，平均模式约 281 MiB；
保留 1024 个位置则约 281 GiB，另需保存最终向量及正负均值。

平均模式的三个张量维度均为 `[concept, step, layer, channel]`；
不平均模式为 `[concept, step, layer, position, channel]`。
metadata 记录目标位置、是否平均以及位置轴对应 text token 还是 image patch。
因此后续可以从已保存均值重新构造方向。缓存匹配包括离线算法版本、模型及 revision、
目标和 anchor、实际正负提示词、seed、目标位置、是否平均、向量构造选项、dtype、步数、分辨率、
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

四种新构造及差分归一化通过数值验证，覆盖异常样本、PCA 不中心化与符号、
全 rank 投影、零差分和逐位置一致性。小型原生 SD3 覆盖两个目标位置 × 两种平均方式，
逐对差分保存、全部构造、从缓存重构且不加载模型、batch=2 向量加载采样和 hook 清理。
默认统计的正负均值与新逐对采集逐元素一致；旧版五组默认向量继续命中缓存。
本次未运行正式模型实验。

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
