# Steer：SD3.5 概念擦除基础实现

从本仓库 CASteer 的 SD3.5 迁移版本中独立出基础算法，作为后续实验起点。
默认 SD3.5 Medium，也可选择同架构的 SD3 Medium；当前不包含 FLUX。
运行时只依赖公共 `benchmark` 和本目录，不导入 `baseline/CASteer`。

## 文件与算法入口

| 文件 | 内容 |
|---|---|
| `method.py` | 向量构造、投影干预、SD3 hooks、离线统计、benchmark adapter |
| `diagnostics.py` | 按评测概念汇总原始 score、保存二维分数图和统计表 |
| `run.py` | 参数入口与 fit → prepare → sample → evaluate 编排 |
| `run.sh` | 转发命令行参数 |
| `imagenet_classes.txt` | 从 CASteer 复制的 ImageNet 类名，保持原配对模板 |
| `prompts/subject.json` | 第 2 种：以目标为主体的短模板 |
| `prompts/matched.json` | 第 3 种：共享具体内容的对照模板 |
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

## Score 与随机方向对照

默认仍计算每个位置的 `score = sum(output * vector, channel)`，
使用 `output += beta * max(score, 0) * vector`；`--no-clipping` 改用有符号 score。
联合擦除按目标顺序依次应用，后一个目标的 score 使用前一个目标干预后的 output。

`--no-score-gate` 将 score 系数固定为 1，直接执行 `output += beta * vector`。
向量仍是原来的方向，不将向量设成全 1；此模式完全跳过点积计算，clipping 不再参与。

`--steer-noise all` 和 `--steer-noise every` 用随机方向替换离线向量，直接跳过 fit，
也不读取 `steering.pt`。从独立 RNG 抽取标准高斯噪声，再按 channel 归一化为单位长度，
使方向长度与离线向量一致；`--seed` 控制这个随机对照，不改变 benchmark 的图片 seed。

- `all`：每个评测类别有自己的固定噪声，同类的所有 prompt、seed、层和时间步复用它。
  IP 的 Snoopy、Pikachu、Mickey 各不相同；style、Object 同理。I2P 和 MS-COCO 各自作为一组。
- `every`：每个样本在每次 steering hook（每层、每步）重新抽取噪声。
  按样本使用独立随机流，因此更改 batch size 或断点补采样不会改变该样本的随机对照。

位置平均模式让各位置共用方向；`--no-average-patches` 为各位置抽取各自的方向。
联合擦除时，每个擦除目标仍有一条独立随机方向，按原目标顺序应用。

```bash
# 学习向量，直接加 beta * vector
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --subset-eval --no-score-gate --offline
# 每个评测类别固定噪声，保留原 score gate
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --subset-eval --steer-noise all --offline
# 每次 steering 重抽噪声；可与 --no-score-gate 组合
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --subset-eval --steer-noise every --no-score-gate --offline
```

默认结果名会标注 `no-score-gate`、`noise-all-seed0` 或 `noise-every-seed0`，
这些选项也会进入方法图与计划的缓存身份。改变 gate 或噪声模式会生成对应的独立图集。
学习向量模式改变 gate 不需要重新 fit；噪声模式不需要离线向量。

## Score 追踪

Score 追踪默认关闭，只有显式传入 `--trace-score` 才开启；使用 score gate 本身不会开启追踪。
`--no-score-gate` 会同时关闭追踪，因为该模式不计算点积。
追踪只读取真实 steering 位置的 conditional CFG 分支，不额外运行模型或评价器。
`text_to_image` 记录投影后的文本到图像贡献与向量的点积；`text_value` 记录文本 Value 与向量的点积。

每个方法图目录下保存 `score-traces/`：

```text
score-traces/
  concept-means.csv         # 各评测概念、各擦除方向的总体平均值
  step-layer-means.csv      # 逐时间步、逐层的平均值
  summary.json             # 参数、统计口径与样本数
  pikachu/                 # IP/style/Object 每个评测概念一组；I2P/COCO 分别为 i2p/mscoco
    scores.pt              # 对样本平均后的全部原始 score，保留每层、每步和每个位置
    Pikachu-overview.png    # step × layer 的 raw score、实际 gate、正分数比例，以及位置均值
    Pikachu-positions.png   # 最多 4 个时间步 × 4 层的位置图预览；完整数据在 scores.pt
```

二维 patch 图来自沿 channel 求和后的点积，保留正负值，不在保存前裁剪。
图像模式下 `score_mean` 为 `[擦除目标, step, layer, y, x]`；
文本模式为 `[擦除目标, step, layer, token]`，绘图按 token 位置展示，不将文本 token 解释成图像网格。
各 prompt 的文本只按位置索引平均，不做词义对齐。PNG 对同一实验的各类别使用相同色标。

I2P 对所有采样统一求平均，COCO 单独求平均；IP、style、Object 分别对各个评测概念求平均。
每张样本权重相同，按批持久化，断点补采样按 `sample_id` 去重，不会重复计入。
缓存命中而没有生成的图片不新增 score；原模型参考图也不计入方法的追踪。
`gate_mean` 与 `positive_fraction` 在逐样本裁剪、判断符号后再平均，
因此不会用“平均 score 的正负”代替真实 gate 的开启比例。

例如读取一个具体位置图：

```python
state = torch.load("/实际方法图目录/score-traces/pikachu/scores.pt", weights_only=True)
score_map = state["score_mean"][0, 10, 5]  # 第一个擦除目标、step 10、layer 5 的二维图
```

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

默认 `--prompt-mode casteer`，提示词模板与 CASteer 一致：IP/Object 为 `ImageNet 类名 with concept`，
style 为 `ImageNet 类名, concept style`，默认各 50 对；I2P 为 15 类人物 × 14 类场景，
共 210 对。默认负 prompt 只保留共享上下文；nudity 的负 prompt 额外包含 `clothed`。
正负两侧使用相同 seed、相同初始噪声、相同原生调度器；统计时保留未干预轨迹。
执行所有去噪步，但 `output_type="latent"` 跳过 VAE 图像解码；不调用外部检测器或评价指标。

### 自定义 target / anchor 对照

默认使用“上下文 + target / 原上下文”，仅 nudity 使用“上下文 + clothed”作为负 prompt：

| 分组 | 正 prompt 示例 | 默认负 prompt |
|---|---|---|
| I2P | `a woman on a beach, nudity` | `a woman on a beach, clothed` |
| IP | `tench with Pikachu` | `tench` |
| style | `tench, Van Gogh style` | `tench` |
| Object | `tench with church` | `tench` |

`--use-anchor` 开启“上下文 + anchor”负 prompt，默认关闭。开启后，
`--target-anchor-config /path/to/config.json` 指定 anchor 配对；
未指定配置文件时读取公共 `benchmark/configs/dit_target_anchor.json`。
例如公共配置下，Pikachu 的负 prompt 是 `tench with Cartoon Character`，
church 的负 prompt 是 `tench with Building`。
单独指定 `--target-anchor-config` 不会开启 anchor 模式，也不会读取该文件。
支持公共配置的平铺 `{ "target": "anchor" }` 格式，以及本目录 `conf.json` 的
`{ "concept_pair": { "target": "anchor" } }` 格式；文件必须包含所选 case 的全部 target，
anchor 必须是字符串，允许空字符串。擦除 target 仍由 benchmark case 定义。

空 anchor 表示负 prompt 只保留共享上下文，同时去掉 `with` 或 `, … style` 后缀；
不会将整条负 prompt 设为空。本目录 `conf.json` 当前设置 `nudity → clothed`，
其余 target 的 anchor 均为空字符串，因此开启 anchor 模式并使用此文件与默认提示词相同。
该设置不修改 CFG 的 `negative_prompt`，也不修改 benchmark 的评测 prompt、seed、
采样协议、擦除/保留划分或指标计算。方法输出随 steering 向量变化。

```bash
# 默认：仅 nudity 负侧包含 clothed，其余负侧只保留上下文
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --offline
# 开启公共配置中的 anchor
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --use-anchor --offline
# 开启自定义 anchor；可加 --phase fit 只统计向量，或 --dry-run 查看计划
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all \
  --use-anchor --target-anchor-config /home/shx/code/ce-new/DiT/steer/conf.json --offline
```

默认结果名追加 `-context`；开启 anchor 后追加 `-anchors`，
使用自定义文件时为 `-anchors-<文件名去扩展名>`，例如 `-anchors-conf`。
这些名称用于区分提示词对照与历史结果；`--method` 仍可显式指定结果名。
修改同一个文件继续做其他对照时，使用不同 `--method` 保存各组指标。
缓存按实际 anchor 和正负 prompt 匹配，不同文件路径但相同有效配置可复用；
默认 IP/style/Object 不会命中旧的非空 anchor 向量或逐对差分缓存；
I2P 的 anchor 未改变时、以及已有空 anchor 配置与默认提示词一致时，仍可复用。
开关不同但实际 anchor 和提示词相同，也可复用同一份离线统计。

### 三种可选提示词构造

`--prompt-mode` 与 `--use-anchor` 独立。前者决定模板，后者决定负侧是否填入配置中的
anchor；默认负侧只保留上下文，nudity 仍使用 `clothed`。

| 模式 | 默认对数 | 正 prompt 示例 | 默认负 prompt |
|---|---:|---|---|
| `casteer`（默认） | I2P 210，其他 50 | `tench with Pikachu` | `tench` |
| `bare`：纯概念词 | 1 | `Pikachu` | 空字符串 |
| `subject`：以目标为主体的短模板 | 8 | `a photo of Pikachu` | `a photo` |
| `matched`：共享具体内容的对照 | 8 | `a scene in a park, with Pikachu as the main subject` | `a scene in a park` |

开启 `--use-anchor` 时，以上后三行的负侧分别变为 `Cartoon Character`、
`a photo of Cartoon Character`、`a scene in a park, with Cartoon Character as the main subject`。
纯概念词模式直接使用 target/anchor 原文；例如 style 为 `Van Gogh` / 空字符串，
nudity 为 `nudity` / `clothed`。只有 `bare` 的空 anchor 会使整条负 prompt 为空。

**第 2 种使用 8 个短模板。** 每对使用相同框架，只添加或替换概念：

| 分组 | 8 个基础框架（按顺序） | 正侧追加；anchor 非空时负侧同样追加 |
|---|---|---|
| IP/Object | `a photo`、`an image`、`a picture`、`a close-up`、`a full view`、`a front view`、`a side view`、`a centered view` | ` of {concept}` |
| style | `a painting`、`an artwork`、`an image`、`a picture`、`an illustration`、`a detailed painting`、`a small painting`、`a large painting` | ` in the style of {concept}` |
| I2P | `a photo of a person`、`an image of a person`、`a picture of a person`、`a portrait of a person`、`a close-up of a person`、`a full view of a person`、`a front view of a person`、`a side view of a person` | `, {concept}` |

例如 style 是 `a painting in the style of Van Gogh` / `a painting`；
I2P 是 `a photo of a person, nudity` / `a photo of a person, clothed`。
空 anchor 时删除整个后缀，避免留下没有宾语的 `of` 或 `in the style of`。

**第 3 种使用 8 个具体内容模板。** 每对共享场景或绘画主体：

| 分组 | 8 个基础内容（按顺序） | 正侧追加；anchor 非空时负侧同样追加 |
|---|---|---|
| IP/Object | `a scene in a park`、`a scene on a city street`、`a scene beside a lake`、`a scene near a forest`、`a scene at the beach`、`a scene in a garden`、`a scene with distant mountains`、`a scene against a plain background` | `, with {concept} as the main subject` |
| style | `a landscape painting with mountains and a river`、`a portrait painting of a person`、`a still-life painting of fruit on a table`、`a painting of a city street`、`a painting of a garden with flowers`、`a painting of a forest path`、`a painting of a beach and the sea`、`a painting of a room with a window` | ` in the style of {concept}` |
| I2P | `a person standing in a room`、`a person sitting on a chair`、`a person walking on a beach`、`a person standing in a garden`、`a person beside a window`、`a person in a studio`、`a person near a swimming pool`、`a person against a plain background` | `, {concept}` |

例如 style 是 `a landscape painting with mountains and a river in the style of Van Gogh` /
`a landscape painting with mountains and a river`。IP/Object 显式把 target 作为主体，
不再沿用 ImageNet 类名作为另一个主体。“共享内容”指提示词中的基础内容相同；
正负轨迹使用相同初始噪声，但不保证其中的图像布局完全一致。

第 2、3 种模板分别保存在 `prompts/subject.json` 和 `prompts/matched.json`，
直接修改 JSON 即可调整。文件按 `ip_object`（IP/Object 共用）、`style`、`i2p` 分组，
每组的 `contexts` 保存共享上下文，`suffix` 保存带 `{concept}` 占位符的概念后缀。
正侧将 `{concept}` 填为 target，负侧仅在 anchor 非空时填为 anchor；否则只用上下文。
`prompt_pairs()` 负责读取和组合。实际正负提示词参与缓存匹配，修改有效模板后会匹配新配置；
只改变 JSON 排版不会触发重新统计。这些模板用于对照实验，效果由后续实验判断。

```bash
# 1. 纯概念词；添加 --use-anchor 可启用 target/anchor 对照
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --prompt-mode bare --offline
# 2. 以目标为主体的短模板
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --prompt-mode subject --offline
# 3. 共享具体内容的概念对照
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --prompt-mode matched --offline
```

`--calibration-prompts N` 选取模板顺序中的前 N 对。新模式省略该参数时分别使用
1/8/8 对；超过模板总数会在入口报错，不重复相同提示词凑数量。
纯概念词只有一对、一个固定 seed，适合先用默认均值差做对照。
单对数据不能提供 PCA、稳健统计所需的多对变化；默认 `pca-filter --pca-rank 4`
因此不适用于 `bare`，如需检查单方向退化情形应显式设 `--pca-rank 1`。
`bare` 的自动 robust 分组数为 1，此时稳健聚合也退化为该单对差分。

新模式在默认结果名及新离线目录标签中追加 `-prompts-bare`、`-prompts-subject` 或
`-prompts-matched`。缓存包含模式与实际正负提示词，不会在不同模式之间混用；
旧 metadata 未记录模式时按 `casteer` 处理，继续复用配置一致的旧向量。
两种 steering 位置、两种位置平均方式以及现有四类向量构造都可与这些模式组合使用，
PCA rank 与 robust 分组数仍须不超过实际对数。benchmark 评测提示词与采样协议不变。

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
`robust-g5-i100-tol1e-05`、`unit-diff`。未选择构造改进时继续使用均值差，
实际提示词及采样配置一致时复用旧均值向量。

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

采样时优先生成被擦除概念，再生成保留概念；方法图先于原模型参考图生成。
`--subset-eval` 开启快速子集：I2P、COCO 默认各取前 20 条；IP、style 每个概念取前 5 个模板，
保留各模板的全部 seed；Object 每个概念取前 5 张。因此 IP 每个 case 共 60 张、style 共 45 张、Object 共 50 张。
I2P 默认从 sexual 测试集取前 N 条，`--with-q16` 时从全量测试集取前 N 条。

```bash
# 使用默认子集
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --subset-eval --offline
# 按任务分别指定数量
bash /home/shx/code/ce-new/DiT/steer/run.sh --cases all --subset-eval \
  --i2p-test-num 20 --mscoco-test-num 20 --ip-test-num 5 --style-test-num 5 --object-test-num 5 --offline
```

只有开启 `--subset-eval` 后，五个 `--*-test-num` 才生效；否则沿用完整评测配置。
子集保留原 prompt、样本 ID 和 seed，不影响离线向量构造数量，仍可复用相同配置的离线向量。
分阶段运行时使用相同的子集参数；`--max-records` 若同时指定，会在子集上进一步截取。

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
