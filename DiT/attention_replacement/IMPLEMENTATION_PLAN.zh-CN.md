# Attention Replacement 实验代码实施计划

日期：2026-10-30。状态：**实现前计划**。本次只创建本文档；下文的代码、配置、命令与实验产物均为后续实施内容。

## 1. 已确认的首版方法

冻结原始 DiT。离线分别用 target / anchor 的简短概念 prompt、多组随机 seed 运行原生生成过程，收集指定层、指定采样步的图像侧 attention 输出，按 seed 求均值。在线只执行 user prompt 的正常生成，在对应位置通过平方 L2 距离生成空间掩码，并以离线 anchor 均值硬替换命中的图像 token。

首版固定以下规则：

1. 特征为 **attention 运算完成后、输出投影之前的图像侧 `AV`**，不是 value 投影 `V`、attention map 或整个 block 输出。
2. target 与 anchor 分别构建均值，按相同的层、采样步和空间位置比较。
3. 使用平方 L2 距离和严格的 `d_anc - d_tar > tau`。
4. 替换源为离线 anchor 均值，执行完整 token 向量的硬替换。
5. 层范围、采样时间窗口、离线 seed 数与 `tau` 均可配置，后续人工选择实验值。
6. 多概念分别计算掩码，再取并集；重叠位置的 anchor 选择规则见第 7 节。
7. 保存离线特征与在线特征、距离、掩码，支持后续逐位置分析。

首版不增加 `d_tar < r` 条件，不增加软替换系数、投影擦除、学习式检测器或自动超参数搜索。未来需要距离上限时，可在已有距离图上增加条件，并形成独立的方法配置。

## 2. 仓库接口与接入方式

先支持 `sd35_medium` 和 `flux1_dev`，默认模型沿用现有 DiT 实验的 `sd35_medium`。本地检查到 Diffusers 为 **0.41.0**；后续运行要记录实际版本与实际模块结构。

| 现有入口 | 后续复用内容 |
| --- | --- |
| `benchmark/dit_adapter.py::DiTAdapter` | 基础模型加载、设备映射、CPU offload、原生 pipeline、释放资源 |
| `benchmark/model_profiles.py`、`benchmark/configs/dit_sampling.json` | 模型、分辨率、步数、guidance、文本长度、原生 scheduler 配置 |
| `benchmark/sampling_profile.py::initial_noise` | 按每条记录的 seed，在 CPU 生成 FP32 初始噪声后转换为模型精度 |
| `benchmark/configs/dit_target_anchor.json` | 默认概念对；允许方法配置或 CLI 显式覆盖 |
| `benchmark/dit_cases.py`、`benchmark/configs/dit_cases.json` | 四类实验、测试 prompt、seed 与保留任务 |
| `benchmark/data.py`、`benchmark/adapters.py`、`benchmark/evaluate.py` | 样本清单、插件采样、原模型对照、已有指标与报告 |
| `DiT/early_semantic/`、`DiT/renoise_self_check/` | 参考其模型接入和实验组织；不依赖其方法逻辑 |

实现文件集中在 `DiT/attention_replacement/`。优先以 benchmark 的 `plugin` 接口接入，不修改其他方法。模型采样继续调用原生 pipeline，通过局部 hooks 和逐步回调完成观察与替换，避免另写 Euler/CFG 采样器。

默认采样设置来自当前 benchmark：SD3.5 Medium 为 40 步、guidance 4.5；FLUX.1-dev 为 28 步、内嵌 guidance 3.5。其他数值均在运行配置中解析和记录，不在方法内部另设采样默认值。

## 3. 数学定义与索引约定

用 `site` 表示一个实际 attention 干预点；它包含 block 编号、模块路径和 attention 分支。对于采样执行序号 `k`、site `s`、空间位置 `p`：

\[
I^c_{n,k,s}\in\mathbb R^{H_s\times W_s\times d_s},
\qquad c\in\{\mathrm{tar},\mathrm{anc}\}.
\]

离线原型为：

\[
\mu^c_{k,s}(p)=\frac1N\sum_{n=1}^{N}I^c_{n,k,s}(p).
\]

在线输入 `x = I_user[k,s]`，按通道求和：

\[
d_{tar}(p)=\sum_{j=1}^{d_s}(x_{p,j}-\mu^{tar}_{p,j})^2,
\qquad
d_{anc}(p)=\sum_{j=1}^{d_s}(x_{p,j}-\mu^{anc}_{p,j})^2.
\]

\[
S(p)=d_{anc}(p)-d_{tar}(p),\qquad M(p)=\mathbf1[S(p)>\tau].
\]

\[
I_{new}(p)=(1-M(p))I_{user}(p)+M(p)\mu^{anc}(p).
\]

具体约定：

- 距离是原始特征空间的**平方距离之和**，不除以通道数，不做 L2 归一化、余弦相似度或方差标准化。
- `d_tar`、`d_anc`、`S`、`M` 的单样本形状为 `(H_s,W_s)`；`I_user`、均值和 `I_new` 为 `(H_s,W_s,d_s)`。批量运行增加最前面的 `B` 维。
- 采样时间用实际执行顺序 `step_index=0,1,...,T-1` 定位，同时记录原生 `timestep` 和 `sigma`。FLUX 内部缩放后的时间输入不作为缓存编号。
- layer 范围和 step 范围在配置中都采用 **0-based、左闭右开 `[start,end)`**。人类可读报告同时显示 1-based 编号。
- 默认 step 范围为前 `ceil(0.30*T)` 步：40 步取索引 `0..11`，28 步取索引 `0..8`。比例指执行步数比例，不是 sigma 区间比例。
- 默认 block 范围为模型全部 blocks。FLUX 的双流 blocks 排在单流 blocks 之前，另存原生列表内编号与完整模块路径。
- 不同 attention 分支独立保存，不把 SD3.5 的主 attention 和额外 self-attention 混为同一份特征。

## 4. 文件组织（实现时按工程要求合并模块）

```text
DiT/attention_replacement/
├── IMPLEMENTATION_PLAN.zh-CN.md  # 方法设计记录
├── README.md                    # 实际命令、配置与验证范围
├── run.py                       # collect / sample / benchmark / analyze / evaluate / dry-run
├── method.py                    # 核心算法、原生 hooks、离线缓存与 benchmark 插件
├── cache/                       # 运行后生成的离线缓存
└── results/                     # 运行后生成的诊断、方法图像与 trace
```

模块之间共用同一套距离与替换函数；独立 prompt 采样和 benchmark 采样调用同一份核心方法。

## 5. 配置与初始值

下表中的数值只是首个可运行版本的起始配置，不是已验证的最佳值。

| 参数 | 初始约定 |
| --- | --- |
| `model` | `sd35_medium`；同时实现 `flux1_dev` |
| `offline_seed_count` | 16，可显式覆盖 |
| `offline_seed_start` | 10000；生成并保存完整 seed 列表 |
| `target_prompt` / `anchor_prompt` | 概念配置中的原始短字符串，不自动添加模板 |
| `layer_start` / `layer_end` | `0` / 全部 blocks 的数量 |
| `attention_branches` | 所选 blocks 中支持的图像 attention 分支；逐项列入 site 清单 |
| `step_fraction` | `0.30`；也可显式指定 `step_start` / `step_end`，两种方式互斥 |
| `tau` | `0.0`；支持全局值和概念级覆盖 |
| `replacement` | `anchor_mean`，硬替换 |
| `overlap_policy` | `max_margin`，具体规则见第 7 节 |
| `batch_size` | 初次调试为 1；张量接口保留 batch 维 |
| 模型精度 | 沿用 benchmark，默认 BF16 |
| 均值累计与距离运算 | FP32 |
| 离线原始特征、在线特征保存精度 | 保留模型输出 dtype |
| 在线完整 trace | 默认开启，记录所选 layer / step；后续可显式关闭以单独测量部署开销 |

默认概念对从 `benchmark/configs/dit_target_anchor.json` 读取，例如仓库现值为 `nudity -> dressed`。可以在方法配置中显式指定 `nudity -> clothed`；两者形成不同缓存，不自动将同义词视为同一 anchor。用户 prompt 原样输入，不加入 target 或 anchor 文本。

## 6. 阶段一：离线收集与原型构建

### 6.1 生成准备

1. 解析模型 profile、概念对、seed 列表、层范围和采样窗口，输出完整展开配置。
2. 加载模型并建立 site 清单，记录完整模块路径、输出维度、图像 token 数、网格形状和分支类型。
3. target 与 anchor 使用相同的 seed 列表，同一 seed 对应相同的初始噪声；两条轨迹分别按自己的 prompt 完整采样。
4. 同 seed 不代表两条中间轨迹或空间语义对齐。首版按用户方案保留固定位置均值，不额外做空间配准。

### 6.2 捕获与保存

离线 hooks 只读，不执行替换。每个采样步仅保存选中 sites 的条件分支图像侧 `AV`，继续使用原始输出完成正常采样。

每个 seed 保存：target / anchor 的最终图像、原始 prompt、初始 seed、所选 sites / steps 的完整激活分片、实际 timestep/sigma 序列。即使只采集前 30% 的激活，也完成整条生成轨迹，以便检查短 prompt 实际生成了什么。

同一个 anchor 字符串在相同模型和采样配置下可以复用已有离线收集结果，不为多个 target 重复生成相同 anchor 数据。

### 6.3 均值计算

原始特征先按原 dtype 保存；均值采用 FP32、按 seed 流式累计。每份原型只对自己的 `(concept, role, step, site, position)` 求平均，不跨 seed 之外的维度平均。保存 FP32 均值及实际累计样本数。

默认所有完成的 seed 均参与均值，不引入目标检测器自动过滤样本。离线最终图像供人工和后续分析使用，不能在没有记录的情况下删 seed 或改变原型。

### 6.4 缓存身份与复用

缓存元数据至少包含：模型来源和 revision、运行依赖版本、attention 接口版本、模型 dtype、分辨率、完整 scheduler 配置及实际时间表、guidance / negative prompt、文本长度、概念原文、seed 列表、site 清单、token 网格与所选收集范围。

- `tau` 不改变离线原型，调整它时复用原型。
- 在线 layer / step 窗口是已收集范围的子集时可直接复用。
- 增加未收集的层或步，需要补收集对应数据。
- 改变模型、分辨率、采样总步数或其他会改变轨迹的配置，需要对应的新缓存；首版不插值旧时间表或缩放旧空间原型。
- 在线使用的均值应引用其元数据和内容摘要，避免同一名称下更换原型而误用旧图像结果。

## 7. 阶段二：在线采样与多概念处理

每个正常 user 前向到达指定 site 时，执行：

```text
I_pre = 当前条件分支图像侧 AV

对每个 target-anchor pair：
    d_tar = sum_channels((float32(I_pre) - mean_tar) ** 2)
    d_anc = sum_channels((float32(I_pre) - mean_anc) ** 2)
    margin = d_anc - d_tar
    M_pair = margin > tau_pair

M_union = 所有 M_pair 的逻辑 OR
确定每个命中位置使用的 anchor
I_post = 在 M_union 内替换为对应 anchor 均值，其余保持 I_pre
按模型原生 dtype 返回，继续原生输出投影和后续网络
```

必须在同一份 `I_pre` 上计算所有概念的距离与掩码，随后统一替换一次；不按概念顺序重复修改输入再计算下一概念。

### 多概念重叠的首版约定

用户已确定掩码并集；对于同一位置命中多个不同 anchor 的情况，采用一个明确的工程规则：

\[
k^*(p)=\arg\max_{k:M_k(p)=1}S_k(p).
\]

使用第 `k*` 个概念的 anchor 均值。相同 margin 时按配置中的概念顺序选择；未触发位置的 `winner` 记为 `-1`。所有概念共用同一个 anchor 时，直接使用其均值。

`max_margin` 仅用于首版冲突处理，不将 margin 解释为概率。记录各概念 mask、并集、重叠次数与 winner 图；首版不同时实现平均融合或顺序替换等其他策略。

窗口之外直接沿用原生输出，不进行距离计算或替换。在线没有额外 target / anchor 完整 DiT 前向，模型调用次数与同一采样配置的原生生成一致。

## 8. Attention 接口与 CFG 分支

### 8.1 优先使用原生模块 hooks

当前本地版本可通过以下位置获取所需特征，优先保留原生 attention processor / kernel，不显式物化完整 attention map：

| 模型与分支 | 捕获和替换位置 |
| --- | --- |
| SD3.5 主 joint attention | `transformer_blocks[i].attn.to_out[0]` 的 forward pre-hook；此时已拆出图像行 |
| SD3.5 额外 `attn2`（若实际存在） | 其 `to_out[0]` 的 forward pre-hook；独立 site，和主 attention 分开缓存 |
| FLUX 双流块 | `transformer_blocks[i].attn.to_out[0]` 的 forward pre-hook；仅图像输出 |
| FLUX 单流块 | `single_transformer_blocks[i].proj_out` 的 forward pre-hook；输入为 `cat([AV, MLP], channel_dim)`，仅处理图像行中前 `attention_dim` 个通道 |

FLUX 单流块的文本行数从当前 block 输入读取，不写死为 256 或 512。捕获时剔除文本行和 MLP 通道；替换后按原布局放回，文本行、MLP 通道及未命中图像行保留原值。SD3.5 与 FLUX 的联合 token 排列不能共用一个假定的切片顺序。

实际模块命名、分支和维度在运行时列出。上述 hook 必须通过空干预对照确认捕获点与 `AV` 一致；不把捕获不到的分支静默当作已支持。

### 8.2 条件分支约定

- SD3.5 原生 CFG batch 为 `[negative, positive]`。离线收集和在线替换均只作用于 positive 部分，negative 部分保持原生结果。
- 无 CFG 时，整个当前 batch 为条件样本。
- FLUX.1-dev 沿用 profile 的内嵌 guidance，默认没有额外 true-CFG negative 前向。首版按此 profile 实现；额外 true-CFG 或 skip-layer-guidance 路径需要独立接入验证，不能混入默认分支判定。
- `step_index` 由采样步上下文维护，不根据 attention 或 transformer 总调用次数推断。开始一次生成时设为 0，原生 step-end callback 完成该步后推进；每次模型前向核对当前 scheduler 时间。
- hooks 只改变当前输出张量，不更新权重。生成结束清理当前样本上下文，`close()` 卸载 hooks 并释放缓存引用。

## 9. 保存格式与分析产物

### 9.1 目录草案

```text
cache/<model>/<collection_id>/
├── metadata.json
├── sites.json
├── trajectories/<concept_id>/<target_or_anchor>/<seed>/
│   ├── metadata.json
│   ├── final.png
│   └── step_<k>/site_<s>.safetensors
└── means/<concept_id>/<target_or_anchor>/step_<k>/site_<s>.safetensors

results/<model>/<run_id>/
├── run.json
├── images/<sample_id>.png
├── traces/<sample_id>/
│   ├── metadata.json
│   ├── statistics.csv
│   └── step_<k>/site_<s>.safetensors
└── analysis/<sample_id>/
    ├── layer_step_mask_ratio.png
    ├── layer_step_margin.png
    └── spatial_maps/
```

使用 `safetensors` 保存张量，JSON/CSV 保存索引、配置和标量；避免把大张量写入 JSON。内部可使用稳定 site 编号，但必须在 `sites.json` 中映射到完整模型路径。

### 9.2 在线默认完整 trace

每个选中 `(sample,step,site)` 保存：

| 字段 | 形状／精度 | 含义 |
| --- | --- | --- |
| `I_user_pre` | `(H,W,d)`，模型输出 dtype | 当前受干预轨迹中，本 site 替换前的输出 |
| `I_new` | `(H,W,d)`，模型输出 dtype | 实际送入后续输出投影的特征 |
| `d_tar/<concept>` | `(H,W)`，FP32 | 对 target 均值的平方 L2 距离 |
| `d_anc/<concept>` | `(H,W)`，FP32 | 对 anchor 均值的平方 L2 距离 |
| `margin/<concept>` | `(H,W)`，FP32 | `d_anc - d_tar` |
| `mask/<concept>` | `(H,W)`，bool | 该概念的阈值判断 |
| `mask_union` | `(H,W)`，bool | 最终需要替换的位置 |
| `winner` | `(H,W)`，整数 | 实际使用哪个概念的 anchor；未命中为 `-1` |

这里 `D_tar/D_anc` 已按用户补充解释为降维后的距离图，不再默认为 `(H,W,d)` 的有符号通道差分。完整 `I_user_pre` 和离线均值可以重建通道差分；若后续需要直接导出，再增加显式诊断选项。

同一 site 的 `I_user_pre` 只保存一次，不为每个概念重复保存。anchor 均值通过离线缓存引用，不复制进每个在线 trace。

### 9.3 标量与热图

逐层、逐步记录 mask 面积比例、每个概念的命中数、重叠比例、margin/距离的分位数、修改量范数、原始与替换后特征范数，以及该层的计算和张量保存耗时。

分析入口仅读取已有 trace，导出空间 mask / margin 热图、时间—层热图和 CSV。所有分析可在 CPU 上运行。插值放大的可视化图片不替代原始 token 网格。

当前 `I_user_pre` 不能标成“未干预原模型输出”，因为它已受到更早层和更早步干预的影响。若要比较原模型轨迹，使用相同 prompt 和 seed 另跑一次 observe-only 对照，并分开保存。

### 9.4 内存和日志开销

离线原始特征量随 `2*N*steps*sites*H*W*d` 增长，均值量随 `2*steps*sites*H*W*d` 增长。启动预览根据实际模型维度估算保存量。

完整特征按 step/site 顺序转移到 CPU 并分片写盘，不在 GPU 中累积所有 seed 或在线轨迹。默认使用简单顺序写入，先完成正确性；GPU 只保留当前计算所需的原型和输出。

完整 trace 的耗时纳入本次运行总耗时，并单独记录 I/O 时间。关闭 trace 的效率测量要显式标记，不能与完整 trace 运行混为同一耗时结果。

## 10. 命令行与 benchmark 规划

以下为拟定接口，文件尚未实现，当前不可执行。

```bash
# 预览：概念对、seed、实际 layer/step 范围、输出位置与缓存量。
python DiT/attention_replacement/run.py collect --targets nudity --anchor clothed --dry-run

# 离线收集与均值；默认模型和采样设置取 benchmark profile。
python DiT/attention_replacement/run.py collect --targets nudity --anchor clothed \
  --offline-seed-count 16 --step-fraction 0.30 --offline

# 指定 user prompt 做诊断，默认使用 anchor 均值硬替换并保存完整 trace。
python DiT/attention_replacement/run.py sample --cache <collection_id> \
  --prompt "<user prompt>" --seed 0 --tau 0 --offline

# 仅观察距离和 mask，不替换；不改变 user prompt 或采样轨迹。
python DiT/attention_replacement/run.py sample --cache <collection_id> \
  --prompt "<user prompt>" --seed 0 --mode observe --offline

# 读取已有结果生成分析图。
python DiT/attention_replacement/run.py analyze --run <run_id>

# 使用原有 case 组织完整实验。
python DiT/attention_replacement/run.py sample --cases all --method-config <config.json> --offline
python DiT/attention_replacement/run.py evaluate --run <run_id> --offline
```

benchmark 插件拟提供 `DiT.attention_replacement.method:create(config)`，返回符合 `generate(records,sampling_config)` / `close()` 的对象，并暴露原生 pipeline 供现有采样计时器使用。

- 方法图像和 trace 位于本方法目录；benchmark 的指标和报告继续通过现有流程组织。
- plugin 配置将完整方法参数、概念对、原型摘要和源码版本纳入方法身份；不同 `tau` 或干预窗口形成不同方法结果，复用同一兼容离线缓存。
- I2P 与对应 COCO 保留任务使用同一组原型和参数；I2P 测试 prompt 不用于离线原型构建或阈值选择。
- IP/style/object 沿用现有原始 prompt、seed、分类器和保留协议。原模型图像按 benchmark 的原有身份复用。
- 默认只计算用户指定配置，不自动批量扫参。原模型参考生成属于显式对照阶段，不作为在线每个样本的附加前向。
- `--dry-run` 不加载模型、不生成图像、不创建缓存或结果目录。

## 11. 后续实施顺序与验收

### 第一步：接通只读捕获

实现 site 枚举、原生 hooks、step 上下文及条件分支切片。先验证 SD3.5，再覆盖 FLUX 双流与单流。用 observe-only 模式确认没有改变网络计算和采样轨迹。

验收：能读取预期形状；文本行、MLP 分支、negative batch 不混入；空干预或仅观察时，使用相同输入和 backend 的输出与原生路径一致。

### 第二步：完成离线产物

实现短 prompt、多 seed 收集、逐位置 FP32 均值、完整原始张量与元数据保存、缓存加载。用小规模原始特征重新计算均值，与保存均值核对。

验收：target/anchor 同 seed 初始噪声一致；每份均值计数等于 N；不会把不同层、时间或分支求在一起；时间表与空间形状能准确还原。

### 第三步：实现精确的距离和硬替换

先用小张量验证距离、阈值和替换语义，再接真实模型。数学检查至少包括：

- `x = mean_tar` 且两个均值不同时，`tau=0` 应触发；`x = mean_anc` 不触发。
- 两个均值完全相同时，margin 为零；严格 `>` 下 `tau=0` 不触发。
- 距离按通道求和，结果为 `(B,H,W)`；未命中位置和未选中分支保持原值。
- 窗口以实际 step 编号选择；窗口结束后直接恢复原生计算。
- 重新载入保存的均值、mask 和 `I_user_pre`，可以复算距离和实际 `I_new`。

真实权重先运行单概念、单 user prompt、单 seed，并保留正常采样总步数。确认没有在线附加 target/anchor transformer 前向。

### 第四步：接入多概念与分析

实现同一 `I_pre` 上的多概念距离、mask 并集、最大 margin 选择和重叠日志。小张量验证概念顺序仅在 margin 完全相同时影响 winner；未触发概念不参与选择。

验收：共享 anchor 与单次并集替换一致；不同 anchor 的重叠结果可由 winner 图重建；热图和统计均来自实际保存张量。

### 第五步：接入完整 benchmark

先运行两种模型的小样本接口检查，再运行所选完整 case。检查 target、邻近保留概念、同图非目标内容和 COCO 保留指标，并记录 mask、距离和修改量。

完整实验之前完成一次 observe-only / 原生对照和一次日志开关对照。数值正确性、接口正确性与擦除效果分别报告；首版可以执行并不等于已经验证其擦除效果。

## 12. 后续实验入口

实现完成后，可用同一份兼容缓存手动调整 `tau`、已有范围内的层子集和 step 子集。扩展到不同采样总步数或更大收集窗口时，再生成对应缓存。

首轮重点回答：

1. 距离 margin 是否能区分目标样本与保留样本，mask 是否随 user prompt 的实际内容变化？
2. 固定空间位置的 anchor 均值替换，能否在擦除目标的同时保留用户原有的姿态、身份与背景？
3. 结束早期干预后，目标是否在后续层和后续采样步恢复？
4. 多概念联合时，mask 并集和 anchor 冲突如何影响保留能力？

相关工作对照入口：[ConceptAttention](https://arxiv.org/abs/2502.04320)、[UVR](https://arxiv.org/abs/2606.06875)、[ActErase](https://arxiv.org/abs/2601.00267)、[PER](https://arxiv.org/abs/2606.31699)。它们用于解释表征位置和比较实验，不改变本计划第 1 节确定的首版算法。
