# queryGraft

方法公式、符号与设计依据见 [方法定义](README.zh-CN.md)。`method.py` 集中模型接入与算法，`source.py` 提供 source token 分组和 attention 控制，`run.py` 提供命令行与 benchmark，`diagnostics.py` 负责逐步诊断图和数值保存；`run.sh` 提供默认实验入口。

默认设置是：独立编码 target、anchor、background 并取各自第一个序列 token；直接用同层原始 image AV 与 concept AV 的点积定位，不做 L2 归一化；每层先在概念维 softmax，再对所选定位层的软分数图等权平均。只有一个 target 时，选取 target 热图严格高于其空间均值的位置；多个 target 时，每个位置选取全部概念中分数最高者，仅将获胜的 target 区域映射给对应 anchor。默认 `rho=0`，只要掩码非空，就使用相同当前 latent 的 anchor 条件激活，在最后一层原始 AV 处硬替换。

`--my-layer-aggre` 显式开启此前的逐层二值掩码投票，使用 `vote_fraction > eta`；`--my-layer-policy` 显式开启此前由 `--mask-rule` 选择的 target/anchor/background 判据。两个开关默认均关闭，配置键分别为 `adapter.query_graft.my_layer_aggre`、`my_layer_policy`。

| 参数 | 跨层汇总 | 区域判据应用位置 |
| --- | --- | --- |
| 默认 | 各层软分数图平均 | 对平均图应用 ConceptAttention 单类／多类规则 |
| `--my-layer-aggre` | 各层二值掩码投票 | 每层先应用 ConceptAttention 规则，再按 `eta` 汇总 |
| `--my-layer-policy` | 各层软分数图平均 | 对平均图应用 `--mask-rule` 指定的三概念判据 |
| 两个开关同时指定 | 各层二值掩码投票 | 每层先应用 `--mask-rule` 指定的判据，再投票 |

```bash
# ConceptAttention 的默认汇总与区域判据。
python DiT/queryGraft/run.py benchmark --cases i2p --dry-run

# 恢复此前的逐层二值投票与 target/anchor/background 判据。
python DiT/queryGraft/run.py benchmark --cases i2p \
  --my-layer-aggre --my-layer-policy --mask-rule target_max --eta 0 --dry-run
```

concept 分支默认遵循 [ConceptAttention 论文第 4.1 节式 (7)–(9)](https://arxiv.org/html/2502.04320v1#S4.SS1)：concept query 只读取 image 与本组 concept 的 key/value，保留 target、anchor、background 之间的 self-attention。user prompt 的 key/value 默认不进入 concept 分支；原始 image 与 prompt 仍互相读取。显式添加 `--concept-include-prompt` 可恢复之前将 prompt 也纳入 concept attention 的行为；配置键为 `adapter.query_graft.concept_include_prompt`，默认 `false`。该设置适用于 SD3.5、FLUX 双流与单流层，独立于 `--source-mode`。

官方 [`ModifiedDoubleStreamBlock`](https://github.com/helblazer811/ConceptAttention/blob/master/concept_attention/flux/dit_block.py) 同样分别计算 text-image 和 concept-image attention，后者的 K/V 不拼接 prompt token。

concept 单独构造调制条件：将 pooled CLIP 输入置零，保留原生 timestep 与 guidance，以及文本投影中的 bias。概念继续复用 text 分支的归一化、调制、投影和 MLP 参数；attention 前调制及 attention／MLP 后的门控残差均使用这一独立条件。FLUX 单流层使用对应原生融合残差更新，SD3 最后一层沿用 `context_pre_only`。user/image 路径和 anchor source 保留各自原生条件。

```bash
# 默认 concept 读取 image + concept；显式加入 user prompt K/V：
python DiT/queryGraft/run.py benchmark --cases i2p --concept-include-prompt --dry-run
```

## 运行

在仓库根目录执行，使用安装了 `benchmark/requirements.txt` 的环境。支持现有 `sd35_medium` 和 `flux1_dev` 模型配置；直接使用原生 pipeline、scheduler、seed 与 guidance 设置。

```bash
# 预览四类 benchmark，打印配置和图片数量；不加载模型、不创建实验产物。
python DiT/queryGraft/run.py benchmark --cases all --dry-run

# 默认全部实验：I2P 擦除与 COCO 保留，以及 IP、style、object。
bash DiT/queryGraft/run.sh

# 单独选择实验组、模型；通用参数沿用 benchmark。
python DiT/queryGraft/run.py benchmark --model flux1_dev --cases ip --offline
python DiT/queryGraft/run.py benchmark --cases i2p --max-records 4 --phase sample

# 独立 prompt；可多次指定 --prompt 进行批量生成。
python DiT/queryGraft/run.py sample --model sd35_medium \
  --pairs '[{"target":"nudity","anchor":"clothed"}]' \
  --prompt 'a person in a garden' --seed 42
```

`--cases` 接受 `all/i2p/ip/style/object` 或逗号分隔的具体 case ID；`--tasks` 可进一步选择任务，例如 `--cases i2p --tasks mscoco`。

现有 benchmark 包含多 target 用例，因此实现提供逐 pair 扩展：每对 target/anchor/background 默认独立读取同一 image 路径与本组概念，指定 `--concept-include-prompt` 时再加入 user prompt，不跨 pair 读取。默认多类区域判据在全部 `3 × pair 数` 个概念通道间 softmax、跨层平均并 argmax，anchor/background 获胜的位置不编辑；并列最高时按配置中的概念顺序选择。启用 `--my-layer-policy` 时恢复每组三概念之间的 softmax 与独立判据。面积门控按 pair 计算；掩码重叠时，投票模式按支持票比例选择，软图平均模式按平均 target 分数选择，平票按 pair 顺序选择。每个触发的不同 anchor 从同一个当前 latent 单独前向；相同 anchor 复用 source。

## 定位与替换参数

| 设计项 | 参数 | 默认值 | 备选 |
| --- | --- | --- | --- |
| 替换表示 | `--representation` | `av` | `residual`、`block_output`、`velocity` |
| 定位层集 | `--localization-layers` | `all` | 逗号分隔的 block 索引，例如 `0,3,6` |
| 替换层集 | `--edit-layers` | `last` | `all` 或逗号分隔的 block 索引 |
| 跨层汇总 | `--my-layer-aggre` | 关闭：软图平均 | 开启：逐层二值投票 |
| 区域判据 | `--my-layer-policy` | 关闭：ConceptAttention 单类／多类规则 | 开启：使用 `--mask-rule` |
| 自定义区域判据 | `--mask-rule` | `target_max`，仅 `--my-layer-policy` 开启时生效 | `anchor_min`、`mean_target_max`（空间均值与 target 最高分区域的交集） |
| 跨层投票比例阈值 | `--eta` | `0`，仅 `--my-layer-aggre` 开启时生效 | `[0,1]` 内数值；`0.5` 为严格多数投票 |
| target 面积阈值 | `--rho` | `0` | `[0,1]` 内数值；`0.2` 为旧面积门控 |
| 更新算子 | `--update-rule` | `replace` | `blend`、`add` 用 `--strength` 控制系数；`zero` 将掩码区域置零 |

层索引从 0 开始；FLUX 的双流 blocks 在前、单流 blocks 在后。每个 block 的主 attention 等权参与软图平均；启用投票时每个 block 投一票。SD3.5 额外的 image-only `attn2` 不参与定位汇总。

开启 `--my-layer-aggre` 后使用严格的 `vote_fraction > eta`：`eta=0` 是逐层掩码的并集，`eta=1` 不产生掩码；默认软图平均不使用 `eta`。所有模式均保留 source 前向的严格面积门控 `target_area > rho`，`rho=0` 在掩码非空时触发，空掩码仍跳过 anchor 前向。概念分数使用 FP32 的原始 AV 点积，不归一化 image 或 concept 向量；用于状态更新与 anchor 替换的激活保持原始表示。

显式指定 `--my-layer-policy` 后，`--mask-rule` 控制本组三个概念的区域判据；软图平均模式比较平均后的软分数：

- `target_max`（默认）：`target > anchor` 且 `target > background`。
- `anchor_min`：`anchor < target` 且 `anchor < background`；anchor 并列最低时不选中，target 和 background 并列高于 anchor 时仍选中。“最低”在同一位置的三个概念分数之间比较。
- `mean_target_max`：`target > mean_spatial(target)` 且 `target > anchor` 且 `target > background`。均值按当前样本、当前采样步和当前 pair 的全部空间位置计算，三个条件均为严格大于。仅指定 `--mask-rule mean_target_max` 不会启用此规则，还需 `--my-layer-policy`。

多 target 时，`mean_target_max` 沿用自定义策略的组内三概念 softmax，逐组取上述交集；各组独立计算 target 均值，候选掩码重叠时再按现有 winner 规则选择。默认不启用 `--my-layer-aggre`，先平均所选层的软图，再计算交集；若显式启用投票，则每层先在其 softmax 软图上计算交集，再按 `vote_fraction > eta` 汇总。原有 `target_max`／`anchor_min` 在投票模式下继续比较每层原始点积分数。

例如，在 I2P 上以 `nude` 定位，开启交集策略并保存 trace：

```bash
python DiT/queryGraft/run.py benchmark --cases i2p --tasks i2p \
  --pairs '[{"target":"nude","anchor":"clothed"}]' \
  --my-layer-policy --mask-rule mean_target_max \
  --device cuda:0 --phase run --trace
```

加 `--update-rule zero` 使用置零更新。交集策略默认关闭，不改变原有默认判据。

三种规则均支持所有替换表示与 source 模式；是否逐层判定后投票由 `--my-layer-aggre` 独立控制。例如，恢复逐层判定与投票，以 `clothed` 为 anchor 选择其分数最低的位置：

```bash
python DiT/queryGraft/run.py benchmark --model sd35_medium --cases i2p --tasks i2p \
  --my-layer-aggre --my-layer-policy \
  --mask-rule anchor_min --anchor-overrides '{"nudity":"clothed"}' \
  --device cuda:0 --phase sample --trace
```

`blend` 的更新为 `X + strength * M * (A - X)`；`add` 为 `X + strength * M * A`；`replace` 不使用 `strength`。最后一层 AV 替换可在当前 user 前向内取得掩码后按需计算 anchor source；多层替换使用固定的当前步掩码并重放 user 路径，实际前向成本随配置变化。

显式指定 `--update-rule zero` 时，更新为 `X' = where(M, 0, X)`：将实际执行掩码内所选表示的全部通道置零。默认仍为 `replace`。`zero` 支持 `av`／`residual`／`block_output`／`velocity`，沿用原有定位、面积门控、编辑层选择和 CFG 正条件分支约定，跳过 source 编码及前向；`--strength` 和 source 参数不参与更新。target／anchor／background 仍参与定位。最后一层或 velocity 每步只需一次 user 前向；其他编辑层配置在掩码触发时重放 user，总计两次前向。

置零发生在 `--representation` 指定的位置；默认 `av` 是输出投影之前，后续投影、门控、残差和 MLP 继续执行。trace 的激活统计也在该位置计算，图中标注 `update=zero` 和 `target -> zero`。

例如，在 I2P 上沿用默认定位及 AV 编辑位置，开启置零与 trace 后采样、评估：

```bash
python DiT/queryGraft/run.py benchmark --cases i2p --tasks i2p \
  --device cuda:0 --phase run --update-rule zero --trace
```

`residual` 在 SD3.5／FLUX 双流层指主 attention 投影、门控、残差更新后的表示。SD3.5 的额外 `attn2` 更新保留在 user 路径；FLUX 单流层则使用 AV 与 MLP 融合后的原生残差输出。定位分数在这些配置下仍统一从 raw AV 计算。

`block_output` 替换整个 block 的最终 image hidden state：在 SD3.5／FLUX 双流层中，位置是 MLP、MLP 门控和第二次残差相加之后；SD3.5 的额外 `attn2` 贡献也包含在该表示中。text 分支保持 user 输出。FLUX 单流层的 `block_output` 与 `residual` 使用同一融合残差输出位置。三种表示均沿用 raw AV 定位、空间掩码和现有更新算子。

`velocity` 替换完整 Transformer 的最终预测。在 SD3.5 中，位置是所有 blocks、最终 modulation、linear projection 和 unpatchify 之后、原生 CFG 之前。source 使用自己的完整输出头及 pooled condition；只替换 user 正条件分支，负条件 velocity 保持原生输出。定位仍使用 raw AV，token 网格中的每个选中位置覆盖对应 patch 内的全部 latent 空间位置与通道，不做平滑插值。FLUX 的最终 velocity 保留原生 packed 布局，在最终输出头之后、scheduler 之前替换，内嵌 guidance 保持原生设置。

`velocity` 只有一个最终输出编辑位置，使用默认 `--edit-layers last`；`all` 或 block 索引不适用并会报错。每步只需一次 user 前向，以及每个触发的不同 anchor 各一次 source 前向，无需重放 user。它兼容 `anchor` 和下述四种 source 模式，以及 `replace`／`blend`／`add`；默认表示仍为 `av`。

```bash
# 定位仍使用全部层，只改变编辑层与融合强度。
python DiT/queryGraft/run.py benchmark --cases ip \
  --edit-layers all --update-rule blend --strength 0.5

# 使用关键定位层、关键编辑层与残差后表示。
python DiT/queryGraft/run.py benchmark --cases style \
  --localization-layers 0,3,6 --edit-layers 6 \
  --representation residual --rho 0.1

# 在最后一个 block 的最终输出处做掩码硬替换；加 --trace 保存逐步诊断。
python DiT/queryGraft/run.py benchmark --model sd35_medium --cases all \
  --representation block_output --edit-layers last --update-rule replace \
  --device cuda:0 --phase sample

# 默认脚本可通过环境变量或额外 CLI 参数修改。
MODEL=flux1_dev CASES=object ETA=0.25 RHO=0.1 UPDATE_RULE=blend STRENGTH=0.5 \
  bash DiT/queryGraft/run.sh --my-layer-aggre --phase sample

# 使用严格多数票和 20% 面积门控；概念分数仍使用原始 AV 点积。
python DiT/queryGraft/run.py benchmark --cases i2p \
  --my-layer-aggre --my-layer-policy --eta 0.5 --rho 0.2
```

额外参数包括 `--background`、`--pairs`、`--anchor-overrides`、`--trace/--no-trace`、`--trace-root`。target/anchor 默认来自 benchmark 的概念配置；可用 `--anchor-overrides '{"nudity":"clothed"}'` 显式覆盖。默认 `--source-mode anchor` 使用完整 anchor 短语作为原生 prompt。

## Source 条件与增强模式

`--source-mode` 默认是 `anchor`，保持原有独立 anchor 条件；下面四个新模式必须显式启用。它们只改变替换 source，不改 user prompt 或定位用的 target／anchor／background。

| 模式 | Source 的构造 | 附加参数 |
| --- | --- | --- |
| `joint` | 每个样本使用原生文本编码的 `user prompt, anchor` | 无 |
| `logits` | 联合条件；掩码内 image query 对 user／anchor 内容 token 的 attention logits 分别加 `log(user_weight)`／`log(anchor_weight)` | `--source-user-weight`、`--source-anchor-weight`，默认均为 1，必须大于 0 |
| `values` | 联合条件；保持当前层 Q/K 和 attention 概率，分别缩放 user／anchor 内容 token 的 Value | 同上，允许 0 |
| `extrapolate` | 在所选替换表示上构造 `H_user + source_scale * (H_joint - H_user)` | `--source-scale`，默认 1；大于 1 放大，0 为 user，1 为 joint |

联合 prompt 按样本构造，不进行语义解析或安全改写。序列条件和 pooled 条件均来自该完整联合 prompt。SD3 的两组 CLIP token 按其原生通道拼接，随后与 T5 序列拼接；FLUX 使用原生 T5 序列。加权只作用于实际 user／anchor 内容 token，特殊和 padding token 不加权，pooled 条件不另行缩放。两个 CLIP 编码器在同一位置分组不一致时，该混合位置保持原权重；T5 独立分组。若 T5 中新增 anchor 被完全截断，加权模式会明确报错，需增加配置中的 `sampling.max_sequence_length`。

`logits`／`values` 在 source 分支的所有主 attention 层生效，空间区域使用当前 user 路径产生的最终执行掩码。当前层的 text query 与掩码外 image query 仍采用原生 attention 结果；后续层可能受到已修改 image 特征的间接影响。SD3 额外的 image-only `attn2` 保持原生计算。多个 pair 共用 anchor 时，source 前向复用，增强区域取这些 pair 最终执行区域的并集。

`extrapolate` 的 `H_user` 始终取同一步、同层未编辑的 user 特征；多层重放时不会改用已受到前层干预的特征。它支持 `av`／`residual`／`block_output`／`velocity`；最后一种在完整输出上构造 `v_source = v_user + source_scale * (v_joint - v_user)`，随后才做掩码更新和原生 CFG。source 外推系数和最终 `--update-rule`／`--strength` 分别控制 source 构造和执行更新。联合条件按生成 batch 编码并缓存到 CPU，各模式仍只为每个触发的不同 anchor 增加一次主干前向。

以下四组命令只运行 I2P 采样并立即计算 NudeNet 指标。这里显式固定 anchor 为 `clothed`；加 `--max-records 4` 可先验证少量样本，加 `--trace` 保存逐步诊断，加 `--dry-run` 只预览配置。

```bash
common=(--model sd35_medium --cases i2p --tasks i2p
        --representation block_output --edit-layers last
        --eta 0 --rho 0 --device cuda:0 --phase run
        --anchor-overrides '{"nudity":"clothed"}')

python DiT/queryGraft/run.py benchmark "${common[@]}" --source-mode joint

python DiT/queryGraft/run.py benchmark "${common[@]}" --source-mode logits \
  --source-user-weight 0.5 --source-anchor-weight 2

python DiT/queryGraft/run.py benchmark "${common[@]}" --source-mode values \
  --source-user-weight 0.5 --source-anchor-weight 2

python DiT/queryGraft/run.py benchmark "${common[@]}" --source-mode extrapolate --source-scale 2
```

`--cases i2p --tasks i2p` 默认是 931 条 sexual I2P，仅评估 NudeNet；`--with-q16` 切换为完整 I2P 并增加 Q16。若同时需要 COCO 保留指标，可改为 `--tasks i2p,mscoco`。实验名称包含 source 模式和实际生效的权重／外推系数；已有 manifest 的 source 配置也保持不可变。`run.sh` 支持 `SOURCE_MODE`、`SOURCE_USER_WEIGHT`、`SOURCE_ANCHOR_WEIGHT`、`SOURCE_SCALE` 环境变量，均可被额外 CLI 参数覆盖。

## Velocity：I2P 采样与评估

在仓库根目录执行。以下命令分别运行四种 source 模式，每组采样后立即评估 NudeNet。anchor 直接读取 `benchmark/configs/dit_target_anchor.json`，当前 `nudity` 对应 `clothed`，无需额外覆盖。

```bash
common=(--model sd35_medium --cases i2p --tasks i2p
        --representation velocity --edit-layers last --update-rule replace
        --eta 0 --rho 0 --device cuda:0 --phase run)

python DiT/queryGraft/run.py benchmark "${common[@]}" --source-mode joint
python DiT/queryGraft/run.py benchmark "${common[@]}" --source-mode logits \
  --source-user-weight 0.5 --source-anchor-weight 2
python DiT/queryGraft/run.py benchmark "${common[@]}" --source-mode values \
  --source-user-weight 0.5 --source-anchor-weight 2
python DiT/queryGraft/run.py benchmark "${common[@]}" --source-mode extrapolate --source-scale 2
```

加 `--max-records 4` 可先验证少量样本，`--dry-run` 仅预览。需要分开采样和评估时，先用 `--phase sample`，再对终端打印的 `[manifest]` 路径执行：

```bash
python DiT/queryGraft/run.py evaluate --manifest /path/to/manifest.json --device cuda:0
```

沿用独立 anchor source 时，将 `--source-mode` 设为 `anchor` 或省略。`--trace` 的空间图仍使用定位 token 网格；激活变化统计使用最终 velocity，统计的通道元素包含每个 patch 的全部 latent 值。`edit_layer_indices=[-1]` 与 metadata 中的 `output.sample` 表示完整 Transformer 输出位置。

## eta／rho 参数扫描

`sweep_eta_rho.sh` 默认按顺序执行 `eta ∈ {0.1,0.2,0.3,0.4,0.5}` 与 `rho ∈ {0.1,0.2,0.3,0.4,0.5}` 的全部 25 个组合。默认模型为 SD3.5 Medium，设置为 `block_output`、所有层定位、末层硬替换，仅运行 I2P 任务的采样阶段。每组使用相同的数据和 seed 协议，结果目录自动包含 eta、rho；任一组失败时脚本退出。

```bash
# 预览全部 25 组配置，每组取 4 个样本，不加载模型。
bash DiT/queryGraft/sweep_eta_rho.sh --device cuda:0 --max-records 4 --dry-run

# 正式运行，每组使用完整 I2P；加 --trace 保存逐步诊断。
bash DiT/queryGraft/sweep_eta_rho.sh --device cuda:0

# 每组采样后立即评估。
bash DiT/queryGraft/sweep_eta_rho.sh --device cuda:0 --phase run

# 自定义参数列表或其他实验设置。
ETA_VALUES="0.2 0.4" RHO_VALUES="0.1 0.3" REPRESENTATION=residual \
  bash DiT/queryGraft/sweep_eta_rho.sh --device cuda:0 --max-records 4
```

沿用 `run.sh` 的 `MODEL`、`CASES`、`LOCALIZATION_LAYERS`、`EDIT_LAYERS`、`UPDATE_RULE` 等环境变量；另可用 `TASKS`、`PHASE` 覆盖默认任务和阶段。额外 CLI 参数传给每组实验，`--eta`／`--rho` 始终以扫描列表为准。

## benchmark 阶段与结果

```bash
# 准备计划，终端输出 manifest 路径。
python DiT/queryGraft/run.py benchmark --cases ip_pikachu --phase prepare --offline

# 复用上述 manifest；现有图片会由 benchmark 自动复用。
python DiT/queryGraft/run.py benchmark --manifest /path/to/manifest.json --phase sample

# 对已经采样的结果运行原有指标。
python DiT/queryGraft/run.py evaluate --manifest /path/to/manifest.json --device cuda:0
```

指定现有 manifest 时，方法、概念和采样科学配置以 manifest 为准，修改配置应准备新计划。运行时设备、batch size、指标选项等沿用 benchmark 原生规则。

方法图片和诊断默认位于 `DiT/queryGraft/results/<model>/`；manifest、指标及统计使用现有 `benchmark/results/<model>/` 流程。新实验名称包含 `concept-dot-zero-pooled`、替换表示、更新算子、投票阈值、面积阈值、层选择和起始时间，目录名不使用哈希。`--result-root` 修改图片与诊断根目录，`--output-dir` 修改 benchmark 结果根目录；独立 prompt 可用 `--run-dir` 指定输出目录。

新实验名称用 `concept-image` 标记默认概念读取范围，用 `concept-image-prompt` 标记显式加入 prompt 的范围；所选 `concept_include_prompt` 写入配置、manifest 和 trace metadata。

实验名称用 `layer-soft-mean`／`layer-vote` 标记汇总，用 `ca-policy`／`my-policy-mask-<rule>` 标记区域判据。两个开关与 `mask_rule` 均保存到配置、manifest 和 metadata；仅设置 `--mask-rule` 不会自动开启 `--my-layer-policy`。

## 每个样本、每个时间步的四联诊断图

诊断默认关闭，仅在显式传入 `--trace` 时开启；`--trace-root` 只设置保存路径，不会开启诊断。开启后每步保存一张 PNG；多概念时每个 target–anchor pair 占一行，每行四个面板：

1. **最终执行掩码**：所选汇总与区域判据、`rho` 和多 pair 重叠选择后的实际编辑区域；标题标出执行面积，底部显示候选面积及门控状态。
2. **实际决策输入**：默认显示所选定位层平均软分数图的 target 通道，开启 `--my-layer-aggre` 时显示所选层的 `vote_fraction`。底部标明实际使用的空间均值阈值、投票阈值、全局 argmax 或三概念比较规则；默认模式不再显示无效的 `eta`。
3. **候选掩码**：所选区域判据或投票阈值作用后的二值区域，即数值文件中的 `mask`；尚未经过 `rho` 面积门控和多 pair 重叠选择。与第一图对照可区分“选中了区域”和“最终执行了编辑”。
4. **替换后激活变化**：三根柱子显示实际执行区域内替换前、替换后及差值的 RMS，并标注相对 L2 改变量、最大绝对改变量和 token–layer 数量。统计位置与 `--representation` 对应。

第二、三项均使用实际决策的定位层范围；图上注明所选层与全部执行层的数量。`localization_scores_mean` 保存所选层的软分数均值，`vote_fraction` 保存所选层的二值支持比例。此前的全执行层辅助统计 `layer_mask_mean`、`concept_scores_mean`、`concept_logits_mean` 继续保存在张量文件中，不再作为默认决策输入绘图。第四项仅统计实际编辑层和各 pair 最终胜出的空间位置，多层结果通过合并平方和与元素数量计算；未执行替换时显示零更新。

采样完成后，在每个样本目录直接保存 `summary.png` 和 `summary.safetensors`。汇总图包含两张热力图，多 pair 按行展示：

- **执行掩码的时间均值**：所有采样步的最终执行掩码等权平均，表示每个位置被编辑的频率，不再二值化。未触发替换的步骤以零掩码参与平均。
- **所选定位层的概念分数时间均值**：每步先平均所选定位层的软图，再平均全部采样步，展示 target 通道。投票模式下此图标为辅助分数。跨时间汇总不重新施加空间均值阈值或 argmax，不由它重新生成执行掩码。

两项均使用实际完成的采样步数作为分母，不只统计触发步骤，也不按模型 timestep 间隔加权。汇总仅累加小型 CPU 空间图，不保留完整激活或增加模型前向。

逐步图和样本汇总图的标题注明实际启用的汇总与区域策略。metadata 每步记录包含两个开关、`mask_rule`，以及解析后的 `aggregation`、`region_policy`、`softmax_scope` 和 `decision_threshold`。单类默认模式的阈值按样本计算；软图平均的 `mean_target_max` 按样本、按 pair 保存 target 空间均值，图上同时标明与 `target > anchor`、`target > background` 取交集。投票模式记录 `eta`；交集投票的图注另标明逐层交集条件。不使用标量阈值的模式记录 `null`。每组的三通道张量仍按 target、anchor、background 排列。

```bash
# CUDA 0：最基本配置，先运行 4 条 I2P，保存逐步诊断和样本汇总。
python DiT/queryGraft/run.py benchmark --model sd35_medium --cases i2p --tasks i2p \
  --representation av --localization-layers all --edit-layers last \
  --source-mode anchor --update-rule replace --eta 0 --rho 0 \
  --device cuda:0 --max-records 4 --phase sample --trace

# FLUX 的五概念 object 实验，自定义诊断根目录。
python DiT/queryGraft/run.py benchmark --model flux1_dev --cases object_multi5 \
  --device cuda:1 --phase sample --trace --trace-root DiT/queryGraft/diagnostic_results

# 独立 prompt；此时诊断位于指定 run 目录的 traces/generated/sample_0000/。
python DiT/queryGraft/run.py sample --prompt 'a person in a garden' \
  --device cuda:0 --run-dir DiT/queryGraft/results/diagnostic_example --trace

# 默认不保存逐步数值与图片诊断；也可显式使用 --no-trace。
python DiT/queryGraft/run.py benchmark --cases i2p --tasks i2p --device cuda:0 --phase sample
```

开启诊断后的输出采用可读目录和文件名：

```text
DiT/queryGraft/results/<model>/<task>/<case>/<method-with-start-time>/
└── traces/<sampling-label>/<sample-id>/
    ├── summary.png             # 两张跨时间步汇总热力图；多 pair 按行展示
    ├── summary.safetensors     # 跨时间步汇总数值
    ├── timesteps/
    │   ├── step_0000.png       # 每步四联图
    │   └── step_0001.png
    ├── tensors/
    │   ├── step_0000.safetensors
    │   └── step_0001.safetensors
    └── metadata.json           # 采样完成后保存配置、逐步统计与实际前向次数
```

`--trace-root` 将诊断根目录改为指定路径，其下继续按 model／task／case／method／sampling／sample 区分。PNG 渲染与保存不增加模型前向，耗时计入实际运行时间。仅保存空间分数、掩码与激活统计，不保存完整 AV。

每步的 `.safetensors` 包含：

| 字段 | 单样本形状／含义 |
| --- | --- |
| `mask`、`vote_fraction` | `[K,H,W]`，所选定位层产生的候选掩码与支持比例 |
| `applied_mask` | `[K,H,W]`，每个 pair 经过门控和重叠选择后的最终执行掩码 |
| `winner` | `[H,W]`，最终 pair 索引；未执行区域为 −1 |
| `layer_mask_mean` | `[K,H,W]`，所有执行层的二值掩码均值 |
| `concept_scores_mean` | `[K,H,W,3]`，逐层 softmax 后的概念分数均值；顺序为 target、anchor、background |
| `concept_logits_mean` | `[K,H,W,3]`，未 softmax 的原始 image/concept AV 点积分数均值 |
| `localization_scores_mean` | `[K,H,W,3]`，仅所选定位层的软分数均值；默认模式的实际决策输入 |
| `decision_threshold` | `[K]`，仅在使用标量阈值时保存：默认单类及软图平均的 `mean_target_max` 为每组 target 图的空间均值，投票模式为 `eta`；按严格大于比较 |
| `concept_argmax` | `[H,W]`，仅默认软图平均、多 target 模式保存；在全部 `3K` 概念通道中的获胜索引，`3k`／`3k+1`／`3k+2` 分别为第 `k` 组 target／anchor／background |
| `edit_layer_indices` | `[E]`，编辑层的实际索引；`velocity` 为 `[-1]`，表示完整 Transformer 输出 |
| `activation_before_rms`、`activation_after_rms`、`activation_delta_rms` | `[K,E]`，逐 pair、逐编辑层的 RMS |
| `activation_relative_l2`、`activation_abs_max` | `[K,E]`，相对 L2 改变量与最大绝对改变量 |
| `activation_value_count`、`activation_token_count` | `[K,E]`，统计涉及的通道元素数与 token 数 |

`K` 为 pair 数，`E` 为编辑层数。`metadata.json` 的每步 `activation_change` 同时记录按 pair 汇总的变化统计和逐层明细。

样本根目录的 `summary.safetensors` 包含：

| 字段 | 单样本形状／含义 |
| --- | --- |
| `applied_mask_mean` | `[K,H,W]`，所有采样步最终执行掩码的均值，取值 `[0,1]` |
| `concept_scores_mean` | `[K,H,W,3]`，逐层 softmax、跨层平均后再跨时间步平均；顺序为 target、anchor、background |
| `localization_scores_mean` | `[K,H,W,3]`，所选定位层软图均值的时间均值；汇总图使用此字段，`concept_scores_mean` 保留全执行层辅助统计 |

## 验证范围

已在本地 Diffusers 0.41.0 下使用真实小型 SD3／FLUX Transformer 验证只读前向一致性、三类判定与跨层投票、面积阈值、两种替换表示、三种更新算子、单层／多层执行和多 pair source 复用。真实小型 pipeline、VAE 与 scheduler 的两步采样也已通过；其中语义文本编码使用确定性替代输入，用于验证计算与接口，不衡量擦除效果。

原始 AV 点积分数使用不同概念范数的已知数值验证，覆盖 FP32／BF16、CFG、多 pair、两种掩码判据；`eta`／`rho` 的严格边界及空掩码跳过保持原检查。独立调制检查位于 `benchmark/tests/test_query_graft_concept_modulation.py`，覆盖 SD3、FLUX 双流／单流和嵌入式 guidance，并将 concept 残差状态与独立的原生 block 前向对照。

软图平均与区域判据检查位于 `benchmark/tests/test_query_graft_layer_policy.py`：用已知数值区分四种开关组合，验证单类空间均值、多类全局竞争、并列分数、面积门控、关键层与诊断层分离，以及 CLI 配置保存。连同现有 queryGraft、独立调制、诊断、source、trace、velocity 和 zero 检查，41 项全部通过。

trace 决策检查位于 `benchmark/tests/test_query_graft_trace.py`，从保存的阈值或 argmax 索引复算候选掩码，并检查绘图实际使用的数组、关键定位层的汇总、未触发步骤的时间均值和关闭 trace 的行为。

置零检查位于 `benchmark/tests/test_query_graft_zero.py`，覆盖 SD3／FLUX、四种表示、单层／多层、多 pair、CFG 负分支、掩码外值保留、空掩码跳过、source 编码与前向跳过、trace 的零激活统计，以及 CLI 默认值与显式开启的独立计划。

逐步诊断已验证 8 组小型 SD3／FLUX、AV／residual、末层／全部编辑层组合：开启与关闭诊断的模型输出一致，前向次数一致；批量 CFG、多 pair、关键定位层下的全层诊断均值和数值文件均通过检查。真实小型 pipeline 的两步采样验证了每步 PNG／张量、最终 metadata，以及未触发时的零更新图；文本编码仍使用确定性替代输入。

样本汇总另用小型 SD3／FLUX Transformer 各两个样本、三个采样步验证：汇总数值与逐步文件重算的均值一致，包含未触发步骤；连续调用采样会重置统计，关闭 trace 不创建诊断文件。现有 queryGraft／velocity 的 8 项检查通过。

首版的现有 DiT／case／plugin 检查共运行 27 项，25 项通过、2 项按原检查条件跳过；离线 manifest 准备通过。本次更新未运行正式权重的擦除指标实验。

`block_output` 的回归检查位于 `benchmark/tests/test_query_graft.py`：使用真实小型 SD3（含额外 `attn2`）／FLUX Transformer，验证末层／全部层的最终输出替换、逐样本空间掩码、CFG 负分支和 text 输出保留、三种表示在空掩码下与原生输出一致，以及最终输出变化的诊断图与 RMS 数值。可运行 `OMP_NUM_THREADS=1 python -m unittest benchmark.tests.test_query_graft -v`。

Source 模式检查位于 `benchmark/tests/test_query_graft_source.py`，覆盖显式 attention 数值、BF16、原生联合文本编码接口、两组 CLIP 分组不一致的位置、各表示与编辑层配置、未编辑 user 基准外推、以及 benchmark 计划的隔离与不可变性。连同现有 DiT case／plugin 检查，共 28 项，26 项通过、2 项按原条件跳过。真实 SD3.5 Medium 的四种模式也各完成了同样 2 条 I2P 记录、40 步采样和 NudeNet 评估；这属于流程验证，不是全量擦除效果结论。另使用真实 CLIP／T5 tokenizer 检查全部 931 条 I2P 的 source 分组，T5 均保留了追加的 `clothed`。
