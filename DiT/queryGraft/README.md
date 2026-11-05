# queryGraft

方法公式、符号与设计依据见 [方法定义](README.zh-CN.md)。`method.py` 集中模型接入与算法，`source.py` 提供 source token 分组和 attention 控制，`run.py` 提供命令行与 benchmark，`diagnostics.py` 负责逐步诊断图和数值保存；`run.sh` 提供默认实验入口。

默认设置是：独立编码 target、anchor、background 并取各自第一个序列 token；对同层 concept AV 沿通道维做 L2 归一化，再与 image AV 点积定位；所有主 attention 层的二值掩码按投票比例阈值汇总。默认 `eta=0`，一个位置只要有一层支持 target 就纳入掩码；默认 `rho=0`，只要掩码非空，就使用相同当前 latent 的 anchor 条件激活，在最后一层原始 AV 处硬替换。

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

现有 benchmark 包含多 target 用例，因此实现提供逐 pair 扩展：每对 target/anchor/background 独立读取同一原始 user/image 路径与本组概念，不跨 pair 读取；每对分别进行投票比例判定与面积门控。区域重叠时，使用支持票比例最高的 pair，平票按配置顺序选择。每个触发的不同 anchor 从同一个当前 latent 单独前向；相同 anchor 复用 source。单 pair 的默认算法与方法文档一致。

## 六组消融参数

| 设计项 | 参数 | 默认值 | 备选 |
| --- | --- | --- | --- |
| 替换表示 | `--representation` | `av` | `residual`、`block_output` |
| 定位层集 | `--localization-layers` | `all` | 逗号分隔的 block 索引，例如 `0,3,6` |
| 替换层集 | `--edit-layers` | `last` | `all` 或逗号分隔的 block 索引 |
| 跨层投票比例阈值 | `--eta` | `0` | `[0,1]` 内数值；`0.5` 为严格多数投票 |
| target 面积阈值 | `--rho` | `0` | `[0,1]` 内数值；`0.2` 为旧面积门控 |
| 更新算子 | `--update-rule` | `replace` | `blend`、`add`，用 `--strength` 控制系数 |

层索引从 0 开始；FLUX 的双流 blocks 在前、单流 blocks 在后。每个 block 的主 attention 投一票，SD3.5 额外的 image-only `attn2` 不增加投票。

始终使用 `vote_fraction > eta` 和 `target_area > rho`：等于阈值时不通过。`eta=0` 是逐层 target 掩码的并集；`eta=1` 不产生掩码。`rho=0` 在掩码非空时触发，空掩码仍跳过 anchor 前向。概念归一化只用于计算分数，采用 FP32 和 `norm + 1e-8`，不修改原始概念状态、image AV 或用于替换的 anchor AV。

`blend` 的更新为 `X + strength * M * (A - X)`；`add` 为 `X + strength * M * A`；`replace` 不使用 `strength`。最后一层 AV 替换可在当前 user 前向内取得掩码后按需计算 anchor source；多层替换使用固定的当前步掩码并重放 user 路径，实际前向成本随配置变化。

`residual` 在 SD3.5／FLUX 双流层指主 attention 投影、门控、残差更新后的表示。SD3.5 的额外 `attn2` 更新保留在 user 路径；FLUX 单流层则使用 AV 与 MLP 融合后的原生残差输出。定位分数在这些配置下仍统一从 raw AV 计算。

`block_output` 替换整个 block 的最终 image hidden state：在 SD3.5／FLUX 双流层中，位置是 MLP、MLP 门控和第二次残差相加之后；SD3.5 的额外 `attn2` 贡献也包含在该表示中。text 分支保持 user 输出。FLUX 单流层的 `block_output` 与 `residual` 使用同一融合残差输出位置。三种表示均沿用 raw AV 定位、空间掩码和现有更新算子。

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
  bash DiT/queryGraft/run.sh --phase sample

# 使用旧的严格多数票和 20% 面积门控；concept 输出仍采用新的 L2 归一化。
python DiT/queryGraft/run.py benchmark --cases i2p --eta 0.5 --rho 0.2
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

`extrapolate` 的 `H_user` 始终取同一步、同层未编辑的 user 特征；多层重放时不会改用已受到前层干预的特征。它支持 `av`／`residual`／`block_output`，source 外推系数和最终 `--update-rule`／`--strength` 分别控制 source 构造和执行更新。联合条件按生成 batch 编码并缓存到 CPU，各模式仍只为每个触发的不同 anchor 增加一次主干前向。

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

方法图片和诊断默认位于 `DiT/queryGraft/results/<model>/`；manifest、指标及统计使用现有 `benchmark/results/<model>/` 流程。实验名称包含 `concept-l2`、替换表示、更新算子、投票阈值、面积阈值、层选择和起始时间，目录名不使用哈希。`--result-root` 修改图片与诊断根目录，`--output-dir` 修改 benchmark 结果根目录；独立 prompt 可用 `--run-dir` 指定输出目录。

## 每个样本、每个时间步的四联诊断图

诊断默认关闭，仅在显式传入 `--trace` 时开启；`--trace-root` 只设置保存路径，不会开启诊断。开启后每步保存一张 PNG；多概念时每个 target–anchor pair 占一行，每行四个面板：

1. **最终执行掩码**：经过 `eta`、`rho` 和多 pair 重叠选择后的实际编辑区域；标题标出执行面积，底部显示候选面积及门控状态。
2. **所有层掩码的平均**：所有实际执行的主 attention blocks 对 target 的二值投票均值，色阶固定为 `[0,1]`。
3. **所有层的 target 概念分数平均**：每层先在本组 target／anchor／background 的分数间 softmax，再跨层平均，展示 target 通道；色阶固定为 `[0,1]`，表示相对概念分数。
4. **替换后激活变化**：三根柱子显示实际执行区域内替换前、替换后及差值的 RMS，并标注相对 L2 改变量、最大绝对改变量和 token–layer 数量。统计位置与 `--representation` 对应。

第二、三项始终汇总所有实际执行的 blocks。使用关键定位层时，它们与决定最终掩码的 `vote_fraction` 可能不同；图上分别标注所有诊断层与所选定位层数量。第四项仅统计实际编辑层和各 pair 最终胜出的空间位置，多层结果通过合并平方和与元素数量计算；未执行替换时显示零更新。

```bash
# CUDA 0：I2P，保存每步诊断，先运行少量样本。
CASES=i2p bash DiT/queryGraft/run.sh \
  --device cuda:0 --tasks i2p --max-records 4 --phase sample --trace

# FLUX 的五概念 object 实验，自定义诊断根目录。
MODEL=flux1_dev CASES=object_multi5 bash DiT/queryGraft/run.sh \
  --device cuda:1 --phase sample --trace --trace-root DiT/queryGraft/diagnostic_results

# 独立 prompt；此时诊断位于指定 run 目录的 traces/generated/sample_0000/。
python DiT/queryGraft/run.py sample --prompt 'a person in a garden' \
  --device cuda:0 --run-dir DiT/queryGraft/results/diagnostic_example --trace

# 默认不保存逐步数值与图片诊断；也可显式使用 --no-trace。
CASES=i2p bash DiT/queryGraft/run.sh --device cuda:0 --phase sample
```

开启诊断后的输出采用可读目录和文件名：

```text
DiT/queryGraft/results/<model>/<task>/<case>/<method-with-start-time>/
└── traces/<sampling-label>/<sample-id>/
    ├── step_0000.png           # 四联图；多 pair 仍合在这一张图中
    ├── step_0000.safetensors   # 对应数值
    ├── step_0001.png
    ├── step_0001.safetensors
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
| `concept_logits_mean` | `[K,H,W,3]`，未 softmax 的归一化 concept 点积分数均值 |
| `edit_layer_indices` | `[E]`，编辑层的实际索引 |
| `activation_before_rms`、`activation_after_rms`、`activation_delta_rms` | `[K,E]`，逐 pair、逐编辑层的 RMS |
| `activation_relative_l2`、`activation_abs_max` | `[K,E]`，相对 L2 改变量与最大绝对改变量 |
| `activation_value_count`、`activation_token_count` | `[K,E]`，统计涉及的通道元素数与 token 数 |

`K` 为 pair 数，`E` 为编辑层数。`metadata.json` 的每步 `activation_change` 同时记录按 pair 汇总的变化统计和逐层明细。

## 验证范围

已在本地 Diffusers 0.41.0 下使用真实小型 SD3／FLUX Transformer 验证只读前向一致性、三类判定与跨层投票、面积阈值、两种替换表示、三种更新算子、单层／多层执行和多 pair source 复用。真实小型 pipeline、VAE 与 scheduler 的两步采样也已通过；其中语义文本编码使用确定性替代输入，用于验证计算与接口，不衡量擦除效果。

本次归一化与阈值更新验证了 concept 范数缩放、FP32／BF16 输入、`eta`／`rho` 的严格边界及空掩码跳过；真实小型 SD3／FLUX 均通过默认触发和 `eta=1` 禁用干预的前向检查，两模型全部 benchmark 配置预览及脚本环境变量覆盖也通过。

逐步诊断已验证 8 组小型 SD3／FLUX、AV／residual、末层／全部编辑层组合：开启与关闭诊断的模型输出一致，前向次数一致；批量 CFG、多 pair、关键定位层下的全层诊断均值和数值文件均通过检查。真实小型 pipeline 的两步采样验证了每步 PNG／张量、最终 metadata，以及未触发时的零更新图；文本编码仍使用确定性替代输入。

首版的现有 DiT／case／plugin 检查共运行 27 项，25 项通过、2 项按原检查条件跳过；离线 manifest 准备通过。本次更新未运行正式权重的擦除指标实验。

`block_output` 的回归检查位于 `benchmark/tests/test_query_graft.py`：使用真实小型 SD3（含额外 `attn2`）／FLUX Transformer，验证末层／全部层的最终输出替换、逐样本空间掩码、CFG 负分支和 text 输出保留、三种表示在空掩码下与原生输出一致，以及最终输出变化的诊断图与 RMS 数值。可运行 `OMP_NUM_THREADS=1 python -m unittest benchmark.tests.test_query_graft -v`。

Source 模式检查位于 `benchmark/tests/test_query_graft_source.py`，覆盖显式 attention 数值、BF16、原生联合文本编码接口、两组 CLIP 分组不一致的位置、各表示与编辑层配置、未编辑 user 基准外推、以及 benchmark 计划的隔离与不可变性。连同现有 DiT case／plugin 检查，共 28 项，26 项通过、2 项按原条件跳过。真实 SD3.5 Medium 的四种模式也各完成了同样 2 条 I2P 记录、40 步采样和 NudeNet 评估；这属于流程验证，不是全量擦除效果结论。另使用真实 CLIP／T5 tokenizer 检查全部 931 条 I2P 的 source 分组，T5 均保留了追加的 `clothed`。
