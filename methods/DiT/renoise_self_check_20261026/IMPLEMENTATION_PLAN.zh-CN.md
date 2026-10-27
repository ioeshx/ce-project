# 重噪自检的速度擦除：代码实现规划与待定配置

规划日期：2026-10-27。对应 [方法提案](/home/shx/code/ce-new/methods/DiT/renoise_self_check_20261026/README.zh-CN.md) 和 [实验计划](/home/shx/code/ce-new/methods/DiT/renoise_self_check_20261026/EXPERIMENTS.zh-CN.md)。

本文规划代码实现，不启动模型采样、不修改现有 benchmark 或 DVE 运行脚本。示例数值不是已确认参数。所有需要用户决定的项目集中列在第 10 节，对应 [待填写配置模板](/home/shx/code/ce-new/methods/DiT/renoise_self_check_20261026/implementation.pending.json)。

## 1. 已由算法约定确定的内容

这些内容来自本次对话中已讨论的算法，属于实现应保持的语义：

| 项目 | 实现约定 |
| --- | --- |
| 干预对象 | 仅修改交给 sampler 的输出 velocity |
| 参数更新 | 无训练、无反向传播 |
| 在线辅助组件 | 仅使用同一个 DiT 及其原生文本编码器；自检不解码给外部模型 |
| 每次处理的擦除目标 | 一个目标概念及其参照；不同概念可以作为独立记录运行 |
| 纠正方向 | 同一当前 latent、同一当前时间、同一生成 guidance 约定下的 `v_anchor-v_forget` |
| 候选 | 两个：`λ=0` 和一个待指定的 `λ_ref` |
| 向前检查 | 各预览当前一步，再在候选新状态上用原用户提示预测一次 velocity |
| 干净 latent 估计 | 使用新状态、新时间、新预测形成局部 clean estimate |
| 重噪比较 | 两候选共享同一份新噪声和同一重噪水平 |
| 打分方向 | `R=参照条件的MSE−目标条件的MSE`；越大表示目标条件相对更匹配 |
| 采用规则 | `R0>θ` 且 `R1<R0−m` 时选纠正，否则选原速度；严格比较，平分选原速度 |
| 正式轨迹 | 每轮只提交当前一步；临时检查不推进正式 sampler |
| 随机性 | 初始生成噪声与自检噪声使用不同随机源；增加自检不会改变原始初始 latent |

不把额外归一化、范数裁剪、动量、低秩投影、自适应 λ 或多概念融合加入这一版本。这些都会改变方法或引入新参数，需要作为新的实验选择提出。

第一版方法文件针对线性 RF 路径。本文用物理噪声系数 `s` 表示噪声比例：纯噪声为 1、干净端为 0。模型实际接收的时间值由 backend 转换，不能把循环索引当作时间。

## 2. 已核对的仓库接入点

| 现有文件 | 能复用什么 | 需要补充什么 |
| --- | --- | --- |
| [benchmark/dit_adapter.py](/home/shx/code/ce-new/benchmark/dit_adapter.py) | FLUX/SD3 模型加载、设备放置、逐记录初始噪声、原生 pipeline 配置 | 当前 `generate()` 包装完整 pipeline，没有本方法需要的候选预览和任意噪声时刻的速度查询 |
| [benchmark/adapters.py](/home/shx/code/ce-new/benchmark/adapters.py:125) | `adapter.kind=plugin`、`generate(records,sampling_config)` 与 `close()` 插件接口 | 本方法插件需自行管理采样循环和逐步诊断记录 |
| [benchmark/sampling_profile.py](/home/shx/code/ce-new/benchmark/sampling_profile.py) | `initial_noise()`：逐样本 CPU float32 随机噪声后转模型精度 | 自检另建随机源，不能复用生成器并消耗其状态 |
| [benchmark/configs/dit_sampling.json](/home/shx/code/ce-new/benchmark/configs/dit_sampling.json) | 现有模型采样 profile | 是否继承必须由用户选择；选定后固化快照，不随原文件更新漂移 |
| [benchmark/configs/dit_target_anchor.json](/home/shx/code/ce-new/benchmark/configs/dit_target_anchor.json) | 现有概念标签和参照标签 | 不是完整条件提示；不得自动将 `Artist Style` 等标签用作方向提示或 probe 提示 |
| [DiT/dve_re/run.py](/home/shx/code/ce-new/DiT/dve_re/run.py) | 参考同状态多条件速度预测、记录统计的方式 | 其三分支同轨迹 forward 替换不能直接处理本方法的候选新状态和异时刻查询 |
| [benchmark/data.py](/home/shx/code/ce-new/benchmark/data.py:81) | 插件配置与入口源码参与结果身份标识 | 方法参数应置于 `adapter` 的专属配置内；多个核心源码也需纳入方法版本记录 |

本地当前 Python 环境读到的版本为 `diffusers=0.41.0`、`torch=2.13.0`、`transformers=5.16.1`、`accelerate=1.12.0`。这是环境观察，不是替用户选定目标运行环境，也不计划自动升级依赖。

本轮核对了已安装的 FLUX pipeline、SD3 pipeline 与 FlowMatchEulerDiscreteScheduler 源码，并对照 [scheduler 官方文档](https://huggingface.co/docs/diffusers/en/api/schedulers/flow_match_euler_discrete) 和 [callback 官方文档](https://huggingface.co/docs/diffusers/en/using-diffusers/callback)。源码对应当前本地版本，最终实现绑定实际执行环境。

### 推荐的接入方式，尚待用户选择

**独立采样入口与 benchmark 插件共用一个算法核心。** 独立入口便于检查分数和候选；插件用于复用现有生成、配对原模型和离线评价流程。算法代码不进入通用 benchmark 的采样/评价模块。

当前 step-end callback 在正常 sampler 提交之后运行，不适合直接承担“比较后再提交当前一步”。推荐显式管理去噪循环，复用 pipeline 的文本编码、latent 准备、时间表和解码辅助函数；不通过递归替换 `transformer.forward` 实现多层查询。

实现目录尚未确定。可以使用 `/home/shx/code/ce-new/DiT/renoise_self_check/`，但本轮不创建该代码目录，不视为已选定。

## 3. 建议的代码划分

在用户选定的实现目录内，建议只保留以下主要部分，不先建设通用控制框架：

```text
<实现目录>/
  config.py        配置解析、待定字段检查、配置快照
  backends.py      选定模型的原生接口适配与模型加载
  algorithm.py     双候选预览、重噪评分、接受规则、逐步记录
  run.py           独立入口：准备、采样、诊断
  plugin.py        现有 benchmark 的 generate()/close() 接口
  tests/          数学/状态测试与选定模型的集成验证
```

其中 `algorithm.py` 不含 attention、block、模态分支索引。backend 使用整个 transformer 的原生条件输出，不操作其内部部件。

建议 backend 提供以下小接口：

```python
encode_conditions(texts, text_config)
prepare_run(records, resolved_sampling_config)
predict_generation_velocity(latents, noise_level, condition, guidance_config)
predict_probe_velocity(latents, noise_level, probe_condition, probe_config)
preview_step(latents, velocity, step_context)
commit_step(latents, velocity, step_context)
estimate_clean(latents, noise_level, velocity)
decode_final(latents)
```

这是一份接口草案，不是已经存在的 API。模型自带的时间输入、FLUX packing/位置标识、pooled embedding、SD3 CFG 合成和 VAE 标度均封装在 backend 内。

## 4. 输入数据与配置如何传递

### 4.1 每条生成记录

```text
sample_id       稳定样本标识；不依赖当前 batch 的位置
prompt          原始用户提示
seed            初始生成噪声种子
concept_id      本次擦除哪个概念
```

`concept_id` 指向用户提供的概念定义；定义中包含方向提示对与自检提示对。原 `prompt` 始终保留，不进行词替换。

如果 benchmark 的一个 case 含多个 target，不能默认取第一个 target，也不能默认全部同时擦除。必须由记录显式给出概念映射，或者将用户明确选择的单目标 case 转为独立运行。

### 4.2 每个概念的四类条件文本

```text
direction_forget_prompt
direction_anchor_prompt
probe_pairs[].forget_prompt
probe_pairs[].anchor_prompt
```

四类文本分别编码并缓存。方向提示与 probe 提示是否相同属于实验选择，不能自动复制。多个 probe 时，还要显式给出其聚合规则。

### 4.3 配置的来源与身份标识

将方法参数放入插件的 `adapter` 专属配置，例如 `adapter.self_check`。现有 `adapter_identity()` 会保留该部分配置，因此更改 λ、ρ、probe 文本或阈值会形成不同的方法身份。

同时记录 `algorithm.py`、`backends.py`、`config.py` 等核心源码的摘要，避免只修改被插件导入的核心文件而错误复用旧结果。使用仓库已有的摘要工具完成，不引入新的结果管理系统。

对于现有 benchmark 自动合并的 profile、数据集种子、任务默认 case、样本量和评价器，先展示完全解析后的配置与来源。没有用户明确选择的值，不应因通用配置合并而变成“已经确认”。

## 5. 完整算法运行流程

### 5.1 准备阶段，不需要运行生成模型

1. 按用户选择的运行阶段检查待指定配置，集中输出缺失字段；不填示例值后直接运行。
2. 检查输入记录、概念映射和四类条件文本。
3. 若用户选择继承 profile，读取指定文件并形成可审阅快照；否则使用用户的显式采样配置。
4. 固化样本列表、种子列表、算法参数、模板文本、软件版本和输出位置。
5. 准备模式到此退出，不下载模型、不执行采样。

### 5.2 模型准备阶段

1. 按指定模型路径/revision、精度和设备加载一次 pipeline。
2. 模型进入 eval 与 inference 模式；无参数更新。
3. 编码当前记录的用户文本及方向/probe 条件。
4. 复用现有逐记录初始噪声函数，得到与原模型基线匹配的初始 latent。
5. 用所选模型的原生流程准备 timesteps 和实际 sigmas，保留 resolution-dependent shift。
6. 根据用户选定的检查策略解析检查步，记录实际循环索引、sigma 和网络时间值。

若使用噪声水平指定检查位置，需要明确如何映射到采样网格，以及多个指定水平落到同一步时如何处理。这是待用户选择的策略，不静默“取最近点后去重”。

### 5.3 每个正常采样步

设当前实际噪声为 `s_i`，下一步为 `s_next`。

**普通步：** 查询原提示的生成 velocity，调用正式 sampler 一次。

**检查步：**

1. 得到原提示速度 `v_user`。
2. 在同一个当前 latent 上查询 `v_forget`、`v_anchor`，构造完整差分 `d`。
3. 构造 `v0=v_user` 与 `v1=v_user+λ_ref*d`。
4. 使用无状态预览得到 `z0_next,z1_next`，此时正式 sampler 未前进。
5. 在两个候选新状态上，用原提示分别预测 `v0_future,v1_future`。
6. 构造 `x0_hat=z0_next-s_next*v0_future` 和 `x1_hat=z1_next-s_next*v1_future`。
7. 从专用自检随机源生成新噪声；在两候选间共享，不使用候选编号作为随机种子的一部分。
8. 用指定的 ρ 构造重噪状态与已知目标速度；分别查询目标/参照 probe 条件。
9. 每个样本单独计算 MSE 与 `R0,R1`。若有多个重噪或模板，按用户确认的规则聚合。
10. 应用 `R0>θ` 且 `R1<R0−m` 的接受规则，得到逐样本选择。
11. 将选中的 velocity 交给正式 sampler，正式提交恰好一次。
12. 记录分数、接受原因、速度/latent 范数及实际查询量；释放临时分支。

前瞻用的新预测不等于额外正式提交一步。第 11 步之后继续正常循环，不把临时分支已经“多走了几步”误算进正式轨迹。

### 5.4 完成后

只解码最终正式 latent。保存最终图像、解析后的配置和逐步统计。独立入口与插件返回的图像顺序都与输入记录一致。

候选完整 rollout、原模型配对对照和外部离线评价是独立阶段，是否包含在第一版交付中由用户指定；不随普通采样自动启动。

## 6. 实现时必须正确处理的几个细节

### 6.1 候选预览不能改变正式 scheduler 状态

已安装 scheduler 的 `step()` 会增加 `_step_index`。直接为两个候选各调用一次正式 `scheduler.step()`，会把正式轨迹提前推进。

对于本文使用的确定性线性 RF Euler，预览按原生计算顺序执行：当前 latent 按 scheduler 规则升精度，计算 `z+(s_next-s_i)*v`，再按原生规则转换回输出精度。不修改 scheduler、随机源或输入张量。最终选定后才调用正式 `step()`。

若用户要求随机采样、多步历史 solver、per-token 时间等配置，需另行定义等价预览。本算法规划不会静默把它们改为确定性 Euler，也不会用当前预览公式假装已经支持。

### 6.2 自检重噪不使用正在采样的 `scale_noise()`

本地 `scale_noise()` 在设置 begin/step index 后可能按当前采样索引取 sigma，且任意 ρ 未必存在于离散时间表。直接拿它重噪，可能实际用了当前 sigma 而非用户指定的 ρ。

因此，在线性路径下显式构造 `z_tilde=(1−ρ)*x_hat+ρ*epsilon`、`u=epsilon−x_hat`。网络输入时间由 backend 从物理 ρ 转换；不在已经 shift 过的物理 ρ 上再次做时间表 shift。

### 6.3 FLUX 与 SD3 的时间输入不同

本地 FLUX pipeline 在 transformer 调用前使用 `timestep/1000`，SD3 pipeline 传递其原生 timestep。两者均通过各自原生适配完成。核心算法不会硬编码“所有模型时间都乘/除 1000”。

### 6.4 生成 guidance 与 probe guidance 分开

用户提示和方向提示使用一致的生成 guidance 约定，保留 `p=f,λ=1` 时候选速度等于参照速度的代数关系。

SD3 自检可使用 raw 条件输出，避免外部 CFG 合成；FLUX-dev 有内嵌 guidance 输入，必须指定 probe 使用哪个值，不能把不做外部 CFG 与没有 guidance 混同。

这些策略及数值见待决项 4、10。不得擅自把生成 guidance 设为 1，也不得把 FLUX 的内嵌 guidance 设为 0 后称其为原始未引导 CFM 输出。

### 6.5 形状、latent 标度与逐样本决策

FLUX 的打包 latent 与 SD3 的 BCHW 形状由 backend 处理。clean estimate 和重噪始终处于模型 latent 标度，自检时不提前应用 VAE 解码的缩放/偏移。

MSE 仅对每个样本的通道/空间或打包 token 维度求平均，输出形状为 `[batch]`。不得把整个 batch 求一个分数后统一接受/拒绝。批处理候选与条件时，显式记录轴顺序，防止错配提示和 latent。

数值计算精度由用户确认，推荐将模型前向精度与差分、clean estimate、评分累积精度分别配置。预览时必须复现正式 sampler 的舍入顺序；在高精度候选上打分后提交一个不同的低精度状态，会使选择与执行不一致。

### 6.6 随机数与恢复运行

自检随机键建议由“用户指定的自检根种子、sample_id、concept_id、采样步、重噪水平编号、噪声重复编号、用途”稳定派生。候选编号不进入随机键，从而共享噪声。

算法选择用的噪声与离线审核噪声用途分开。逐样本初始噪声继续复用现有规则，不能因 batch 大小、跳过已完成图像或增加 probe 查询而改变。

首版不默认做采样过程的跨进程中途恢复；可复用现有按已完成图像跳过的机制。若需要保存并恢复中间轨迹，应明确增加保存范围与磁盘预算，属于待决项 11 的记录要求。

### 6.7 边界条件

检查步需要存在有效的下一个模型时间，不将终点 `s_next=0` 的一步作为本流程的检查步。给定的检查位置若无法满足，应在配置检查时告知，不静默减少检查次数。

首版不自动添加范数阈值、低置信回退、clean-latent clipping 或图像质量过滤规则。发现非有限预测时使该样本失败并记录原因，不能把数值失败伪装成“自检选择不纠正”。

## 7. 尚未指定的配置应如何表达

待填写模板不是可运行 benchmark 配置，带有 `status=pending_user_choices`。需要指定的字段使用 `null`；后续代码必须一次列出所有缺失字段，不从提案示例或 Python 函数默认值补齐。

配置检查区分用户选择、派生信息和条件字段：实际 sigma 网格、模型 shape、选定 profile 的解析快照由代码计算并记录，不要求用户手填；没有启用的校准、张量保存等功能，其专属字段可以保持未启用状态，不能因 `null` 而被自动开启。

验证还应区分运行阶段：只收集候选分数的 `diagnose_scores` 阶段不要求 θ、m 已确定；真正按分数控制采样的 `sample_self_check` 阶段必须有明确阈值，或用户授权的校准流程产生的参数文件。先诊断后定阈值，不等于诊断时临时采用 `θ=0,m=0`。

基本取值约束由公式确定：`λ_ref>0`、`0<ρ<1`、`m≥0`，重复次数为正整数，θ 为有限实数。具体值由用户选择；若有意做 λ=0 等退化对照，应明确标记为对照模式，而不是正式纠正配置。

| 类别 | 必须显式决定的字段 | 说明 |
| --- | --- | --- |
| 代码交付 | 实现目录、独立入口/插件范围、首次运行阶段 | 推荐方案不等于已确认方案 |
| 模型 | 模型名称、权重路径/版本、目标运行环境 | 多模型分别维护解析配置 |
| 任务 | 概念定义、原提示来源、概念映射 | 不默认采用 i2p case 或第一个 target |
| 正常采样 | profile 继承策略，或步数/尺寸/scheduler/guidance/文本长度等显式值 | 原提案的 30 步与现有 profile 的 28/40 步均未被本轮选定 |
| 检查位置 | 按索引还是按实际 sigma、位置列表或次数与区间、网格映射规则 | 不隐式等间隔，不自动去重 |
| 纠正强度 | λ_ref 或明确的离线扫描值列表 | 每次运行仍只有两个候选；扫描是多个独立运行 |
| 重噪 | ρ 的值/列表、与当前噪声重合时的显式规则 | 不静默替换为 0.5 或别的噪声水平 |
| 重复次数 | probe 提示对数量、每水平噪声样本数、聚合规则、自检种子 | 直接影响方差和查询量 |
| 接受规则 | θ、m；全局/按概念的参数范围 | 原提案提过 0，但不是已选值 |
| 校准 | 是否先收集分数再校准；开发样本与选择标准 | 不自动决定假阳性容忍率或质量权重 |
| Probe guidance | SD3 raw/外部 CFG，FLUX 内嵌 guidance 值 | 与生成 guidance 分开记录 |
| 执行与精度 | GPU、放置/卸载、样本 batch、候选执行方式、前向/评分精度 | 不自动开启多卡、offload 或改变精度 |
| 数据规模 | 提示清单/筛选、每提示种子及数量 | 不继承 benchmark 的默认种子或默认 limit |
| 输出与评价 | 输出目录、统计/张量保存范围、是否做原模型对照/完整 rollout/外部评价 | 最终标量统计与配置需保留；大张量与额外生成待指定 |

现有模型 profile 的可见示例：FLUX.1-dev 为 512×512、28 步、guidance 3.5；SD3.5-medium 为 512×512、40 步、guidance 4.5，二者使用 bfloat16。这只是仓库现状。若用户选择“继承指定 profile”，需要把这些解析值完整展示并存档。

## 8. 查询量与输出内容

令 `K` 为检查次数、`L` 为重噪水平数、`P` 为 probe 对数量、`M` 为每水平噪声样本数。单分支条件场口径下，每检查步新增查询为：

```text
方向条件：2
候选新状态的用户条件：2
两个候选 × 目标/参照 × L × P × M：4*L*P*M
合计：4 + 4*L*P*M
```

基本采样 T 步的名义总量为 `T+K*(4+4*L*P*M)`。外部 CFG 另外展开，batch 调用数和按样本条件计算的查询数分别记录。暂不把未来预测跨步复用纳入首版必要优化，防止缓存与状态检查使实现复杂化。

建议每个检查点记录：

```text
sample_id, concept_id, step_index
sigma_current, sigma_next, model_time, rho_values
lambda_ref, R0, R1, theta, margin, accepted, reason
各 probe/重噪重复的原始分数
direction_norm, correction_norm, clean_latent_norms
generation_seed, probe_noise_keys
branch_query_counts, actual_forward_calls, elapsed_seconds
```

保存完整 velocity、候选 latent 和候选图像会明显增加存储和额外工作量，是否启用由用户指定。统计日志不自动升级为完整张量存档。

## 9. 实现顺序与验证

第一步完成配置验证和只准备不采样的入口；第二步完成选定 backend 的原模型采样一致性；第三步接入双候选和重噪打分；第四步接入 benchmark 或独立诊断。具体交付范围由待决项 1 指定。

与算法正确性直接相关的验证如下：

- **原模型一致性：** 关闭检查后，文本编码、初始 latent、时间表、逐步更新及最终输出与原生 pipeline 对齐。
- **无效干预一致性：** 在测试中将 λ 设为 0 或强制全部拒绝，确认额外查询不会推进 scheduler 或改变生成随机源。
- **预览一致性：** 同一 state 和 velocity 的无状态预览，与独立原生 scheduler 单步结果对齐；正式 scheduler 每轮只前进一步。
- **评分方向：** 用可控小型测试场验证 `R=E_anchor−E_forget` 的符号、逐样本 reduction，以及接受条件的严格边界。
- **分支匹配：** 改变 batch 分组和候选执行顺序时，记录、提示与共享噪声映射保持正确；数值比较容差与执行精度一起记录。
- **独立重噪：** 查询使用指定的物理 ρ，没有从正式 scheduler 状态误取噪声系数。
- **结果身份：** 修改任一方法参数或核心实现后，不复用旧方法输出。

真实模型验证的样本、数量、硬件和可接受误差在执行前明确。当前不启动测试生成，也不把“没有崩溃”当成方法有效性验证。

## 10. 需要用户决定的问题

1. **实现范围与位置：** 代码放在哪里？做独立采样入口、接入现有 benchmark，还是两者共用核心？建议目录为 `DiT/renoise_self_check/`，但尚未创建。
2. **首批模型与运行环境：** 先支持哪个/哪些模型？使用什么权重路径或 revision，以及哪个 Python 环境？可选现有 FLUX.1-dev、SD3.5-medium，暂不自动扩展其他模型。
3. **任务与文本条件：** 先擦除哪些概念？原用户提示来自哪个文件/样本清单？方向目标/参照提示及 probe 提示由用户提供，还是先由助手拟定后再审阅？
4. **基础采样与样本规模：** 继承指定的现有模型 profile，还是另行给出步数、分辨率、scheduler、guidance、负提示和文本长度？每个提示生成几个种子、具体种子或抽样种子是什么？
5. **检查时机：** 按采样步索引还是实际 sigma 指定检查点？具体位置列表，或次数与覆盖区间是什么？若按 sigma 指定，网格映射和重合处理采用什么规则？
6. **纠正强度：** `λ_ref` 取什么值？若希望比较多个值，请指定独立实验的扫描列表；不自动采用提案示例。
7. **重噪水平：** `ρ` 取一个值还是多个值，分别是多少？若与候选当前噪声重合，替代水平或跳过策略如何规定？
8. **自检重复与随机性：** 使用多少组 probe 提示、每个重噪水平采样几份噪声，如何聚合？自检根种子及离线审核种子如何指定？
9. **阈值与校准：** `θ` 和改善余量 `m` 由用户给数值，还是先收集诊断结果再用开发集确定？若校准，请指定开发数据与选择标准，以及按全局还是按概念设置阈值。
10. **自检 guidance：** SD3 是否采用 raw 条件输出进行自检？FLUX-dev 的自检内嵌 guidance 使用哪个值？方向分支是否与原提示采用同一生成 guidance？当前方法建议后者一致，不自动改变。
11. **执行资源与记录：** 使用哪些 GPU、什么前向/评分精度、batch 大小，两个候选顺序计算还是合批？输出放哪里，是否保存完整 velocity/latent 张量或只存图像与标量统计？
12. **第一轮验收范围：** 只实现自检采样与记录，还是同时生成原模型对照、候选完整 rollout 诊断和离线评价？如果包含诊断/评价，样本量及评价方式是什么？

这些问题是方法配置与交付范围的选择，不是启动运行的批准流程。未回答前可以继续完善接口和纯逻辑设计；依赖这些选择的模型运行与实验配置保持待定。
