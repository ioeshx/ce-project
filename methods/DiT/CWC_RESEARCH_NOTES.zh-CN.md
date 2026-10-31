# CWC 研究依据、预算与验证记录

检索日期：2026-10-29。本文配合 [CWC 方法说明](CWC_METHOD.zh-CN.md)，记录公开研究与本次设计的关系。所有性能数字与预算目标均需通过实现测量；本次没有运行生成模型或得到擦除实验结果。

## 1. 对七项要求的落实

| 要求 | CWC 的具体安排 |
|---|---|
| 单概念和多概念 | 单目标与联合目标使用相同模块；联合样本一次移除所有相关目标 |
| flow matching DiT | 在 DiT block 内定义图像残差，保持原 velocity 参数化、guidance 与采样器 |
| train-free 或少量训练／计算 | 冻结生成模型，只拟合缓存上的低维预测头；计入生成与局部重放成本 |
| 在线额外计算少 | 少数 block 和时间窗内做低秩读写；无额外 DiT 调用 |
| 在线无 target／anchor 输入 | 目标集合、门控、纠正参数在部署前写入擦除包 |
| 适度建模 | 以局部条件替换实验定义监督，使用 SVD/PCA 和小 MLP 压缩；以完整采样检验有效性 |
| 参考 DiT 可解释性 | 依据概念表征与 block 定位研究选候选位置，再用生成干预验证 |

## 2. 直接相关的一手文献

### ConceptAttention：内部表示可读，但定位查询需要概念条件

[ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features，ICML 2025](https://proceedings.mlr.press/v267/helbling25a.html) 从 DiT attention 输出空间构造概念定位图，展示该空间中的线性投影能够产生清晰的概念区域。

对 CWC 的启发：内部图像表示可以作为概念识别的输入。CWC 使用普通前向中已有的状态训练门控；在线识别能力需要独立评估。若离线使用 ConceptAttention 辅助观察，概念查询只属于离线分析工具。

### Localizing Knowledge：位置需要验证，概念分布随模型变化

[Localizing Knowledge in Diffusion Transformers，NeurIPS 2025](https://arxiv.org/abs/2505.18832) 研究 PixArt-α、FLUX 和 SANA 中的知识定位，并用条件干预分析 block 的作用。其概念擦除应用对定位后的 block 做定向微调；文中应用实验的具体模型设置需要与 FLUX 定位结果区分。

CWC 采用“定位候选—完整生成干预验证”的路径。新的监督操作是保留原图像输入，重放当前 block 的去目标条件写入；纠正器仅在缓存上拟合。

### DVE：预计算差分、在线判断和多概念汇总已有直接先例

[Differential Vector Erasure: Unified Training-Free Concept Erasure for Flow Matching Models，2026-02](https://arxiv.org/abs/2602.01089) 构造 target 与 anchor 条件下的 velocity 差分，通过投影分数选择性修正。第 4.3 节提出预计算并平均每个时间步的差分，省去在线 target／anchor 前向；第 4.4 节汇总多概念纠正。

因此，“离线缓存概念信息、在线无额外模型前向”已有明确实现。CWC 的对照重点是：局部且固定图像输入的监督、随状态变化的纠正方向，以及联合目标写入的学习。在同一 flow DiT 上应同时比较 DVE 的完整版本与预计算版本，分别报告质量和成本。

### SafeRoPE：DiT 头级低维风险检测与轻量干预已有先例

[SafeRoPE: Risk-specific Head-wise Embedding Rotation for Safe Generation in Rectified Flow Transformers，2026-04](https://arxiv.org/abs/2604.01826) 对关键 attention head 构造低维风险子空间，用投影评分识别风险，并学习小规模旋转以干预 query/key。

这为低维内部风险读出提供直接参照。CWC 干预 block 的图像写入残差，纠正监督来自固定图像输入的局部条件替换。应在相同擦除集合和保留集上比较二者的检测、擦除和延迟。

### AcFlow：状态相关的图像 token 更新也已有先例

[AcFlow: Controlling Text-to-Image Diffusion Transformers via Learned Conditional Activation Flow，2026-09](https://arxiv.org/abs/2609.10723) 用概念描述驱动图像 token 的 activation flow，并通过冻结生成器的 velocity 蒸馏训练控制模块。其更新随 activation 和 token 改变。

CWC 的设计差别是：固定离线擦除集合，在缓存上拟合小预测头，在线直接预测一次低秩写入纠正。与 AcFlow 的比较应匹配使用接口：如将固定概念条件提前绑定到 AcFlow，也应计入其实际计算量，并单独评估是否具备自动触发。

### 擦除与保留的特征重叠

[RASteer: Retain-Aware Activation Steering for Concept Erasure in Diffusion Models，2026-10](https://arxiv.org/abs/2610.01969) 通过保留子空间调整擦除方向。其主要实验证据来自 U-Net 系模型。它说明“利用保留信息改善方向”已有先例；CWC 将近邻负例与联合去目标条件纳入缓存监督，并在 flow DiT 上验证效果。

[Sparse Autoencoder as a Zero-Shot Classifier for Concept Erasing in Text-to-Image Diffusion Models，2025](https://arxiv.org/abs/2503.09446) 的 ItD 使用 SAE 特征识别与停用目标，支持多概念扩展。“识别目标特征后自动干预”同样已有先例。CWC 的成本比较应包含自己的全部离线缓存与拟合开销。

### 训练式 flow DiT 基线

[EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers，ICML 2025](https://proceedings.mlr.press/v267/gao25j.html) 用 LoRA、attention 正则与对比策略执行 flow transformer 概念擦除。可作为擦除效果与离线成本的训练式参照。

## 3. 本次方案的研究主张

主张集中在一个能力：**将局部条件替换测得的概念写入变化，编译成无在线概念参数、状态相关、支持联合目标的低秩纠正器。**

对应三个证据链：

1. 在相同图像输入下测得的写入差分，是否比独立轨迹差分提供更好的保留效果。
2. 少数位置、低秩压缩和缓存拟合，分别能保留多少精确局部重放的擦除效果。
3. 联合目标监督能否在相同纠正秩和算力下，改善单概念模块叠加的组合擦除效果。

方法中的 SVD、PCA、门控和 MLP 都是已有工具。论文需要用上述证据说明新的监督单位和编译流程带来了什么能力。

## 4. 可控的首轮离线预算

先用一个角色目标验证机制，例如 Pikachu。第二阶段加入 nudity 和一个风格目标，检查跨类型组合。

单概念最小试验可从 24 对 prompt、24 个保留 prompt、每个 prompt 一个种子起步。将显式与间接表达分入训练和留出场景，验证集使用独立种子。这个规模用于判断方法链条是否成立。

建议给最小试验设定约 **160 次完整采样等价的 DiT 计算预算**。以下为预算分配：

| 项目 | 完整采样等价预算 |
|---|---:|
| 24 个目标、24 个去目标与 24 个保留样本的基础生成及缓存 | 72 |
| 少数噪声水平的同 latent 前缀及局部重放 | 8 |
| 6 个候选位置，各 4 个 prompt 的精确干预；含额外去目标前缀 | 48 |
| 位置组合验证；含重放开销 | 16 |
| 小模块上线后的检查与一次补采 | 16 |

预算按实际前向量核算。局部 block 重放单独记录次数和时间；如使用完整采样等价值换算，应公布基准时间。精确干预每步所需的去目标前缀同样计入预算。提前停止无效候选可以把计算留给留出种子。

起步参数：`U` 与 `V` 各 8 维，纠正 MLP 隐藏宽度 32，2–4 个干预 block。小模块最多拟合约 2,000 个 mini-batch 更新，按留出数据选择停止点。记录实际参数量、缓存大小和拟合耗时。

多概念复用保留缓存，新增单目标和组合样本。在 1、2、4、8 个目标下分别记录所需位置数、纠正秩、离线成本和在线延迟。目标数增加后，所需联合秩与位置数是否仍小，是扩展性实验的内容。

## 5. 实现时需要精确保持的条件

- **同 latent**：去目标前缀使用当前目标轨迹的 latent；局部重放再固定原 block 的图像输入。
- **调制条件完整**：有 pooled text conditioning 的模型同时替换该文本分量；噪声时间和 guidance 设置保持相同。
- **输出选择明确**：精确干预只将重放后的图像残差写回主轨迹；文本输出采用正常前向结果。
- **双流与单流分别接入**：单流 block 按 token 索引提取图像部分；含并行 attention/MLP 支路的 block 以整个 block 的图像输出减输入定义残差。
- **联合监督一致**：组合 prompt 的去目标条件移除所有已配置且相关的目标，并保留其他内容；训练一个总纠正头。
- **时间窗可复现**：记录实际 scheduler 噪声水平和选中窗口。模型、guidance 或采样配置改变时重新检查校准。
- **guidance 对齐**：显式 CFG 模型在现有条件分支上纠正，按原公式组合 velocity；离线验证使用相同 guidance 设置。FLUX.1-dev 沿用模型的 guidance 条件。

无需在在线路径构造完整 attention map。均值和逐维最大值在低维读数上计算；模块按 batch 样本分别门控。

## 6. 结果如何判断

擦除效果按生成图像中的目标出现率评价。角色和风格使用适合该目标的独立识别方法，并抽样人工检查；nudity 使用相应视觉检测并复核误报。所有方法使用相同判定设置。

保留评价同时覆盖近邻概念、通用 prompt 和混合图像中应保留的主体。使用去目标后的语义要求评价擦除样本，避免把原 prompt 中目标名称对应的对齐降低直接视为内容损失。另报告非目标语义保留、图像质量、同种子变化与生成多样性。

门控分别报告显式目标、间接描述和近邻保留样本的召回与误触发。最终图像的目标出现率单独报告，二者共同说明识别和纠正各自的作用。

关键消融采用相同数据量、干预位置和小模块预算：固定图像输入对比独立轨迹差分；精确重放对比低秩重放和预测纠正；常量平均纠正对比状态相关纠正；联合拟合对比单概念编辑器叠加；补采前后对比。

若精确局部重放无效，说明当前位置预算覆盖不了目标生成；若低秩重放损失明显，说明当前纠正秩不足；若预测版本损失明显，说明缓存或小模块未覆盖所需状态。按这三个环节报告结果，可以直接确定后续工作落点。

## 7. 论文表述范围

生成模型参数始终冻结，部署时通过固定擦除包干预内部状态。擦除效果以启用该包时的生成行为定义。初始实验证据应报告所评估的概念、模型和采样设置；间接描述与对抗 prompt 使用留出集合独立评估。

预期的成本目标是无额外 DiT 前向、小于 1 MiB 的起步模块，以及不超过 5% 的端到端延迟增加。这些是首轮设计目标；最终论文填写实测结果，并与擦除率和保留指标一同报告。
