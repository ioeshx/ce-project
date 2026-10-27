# 文献依据与创新性边界

核对日期：2026-10-27。历史报告用来定位，方法判断优先依据论文自身正文。以下区分原文事实、由事实得到的推断与本轮设计；不以旧文件名确认最新发表状态，也不主张查新穷尽。

## 1. 最接近的已有方法

| 原始来源 | 已有机制及正文定位 | 本轮需要体现的增量 |
|---|---|---|
| [DVE](https://arxiv.org/html/2602.01089v1) | §4.1–4.4：同状态 anchor−target velocity 差、单侧投影纠正、逐时间缓存、前期纠正、多概念叠加 | 差分与原 gate 作为共同起点；新增当前用户状态中的短续程响应反求。缓存、时间窗与输出接口均不算新意 |
| [Semantic Steering](https://arxiv.org/html/2608.12829v1) | §3.2–3.3：从变化文本 token 的中间层表示形成安全／目标方向，注入连续早中 blocks，跨采样步复用 | 不依赖文本分支或层号；不重新提出成对语义差向量。比较在相当擦除水平下的保留与实际成本 |
| [Steering Fields](https://arxiv.org/html/2609.39573v1) | §3.2–3.3：每步重估多条件速度，以吸引／排斥的解析组合控制生成 | 架构无关、自适应速度场与二次目标已有。区别需落在“同一次编辑经过原提示后续计算的实际响应” |
| [VESFlow](https://arxiv.org/html/2606.23267v2) | §4–5：从干净预测与安全验证器梯度构造速度编辑；任务不同，验证器训练依赖也不同 | 本轮不用安全 scorer 或其梯度，比较条件短续程与脉冲响应。不能把速度编辑本身当新贡献 |
| [IAM](https://arxiv.org/html/2607.14945v2) | §3.3、Eq.(8)/(11)：学习危险 attention 方向，逐步进行 attention PGD | 本轮不学判别方向，也不优化 attention；“DiT 自省”“冻结主干”本身不新 |
| [UVR](https://proceedings.mlr.press/v306/yang26an.html) | §4.1–4.2：危险 patch anchors 定位、注意力信息流限制和注噪；离线区域标注用到 Grounded-SAM | 本轮没有区域检测器或特定 attention 接口；不宣称自带空间擦除能力 |
| [SafeRoPE](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_SafeRoPE_Risk-specific_Head-wise_Embedding_Rotation_for_Safe_Generation_in_Rectified_CVPR_2026_paper.pdf) | §3.2–3.4：逐头危险子空间、风险分数和学习型低秩旋转 | 不将 SVD、正交补保留或小参数训练包装成空白；本轮避开 Q/K 和 RoPE 依赖 |

DVE 的成本缩减尤其需要完整比较：其 §4.3 已有缓存与分阶段干预，不能只用最昂贵实时版本对比本方法。上述论文中，没有在本次核对位置找到 SCRM 的完整计算链；这只支持把该组合列为研究候选，不能证明它是首次提出。

## 2. 短续程、控制与数值校正的已有基础

[Lookahead Sample Reward Guidance](https://proceedings.mlr.press/v306/kim26bo.html) 已研究少步 lookahead 和无需神经反向传播的 reward guidance。[Flow Map Reward Guidance](https://proceedings.mlr.press/v306/huang26ae.html) 已用生成 flow map 研究少步控制。因此“看未来”“冻结模型”“用 flow map 控制”都不能单独认领为创新。本轮在线接受依据是同模型的条件短续程参照及其响应误差，不是外部 reward。

残差 `b-R(delta)` 回加到 `delta`，属于已有迭代反求思想的一步近似。式子简单并不等于具有新数学；本轮没有建立精确最优控制、目标条件分布或全局收敛定理。

与 Heun 等高阶 solver 也要区分：solver 的额外评估用于更准确地积分既定速度场，本方案跨条件形成编辑参照、跨候选测量编辑响应。二者不是同一操作，但同等前向预算下“多走几步普通差分／提高积分精度”可能更划算，必须作为实验对照。

## 3. 表征研究到底支持什么

| 一手论文 | 支持的观察 | 不能据此推出的结论 |
|---|---|---|
| [REPA](https://arxiv.org/html/2410.06940)，§3.2 | velocity-prediction SiT 的内部语义读出随深度变化，较后层下降；外部表示对齐可改变训练 | 某一层具有纯净可删的概念方向；REPA 本身也不是免训练擦除组件 |
| [SRA](https://arxiv.org/html/2505.02831)，Fig.2、附录 A | 自身较深／低噪声特征可为表征学习提供教师；可读性依赖噪声与深度 | 跨噪声特征可以直接相减；自监督表示训练等于 train-free 擦除 |
| [ConceptAttention](https://proceedings.mlr.press/v267/helbling25a.html)，[正文](https://arxiv.org/html/2502.04320v2) §5.3–5.5 | DiT attention-output 表征能用于概念定位；存在近邻混淆及目标缺席仍选相近概念的现象 | 热图是概念存在概率；删除热图高值位置必然因果擦除 |
| [Localizing Knowledge in DiTs](https://proceedings.neurips.cc/paper_files/paper/2025/hash/157a6107a95883f6ccc0e38e80a24295-Abstract-Conference.html)，[正文](https://arxiv.org/html/2505.18832v2) §3–4、附录 D | attention contribution 与条件 withholding 可帮助定位因果影响；定位涉及 PixArt、FLUX、SANA，正式遗忘微调实验在 PixArt | FLUX 已完成同等局部遗忘验证；固定层号适用所有模型；内部归因自动给出删除程度 |

由此得到的本轮推断是：**可读性、瞬时因果作用、后续完整生成结果，应分别检验。** SCRM 因而把内部表示留给可选离线分析，核心方法直接测量输出干预经过后续计算的效果。它并不需要运行 REPA、SRA 或 ConceptAttention。

## 4. 为什么桥接训练没有成为主线

候选曾考虑：缓存原模型实际目标轨迹的中间状态 `z_b`，与同 seed 的安全终点 `x_a` 配对，在 `s∈[0,b]` 上训练线性桥 `z_s=(s/b)z_b+(1-s/b)x_a`，监督解析导数 `(z_b-x_a)/b`，只更新少量通用低秩参数。

这可以降低对专用内部层和外部判断模型的依赖，但它是标准 CFM 在选定 coupling 与局部时间区间上的应用。[Flow Matching](https://arxiv.org/html/2210.02747)、[Conditional Flow Matching](https://arxiv.org/abs/2302.00482) 已提供相关基础，不能声称新的桥理论。

更重要的是，[VisualErase](https://arxiv.org/html/2610.05000v1) §3.1–3.4 已经从概念承载视觉状态、明确的安全视觉终点构造解析重定向监督，再加条件／无条件分支与保留训练。它的对齐图像构造依赖图像编辑和多个外部组件，且主实验是 SD1.4，但**这不消除方法思想上的重叠**。仅替换成 flow DiT 或原模型自生成配对，暂不足以支撑本轮主线。

此外，[GEM](https://arxiv.org/html/2606.00140) §4 已在真实目标轨迹状态上做 safe/unsafe teacher velocity 对比；[EraseFlow](https://arxiv.org/html/2511.00804v3) §4.1 通过安全 anchor 轨迹进行训练；[FlowErase-OPD](https://arxiv.org/html/2608.07620v1) 使用学生轨迹和多擦除教师进行蒸馏。这些工作进一步限制了“目标状态覆盖”“轨迹训练”“少量 LoRA”可单独主张的空间。

本轮保留桥接方向的否决理由，不把它发展成另一套复杂模块。

## 5. 与本目录已有构想的关系

| 已有提案 | 已覆盖的问题 | SCRM 的具体差异 |
|---|---|---|
| [FRS](../FRS_PROPOSAL.zh-CN.md) | 用两步真实生成响应，离线联合拟合内部注入候选 | 同一科学问题的输出端延伸；在线、当前 prompt／状态特异；一次全维残差修正，无内部选层和系数求解 |
| [SIDV](../SIDV_PROPOSAL.zh-CN.md) | 内部内容索引的局部 velocity 差分缓存 | 不建立字典，不假设内部相似度能预测所需差分 |
| [IRDE](../input_realizable_erasure_20261026/README.zh-CN.md) | 用少量条件输入动作实现目标 velocity 响应 | 不优化输入条件；实际动作直接作用于输出速度 |
| [重噪自检](../renoise_self_check_20261026/README.zh-CN.md) | 候选结果重噪，用自身重建误差选择 | 无重噪分类评分，构造新的校正方向，以短续程响应误差接受 |
| [PKVT](../velocity_geometry_20261026/README.zh-CN.md) | 多组即时速度参照、核权重与有限支持选择 | 不更换即时门控或几何插值，重点测量干预后的生成响应 |

因此这里使用“输出端延伸／待验证候选”的定位，不以新缩写暗示新的擦除原理已经成立。

## 6. 实际使用的本地正文入口

以下均位于仓库内，不依赖临时提取文件。文本行号绑定本次已有版本；需要复核时优先看节与公式。

| 文献 | 本地文件与阅读位置 |
|---|---|
| DVE | [完整文本](../../../papers/dit_primary_expanded_20261024/text/2602.01089.txt)：约 283–295 行差分、376–436 行 gate、443–467 行缓存、1236–1297 行成本 |
| Semantic Steering | [PDF](../../../papers/pub/2026_ACCEPTED_ACM_MM_2026_2608.12829_Semantic_Steering_for_Controllable_Generation_Tuning_Free_Concept_Erasure_in_Multimodal_Diffusion_Transformers.pdf)：§3.2–3.3、Table 2–3 |
| VESFlow | [完整文本](../../../papers/inference_preprints_20261023/text/2606.23267v2.txt)：§4–5.1 |
| IAM | [完整文本](../../../papers/inference_preprints_20261023/text/2607.14945.txt)：约 298–326 行、358–368 行 |
| UVR | [PDF](../../../papers/pub/2026_PUBLISHED_ICML_2026_Unified_Safe_In_context_Image_Generation_in_Multimodal_Diffusion_Transformers_via_Restricting_Unsafe_Information_Flows.pdf)：§4.1–4.2 |
| SafeRoPE | [PDF](../../../papers/pub/2026_PUBLISHED_CVPR_2026_2604.01826_SafeRoPE_Risk_specific_Head_wise_Embedding_Rotation_for_Safe_Generation_in_Rectified_Flow_Transformers.pdf)：§3.2–3.4、补充 §9.1 |
| GEM | [完整文本](../../../papers/dit_primary_expanded_20261024/text/2606.00140.txt)：约 526–603 行，时间窗、target trajectory 与对比监督 |
| FlowErase-OPD | [完整文本](../../../papers/dit_primary_expanded_20261024/text/2608.07620.txt)：约 209–216、344–406 行 |
| VisualErase | [完整文本](../../../papers/dit_primary_expanded_20261024/text/2610.05000.txt)：§3.1–3.4；另核对上方作者 HTML 全文 |

Steering Fields、表征论文、lookahead／flow-map guidance 通过上面链接的一手正文、正式论文页或作者代码核对；未把只有摘要的搜索结果当作已读全文。旧报告中的录用标签不用于支撑本轮方法新颖性判断。

## 7. 能提出的论文主张

可检验的主张是：**输出差分的即时方向与原提示下的实际短续程响应存在有意义的失配；用当前状态的响应做一次向量补偿，在同预算下优于只调强度，并改善最终擦除／保留。**

当前不能声称首个 flow DiT 擦除、首个架构无关 velocity 控制、首个 lookahead、自适应强度、无外部模型的概念判断或严格遗忘保证。定向搜索没有发现相同标题／术语，也不构成原创性证明。
