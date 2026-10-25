# 文生图扩散模型的推理时概念擦除与 Adapter：分类、研究问题及跨领域启发

检索与核验日期：**2026-10-22 至 23**。本文是选题调研，不包含新实验结果。先检索本地 `papers/codes/inventory.json` 的 185 条记录与前轮全文，再补查 CVF、PMLR、ICLR/NeurIPS 论文集、ACL Anthology 和会议官方日程。

范围沿用“已录用论文”，不把仅有预印本记录的候选列为正式文献；主会、Findings 和 Workshop 分别注明。继续排除闭式权重编辑及解析投影／解析构造擦除算子的路线。由于本次明确讨论 **adapter**，将“骨干冻结、另外训练少量参数”的方法纳入研究比较，但不称其为全流程 training-free，也不改写上一轮严格免训练清单。文中“可探索”指值得验证的研究问题，不能据此声称全球首次。

**优先判断：这个领域已经足够分类。最值得先验证的方向，是面向最终生成结果的风险预测与干预决策；其次是减少近邻概念误伤的局部 adapter，以及预算受限的回退／分支采样。单纯增加动态强度、检测器、空间 mask 或 LoRA，已有非常接近的工作。**

## 1. 当前方法如何分类：五类执行机制，配合两个独立维度

### 1.1 不应把 inference-time 和 adapter 当作互斥类别

**Inference-time 描述干预发生在生成过程中，adapter 描述干预由什么模块实现。** 一个训练好的 adapter 可以在推理时改变激活；一个完全免训练的方法也可以在每个采样步反向传播，以优化当前输入而非模型参数。分类时应分别记录：执行机制、是否学习新参数、控制信号是否随生成状态更新。

下面的五类按主要执行机制归纳，允许交叉归类，不是论文数量的互斥统计。

|类别|主要改动对象|代表工作|共同优势|共同困难|
|---|---|---|---|---|
|A. 采样轨迹／负引导|噪声预测、score、干净样本预测或 latent 更新|SLD、DNG、Safe Denoiser、SGF、CASG+SLD|容易接到采样器；直接改变生成分布|负方向不一定只含目标语义；强度、时间窗与计算成本耦合|
|B. 文本条件控制|本次生成的文本嵌入，或离线学到的 soft prompt|STG、Prototype-Guided|骨干冻结；可以利用现成安全评估器|条件变化可能影响多个语义；梯度与评估器调用有开销|
|C. 注意力与空间信息流控制|注意力列、key/value、局部 patch 或 latent|Concept Corrector、UVR、DAG、GIE、ARCE|能够定位和局部干预；适合实体或局部属性|定位错误会误伤；全局风格、身份与关系概念不一定对应一个区域|
|D. 中间语义特征控制|隐藏激活、SAE 编码后的语义特征|Semantic Steering、SAeUron、SAEmnesia|可解释、可组合；可绕开表面关键词|表示纠缠、特征分裂、层与时间依赖；SAE 本身要训练|
|E. 可学习残差适配|冻结层旁的低秩／小型残差分支|SPM、Receler；GIE 的强度预测器属于辅助控制模块|可保存、装卸和组合概念模块|增加训练成本；组合、迁移和非目标保持需要单独验证|

E 类描述得更偏实现方式，因此与 A—D 有交叉：例如，GIE 的执行器属于 C 类，控制强度由学习模块产生；SAE 是 D 类执行器，也是一种广义外挂模块。**用于论文 related work 时，推荐以 A—D 的干预对象为主线，再单列可学习 adapter；用于实验设计时，使用下面两条轴更有区分度。**

|独立维度|需要区分的情况|例子与含义|
|---|---|---|
|学习成本|无新参数训练／推理期输入优化／离线模块训练|SLD 属于第一种；STG 优化当前条件；SPM、DAG、GIE、SAeUron 属于第三种。调用已有预训练评估器仍有外部依赖|
|反馈方式|固定干预／提示条件化调度／当前生成状态反馈|固定 steering 向量与根据风险重新选择动作不同；由提示预测一整条时间曲线，也不同于逐步读取当前状态的闭环控制|

因此，不能仅根据题名中的 “tuning-free”“inference-time” 判定是否需要训练，也不能把所有“adaptive”都当作相同的闭环控制。

### 1.2 支撑分类的已录用代表论文

以下逐篇给出发表场所、会议时间和方法概括。时间以会议举办年月为主；出版社上线时间与会期不同时单独注明。

**A 类：采样轨迹与负引导。**

1. **[Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models（SLD）][D01] — CVPR 2023，2023-06。** 对无条件、用户提示和不安全概念分别预测噪声，在 classifier-free guidance 中扣除不安全方向，并用逐维阈值、预热和动量调节干预。它是冻结模型的安全采样方法，适合作为后续控制策略共享的基础执行器；不能将其描述为永久删除模型知识。

2. **[Dynamic Negative Guidance of Diffusion Models（DNG）][D02] — ICLR 2025，2025-04。** 利用反向扩散链估计当前样本属于待排除类别的后验概率，据此随采样状态和时间调整负引导，缓解固定强度造成的过度或不足干预。方法不训练额外分类器；论文包含基础类别生成和 Stable Diffusion 实验。它已覆盖“根据当前风险动态调强度”这一想法。

3. **[Training-Free Safe Denoisers for Safe Use of Diffusion Models][D03] — NeurIPS 2025，2025-12。** 以不安全图像集定义负分布，用核加权近似不安全去噪器，在样本接近该分布时纠正干净图像预测，并选择关键采样步实施干预。它不依赖骨干微调；此处讨论不搭配投影方法的实现。负样本库的覆盖程度与检索／核计算成本是设计时需要考虑的因素。

4. **[Safety-Guided Flow: A Unified Framework for Negative Guidance in Safe Generation（SGF）][D04] — ICLR 2026，2026-04。** 用 MMD 势能梯度使生成轨迹远离不安全参考图像，并统一解释已有排斥式引导；通过控制屏障函数分析关键干预时间窗。本文只讨论其非投影实现。它意味着“给负引导加控制论解释”“只在关键时间窗干预”也已有直接先例。

5. **[When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance（CASG）][D05] — CVPR 2026，2026-06。** 分析不同有害类别的引导冲突，在每步比较各类别方向与提示方向的相关性，选择当前最相关类别进行纠正。本调研只纳入 **CASG+SLD**；另一个 CASG+SAFREE 分支涉及投影，仍排除。它已研究多类别冲突，但“选择一个类别”与“同时约束所有类别的最终风险”仍是不同目标。

**B 类：条件嵌入控制。**

6. **[Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models（STG）][D06] — NeurIPS 2025，2025-12。** 由当前 noisy latent 预测最终图像，用安全函数评价，再把安全梯度反传到本次生成的文本条件嵌入。骨干和评估器参数保持冻结，更新的是当前输入条件。其裸露、暴力和风格实验说明“冻结扩散模型，在推理时优化输入”本身已经是成熟的候选基线。

7. **[Prototype-Guided Concept Erasure in Diffusion Models][D07] — CVPR 2026，2026-06。** 将目标概念的不同视觉表现组织为原型，并优化可学习 soft prompt 来表示这些视觉模式；推理时选择与当前提示相关的原型用于负条件引导。骨干不变，但原型提示需要离线梯度优化，因此属于学习小模块的路线。它已触及“单个负概念向量不足以覆盖多模态概念”这一问题。

**C 类：注意力与空间信息流。**

8. **[Concept Corrector: Erase Concepts on the Fly for Text-to-Image Diffusion Models][D08] — PRCV 2025，2025-10；Springer 上线于 2026-01-23。** 在两个生成检查点预测干净图像，使用已有视觉检测器或 VLM 判断目标是否出现，随后通过概念掩码融合原提示与替代提示对应的注意力输出。不同阶段采用不同纠正策略，兼顾结构与细节。它是“观察中间结果再纠正”的直接先例；本地使用已录用工作的公开作者版。

9. **[Unified Safe In-context Image Generation in Multimodal Diffusion Transformers via Restricting Unsafe Information Flows（UVR）][D09] — ICML 2026，2026-07。** 从不安全示例缓存视觉 patch 的注意力输出作为语义锚点，推理中根据相似度、连通域与区域扩张定位风险，再局部衰减注意力信息流并注入噪声。该方法冻结模型，同时覆盖文生图与图像编辑；因此不能再把“首次在 MM-DiT 上局部控制”或“首次同时处理图文条件”作为宽泛动机。

10. **[Growth Inhibitors for Suppressing Inappropriate Image Concepts in Diffusion Models（GIE）][D10] — ICLR 2025，2025-04。** 用当前视觉 query 与待擦除概念计算额外注意力图，提取目标对应的空间响应，将负向加权后的响应注入原注意力计算。另训练小型 MLP，根据初始采样阶段的跨层统计预测抑制强度；训练标签来自对不同强度结果的比较。它已经是“冻结骨干＋学习强度预测 adapter”，并讨论相似概念叠加造成的过擦除。

11. **[Detect-and-Guide: Self-regulation of Diffusion Models for Safe Text-to-Image Generation via Guideline Token Optimization（DAG）][D11] — CVPR 2025，2025-06。** 用少量带区域标注的示例学习 guideline token 和跨层聚合权重，使注意力图更准确地检测不安全属性与对象的组合；再根据检测区域、响应强度及面积调节局部负引导。扩散模型冻结，检测 token 需要训练。其方法已明确考虑背景泄漏、属性—对象组合和安全内容保持。

12. **[Concept Erasure via Attention Redirection（ARCE）][D12] — CVPR Findings 2026，2026-06。** 在冻结 FLUX 的注意力层中追加可学习 eraser key 与缩放参数，使其竞争目标概念的注意力概率；推理时用中性 value 引导到替代语义。训练同时约束目标注意力下降、非目标注意力保持和重建。因此，“增加一个可学习 token，并保留其他 token 的注意力”也已有明确实现。这里的 venue 是 Findings，不能写成 CVPR 主会。

**D 类：中间语义特征。**

13. **[Semantic Steering for Controllable Generation: Tuning-Free Concept Erasure in Multimodal Diffusion Transformers][D13] — ACM Multimedia 2026 主会，已录用，计划会期 2026-11-10 至 14。** 从目标／替代提示对提取 MM-DiT 中间块文本表示之差，平均得到 steering 向量，并在生成时加到选定文本分支激活。这里是激活平移，不是将当前激活分解后投影删除。录用已由官方日程数据确认；截至检索日会议尚未举行。

14. **[SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders][D14] — ICML 2025，2025-07。** 先训练 SAE 重建扩散模型中间激活，再按目标／非目标样本的响应差筛选相关稀疏特征；采样时对选中特征实施抑制，解码回原空间并保留重建误差项。骨干冻结，但 SAE 要训练。正文明确讨论相似概念保持、抽象概念、激活存储及可被移除的外挂机制等局限。

15. **[SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders][D15] — ICML 2026，2026-07。** 针对无监督 SAE 将一个概念分散到多个特征的现象，在 SAE 训练中加入概念标签，促使指定概念集中到对应潜在单元，再通过单元级干预实现擦除。它已有多概念、顺序擦除和鲁棒性实验。因此，“让概念特征更集中”需要提出比监督 SAE 更具体的改进才可能形成新贡献。

**E 类：可学习残差适配。**

16. **[One-dimensional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing Applications（SPM）][D16] — CVPR 2024，2024-06。** 在冻结层旁增加一维低秩分支，学习擦除信号及输入相关的调节向量；训练时通过 latent anchoring 约束非目标行为，推理时用 Facilitated Transport 根据提示调节模块作用。论文还研究模块组合及向相关扩散模型迁移。它是 adapter 路线必须对照的早期核心工作，不能因前轮只收 training-free 而继续忽略。

17. **[Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers][D21] — ECCV 2024，2024-09 至 10。** 在冻结扩散模型的 cross-attention 层后插入轻量 eraser，将其输出作为残差作用于视觉特征；通过目标概念的注意力空间信息约束区域外残差，并交替学习试图恢复目标的对抗提示与擦除器。它同时研究 locality、鲁棒性及擦除器组合。因此，“轻量残差＋空间约束＋对抗训练”已经是已有方法，不能作为新 adapter 方案的全部贡献。

**已录用 Workshop 补充：调度策略的学习。**

18. **[SafeGuide: Adaptive Inference-Time Safety Control for Diffusion Models][D17] — ICLR 2026 的 Principled Design for Trustworthy AI Workshop，2026-04。** 官方摘要说明：将安全引导参数化为钟形时间曲线，用风险感知强化学习训练策略网络，依据提示语义和生成阶段决定时间与强度。**当前核验深度仅为官方录用页面与摘要**：会议 PDF 返回 403，故不进一步断言其状态输入、训练损失细节或安全保证。它足以构成“学习一个动态调度器”方向的直接相关工作；不能误写成 ICLR 主会论文。

**三个相关边界工作，单独使用。**

**[SEGA: Instructing Text-to-Image Models using Semantic Guidance][D18] — NeurIPS 2023，2023-12。** 计算概念条件与无条件噪声预测之差，按响应强度阈值化并加入动量，支持正向增加与负向减少语义。它无需训练，是通用编辑基础；概念减少功能不能直接等同于专门的安全擦除系统。

**[Adaptive Median Smoothing: Adversarial Defense for Unlearned Text-to-Image Diffusion Models at Inference Time（AMS）][D19] — ICML 2025，2025-07。** 按 token 与待移除概念的相关性配置各向异性扰动，并对扰动后的预测或文本编码实施中值平滑，以缓解对抗提示恢复目标的风险。插件自身不更新权重，但论文实验依赖已擦除骨干。

**[M-ErasureBench: A Comprehensive Multimodal Evaluation Benchmark for Concept Erasure in Diffusion Models][D20] — WACV 2026，2026-03。** 评测普通文本、学习嵌入与反演 latent 下的概念恢复，提出 IRECE 防御：由原模型注意力定位区域，在指定步骤局部替换为噪声，再由擦除模型继续生成。该工作同时贡献基准和外挂防御。它与 AMS 都不能作为“从原始模型开始完全免训练擦除”的同口径基线。

### 1.3 哪些宽泛创新点已经被覆盖

|不能单独作为新贡献的表述|已有直接先例|仍可能需要研究的更窄问题|
|---|---|---|
|根据风险动态调节强度|DNG、DAG、GIE；SafeGuide 的调度学习|基于动作后最终风险、而非仅当前分数选择干预|
|中途看图，再纠正|Concept Corrector、STG|评估器在不同噪声阶段的不确定性；漏检后的恢复策略|
|只修改危险区域|DAG、UVR、Concept Corrector、Receler|相关但应保留属性、对象关系、全局风格的选择性控制|
|处理多个待擦除概念|SPM、GIE、CASG、SAeUron、SAEmnesia|同时满足各类别风险上限及保留约束，检查最差类别|
|冻结模型，只训练小模块|SPM、Receler、GIE、DAG、ARCE、SAE 路线|新模块学习何种信息，能否在同等预算下改善控制决策|
|适配新模型或 MM-DiT|SPM 的相关模型迁移；UVR、Semantic Steering、ARCE|跨骨干／采样器／少步模型的迁移边界与低成本校准|
|控制论解释或关键时间窗|SGF|带不可靠观测、计算预算及恢复动作的可检验控制问题|

这些先例不代表问题已经彻底解决，但要求 motivation 从“有没有这个功能”转向“在哪些条件下现有机制仍系统性失效”。

## 2. 尚未充分解决的问题，以及可形成论文的研究方向

### 2.1 先明确研究对象与可声称的目标

对冻结骨干且可卸载的系统，更准确的研究目标是：**在受控推理流程内，降低目标概念的输出概率，同时保持非目标能力。** 卸下模块即可恢复原模型，因此不能仅凭生成结果宣称永久遗忘；SAeUron 的局限部分也明确指出这一部署边界。[SAeUron][D14]

实验中应分别定义“不要生成某身份／风格”和“不要生成违反给定规则的内容”。前者可能要求无条件抑制一个命名概念，后者往往依赖上下文。二者不能共用一个含糊的安全标签。攻击者可以优化文本、seed 或已开放的图像条件，但是否能关闭外挂控制器，必须在威胁模型中写清楚。

下表中，**文献事实**表示原文已有相应机制或局限；**研究假设**表示需要自行实验验证，不能直接写成已有方法全部失败的结论。

|问题|已有证据及已做部分|值得验证的剩余问题|证据强度|
|---|---|---|---|
|① 当前风险指标不一定足以比较动作后果|DNG 的类别后验本已有最终类别的概率含义；Concept Corrector／STG 使用中间干净图像预测；SGF 分析干预时窗|相同风险分数，在不同动作、剩余步数、seed 和采样器下是否对应不同最终风险？能否用动作条件化预测改进决策？|机制有全文依据；差异的大小是待验证假设|
|② 非目标保持仍容易被过粗指标掩盖|DAG、ARCE 已做局部保持；SAeUron 明确指出相似／抽象概念的困难；SAEmnesia 缓解特征分裂|擦除某身份时能否保留其他身份；去除一种属性时能否保留对象、姿态、关系及相近的合法属性？|局限有直接依据；具体新方法收益未验证|
|③ 多风险平均分不能代表逐类满足约束|CASG 已指出方向冲突；GIE 合并相似概念以防过擦除；SAEmnesia 研究顺序移除|抑制 A 后是否提高 B 的风险？当两个类别同时出现时，能否保证两者都得到足够抑制，而不只改善宏平均？|冲突有直接依据；约束式解决方案是研究建议|
|④ 检测器和控制器可能共享盲点|STG 优化安全分数；DAG／UVR 依赖内部响应定位；AMS、M-ErasureBench 已研究恢复攻击|优化后分数下降，是真正擦除了目标，还是改变了评估器易识别的特征？未见攻击、同义改写和视觉条件是否绕过监测？|依赖关系有依据；具体绕过需适配攻击验证|
|⑤ 额外计算往往缺少统一预算比较|STG 有反传；Concept Corrector 有图像检查；参考样本引导和 SAE 有不同成本|在相同延迟／显存预算下，何时多检查一次、加一次干预或保留一个分支更划算？少步采样是否仍有足够恢复空间？|开销来源明确；优劣必须测量|
|⑥ 模块复用与控制策略迁移仍是不同问题|SPM 已做相关模型迁移；UVR 和 ARCE 已面向新架构；SAeUron 有跨步数实验|换骨干、guidance、分辨率或采样器后，风险阈值、动作强度和时间窗是否需要重校准？|已有迁移不能忽略；更广迁移结论尚待验证|

对应文献见：[DNG][D02]、[Concept Corrector][D08]、[STG][D06]、[SGF][D04]、[DAG][D11]、[ARCE][D12]、[SAeUron][D14]、[SAEmnesia][D15]、[CASG][D05]、[GIE][D10]、[AMS][D19]、[M-ErasureBench][D20]、[SPM][D16]、[UVR][D09]。

### 2.2 优先方向 A：预测最终风险，再选择最小必要干预

**建议的核心问题：相同预算下，动作条件化的最终风险预测，能否比当前图像分数／注意力响应，更准确地选择干预时机与强度？** 这比“加一个动态 gate”更具体，也更容易被证伪。

将生成步骤按时间顺序记为 (k=0,\ldots,T)，冻结骨干参数为 (\theta)，候选干预为 (u_k)：

\[
z_{k+1}=F_\theta(z_k,c,k,u_k).
\]

小型预测模块估计的对象不是“现在看起来有多像目标”，而是：

\[
Q_\psi(z_k,c,k,u;\pi)
\approx
\Pr\bigl[ R(x_T)>\tau\mid z_k,c,\ u_k=u,\ \text{之后按策略 }\pi\text{ 继续}\bigr].
\]

这里 (R) 是预先定义的目标检测／违规评价，\(\pi\) 是后续控制策略。**如果策略、模型或采样器改变，预测目标的分布也会改变**；这不能省略，否则容易把一个只在训练轨迹上准确的分类器误当成通用风险估计器。

**最小实现。** 先固定一个已有的非投影执行器，例如 SLD 或局部注意力纠正；只允许“不干预、弱干预、强干预”三个动作。在若干检查点缓存骨干状态，由小型时序 head 读取时间、概念条件和少量激活统计，预测各动作后的最终风险及保持损失。根据风险上界和代价选动作，而不是同时重写执行器、定位器和采样器。可以先用实际继续采样的少量分支建立参考，再蒸馏到预测 head；后者属于 adapter 训练。

**数据与训练。** 从同一中间状态分别采取候选动作，按指定后续策略完成生成，得到动作—终点标签。训练集、校准集和测试集按提示语义或概念拆分，避免同模板不同 seed 泄漏。独立终点评估器及人工抽查用于检查标签质量。对受干预后的状态补充训练样本，检查由控制行为引起的分布变化。

**与最近工作的差别。** DNG 的类别后验、GIE 的初期强度预测、DAG 的区域响应、Concept Corrector 的检查点纠正、SafeGuide 的调度学习都必须比较。特别是，DNG 的后验本来就可关联最终类别，不能声称“首次考虑最终风险”；动作价值函数也不是新的数学概念。潜在新增部分是**同一状态下不同动作的后续风险估计、预测不确定性与干预收益的联合验证**；“使用一个 MLP／强化学习／动态曲线”本身不是新增贡献。跨领域可借鉴 [ACoFi][X06] 的不确定性控制和 [DSA][X08] 的廉价旁路读出。

**需要严格限定的保证。** 校准覆盖率不自动等于目标概念永不出现；ACoFi 给出的也是特定条件下的长期软保证。图像生成缺少每步可观测的真实安全标签，不能直接照搬其在线更新定理。第一版论文应报告分组校准误差、漏检率与最终风险；若要理论保证，必须写出标签正确性、校准分布、策略固定性及多检查点选择的条件。

**否定性实验。** 如果在固定 NFE／延迟下，当前检测分数加一个简单时变阈值已达到相同风险—保持曲线，或用真实 rollout 获得的理想风险估计也无法改善动作选择，那么这个方向缺少继续复杂化的依据。

### 2.3 优先方向 B：以近邻保留为目标的局部残差 Adapter

**建议的核心问题：擦除目标概念时，如何保留与它高度相关、但规则允许的语义？** 例如擦除特定人物身份，同时保留其他人物、服装、姿态及场景；去除某艺术家风格，同时保留画面对象与布局。这比泛称“降低 FID”更接近概念擦除的选择性要求。

**最小实现。** 在一到两个已定位的模块旁放置零初始化的小型残差 adapter，用独立检测 head 估计目标证据；局部实体任务采用 patch gate，全局风格任务采用不同的全局分支，不预设所有概念都能空间分割。检测 head 读取该次动作施加前的激活，以减少控制器通过直接改变同一读出而获得低分的捷径；终点仍用独立评估器复核。这种设计只能减少一种耦合，不能宣称完全独立监测。

**数据重点。** 使用成组的近邻正负例：保持场景、对象或构图，只改变被禁止的身份／属性／风格条件；同时加入“出现相关词但并不含目标”和“不出现目标词却生成目标”的例子。对有可靠局部标注的样本约束区域外保持；对全局概念使用对象、关系和布局指标。训练目标可包含目标移除、非目标保持、安全输入零干预及时间一致性，但这些常见损失项本身不能当作新颖性。

**与最近工作的差别。** SPM 已有 latent anchoring 和提示门控，Receler 已将轻量残差、概念局部正则和对抗提示学习结合；DAG 已做属性—对象结合及局部定位，ARCE 已有非目标注意力保持，SAEmnesia 已做特征集中。可争取的贡献应落在**针对近邻语义的可复现实验设定、针对失败原因的模块设计，以及跨未见组合的保持效果**。仅把 HiddenGuard 的 token mask 换成图像 patch mask，或重复 Receler 的训练组成，不足以构成强贡献。[Receler][D21]

**评价与否定性实验。** 与 SPM、Receler、DAG／UVR、ARCE 或适用的 SAE 方法在同一骨干、数据量和参数预算下比较。至少区分：目标出现率、相关合法概念保持率、区域外结构变化、完全安全提示误干预率。如果优势只体现在无关的 COCO 提示，而在近邻合法概念上仍同样退化，motivation 就没有被解决。

**适用边界。** 配对提示或相同 seed 只是构造近似对照，不能自动证明因果解耦；两个提示也不保证得到像素对齐图像。需要人工检查配对有效性，且不要对未对齐输出强行使用像素差作为核心保持指标。

### 2.4 优先方向 C：预算受限的回退与分支采样

**建议的核心问题：当目标结构已形成、单次纠正效果很差时，能否回到更合适的中间状态，以较少额外计算保留一个安全且语义接近的生成路径？** 这是更适合严格 training-free 的候选，但算力成本风险较高。

**最小实现。** 缓存少量中间检查点；当终点风险预测或多个检测信号不一致时，从最近的候选检查点展开 2—4 条带不同干预强度／安全替代条件的路径。使用目标风险、非目标保持和已耗预算共同选择继续路径，并限制回退次数。在随机采样器上可以使用受控噪声分支；确定性采样器若重复相同状态和相同动作只会重放同一路径，需要显式改变动作或合法的随机化方式。

**与最近工作的差别。** Concept Corrector 已有分阶段纠正；[ABack][X04] 已在文本隐私保护中研究定位回退点；[FK steering][X05] 已在包括文生图在内的扩散生成中研究粒子重采样。因此创新点不能写成“首次用搜索做扩散控制”，而应是**针对目标概念恢复的回退位置选择、保持约束及计算分配**。单纯把安全分数作为 FK reward 是很强的基线，不应直接当作完整新方法。

**评价与否定性实验。** 同时比较相同总 NFE 的 best-of-N、固定检查点重启、FK 重采样、单路径更强引导和自适应分支。报告检查器耗时、失败重试、输出多样性和最坏延迟。如果在等算力下仅优于单路径而不优于 best-of-N，或各分支收敛到相同的安全替代图像，则收益可能只是多花算力或牺牲多样性。

**适用边界。** 文本可删除已生成 token 并重写，图像结构在多个去噪步中共同形成；两者的回退位置不能直接类比。少步模型可回退的空间也更小，需要先做时间点诊断。

### 2.5 可作为后续扩展的方向：多概念约束组合

在前面任何一个单概念方案有效后，再考虑同时限制多个目标：为每个概念分别估计最终风险，检查某个动作对其他目标的副作用，按最差风险及保持预算选动作。可以用迭代的标量权重更新或学习的路由器实现，不必引入被排除的解析投影。

这里的研究价值是**约束满足和概念组合规律**，不是“支持多概念”本身。需要特别比较 CASG 的逐步单类别选择、GIE 的相近概念去重和多个 SPM 的组合；依次增加目标数量，记录逐类风险、误伤与顺序效应。若所有约束无法同时满足，应把失败／拒绝输出的比例单列，不能用大量拒绝换取表面上的完美擦除率。

### 2.6 推荐的启动顺序与实验规范

**先做诊断，优先验证 A，保留 B 作为较直接的 adapter 路线；需要严格免训练时选 C。** 不建议一开始把三个方向全部拼成复杂系统。以下是建议实验计划，并非本次已运行的实验。

1. **机制诊断。** 在一个能稳定运行相关基线的骨干上，先选局部实体／属性、身份、全局风格中至少两类概念。用约 60—100 个含近邻对照的提示、每个 4 个 seed 做初筛；在若干检查点记录当前分数、不同干预后的终点结果和保持损失。这个规模用于发现现象，不足以支持最终性能结论。
2. **先检验“选择动作”是否有空间。** 对同一状态试验不同动作，比较当前分数阈值、时间调度和知道实际后果的理想选择。如果理想选择也没有改善，先换问题；如果差距明显，再训练小预测器逼近它。
3. **控制比较条件。** 第一阶段先共用执行器与评估器，只比较控制策略；第二阶段再比较不同执行器。训练方法报告骨干是否冻结、参数量、训练图像数与离线计算；免训练方法报告前向、反向、VAE 解码与外部评估器调用成本。
4. **设置四组测试。** 显式目标、隐式或对抗目标、与目标高度相关但合法的提示、一般安全提示。若声称支持图像编辑，再加入视觉条件及反演 latent 的恢复测试；仅做文生图时不要夸大到所有入口。
5. **同时报告擦除、保持、鲁棒性和成本。** 使用逐概念出现率／ASR、相关合法概念保持率、提示与对象关系遵循、质量／多样性、误干预率、延迟及显存。画风险—保持—计算的权衡曲线，不只挑单个超参数结果。统计区间应以提示为聚类单位，避免把同一提示的多个 seed 当作完全独立样本。
6. **避免评估器自证。** 控制信号与最终评价至少使用不同模型／特征来源，并人工抽查分歧样本。对启用防御后的系统重新优化攻击；随机化方法应评估跨随机种子的攻击迁移。报告拒绝、重采样失败与被丢弃候选，不能只评价成功输出。

|路线|主要额外成本|最先需要证明的现象|不应作为论文主张的内容|
|---|---|---|---|
|A：最终风险与动作选择|离线完成轨迹、预测 head、校准数据|相同当前风险下，不同动作后果存在可学习差异|“第一个动态安全 gate”；无条件安全保证|
|B：近邻保持 adapter|成组数据、局部标注或结构评价、模块训练|现有方法在相关合法概念上出现系统性误伤|“冻结骨干所以完全免训练”；mask 本身的新颖性|
|C：回退／分支|额外采样、检查点缓存、检查器调用|回退到特定阶段比末端加大强度更有效|“首次把粒子搜索用于文生图”；不计额外算力的比较|

对选题的暂定判断是：**A 的问题定义最值得先做诊断；B 更容易连接现有 adapter 实现；C 更符合全流程免训练偏好，但必须先证明等预算收益。** 这只是基于文献的研究判断，不是已经验证的性能排名。

## 3. 其他领域近期已录用工作：可以借鉴什么，迁移时要补什么

### 3.1 语言模型安全：从整体拒绝到局部控制与模式切换

**X01 — [SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding][X01]，ACL 2024 主会，2024-08。**

方法先用少量安全拒绝数据通过 LoRA 等方式构建安全专家，再让原模型与专家共同预测下一 token；在双方高概率候选的交集中重构分布，放大专家偏好的安全输出，通常只干预开头若干 token，随后恢复普通解码。**它包含专家训练，不是全流程 training-free。** 这篇较早，但仍是“推理时控制＋小模块训练”必须理解的基础。[正式论文及全文][X01]

**可迁移点：** 一个完整生成器加一个安全专长分支，不一定要复制两套完整模型；可以只在关键阶段调用小型安全分支，并把原模型的语义保留能力用于约束候选动作。**限制：** 扩散状态没有可直接枚举的下一词词表，不能把 token 概率相减公式直接替换成噪声预测相减，再声称二者等价；相关想法也会与 SLD、SPM 重叠。它更适合作为 adapter 设计启发与消融参照。

**X02 — [Please refuse to answer me! Mitigating Over-Refusal in Large Language Models via Adaptive Contrastive Decoding（AdaCD）][X02]，ACL 2026 主会，2026-07。**

AdaCD 比较同一模型在普通输入与极端拒绝提示下的输出，提取与拒绝相关的分布差异；再结合候选 token 的排序一致性和置信度，动态选择增强还是减弱拒绝倾向，以同时处理恶意请求与对正常请求的过度拒绝。它不训练新模型参数。论文研究的是安全与过拒绝之间的平衡，不是语言模型永久遗忘。[正式论文及全文][X02]

**可迁移点：** 概念擦除也可以同时研究“漏擦”和“过擦”，加入保留正常语义的对照条件，检查控制器是否应继续加力或停止。**限制：** 对裸露等风险主动反向增强并不合适；图像侧更稳妥的起点是“不干预／较弱干预／较强干预”之间切换。LLM 的 top-token 排名指标在连续扩散中没有直接对应物，需要重新定义状态或候选动作间的一致性。

**X03 — [HiddenGuard: Fine-Grained Safe Generation with Specialized Representation Router][X03]，ACL 2026 主会，2026-07。**

HiddenGuard 的 PRISM 模块复用语言模型的中间表示，学习低秩 activator 信号及专门的 router，用 token 级标注识别局部敏感内容，并在生成流中进行选择性删改。它希望保留一段回答中的正常信息，而不是因为局部敏感内容拒绝整段回答；骨干参数冻结，新增模块需要训练。正文将低秩 activator 描述为信号分支，不宜笼统理解成把标准 LoRA 合并进骨干。[正式论文及全文][X03]

**可迁移点：** “检测信号分支＋内容执行分支”的职责分离，以及局部粒度的保留评价，最适合启发方向 B。**限制：** 图像中删除一个 patch 并不等于删除一个词；人物身份与风格还可能分布在全图。需要处理空间交互、跨步一致性及合理的安全替代内容，并与已有 DAG／UVR 的局部控制直接比较。

**X04 — [Adaptive Backtracking for Privacy Protection in Large Language Models（ABack）][X04]，Findings of ACL 2026，2026-07。**

ABack 针对 RAG 中的隐私泄漏，先从检索资料提取受保护实体，用字符串及语义匹配配合短程 lookahead 监测输出；发生风险时，在有限上下文窗口内用语言模型反向推断最早的泄漏倾向，再从该位置改写并继续生成。它复用预训练模型，并离线构建用于推断的状态原型，不训练目标生成器。文中的“mental state”是其建模变量，不能据此理解为已经测得模型真实心理状态。[正式论文及全文][X04]

**可迁移点：** 干预不一定发生在风险首次被检测到的位置，可能需要恢复到更早且仍可纠正的状态；这直接启发方向 C。**限制：** ABack 的状态推断依赖可读文本和受保护实体，图像侧需要用轨迹证据定位回退点。不要仅把“文本片段”换成“去噪步”就声称完整迁移。

### 3.2 扩散采样与安全控制：从单步打分到路径和不确定性

**X05 — [A General Framework for Inference-time Scaling and Steering of Diffusion Models（FK steering）][X05]，ICML 2025，2025-07。**

FK steering 同时维护多条生成轨迹，以中间状态的奖励势函数为轨迹赋权，并在若干时刻重采样，使计算更多地投入预计能产生高奖励结果的路径。其框架支持不需要奖励梯度的实现，也讨论不同提议采样器与中间奖励。它已经包含文生图实验，因此属于**邻近的通用生成控制工作**，不能冒充从纯外部领域首次移植；文本实验还包含为红队测试增加罕见毒性内容，并非全部是抑制毒性。[正式论文及全文][X05]

**可迁移点：** 不必把擦除限制为“沿一个向量推开”，可以对整个路径分布进行选择；不同安全替代模式可由不同粒子保留。**限制：** 直接改成负安全奖励只是基线。需要研究预算分配、粒子退化、保留约束和最终评估器一致性；也不能将有限粒子的结果当作精确满足某个目标分布。

**X06 — [Safe Control using Learned Safety Filters and Adaptive Conformal Inference（ACoFi）][X06]，L4DC 2026，2026-06-17 至 19。**

ACoFi 把学习得到的 Hamilton–Jacobi 可达性安全过滤器与自适应 conformal inference 结合。系统根据已观测的安全值预测误差更新不确定性范围及切换阈值；当名义动作的安全下界不足时，切换到学习的安全策略。它在车辆与 Safety Gymnasium 环境验证，依赖已训练的价值／策略模块；理论约束是长期误覆盖或相应安全值事件的软保证，而非每一步绝不违规。[正式论文及全文][X06]

**可迁移点：** 控制决策应考虑“预测有多不可靠”，而不只看一个风险均值；它为方向 A 提供比手工阈值更明确的研究框架。**限制：** 原任务能够在后续时刻获得用于更新的目标量，图像去噪中往往只有代理检测分数。标签不可得、策略变化和最终概念识别错误都需要重新处理，不能移植定理后直接宣称有保证的概念擦除。

### 3.3 模块化与路由：安全能力不一定写回骨干

**X07 — [Steering MoE LLMs via Expert (De)Activation（SteerMoE）][X07]，ICLR 2026，2026-04。**

SteerMoE 比较行为相反的成对输入在各专家上的激活频率差，筛出与目标行为相关的专家；推理时修改 router 的打分，使指定专家进入或退出 top-k 路由，从而控制安全性、忠实性等行为。其模型权重无需微调，改变的是每次推理的专家选择，并非通过子空间投影删除激活。[正式论文及全文][X07]

**可迁移点：** 与其让一个统一擦除器覆盖所有目标，可以研究有明确功能分工的小模块，以及状态相关的路由与组合。**限制：** 普通稠密 DiT 没有同构的原生 MoE 专家；若自行增加专家，就引入架构与训练成本。对于当前选题，借鉴的应是**对新增 adapter 的选择策略**，而非将其等同于此前排除的权重剪枝／连接屏蔽。

**X08 — [Disentangled Safety Adapters Enable Efficient Guardrails and Flexible Inference-Time Alignment（DSA）][X08]，ICLR 2026 的 Principled Design for Trustworthy AI Workshop，2026-04。**

DSA 把安全计算放在与冻结骨干并行的侧支网络，复用骨干逐层表示；一类侧支做安全分类，另一类预测用于安全生成的 logits，再与原模型输出插值。将分类侧支与对齐侧支结合，可以按输入调整对齐强度。文中比较线性读出、瓶颈侧支及 ladder side-tuning 等实现，均应计入模块学习成本。录用由 ICLR 官方日程和作者机构页面确认；本地全文为作者公开版本，首页亦标明该 Workshop。[官方录用页面][X08]、[作者机构说明][X08A]

**可迁移点：** 尽量复用已有去噪特征，以小模块同时支持风险读出和安全控制，减少逐步调用完整 VLM 的成本。这对方向 A、B 都有帮助。**限制：** 扩散噪声预测不是词表 logits，侧支应输出风险、候选动作价值或残差等与扩散过程相容的量；“分离检测和控制”以及“按输入调整 adapter 强度”已被 DSA／GIE 等做过，不能直接作为新颖性。

### 3.4 两篇应影响实验设计的近期评估论文

**X09 — [Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in Inference-Time Interventions][X09]，EACL 2026 主会，2026-03。** 论文将特异性分为保持一般能力、保持相关控制属性，以及在分布变化下继续保持这些属性；在减少过度拒绝和忠实性幻觉的任务中，发现一些干预能通过常规能力与控制测试，却削弱受到攻击时的安全性。对图像擦除的启发是：正常提示保持不等于对抗条件下保持，尤其要评测“为了减少过擦而增加的机制是否重新打开目标生成路径”。这是评估框架启发，不是现成图像擦除方法。[正式论文及全文][X09]

**X10 — [SteeringSafety: Benchmarking Representation Steering in LLMs Across Safety Perspectives][X10]，ICML 2026，2026-07。** 论文统一评测多个 representation steering 方法、模型和安全维度，显示某个维度的改善可能伴随其他维度变化，效果依赖具体方法—模型—目标的组合。对概念擦除可借鉴其“干预目标 × 被影响属性”矩阵：擦除概念 A 后，不仅测 A，还要测相关合法概念和其他禁止概念 B。本文只借鉴评估设计，不将其评测的 PCA 等投影算法重新引入候选方案。[正式论文及全文][X10]

### 3.5 将跨领域启发落到本课题的具体问题

|外部机制|迁移到概念擦除时的具体问题|最接近的图像侧已有工作|需要超越的部分|
|---|---|---|---|
|ACoFi 的不确定性与策略切换；DSA 的低成本读出|如何估计某个干预动作后的最终风险，并在不确定时采取保守动作？|DNG、DAG、GIE、SafeGuide|动作条件化、校准有效性、保持代价与等预算收益|
|HiddenGuard 的局部处理；AdaCD 的减少误干预|能否保留与目标相似的合法属性，并对完全安全输入趋近于不操作？|SPM、DAG、UVR、ARCE、SAEmnesia|近邻概念与未见组合上的保持效果，不只是一般图像质量|
|ABack 的回退点定位；FK 的中间轨迹选择|晚期识别风险后，回到哪里重采样最有效？|Concept Corrector、STG；FK 本身已有 T2I|目标特定的恢复机制，及相同计算预算下的优势|
|SteerMoE 的模块选择；DSA 的安全旁路|如何只调用有用的安全模块，避免所有模块同时干扰生成？|SPM 组合、GIE 强度预测、CASG 类别选择|未见目标组合、多风险约束及路由错误分析|
|两篇 specificity／多维评估工作|擦除成功是否以破坏相关合法概念或其他安全约束为代价？|M-ErasureBench、CASG 及现有保持评测|系统的交叉影响矩阵、适配攻击、误干预与失败输出统计|

**建议的精读顺序：** 图像侧先读 SPM／Receler → GIE／DAG → DNG／Concept Corrector → STG／UVR → SAeUron／SAEmnesia；跨领域先读 ACoFi → DSA → HiddenGuard → ABack → FK steering。若最终选择完全免训练路线，先做 C 的等预算基线；若允许小模块训练，先做 A 的动作后果诊断。

**检索边界与可复查性。** 这是一份围绕选题的代表性调研，不是穷尽论文目录。外部文献以 2025—2026 年为主，保留 SafeDecoding 作为 2024 年基础；正文明确注明两个 Workshop 条目。SafeGuide 暂只有官方摘要级方法核验，其余新增重点工作已检查全文。此前排除的解析擦除路线不作为候选创新方案；仅有预印本或未明确发表场所的记录不列入正式推荐。这样的筛选也意味着：正式投稿前仍需补做更广的新颖性排查，不能把本报告中的研究假设写成“从未被探索”。

配套文件：[`sources/catalogue.json`](sources/catalogue.json) 记录出处、会期、核验深度及范围；[`sources/downloads.json`](sources/downloads.json) 记录新增全文下载状态与 SHA-256；`sources/pdfs/` 与 `sources/text/` 保存公开全文及提取文本。已在 `papers/` 中的原文复用原路径，前轮报告保留不变。

[D01]: https://openaccess.thecvf.com/content/CVPR2023/html/Schramowski_Safe_Latent_Diffusion_Mitigating_Inappropriate_Degeneration_in_Diffusion_Models_CVPR_2023_paper.html
[D02]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/5647763d4245b23e6a1cb0a8947b38c9-Abstract-Conference.html
[D03]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/c66e1fcc9691aae706250638f36f681b-Abstract-Conference.html
[D04]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/a1722a6bd1023c026a3d6a570fb3af75-Abstract-Conference.html
[D05]: https://openaccess.thecvf.com/content/CVPR2026/html/Xiang_When_Safety_Collides_Resolving_Multi-Category_Harmful_Conflicts_in_Text-to-Image_Diffusion_CVPR_2026_paper.html
[D06]: https://papers.nips.cc/paper_files/paper/2025/hash/7c5f910828177d484f1ecd4bdc357585-Abstract-Conference.html
[D07]: https://openaccess.thecvf.com/content/CVPR2026/html/Cai_Prototype-Guided_Concept_Erasure_in_Diffusion_Models_CVPR_2026_paper.html
[D08]: https://link.springer.com/chapter/10.1007/978-981-95-5567-3_7
[D09]: https://proceedings.mlr.press/v306/yang26an.html
[D10]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c48dcd605cfb17d5b4f94ce106a915f7-Abstract-Conference.html
[D11]: https://openaccess.thecvf.com/content/CVPR2025/html/Li_Detect-and-Guide_Self-regulation_of_Diffusion_Models_for_Safe_Text-to-Image_Generation_via_CVPR_2025_paper.html
[D12]: https://openaccess.thecvf.com/content/CVPR2026F/html/Schechter_Concept_Erasure_via_Attention_Redirection_CVPRF_2026_paper.html
[D13]: https://2026.acmmm.org/site/technical-programme.html
[D14]: https://proceedings.mlr.press/v267/cywinski25a.html
[D15]: https://proceedings.mlr.press/v306/cassano26a.html
[D16]: https://openaccess.thecvf.com/content/CVPR2024/html/Lyu_One-dimensional_Adapter_to_Rule_Them_All_Concepts_Diffusion_Models_and_CVPR_2024_paper.html
[D17]: https://iclr.cc/virtual/2026/10019138
[D18]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ff83037e8d97b2171b2d3e96cb8e677-Abstract-Conference.html
[D19]: https://proceedings.mlr.press/v267/han25k.html
[D20]: https://openaccess.thecvf.com/content/WACV2026/html/Weng_M-ErasureBench_A_Comprehensive_Multimodal_Evaluation_Benchmark_for_Concept_Erasure_in_WACV_2026_paper.html
[D21]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5685_ECCV_2024_paper.php
[X01]: https://aclanthology.org/2024.acl-long.303/
[X02]: https://aclanthology.org/2026.acl-long.1823/
[X03]: https://aclanthology.org/2026.acl-long.1482/
[X04]: https://aclanthology.org/2026.findings-acl.1857/
[X05]: https://proceedings.mlr.press/v267/singhal25b.html
[X06]: https://proceedings.mlr.press/v331/huriot26a.html
[X07]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d61c5f5ef91e7e8a091b7b8f72b853c-Abstract-Conference.html
[X08]: https://iclr.cc/virtual/2026/10019214
[X08A]: https://machinelearning.apple.com/research/disentangled-safety
[X09]: https://aclanthology.org/2026.eacl-long.268/
[X10]: https://proceedings.mlr.press/v306/siu26a.html
