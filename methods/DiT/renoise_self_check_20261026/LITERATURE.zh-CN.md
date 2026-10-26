# 原文依据、创新边界与容易重复的路线

核查日期：2026-10-26。服务于 [方法提案](README.zh-CN.md)。本文件区分“论文明确做了什么”和“本提案由此提出的研究假说”。

本轮优先读 `papers/` 中完整 PDF，而非只读已有统计报告或 `papers/metadata/` 中的摘要文本；两篇重点论文的 metadata 文本只包含前部内容，不能用它们代替方法节和附录。未引用项目其他对话的方法提案。

## 1. Semantic Steering：内部文本表征转向

- 完整题名：*Semantic Steering for Controllable Generation: Tuning-Free Concept Erasure in Multimodal Diffusion Transformers*。
- 阅读版本：arXiv:2608.12829v1。
- [公开全文](https://arxiv.org/html/2608.12829v1)；[摘要与版本记录](https://arxiv.org/abs/2608.12829)。
- [本地完整 PDF](../../../papers/pub/2026_ACCEPTED_ACM_MM_2026_2608.12829_Semantic_Steering_for_Controllable_Generation_Tuning_Free_Concept_Erasure_in_Multimodal_Diffusion_Transformers.pdf)。
- 定位：§3.2、§3.3；§4.1；§4.3 与 Tables 2–3。

### 原文机制

1. 为目标概念与安全替代内容建立成对提示。
2. 在中间时刻、中间 block 提取配对提示中有差异的文本 token 表征，求和、求平均差并归一化缩放。
3. 把这个向量加到多个连续早中期 blocks 的文本 token 输出。
4. 同一构造向量跨采样步复用；无需学习模型参数。

§4.3 的消融以 SD3.5 为对象，讨论构造 block `b=10`、注入 blocks `3…10`、构造时刻 `t=0.5`。这些数值是该实验的具体设置，不能直接当成任意 DiT 的通用规则。论文主实验使用 SD3.5-medium 和 FLUX.1-dev。

### 对本提案的启发与边界

继承“只改变一个因素的目标/安全参照”，不继承文本分支定位和 block 注入。架构迁移时仍可使用模型自带文本编码接口，但无需知道内部 token 如何进入双流或单流结构。

原文以 RF 的轨迹性质解释跨步静态 steering 的有效性；本提案只接受其报告的经验结果，不据此建立“所有条件速度方向在所有状态上相同”的假设。

论文用 GPT-4o 生成测试提示，用 GIPHY、LLaVA-1.5、NudeNet 等评价不同任务。**离线数据构造/评价用的模型与在线方法依赖必须分开统计。** 不能因实验用 LLaVA，就说其 steering 算法每步依赖 VLM。

## 2. DVE：本轮必须超出的直接基线

- 完整题名：*Differential Vector Erasure: Unified Training-Free Concept Erasure for Flow Matching Models*。
- 阅读版本：arXiv:2602.01089v1。
- [公开全文](https://arxiv.org/html/2602.01089v1)；[摘要与版本记录](https://arxiv.org/abs/2602.01089)。
- [本地完整 PDF](../../../papers/downloads/2026_ARXIV_2602.01089_Differential_Vector_Erasure_Unified_Training_Free_Concept_Erasure_for_Flow_Matching_Models.pdf)。
- 定位：§4.1–4.4，Algorithms 1–2，Appendix A、C。

### 原文核心算子

令 `d=v_anchor-v_forget`、`d_hat=d/||d||`，则其即时分数为 `q=<v_user,d_hat>`；只在 `q<τ` 时增加 `γ(τ-q)d_hat`。这等于对用户 velocity 在一个概念差分方向上的分量进行有条件校正。

论文的理论分析把目标提示、安全提示、无关提示分为三种几何情况。无关语义与差分方向近似正交，是解释这个判据的理想情况，不是对真实模型每个状态的保证。

### 已有内容，不能重复声称为新贡献

| 内容 | 原文定位 |
| --- | --- |
| 同状态的目标/anchor velocity 差 | §4.1 / Eq. (6) |
| 即时投影、阈值与擦除强度 | §4.2 / Eqs. (8)–(9) |
| 代表提示轨迹上按时间聚合并预缓存差分 | §4.3 / Eq. (10) |
| 只在前中期纠正 | §4.3 / Eq. (11) |
| 多概念分别计算，并基于原始用户 velocity 求和 | §4.4 / Eq. (12)，Appendix B.1 |
| 扩展到 FlowEdit | §4.4，Appendix B.2 |

在线单概念版本每个纠正步比原生成多两次条件场查询。缓存版本可减少在线查询，但需要计入预处理轨迹、概念数、时间点、分辨率以及缓存规模。不能把 DVE 描述成只能在线计算全时段差分的昂贵方法。

§5.1 的主设置是 FLUX.1-dev，Euler 28 步、512×512、guidance scale 3.5。Appendix C/Table 9 的 T2I 设置为 `γ=3,τ=-15`。这些数值绑定原实现的速度单位和 guidance 设置，新接口不能机械照搬。

### 本提案提出的质疑，尚未由实验证实

用户 velocity 可能包含与目标概念无关的共享运输成分。若在同一组比较里给目标、anchor 和用户速度都加上共同分量 `b`，差分 `d` 不变，而用户投影改变 `<b,d_hat>`。这只说明即时投影不具有对共同偏移的不变性；**不能单凭这个代数事实断言其在实际生成中必然错判。**

需要用真实模型区分：整体运输的干扰、概念本身未激活、prompt/anchor 选择不合适、时间和符号实现错误。这些因素不能混成一个“DVE 不懂语义”的结论。

### 两个容易被重新命名的等价关系

**安全参照中心化投影。** 如果对 `v_user-v_anchor` 而不是 `v_user` 做投影，并将超出的目标方向分量推回参照，那么它通常可重写为 DVE 使用 `τ(z,s)=<v_anchor,d_hat>` 的状态相关阈值。可能有实用改进，但不应仅因改了中心点就声称新范式。

**同状态的“自检”退化。** 固定同一状态上的 `v_a,v_f`，对一个候选向量 `u` 比较两种重建残差，有：

\[
\|v_a-u\|^2-\|v_f-u\|^2
=\|v_a\|^2-\|v_f\|^2-2\langle u,v_a-v_f\rangle.
\]

右边仍是对同一差分方向的仿射投影。主方案必须在候选新状态上重新预测，再对其 clean estimate 独立重噪并查询模型，不能省掉这些步骤后把同状态打分重新命名为后果验证。

## 3. 其他近邻：为什么只比较两篇还不够

以下仅保留影响本次选题判断的内容。会议状态不作为方法优劣或创新性的依据。

| 论文 | 原文依据与已覆盖机制 | 本提案需要区分的部分 |
| --- | --- | --- |
| Steering Fields: Adaptive Vector Fields for Safe Image Generation and Beyond | [全文 §3.2–3.3](https://arxiv.org/html/2609.39573v1)；在当前状态组合源、吸引、排斥条件 velocity，二次目标给出解析更新；有 FLUX1、SD3.5 实验 | 多条件输出速度组合、状态相关方向已经存在；其系数调节不等于本方案的候选后果自检 |
| Safe Few-Step Generation via Velocity Editing / VESFlow | [全文 v2 §4.1–4.2、§5.1](https://arxiv.org/html/2606.23267v2)；[本地 PDF](../../../papers/inference_preprints_20261023/pdfs/2606.23267v2.pdf)；由预测干净样本的安全验证器梯度纠正速度，并跳过低风险样本 | “检查干净预测再调速度”已有；本提案的差异在同 DiT 的重噪残差、前向候选比较、不依赖在线外部验证器；不声称首创闭环纠正 |
| Modular Energy Steering for Safe Text-to-Image Generation with Foundation Models | [全文 §3](https://arxiv.org/html/2604.02265v1)；对 clean estimate 使用 CLIP/VLM 能量并反馈梯度，包含 SD3 实验 | 预测终点的概念能量已有先例；本提案不解码给外部模型评价，也不反传该能量 |
| Orthogonal Negative Guidance in Attention Feature Space | [全文 §3.2](https://arxiv.org/html/2605.29390v1)；[本地 PDF](../../../papers/downloads/2026_ARXIV_2605.29390_Orthogonal_Negative_Guidance_in_Attention_Feature_Space_for_Text_to_Image_Generation.pdf)；在注意力输出空间对负特征正交化 | “正交化以保留语义”已有，且需要内部注意力接口；本方案不沿此方向增加组件 |
| Unified Safe In-context Image Generation in Multimodal Diffusion Transformers via Restricting Unsafe Information Flows / UVR | [PMLR 原文入口](https://proceedings.mlr.press/v306/yang26an.html)；[本地 PDF](../../../papers/pub/2026_PUBLISHED_ICML_2026_Unified_Safe_In_context_Image_Generation_in_Multimodal_Diffusion_Transformers_via_Restricting_Unsafe_Information_Flows.pdf)，§4；通过不安全 patch anchors 定位并限制注意力信息流 | 生成状态中的概念检查与选择性干预已有；本方案研究输出速度接口下能否完成相近任务，不引入注意力定位 |

VESFlow 的完整训练依赖要按配置区分：v2 裸露任务复用现有 CLIP NSFW verifier，暴力任务另训练 MLP。不能将它统一说成“所有任务都需要训练”，也不能将其 verifier 的已有训练成本隐去。

## 4. 自检评分的思想来源与技术边界

### 4.1 Diffusion Classifier 是识别思想来源

[Your Diffusion Model is Secretly a Zero-Shot Classifier](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.html)，ICCV 2023，使用条件去噪重建误差比较类别，无需额外训练分类器。这里只借鉴“复用生成模型进行条件识别”，不将旧 U-Net 纳入本提案的模型适用或实验目标。

本方案使用 RF 的速度重建残差，而非直接搬用 DDPM 噪声误差的似然解释。CFM 的 MSE 不是任意样本、任意 guidance/蒸馏模型上的精确负对数似然；候选排序的可靠性只能由诊断建立。

### 4.2 基础 FM/RF 原文约束

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)：FM 是对条件概率路径向量场的回归框架，覆盖多种路径；不能把 FM 一律等同于一种线性 RF 路径。
- [Flow Straight and Fast](https://arxiv.org/abs/2209.03003)：RF 学习尽量沿连接端点的直线路径运输；不能因此断言训练后的全部实际轨迹完全笔直。

主方案选择线性 RF，是为了直接构造已知重噪速度目标并复用公开 sampler。它的内部结构无关性不等同于路径参数化无关性。

### 4.3 Guidance 与模型来源

[FLUX.1-dev 官方模型卡](https://huggingface.co/black-forest-labs/FLUX.1-dev) 明确标注 guidance distillation；[SD3.5-medium 官方模型卡](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) 提供其生成配置。主方案区分生成 guidance 与原始条件残差，是由这些模型差异引出的设计选择；不能把蒸馏输出上的评分可靠性当成已有事实。

### 4.4 需要一个真正不同内部结构的模型

仅在 FLUX.1-dev 与 SD3.5 上测试不足以充分证明 single-stream 与 double-stream 都能迁移。建议加入 Lumina-Image 2.0：其 [ICCV 2025 原论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Qin_Lumina-Image_2.0_A_Unified_and_Efficient_Image_Generative_Framework_ICCV_2025_paper.pdf) 描述统一的 single-stream block，官方 [作者仓库](https://github.com/Alpha-VLLM/Lumina-Image-2.0) 提供生成实现。

这里仅把它作为输出接口迁移实验候选，不预设该模型拥有与 FLUX 完全相同的目标概念覆盖率。迁移时保留其原生文本编码器，不增加 captioner 或在线改写器；只在原模型确实可生成的概念上评价擦除。

## 5. 新贡献应如何陈述

可以作为本轮研究目标的表述：

> 研究原 flow-matching DiT 能否通过独立重噪的条件速度重建比较，评价候选速度修改的概念效果，并在不使用额外在线验证器、不访问内部结构的情况下改进推理期擦除。

必须通过实验建立的差异：候选新状态的检查优于即时投影；重噪检查优于同状态残差；效果不是更多模型查询或更强纠正的结果；评价器以外的人工/独立指标仍认可改善。

不能作为已有结果写入摘要的表述：精确估计概念后验、保证无关语义不变、保证轨迹不会恢复概念、所有 FM/DiT 即插即用、永久遗忘、零额外开销，以及未经真实实验的 SOTA。

本轮文献阅读给出的是选题与实验边界；没有把组合一个新名字视为已完成论文创新性证明。
