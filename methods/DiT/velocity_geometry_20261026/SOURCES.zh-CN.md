# 本轮原文核查与新颖性边界

核查日期：2026-10-26。以下区分论文已提出的内容与本轮推断。主要材料来自本地 `papers/` 全文，网络原文用于核对身份与方法；没有将旧研究提案当作方法依据。此清单是围绕本轮候选的定向查新，不是全部概念擦除文献的穷尽检索。

## 重点论文

### S1. Semantic Steering for Controllable Generation: Tuning-Free Concept Erasure in Multimodal Diffusion Transformers

- [作者原文](https://arxiv.org/html/2608.12829v1)，重点 §3.2、§3.3、§4.1。
- [本地 PDF](/home/shx/code/ce-new/papers/pub/2026_ACCEPTED_ACM_MM_2026_2608.12829_Semantic_Steering_for_Controllable_Generation_Tuning_Free_Concept_Erasure_in_Multimodal_Diffusion_Transformers.pdf)。本次阅读对应全文文本 `papers/pub/experiment_audit/text/49.reading.txt`。
- 核实：从成对提示中变化 token 的中间层文本表示构造平均差分；跨连续早中期 blocks、跨去噪时间注入。
- 对本轮的意义：多组成对上下文、差分平均和跨步复用已有先例。不能声称“首次使用多组成对提示”。本轮输出速度方案不采用其内部文本分支接口。

### S2. Differential Vector Erasure: Unified Training-Free Concept Erasure for Flow Matching Models

- [作者原文](https://arxiv.org/html/2602.01089v1)，重点 §4.1–4.4、Appendix A、§5.3。
- [本地 PDF](/home/shx/code/ce-new/papers/downloads/2026_ARXIV_2602.01089_Differential_Vector_Erasure_Unified_Training_Free_Concept_Erasure_for_Flow_Matching_Models.pdf)。全文文本 `papers/dit_primary_expanded_20261024/text/2602.01089.txt`。
- 核实：Eq.(6) 为 `v_anchor−v_target`；Eq.(8) 只对差分向量归一化，用户速度没有归一化；Eq.(9) 进行单侧阈值修正。
- 核实：§4.3 已有逐时间平均差分缓存和早阶段纠正；§4.4 已有多概念独立计算修正再求和。
- 对本轮的意义：输出 velocity 编辑与在线状态相关差分本身已有先例。PKVT 需与它比较相对两端的判断、有限支持和同概念多种配对模式。

## 限制“只换数学算子”新颖性的相邻工作

| 文献及原文 | 本次核查位置 | 与本轮直接相关的已有内容 |
|---|---|---|
| [Controlling Language and Diffusion Models by Transporting Activations / AcT](https://arxiv.org/html/2410.23054)；[作者项目](https://machinelearning.apple.com/research/transporting-activations) | §3，§5，Appendix M.8 | 激活分布传输、低预算映射估计；含 FLUX 条件处理。传输并非未研究的工具 |
| [MidSteer](https://arxiv.org/html/2605.05220v3) | §4.2–4.3，§5；本地全文 `papers/dit_primary_expanded_20261024/text/2605.05220.txt` | 仿射概念切换、最小扰动变换；仅把均值差分升级成仿射映射，已有密切先例 |
| [Orthogonal Concept Erasure / OCE](https://arxiv.org/html/2605.28902) | §5.1、§5.2、§6.4 | 正交 Procrustes 参数编辑、子空间目标、DiT 扩展；“由加法改乘法/旋转”不是空白 |
| [SafeRoPE](https://arxiv.org/html/2604.01826v1) | §3.2–3.4 | rectified-flow Transformer 内部 head 风险分析及旋转；与输出速度平面旋转位置不同，但数学工具相邻 |
| [RASteer](https://arxiv.org/abs/2610.01969) | §3；本地全文 `papers/dit_primary_expanded_20261024/text/2610.01969.txt` | 从差分方向减去 retain 子空间分量，以及随重叠程度调节；不能把 retain-orthogonal steering 本身当作新点 |
| [Dynamic Negative Guidance](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5647763d4245b23e6a1cb0a8947b38c9-Abstract-Conference.html) | §3.1–3.2；本地 PDF `papers/inference_no_projection/pdfs/2025_DNG.pdf` | 状态、时间相关的负引导，借离散扩散链近似后验。自适应强度已有先例；本轮不将其离散推导直接照搬到确定性 flow |
| [Safety-Guided Flow / SGF](https://arxiv.org/abs/2603.13300)；[官方实现](https://github.com/MingyuKim87/SGF) | §4.1–4.4、Appendix Algorithm 1；本地 PDF `papers/inference_no_projection/pdfs/2026_SGF.pdf` | 负参考数据的 MMD 势能、核排斥与时间窗口；算法在预测干净样本空间施加负引导，含 flow 适配 |
| [Safe Few-Step Generation via Velocity Editing / VESFlow](https://arxiv.org/html/2606.23267v2) | §4、§5.1、Appendix B.1；本地全文 `papers/inference_preprints_20261023/text/2606.23267v2.txt` | 速度编辑、少步 flow/MeanFlow；实现用安全 scorer 的图像梯度。证明“速度编辑”本身也不能作为新颖性声明 |
| [GEM](https://arxiv.org/abs/2606.00140) | 本地全文 `papers/dit_primary_expanded_20261024/text/2606.00140.txt`，方法与引言 | 教师提供吸引和排斥的速度信号、训练学生模型。与本轮冻结模型的推理干预在成本和目标上不同 |

部分相邻工作面向较早模型；此处只用它们定位数学工具的先例，不要求本轮方法兼容 U-Net。

## 本轮新增的是哪些推断

1. **共同速度敏感性。** 由 DVE 原式直接得到：条件间共同的速度分量不改变差分，但改变原始投影分数。这是代数观察；实际模型中影响有多大尚未验证。
2. **相对两端的局部选择。** 用同一状态下的 target/anchor 距离与正的无支持底量构造选择系数，保留多种配对位移。属于本文的候选设计，不是对 S1/S2 的方法描述。
3. **底量的几何作用。** 单对核归一化在没有正底量时会约掉垂直距离；这是本轮构造的解析检查，不是生成实验结果。
4. **论文潜力。** 若上述选择在等预算、等擦除或等保留水平下持续改善结果，才有理由进一步研究其组合新颖性。当前不主张“首个”“最优”或“已超越现有方法”。
5. **输出空间的等价检查。** 在线性路径、相同状态时间上，`x̂_0=x_t−t v` 使归一化速度距离及成对位移可重写为干净预测空间的操作。因此与核排斥工作的区别必须核对参考来源和规则，单纯换成 velocity 表述不能证明创新。

## 检索记录与证据范围

本次检索围绕以下组合核对相邻工作：两篇重点论文的完整标题，以及 `concept erasure + velocity + rotation / affine / transport / covariance / kernel / paired / barycentric / Procrustes / difference-in-differences`。随后读 AcT、MidSteer、OCE、SafeRoPE、RASteer、SGF、DNG、VESFlow 原文或对应本地全文。

搜索结果还可能包含转述、聚合页或晚于当前研究日期的条目；这些没有作为本轮方法事实的依据。一个搜索无结果也不能证明某种组合不存在。

本轮查新能支持的结论是：**相关工具已有明确先例；成对条件速度中的局部选择与有限支持是一条可以检验的候选路线。** 完整新颖性仍需在候选产生有效实验后，围绕其最终实现继续核对。
