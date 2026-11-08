**补偿式非线性 Steering：面向 DiT 概念擦除的研究方案**

日期：2026-11-07。状态：已退出当前研究方向；用户要求优先干预 text 分支，本方案直接干预 image FFN。当前修订见[文本效应 Steering 方案](/home/shx/code/ce-new/methods/DiT/text_effect_steering_20261107/METHOD.zh-CN.md)。本文保留为历史设计记录，尚未运行真实模型的擦除实验。

建议研究的核心是：**在图像分支 FFN 的非线性之前减去统计向量，再在 FFN 输出处加入由正常样本响应统计得到的补偿向量，使最终干预集中于目标与正常内容之间的非线性响应差异。** 下文暂称 Compensated Nonlinear Steering，CNS。

核心研究假设是：存在一些层和噪声区间，同一固定扰动在正常内容上产生近似共同的响应，在目标概念上产生额外响应。补偿前者后，后者仍能改变目标生成。方案的价值取决于这个假设在 DiT 上是否成立。

**已有工作的边界。**

| 工作 | 原文中的主要机制 | 对本方案的约束 |
|---|---|---|
| [CASteer](https://arxiv.org/html/2503.09630) | 从正负提示词统计 CA 输出的均值差；逐 patch 按正投影强度施加干预 | 将“模型 output”准确理解为所选 CA 层的输出；单独换到 MLP 输出不足以形成创新，原文已有相关消融 |
| [SHIFT](https://arxiv.org/html/2604.09213) | 修改 pooled text embedding 和 DiT attention 后的文本 token；采用均值差或 SVM 方向，并用分类信号调节强度 | DiT 适配、时不变向量、SVM 门控已经有直接先例 |
| [RASteer](https://arxiv.org/html/2610.01969) | 从擦除方向中削弱保留子空间分量，并依据重叠程度校准 | 保留子空间正交化和重叠自适应属于已有核心机制 |
| [MidSteer](https://arxiv.org/html/2605.05220) | 利用协方差约束求解最小扰动的仿射概念操控 | 白化、协方差加权和最小扰动仿射映射需要与其比较 |

进一步核对了 [BARRIER](https://arxiv.org/html/2605.15737) 的激活区域约束、[ConceptPrune](https://arxiv.org/html/2405.16534) 的参数剪枝，以及 [Weight Updates as Activation Shifts](https://arxiv.org/html/2603.00425) 对干预位置和非线性的分析。CNS 拟研究的具体贡献是“跨 FFN 非线性的成对向量干预＋正常响应的闭式补偿＋依据补偿后响应选择干预位置”。上述工作中的通用思想，例如统计对比、功能保持和使用非线性，本身不作为原创贡献。此次检索支持与这些工作的机制区分；全局新颖性还需在立项和投稿时继续查新。

[Activation Flow](https://arxiv.org/abs/2609.32530) 已使用 Jacobian 反演构造激活向量，[Principal Steering Subspaces](https://arxiv.org/abs/2609.33765) 已使用离线响应构造控制基。响应反演和低维求解属于已有技术。本方案的具体研究对象是正常响应被补偿后留下的 FFN 非线性差异，以及这一差异能否支持 DiT 的概念擦除。

**1. 在哪里构造向量。**

取 DiT block 的图像 FFN。令 x 为经过 LayerNorm 和 AdaLN 调制后、进入 FFN 的一个图像 token，定义

\[
a=W_{\mathrm{in}}x+b_{\mathrm{in}},\qquad
F(a)=W_{\mathrm{out}}\phi(a)+b_{\mathrm{out}}.
\]

其中 a 属于 FFN 展开维度，phi 为该模型实际使用的激活函数。令 g 为模型原有的 FFN residual gate，则该支路写回 residual 的量为

\[
r(a,g)=g\odot F(a).
\]

构造两个离线向量：

- v：与 a 同维，在第一层线性投影之后、激活函数之前相减。
- c：与 F(a) 同维，在第二层线性投影之后、原生 residual gate 之前相加。

在线计算为

\[
\boxed{
r_{\mathrm{CNS}}(a,g)
=g\odot\left[W_{\mathrm{out}}\phi(a-v)+b_{\mathrm{out}}+c\right].
}
\]

两个向量都按层和噪声区间预先保存，在该区间内对所有图像 token 广播。向量的幅度包含在 v 中，c 与该幅度一起标定。

这两次加减在函数上可以等效写成 FFN 两个偏置的改动；该参数化本身属于常规操作。使用独立缓存和 hooks 可以在运行时切换向量。

当前安装的 Diffusers 中，FLUX double-stream 图像支路使用 `ff.net[0].proj → activation → ff.net[2]`，外部再乘 `gate_mlp`；SD3.5 的 JointTransformerBlock 也有对应的图像 FFN。FLUX single-stream 使用 `proj_mlp → act_mlp`，再将其与 attention 输出拼接后送入 `proj_out`。single-stream 中应只对图像 token 的 MLP 部分施加 v，使用 `proj_out` 的 MLP 列作为 W_out，并在总投影输出的图像部分加入 c，再乘原生 gate。当前版本 double-stream 使用近似 GELU，single-stream 使用 tanh 近似 GELU，离线必须调用各自真实函数。

对应实现可核对 [FLUX blocks](/software/anaconda3/envs/shx/lib/python3.13/site-packages/diffusers/models/transformers/transformer_flux.py:359)、[SD3 JointTransformerBlock](/software/anaconda3/envs/shx/lib/python3.13/site-packages/diffusers/models/attention.py:690) 和 [FFN activations](/software/anaconda3/envs/shx/lib/python3.13/site-packages/diffusers/models/activations.py:65)。

**2. 为什么这种固定向量能够产生选择性。**

定义相减 v 引起的、尚未经过 residual gate 的 FFN 输出变化

\[
R(a;v)=W_{\mathrm{out}}\left[\phi(a-v)-\phi(a)\right].
\]

最终扰动为

\[
\Delta r(a,g;v,c)=g\odot[R(a;v)+c].
\]

如果在正常样本上 R(a;v) 接近同一个向量，那么 c 可以消去它的共同部分。目标概念处于不同激活状态时，R 的额外部分仍然存在。因此，实际写回 residual 的变化随 a 而变，离线保存的 v 和 c 则固定。

在 g 恒定、v 较小时，有

\[
\Delta r(a,g)
\approx-g\odot W_{\mathrm{out}}
\left\{\left[\operatorname{diag}(\phi'(a))
-\mathbb E_{\mathcal R}\operatorname{diag}(\phi'(a))\right]v\right\}.
\]

这里的选择性来自目标与正常状态的激活函数响应差。每个 token 沿用原网络的非线性运算。

这个机制有明确的可检验条件：如果 phi 在所有相关状态和位移区间内都是同一个线性函数，R(a;v) 就与 a 无关，补偿会将干预完全抵消。目标擦除也随之消失。因此，“激活均值能区分概念”不足以支持 CNS；必须观察到补偿后的目标响应与正常响应可以分离。

**3. 如何离线统计 v。**

首先构造概念有无成对的提示词，尽量固定主体、场景、姿态和描述长度。风格任务可以比较“同一场景，有目标风格／中性风格”；实例身份任务可以比较“Snoopy／普通狗”。实例身份擦除与把整个对象从场景中删除是两个不同目标，应分别构造负提示词和评价，避免把身份替换记为对象消失。

正负对至少覆盖多类场景、近邻概念、同义表达和隐式描述。正常校准集合 R 另外包含泛化内容、语义近邻和视觉近邻；例如擦除 Snoopy 时，应覆盖普通狗、其他卡通角色、黑白线稿等正常生成需求。

**用同一中间 latent 比较正负条件。** 从原模型正、负提示词的真实采样轨迹中各取一半状态 z_t。固定 z_t、t、采样配置和其他输入，分别用 p+、p- 做一次原模型 forward，采集 a+、a-。只固定最初 seed、随后让两条轨迹分别演化，仍会将轨迹与布局差异混入统计；同状态比较直接控制了这一差异。

这些对比 forward 都在离线阶段完成。正常补偿统计主要使用正常提示词自身真实轨迹上的状态，确保它描述部署时正常生成的响应分布。

在每个模板组 k、层 l 和噪声桶 b 内，先计算提示词对和 token 的平均差

\[
d_{k,l,b}=\mathbb E_{i,n\mid k,l,b}[a^+_{i,n}-a^-_{i,n}].
\]

每个模板组等权，避免模板数量较多的组主导向量。推荐起步估计为

\[
d^0_j=\operatorname{median}_k(d_{k,j}),\qquad
\rho_j=\frac{|\sum_k d_{k,j}|}{\sum_k|d_{k,j}|+\epsilon},\qquad
v(\alpha)=\alpha\,\rho\odot d^0.
\]

median 为逐通道的组均值中位数，rho 为跨组符号一致性的连续权重。它们用于降低模板偶然相关的影响。这些是统计可靠性的实现选择，需要与普通均值差做消融。

该 v(alpha) 可用作最小原型和消融。完整方案进一步将稳健均值差及不同模板组的差异构成 4–8 列候选基 B，通过下一节的响应统计闭式求解组合系数。组差异只负责限定概念相关的候选空间，实际写入向量由响应而非均值距离决定。

保留 preactivation 差异的原始尺度，用闭式解确定组合，再搜索无量纲 alpha。preactivation 的绝对位置决定是否跨过非线性区间，统一单位范数会丢掉这部分尺度信息。初始幅度网格可取 `{0, 0.25, 0.5, 1, 1.5, 2}`；它是试验起点。

token 只用于增加每条轨迹的统计精度。置信区间和 bootstrap 按提示词模板、seed 或轨迹分组，避免将高度相关的 token 当成独立样本。

**4. 如何统计补偿向量 c。**

固定一个候选 v，在正常集合 R 上离线计算 R(a;v)。若忽略 gate 的变化，最小二乘补偿就是负均值：

\[
c=-\mathbb E_{\mathcal R}R(a;v).
\]

DiT 的原生 g 随提示词和时间改变，使用经过 gate 的实际 residual 扰动作为目标更合适：

\[
c^*(v)=\arg\min_c\mathbb E_{\mathcal R}
\left\|g\odot[R(a;v)+c]\right\|_2^2.
\]

逐通道闭式解为

\[
\boxed{
c_j^*(v)=-\frac{\mathbb E_{\mathcal R}[g_j^2 R_j(a;v)]}
{\mathbb E_{\mathcal R}[g_j^2]}.
}
\]

若分母为零，该通道在校准样本上不写回 residual，取 c_j=0；实际数值计算对小分母使用稳定处理。不同正常类别在期望中等权。

该解给出固定 v 下，当前校准分布中使局部 residual 均方变化最小的常数补偿。它消除共同扰动后留下的样本波动仍需测量，不能由均值抵消推导每张正常图像保持不变。

每个 alpha 都对应不同的非线性响应，必须重新统计 c。在线使用另一强度时，不能直接将一对 v、c 同比例缩放并假定补偿关系保持成立。

**在统计候选空间里求最终 v。** 令 v=Bw，B 的列来自上一节的成组对比统计。对每个离线样本计算

\[
J(a)=W_{\mathrm{out}}\operatorname{diag}(\phi'(a))B
\in\mathbb R^{d\times K}.
\]

它只涉及 K 个方向的 FFN 局部响应，可直接用真实激活函数的导数和现有 W_out 计算。统计正常响应的 gate 加权均值矩阵

\[
C_{j,:}=\frac{\mathbb E_{\mathcal R}[g_j^2J_{j,:}(a)]}
{\mathbb E_{\mathcal R}[g_j^2]},\qquad
S(a,g)=\operatorname{diag}(g)[J(a)-C].
\]

小位移下 c 约为 Cw，补偿后 residual 扰动约为 -Sw。对同状态正负提示词对，令 d_i=r_i^+-r_i^-，求解

\[
\min_w\quad
\mathbb E_+\|d_i-S_i^+w\|_2^2
+\lambda\mathbb E_{\mathcal R}\|S_rw\|_2^2
+\eta\|w\|_2^2.
\]

其闭式解为

\[
\boxed{
w^*=\left(
\mathbb E_+[(S^+)^\top S^+]
+\lambda\mathbb E_{\mathcal R}[S_r^\top S_r]
+\eta I
\right)^{-1}
\mathbb E_+[(S^+)^\top d_i],\qquad v_0=Bw^*.
}
\]

这是 K×K 的响应统计问题，eta>0 用于处理候选相关性。各模板组与正常类别等权积累统计。lambda 在验证集上依据擦除—保留曲线选择。该求解依赖一阶近似，实际部署用候选 v=alpha v_0 重新计算完整非线性响应 R、闭式补偿 c 和生成结果。最小原型 v(alpha)=alpha rho⊙d0 与响应校准版本的对比，应单独列为消融。

这个步骤保护的是“候选扰动经过 FFN 和原生 gate 后的实际作用”。它与将擦除向量投影到保留概念子空间的几何约束具有不同的优化对象。

**5. 用什么标准选择层、噪声桶和幅度。**

首先选择 double-stream 和 single-stream 各三个代表位置，覆盖各自深度的前、中、后段；按模型实际 noise level 或 sigma 分三个桶采集统计。先计算候选的局部响应，再对少量通过筛选的配置做完整生成。代表位置与三个桶用于控制首轮开销，不预设某一层或某一阶段最有效。

对每对同状态输入，记原模型的图像 FFN residual 为 r+、r-，目标条件下的干预为 Delta r+。计算目标对比剩余比例

\[
E_{\mathrm{target}}=
\frac{\mathbb E\|r^++\Delta r^+-r^-\|_2^2}
{\mathbb E\|r^+-r^-\|_2^2+\epsilon}.
\]

同时计算正常 residual 的相对变化

\[
E_{\mathrm{retain}}=
\frac{\mathbb E_{\mathcal R}\|\Delta r\|_2^2}
{\mathbb E_{\mathcal R}\|r\|_2^2+\epsilon}.
\]

分别报告各模板组和正常类别的结果；用较差组的结果筛选。保留目标对比确实缩小、正常扰动小的候选。分母接近零的局部对比不用于排序，避免数值放大。

局部指标用于筛选位置和验证机制。生成能力由最终图像检验：冻结全部向量和窗口，在未参与统计的提示词和 seeds 上比较擦除率、保留率以及生成质量。

联合使用多个位置后，上游干预会改变后续激活分布。选定一至两个初始窗口后，运行该联合配置并重新标定下游 c，再冻结验证；单层最优配置不能直接推导为联合配置最优。

**6. 在线如何使用。**

采样到离线选定的层和噪声桶时，读取相应的 v、c：

```python
a = W_in(x) + b_in
a = a - v[layer, noise_bin]
y = W_out(phi(a)) + b_out
y = y + c[layer, noise_bin]
residual = residual + native_gate * y
```

FLUX single-stream 按前述 MLP 列与图像 token 切片实现。SD3.5 使用 CFG 时，首轮方案在 conditional 分支施加干预，保留标准 CFG 组合公式；FLUX guidance-distilled 路径按其实际单分支执行，不能根据 batch 大小猜测 CFG。标准 CFG 会放大 conditional 输出的变化，校准与评估应固定并记录 guidance 配置。

直接开销为两次向量广播加减，量级 O(N(m+d))。额外存储为每个选定位置、噪声桶的 m+d 个标量。线上沿用一次正常模型 forward。层、时间窗口和向量在离线验证后固定。

**7. 第一轮实验如何决定是否继续。**

先选一个风格概念和一个实例身份概念，在 FLUX.1-dev 上验证；再用相同统计流程扩展到 SD3.5 Medium。建议首轮每个概念使用约 40–80 个构造模板、每模板两个 seed，并准备独立的正常校准集合。按模板组划分构造集、调参集和测试集；测试集包含未参与构造的隐式表达与近邻概念。样本数量是起始预算，结果按实际覆盖范围解释。

| 对照 | 回答的问题 |
|---|---|
| 原模型与同预算 CASteer、SHIFT、RASteer | 在相同擦除水平下，正常生成保持是否改善 |
| 同一 FFN 的普通输出 steering | 增益是否来自跨非线性的成对干预 |
| 只做 a-v，去掉 c | 正常响应补偿是否降低生成扰动 |
| 只加 c，去掉 a-v | 补偿向量本身是否造成了主要语义变化 |
| 完整 CNS | 两个向量联合是否形成预期的目标选择性 |
| 稳健均值差向量／响应统计闭式解 | 响应校准是否比直接使用统计均值方向更有效 |
| 用等能量随机 v，并为它重新统计 c | 作用是否依赖概念统计方向 |
| 独立演化轨迹统计／同 latent 统计 | 同状态配对是否提高向量可靠性 |

统计上同时保留每类正常内容的平均扰动和尾部扰动，防止总体均值掩盖近邻类别损伤。最终图像报告目标概念成功生成率、近邻保留率、原 prompt 中其余实体和属性的符合程度、正常提示词的同 seed LPIPS、图文一致性，以及达到足够样本量后的 FID。LPIPS 用于正常样本；目标样本本来就需要发生语义变化。

比较各方法的擦除—保留曲线，并在相同目标擦除率下比较正常内容。在目标提示词中保留其他主体、背景和属性是准确性的另一部分，应单独评价。

继续投入的依据是：独立验证上存在补偿后目标响应明显、正常响应较小的窗口，并且最终图像在相同擦除率下具有更好的内容保持。若只有 hidden contrast 下降，图像概念仍保留，说明该 FFN 局部对比不是有效控制量；若所有窗口都需要较大正常扰动才能擦除，说明当前数据和候选向量不支持该路线的选择性假设。

**8. 数学检查与证据状态。**

已检查补偿解的一阶条件，以及将 phi 替换为线性函数时干预完全抵消的性质。另在一维 GELU 合成样本上做了数值检查：正常 preactivation 取 N(-2, 0.12^2)，目标取 N(2, 0.12^2)，位移 v=4，原生 gate 在 [0.5, 1.5] 中随机取值。补偿使正常 residual 变化 RMS 从约 0.0494 降到 0.0107；目标与原正常特征的均值差从约 2.0001 变为约 -0.0462。该例展示所需机制确实能够存在，不代表真实 DiT 的概念擦除结果。真实模型需要验证的核心就是这种响应分离是否存在、是否能传递到最终生成。

建议将论文主张聚焦为：**利用正常生成的响应统计，构造跨 FFN 非线性的补偿式 steering，并依据补偿后的响应求解向量、选择干预位置，以改善 DiT 概念擦除的擦除—保留权衡。** 正负提示词、稳健均值和噪声分桶分别是构造与验证手段；小矩阵求解是响应校准的具体实现。
