# PairCut-FM：基于 Q/K 聚类的 attention score 编辑

PairCut-FM 是本研究设计的暂定名称：Pair 指 query 聚类编号与 key 聚类编号的组合；Cut 指降低这些组合对应的注意力分数；FM 指 flow matching。该设计尚待模型实验验证。

**核心算法：离线验证哪些 query–key 聚类组合可以被抑制，并保存二值表；在线根据当前 Q/K 的聚类编号查表，降低对应 attention logits，再执行原生 softmax、value 聚合与 flow 采样。**

## 1. 先说明唯一的在线编辑

正常 attention 的分数为 \(S_{ij}\)。如果图像 token \(i\) 的 query 被分到第 \(a_i\) 簇，token \(j\) 的 key 被分到第 \(b_j\) 簇，编辑为：

\[
\widetilde S_{ij}=S_{ij}-\beta_s B_s(a_i,b_j).
\tag{1}
\]

其中，\(B_s(a,b)\in\{0,1\}\) 是离线保存的小表，\(\beta_s>0\) 是有限的抑制强度，\(s\) 指定 block、head 和采样时间区间。表项为 1 时降低分数，为 0 时保留原分数。

因此，离线工作的目标很具体：**确定聚类中心、聚类匹配阈值、表 \(B_s\) 与强度 \(\beta_s\)。**

## 2. 符号与正常 attention

| 符号 | 定义 |
|---|---|
| \(p,e=E(p)\) | 普通用户 prompt 与原生文本编码结果 |
| \(\mathcal C=\{c_1,\ldots,c_M\}\) | 离线确定的待擦除概念集合 |
| \(\xi,x_t\) | 初始噪声与采样时刻 \(t\) 的图像 latent |
| \(v_\theta(x_t,t,e)\) | 原模型的速度预测；\(\theta\) 始终冻结 |
| \(\ell,h,\omega(t)\) | block 编号、head 编号、采样时间区间编号 |
| \(s=(\ell,h,\omega)\) | 一个进行标定或编辑的注意力位置 |
| \(\mathcal I,\mathcal T\) | 图像 token、文本 token 的索引集合 |
| \(q_i,k_j,V_j\) | 当前 head 的 query、key、value 向量 |
| \(d_h\) | 当前 head 的向量维度 |
| \(a_i,b_j\) | query 与 key 的聚类编号；0 表示不匹配任何已有簇 |
| \(B_s,\beta_s\) | query–key 聚类组合的二值表与抑制强度 |

\(i,j\) 是 token 索引，\(a,b\) 是聚类编号。\(h\) 表示 head，\(\omega\) 表示采样时间区间；它们均不表示图像空间坐标。

对一个 attention head，省略归一化、bias 与模态差异，Q/K/V 投影可写为：

\[
q_i=W_Qz_i,\qquad k_j=W_Kz_j,\qquad V_j=W_Vz_j.
\tag{2}
\]

\(z_i\) 是当前 token 在该层的输入表征。联合注意力中，同一个 token 通常同时具有自己的 Q、K、V。

令 \(\bar q_i,\bar k_j\) 为模型按原生规则完成 QK 归一化及位置编码后的向量，则正常计算为：

\[
S_{ij}=\frac{\bar q_i^\top\bar k_j}{\sqrt{d_h}},\qquad
A_{ij}=\operatorname{softmax}_j(S_{ij}),\qquad
O_i=\sum_j A_{ij}V_j.
\tag{3}
\]

求和只包含原生 attention mask 允许的位置。\(A_{ij}\) 决定位置 \(j\) 的 value 对位置 \(i\) 的 attention 输出贡献多少。

本方法只修改 \(i\in\mathcal I\) 的分数行，\(j\) 可以属于文本或图像 token。当前位置的 Q/K/V、位置编码与模型参数保持原值；后续层和后续时间的表征会随修改后的 attention 输出变化。

## 3. 离线步骤一：准备样本并建立 Q/K 聚类

对每个概念 \(c\)，准备：

- \(\mathcal D_c^+\)：原模型确实生成目标概念的 prompt—噪声对。
- \(\mathcal D_c^-\)：与目标共享部分属性、但生成结果不含目标的相邻正常样本。
- \(\mathcal D_{\mathrm{keep}}\)：需要保留的相邻与普通样本集合。

例如，擦除 Pikachu 时，阳性组包含名称、别名与属性描述；相邻正常组包含其他卡通角色、鼠类与黄色玩具。样本按实际生成图像标注，并按模板划分发现、标定与独立测试集合。

在选定的 \(s=(\ell,h,\omega)\) 记录 Q/K 副本。聚类使用 QK 归一化之后、RoPE 之前的向量，并将副本转换为单位向量。分别建立三组中心：

\[
\mathcal M_s^Q=\operatorname{SphericalKMeans}(\mathcal Q_s,K_Q),\quad
\mathcal M_s^{K,T}=\operatorname{SphericalKMeans}(\mathcal K_s^T,K_T),\quad
\mathcal M_s^{K,I}=\operatorname{SphericalKMeans}(\mathcal K_s^I,K_I).
\tag{4}
\]

\(\mathcal Q_s\) 是缓存的图像 query 集合，\(\mathcal K_s^T\)、\(\mathcal K_s^I\) 是文本与图像 key 集合。首轮各用 16 个中心；多概念在平衡汇总的数据上建立共享中心。

这些中心划分模型内部向量，**不直接对应“Pikachu 类”“正常类”等概念类别**。V 沿用模型正常计算，无需聚类。

以 query 为例，设 \(\widehat q_i\) 为单位向量，中心 \(\mu^Q_{s,a}\in\mathcal M_s^Q\)。聚类编号为：

\[
a_i^*=\arg\max_a\widehat q_i^\top\mu^Q_{s,a},\qquad
a_i=
\begin{cases}
a_i^*,&\widehat q_i^\top\mu^Q_{s,a_i^*}\geq\tau^Q_{s,a_i^*},\\
0,&\text{其余情况}.
\end{cases}
\tag{5}
\]

\(\tau\) 从标定样本与中心的相似度分位数确定。key 使用相同规则，按 token 模态选择相应中心。

统一编号时，文本 key 使用 \(1,\ldots,K_T\)，图像 key 使用 \(K_T+1,\ldots,K_T+K_I\)，不匹配已有簇的向量编号为 0。因此：

\[
B_s\in\{0,1\}^{(K_Q+1)\times(K_T+K_I+1)},\qquad
B_s(0,b)=B_s(a,0)=0.
\tag{6}
\]

## 4. 离线步骤二：预筛 query–key 聚类组合

设 \(z=(p,\xi)\) 表示一个 prompt—噪声对。在原模型生成轨迹上，定义编号组合 \((a,b)\) 对应的平均注意力质量：

\[
m_{s,a,b}(z)=
\underset{t\in\omega,\ i\in\mathcal I}{\operatorname{Avg}}
\left[\sum_j A^{\ell,h,t}_{ij}
\mathbf 1[a_i=a]\mathbf 1[b_j=b]\right].
\tag{7}
\]

然后计算目标组与相邻正常组的差异：

\[
F_c(s,a,b)=
\operatorname{Avg}_{z\in\mathcal D_c^+}m_{s,a,b}(z)
-\operatorname{Avg}_{z\in\mathcal D_c^-}m_{s,a,b}(z).
\tag{8}
\]

按 \(F_c\) 保留少量候选 \(r=(s,a,b)\)，减少后续生成验证次数。式 (7) 定义完整统计量；预筛可用缓存 token 与时间点的抽样近似。\(F_c\) 用于候选排序，规则是否保留由下一步决定。

## 5. 离线步骤三：实际修改 attention，重新生成并选规则

一个候选 \(r=(s,a,b)\) 的操作是：在位置 \(s\)，对所有聚类编号满足 \(a_i=a,b_j=b\) 的连接，应用式 (1)。

记 \(\mathcal R\) 为当前规则集合，\(G_{\mathcal R}(z)\) 为安装规则后完整采样并解码的图像。定义：

- \(f_c(x)\in[0,1]\)：目标概念的图像检测结果或置信度，仅供离线筛选与评估。
- \(U(x,p)\in[0,1]\)：正常内容的保留评分，例如预先标注的对象和属性完成率；图像质量另行记录。

在相同 prompt、噪声与采样配置下，候选规则的增量擦除收益为：

\[
\Delta_c(r\mid\mathcal R)=
\operatorname{Avg}_{z\in\mathcal D_c^+}
\left[f_c(G_{\mathcal R}(z))-
f_c(G_{\mathcal R\cup\{r\}}(z))\right].
\tag{9}
\]

规则集合相对于原模型的累计保留损失为：

\[
L_{\mathrm{keep}}(\mathcal R)=
\operatorname{Avg}_{z=(p,\xi)\in\mathcal D_{\mathrm{keep}}}
\left[U(G_\varnothing(z),p)-U(G_{\mathcal R}(z),p)\right]_+.
\tag{10}
\]

\([x]_+=\max(x,0)\)，\(G_\varnothing\) 表示原模型。给定一个初始有限惩罚强度，以有限候选贪心搜索选择规则：

\[
r^*=\underset{r\in\mathrm{Candidates}_c\setminus\mathcal R}
{\arg\max}\ \Delta_c(r\mid\mathcal R),\qquad
\text{subject to }L_{\mathrm{keep}}(\mathcal R\cup\{r\})\leq\varepsilon.
\tag{11}
\]

\(\varepsilon\) 是保留损失容忍度。当最大收益不超过阈值 \(\eta>0\)、没有可行候选或预算用完时停止。\(\eta\) 用于排除标定样本上的微小变化。

通过这一步选出规则后，编译为表：

\[
B_s^c(a,b)=\mathbf 1[(s,a,b)\in\mathcal R_c].
\tag{12}
\]

例如，**假设**标定发现某个位置上，将 query 第 3 簇与 key 第 7 簇之间的注意力分数降低，可以减少 Pikachu，同时保留正常对象，则保存 \(B_s^c(3,7)=1\)。聚类编号本身没有预设的目标含义，是生成干预赋予这个编号组合擦除用途。

为验证按编号组合编辑的价值，比较相同码本上的 key 簇 value 屏蔽、query 簇 head 输出屏蔽，以及相近注意力质量的随机连接抑制。在相近擦除率下评估相邻概念保留。

## 6. 离线步骤四：合并概念并联合标定

单概念直接使用对应表；多概念在共享聚类中心上取布尔并集：

\[
B_s(a,b)=\max_{c\in\mathcal C}B_s^c(a,b).
\tag{13}
\]

同一编号组合命中多个概念时，在线仍只惩罚一次。随后在完整生成轨迹上，使用小型有限候选网格标定 \(\beta_s\)，删除没有增量收益或导致累计误伤的表项，并在多目标与正常对象共存的样本上联合验证。

最终保存：

\[
\mathcal P=
\{s,\mathcal M_s^Q,\mathcal M_s^{K,T},\mathcal M_s^{K,I},
\tau_s,B_s,\beta_s\}_{s\in\mathcal S}.
\tag{14}
\]

\(\mathcal S\) 是最终选择的注意力位置集合。部署包 \(\mathcal P\) 包含判断与编辑所需的全部离线信息。

## 7. 在线步骤：普通 prompt 驱动的正常采样

每次生成只输入 \(p\)、seed 与原生采样参数。在选定位置按式 (5) 得到当前 Q/K 的聚类编号，查询 \(B_s\)，再执行：

\[
\widetilde A_{ij}=
\operatorname{softmax}_j\left(S_{ij}-\beta_sB_s(a_i,b_j)\right),\qquad
\widetilde O_i=\sum_j\widetilde A_{ij}V_j.
\tag{15}
\]

之后继续该次 DiT 前向，得到修改后的速度预测 \(\widetilde v_\theta\)。以 \(t=1\) 为噪声、\(t=0\) 为图像，Euler 更新示意为：

\[
x_{t_{n+1}}=x_{t_n}+
(t_{n+1}-t_n)\widetilde v_\theta(x_{t_n},t_n,E(p)).
\tag{16}
\]

实际实现使用模型原生的时间约定、guidance 与积分器，保持原有模型调用安排。

目标相关信息已经写入聚类中心、阈值和表项。运行时按当前 Q/K 的编号组合触发编辑，无需额外输入 target、anchor，也不运行离线检测器 \(f_c\)。隐式 prompt 能否触发，取决于它的内部模式是否被已标定的编号组合覆盖。

## 8. 编辑公式的两个直接性质

令 \(\rho_i\) 为当前行命中表项的原始注意力质量，则：

\[
\rho_i=\sum_j A_{ij}B_s(a_i,b_j),\qquad
\widetilde\rho_i=
\frac{e^{-\beta_s}\rho_i}{1-\rho_i+e^{-\beta_s}\rho_i}.
\tag{17}
\]

在 \(0<\rho_i<1\) 时，被选连接的总注意力质量下降，其他连接的权重增加；\(\rho_i=0\) 时保持原样；\(\rho_i=1\) 时所有 logits 等量平移，softmax 同样保持原样。离线筛选需要排除仅产生整行平移的无效规则。

计算上，原生 DiT 调用次数保持不变。新增工作是少量 head 上的聚类匹配与 attention score 查表。查询应融合到 attention tile 内，实际延迟和显存与原生优化实现对比测量。

## 9. English pseudocode

~~~text
Algorithm 1: Offline construction

Input:
    Frozen model G, target concepts C
    Offline image evaluators f_c and U
    Discovery data D_discovery, calibration data D_calibration
    Candidate sites S_candidate, initial penalty beta_initial
    Retention tolerance epsilon, minimum gain eta, search budget
Output:
    Deployment package P

cache_discovery = CollectQKAndAttention(G, D_discovery, S_candidate)
cache_calibration = CollectQKAndAttention(G, D_calibration, S_candidate)
centers = FitSharedQKClusters(cache_discovery)  # Image Q, text K, and image K
thresholds = CalibrateClusterThresholds(cache_calibration, centers)
candidates = RankClusterPairs(cache_discovery, centers, thresholds, C)

for each concept c in C:
    R_c = empty set
    while search budget remains:
        gains, retention_losses = EvaluateCandidateEdits(
            G, c, R_c, candidates[c], centers, thresholds,
            beta_initial, D_calibration
        )  # Equations 9–10; use matched prompts and seeds
        feasible = evaluated candidates satisfying retention_loss <= epsilon
        if feasible is empty:
            break
        r = feasible candidate with the largest gain
        if gain[r] <= eta:
            break
        add r to R_c
        remove r from candidates[c]
    B_c = CompileBinaryTables(R_c)

B = LogicalORAcrossConcepts({B_c})
B, beta = JointlyCalibrateAndValidate(G, B, centers, thresholds, D_calibration)
return P = {active sites, centers, thresholds, B, beta}
~~~

CollectQKAndAttention 使用正常生成轨迹；EvaluateCandidateEdits 实际应用式 (1) 并完成生成；JointlyCalibrateAndValidate 在有限强度网格上评估完整组合、删除有害或无效表项。发现与标定按模板分开，最终测试集独立保留。

~~~text
Algorithm 2: Image generation

Input: Ordinary prompt p, seed, deployment package P, native sampling settings
Output: Generated image

e = NativeTextEncode(p)
x = SampleInitialNoise(seed)
for each native sampling time t:
    velocity = DiT(x, t, e, attention_hook=BindAttentionHook(P, t))
    x = NativeSamplerStep(x, velocity, t)
return NativeDecode(x)

Procedure EditAttention(Q_pre, K_pre, V, metadata, P, t):
    s = (metadata.block, metadata.head, TimeBin(t))
    Q_rot, K_rot = NativePositionEncoding(Q_pre, K_pre, metadata)
    if s is not active:
        return NativeAttention(Q_rot, K_rot, V, metadata.mask)
    a = AssignQueryClusters(Q_pre[image_tokens], P.centers, P.thresholds)
    b = AssignKeyClusters(K_pre, metadata.modalities, P.centers, P.thresholds)
    compute scores from Q_rot and K_rot in tiles, preserving metadata.mask
    for each score tile:
        for each allowed pair (i, j) with i an image token:
            score[i, j] = score[i, j] - P.beta[s] * P.B[s][a[i], b[j]]
        aggregate values with the native streaming softmax
    return the attention output
~~~

BindAttentionHook 将部署包与当前时间绑定到 EditAttention；Q_pre、K_pre 是 QK 归一化后、位置编码前的向量，metadata 包含 block/head 编号、token 模态、图像 token 索引与原生 mask。这里的 attention hook 是嵌入原生 attention kernel 的操作示意。实现保留原有 attention mask，并采用原生流式 softmax 的跨 tile 归一化；各 tile 不独立做 softmax。若原生采样含多个 guidance 分支，保持其调用结构，在相应前向内应用编辑。
