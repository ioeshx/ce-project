# queryGraft：只读概念定位与 anchor 激活替换

本文记录 queryGraft 的方法定义与首版实验配置；实现入口与命令见 [运行说明](README.md)。下文核心算法定义一个 target–anchor 对，另加一个 background 概念；现有多 target benchmark 的逐 pair 扩展见运行说明。模型参数冻结，干预发生在推理过程中。

核心流程是：在 user 前向中加入只读概念 token，利用模型自身的 attention 输出定位 target；跨层汇总得到当前采样步的空间掩码；达到面积阈值时，以同一当前 latent 下的 anchor 条件前向激活，替换目标区域内的 image attention 输出。

## 1. 当前默认配置

| 设计项 | 当前默认值 | 后续保留的选择 |
| --- | --- | --- |
| 定位概念 | target、anchor、background，共 3 个 | 本文固定单对概念 |
| 概念初始化 | 每个完整概念短语独立编码，取输出的第一个 token | 沿用当前约定 |
| 概念读取范围 | 原始 user text、image，以及概念 token | 单向读取，原始 token 不能读取概念 token |
| 定位特征 | 同层原始 image AV 与沿通道 L2 归一化后的 concept AV 点积 | 归一化只作用于分数读出 |
| 定位层集 | 全部具有概念读出路径的主 attention 层 | 与关键层子集比较 |
| 层间汇总 | 逐层二值掩码的支持比例严格大于 \(\eta=0\) | 对 \(\eta\) 做消融；\(0.5\) 为严格多数 |
| 面积阈值 | \(\rho=0\)，掩码非空即触发 | 对 \(\rho\) 做消融 |
| 替换表示 | 原始 AV，位于输出投影之前 | 与 attention 残差更新后的表示比较 |
| 替换层集 | 模型最后一层的主 attention | 与全部层、关键层子集比较 |
| 更新算子 | 硬替换 | 与比例融合、缩放相加比较 |
| 干预采样步 | 每个采样步都重新定位并判定触发 | 采样配置独立记录 |
| anchor 前向条件 | anchor 短语本身作为完整 prompt | 直接使用配置中的 anchor 原文 |

这里的 AV 是 softmax attention 权重乘 value 后的输出，不是 value 投影 \(V\)、attention 权重矩阵，也不是整个 Transformer block 的输出。本文以 `nudity`、`clothed`、`background` 为概念文本示例；实际字符串随实验配置保存。

## 2. 符号与张量定义

下文先定义单样本，批量运行时增加最前面的 batch 维，掩码、面积和触发条件均按样本分别计算。

| 符号 | 含义 |
| --- | --- |
| \(\theta\) | 冻结的生成模型参数 |
| \(F_\theta\) | 模型原生条件预测网络；内部可通过指定位置捕获 AV |
| \(T\) | 采样步数 |
| \(t=T,\ldots,1\) | 按执行顺序递减的采样步索引；不是模型原生时间值 |
| \(\tau_t\) | scheduler 在第 \(t\) 步传给模型的原生 timestep 或等价噪声时间 |
| \(x_t\) | 当前采样步的 image latent，已包含之前采样步的干预结果 |
| \(c_u,c_{\rm tar},c_{\rm anc},c_{\rm bg}\) | user prompt、target、anchor、background 文本 |
| \(P_u\) | 完整 user prompt 的原生序列条件，映射到 Transformer 输入空间后得到的 text token |
| \(X_t^0\) | 当前 \(x_t\) 经原生图像输入映射得到的 image token |
| \(C^0\) | 三个概念 token 的初始表示，形状为 \(3\times D_{\rm model}\) |
| \(N_p,N_x\) | user text token 数、image token 数；\(N_x=H W\) |
| \(H,W\) | image token 网格的高、宽，不是最终图像的像素高、宽 |
| \(N=N_p+N_x,\;N_c=3\) | 原始 token 总数、附加概念 token 数 |
| \(\ell=1,\ldots,L\) | 按模型执行顺序编号的 Transformer 层；每层指定一个主 attention 读出位置 |
| \(h_\ell,d_\ell,D_\ell=h_\ell d_\ell\) | 第 \(\ell\) 层的 attention 头数、每头通道数、多头拼接后的 AV 通道数 |
| \(\kappa_{u,t},\kappa_{a,t}\) | user、anchor 前向各自的原生条件，包括原生文本条件、pooled 条件、时间和 guidance 设置 |
| \(O^x_{t,\ell},O^c_{t,\ell}\) | user 前向中的 image、concept 原始 AV，形状为 \(N_x\times D_\ell\)、\(3\times D_\ell\) |
| \(O^a_{t,\ell}\) | anchor 前向在对应层的 image 原始 AV，形状为 \(N_x\times D_\ell\) |
| \(\mathcal L_q\) | 用于计算定位掩码的层集，默认 \(\{1,\ldots,L\}\) |
| \(\mathcal L_e\) | 执行激活替换的层集，默认 \(\{L\}\) |
| \(m_{t,\ell}(p),M_t(p)\) | 第 \(\ell\) 层的 target 二值掩码、跨层汇总后的 target 二值掩码 |
| \(r_t,g_t\) | target 掩码面积比例、当前步是否触发替换的二值变量 |
| \(\eta\) | 跨层支持比例阈值，默认 \(0\) |
| \(\rho\) | 面积触发阈值，默认 \(0\) |
| \(\lambda\) | 融合或相加时的强度系数；硬替换不使用该参数 |
| \(\widetilde f_{u,t},\widetilde y_t\) | 干预后的 user 条件预测、交给 scheduler 的原生 guidance 组合后预测 |
| \(w,f^-_t\) | 原生 CFG 的 guidance scale、同一步负条件预测；仅在模型启用原生 CFG 时使用 |

本文的“层”按模型主干的 Transformer block 计数。SD3.5 若存在额外 image-only `attn2`，默认不把它作为第二个定位投票来源。FLUX 的双流层、单流层按实际执行顺序统一编号，保留完整模块路径与原生列表内编号。

## 3. 独立编码 user 与三个概念

user prompt 沿用模型的完整原生文本编码流程，得到 \(P_u\) 和原生 pooled 等条件。三个概念分别编码：

\[
e_k=E_{\rm seq}(c_k)_{0,:},
\qquad k\in\mathcal K
=\{\mathrm{tar},\mathrm{anc},\mathrm{bg}\},
\tag{1}
\]

\[
C^0=
\begin{bmatrix}
\Pi_c(e_{\rm tar})\\
\Pi_c(e_{\rm anc})\\
\Pi_c(e_{\rm bg})
\end{bmatrix}
\in\mathbb R^{3\times D_{\rm model}}.
\tag{2}
\]

\(E_{\rm seq}\) 表示用于概念初始化的序列文本编码器；按照 ConceptAttention 的 FLUX 实现使用 T5，并由 \(\Pi_c\) 复用或适配模型的原生 text 输入投影。迁移到 SD3.5 时，同样将概念序列表示映射到 joint attention 的 text 接口。

“取第一个 token”指：**先编码整个多词短语，再取输出序列索引 0 的向量**，不是只输入短语的第一个词。三个概念独立编码，避免初始化时因拼接成一句话而互相改变文本表示。

概念表示在 DiT 输入处加入；user 的原始序列编码、pooled 条件、image/text 位置编号保持原生定义。概念 token 使用对应模型的 text 位置编码约定；采用 ConceptAttention 的 FLUX 约定时其 position IDs 为零。概念不加入 user pooled 条件，概念路径沿用当前 user 前向的原生时间、guidance 与调制条件。

\(C^0\) 可预先编码并复用；**每个采样步从 \(C^0\) 开始重新演化概念状态**，不把上一步的末层概念状态直接传到下一采样步。

## 4. 只读概念 attention

### 4.1 单向信息流的块掩码

为定义方便，将原始 token 记为 \(B^0_t=[P_u;X_t^0]\)，逻辑 token 顺序写为：

\[
Z_t^0=[B_t^0;C^0],
\qquad |B_t^0|=N,\quad |C^0|=N_c=3.
\tag{3}
\]

对每一层、每一个 attention head，在 softmax 前加入下列偏置：

\[
\mathcal B=
\begin{bmatrix}
0_{N\times N} & -\infty_{N\times N_c}\\
0_{N_c\times N} & 0_{N_c\times N_c}
\end{bmatrix}.
\tag{4}
\]

行是 query，列是 key。式 (4) 表示：

- 原始 user text、image query 只能读取原始 text、image 的 key/value。
- concept query 可以读取 user text、image 和全部 concept 的 key/value。
- target、anchor、background 在定位路径内可以互相读取；这个路径只负责定位。

如果原生 attention 已有 padding 或其他限制，保留原始 \(B\!\to\!B\) 掩码，并对 concept 读取的无效 padding 列施加相同限制。式 (4) 展示的是所有原始 token 均有效时的概念隔离偏置。

真实模型可以使用不同 token 排列；实现时根据 text、image、concept 的实际索引映射设置掩码与切片。只读限制应用于**所有继续携带 concept token 的层**，包括接入概念路径后的 FLUX 单流层。

### 4.2 原始路径与概念路径的等价分解

省略 head 维与 \((t,\ell)\) 下标，在应用原生 Q/K 归一化、位置编码后，式 (4) 等价于：

\[
O_B=\operatorname{Attn}(Q_B,K_B,V_B),
\tag{5}
\]

\[
O_C=
\operatorname{Attn}
\left(Q_C,[K_B;K_C],[V_B;V_C]\right),
\tag{6}
\]

\[
\operatorname{Attn}(Q,K,V)
=\operatorname{ConcatHeads}
\left[
\operatorname{softmax}_{\rm key}
\left(\frac{Q_jK_j^\top}{\sqrt{d_\ell}}+\mathcal B_j\right)V_j
\right]_{j=1}^{h_\ell}.
\tag{7}
\]

式 (7) 中的 \(\mathcal B_j\) 是该次 attention 所需的有效位置掩码；只读分解后的两次调用分别使用对应的行列范围。图像与文本使用各自的原生 Q/K/V 投影；在有独立 text 参数的层，概念复用 text 侧投影、归一化和更新路径。

从 \(O_B\) 取出 image 行得到 \(O^x_{t,\ell}\)，从 \(O_C\) 取出三个 concept 行得到 \(O^c_{t,\ell}\)。这里取的是输出投影前的原始 AV。

当前层完成读出后，原始 token 和概念状态各自通过对应的原生投影、门控、残差与 MLP 更新，再进入下一层。可统一记为：

\[
B_t^\ell=\mathcal U^B_\ell
\left(B_t^{\ell-1},O_B;\kappa_{u,t}\right),
\qquad
C_t^\ell=\mathcal U^C_\ell
\left(C_t^{\ell-1},O_C;\kappa_{u,t}\right).
\tag{8}
\]

\(\mathcal U\) 保留该层实际存在的其他原生运算；FLUX 单流层中也保留并行 MLP 与 attention 输出的融合。概念只通过自己的 query 路径读取，不写回 \(B\) 的更新。

因此，在执行显式激活替换前，原始 token 的计算在数学上与原生 user 前向一致。实现上可以使用式 (5)–(6) 分别计算，无需物化完整的 \((N+3)\times(N+3)\) attention 矩阵；数值对照允许不同 kernel 的浮点误差。

## 5. 同层输出空间的概念定位

对每个定位层 \(\ell\in\mathcal L_q\) 和概念 \(k\in\mathcal K\)，先沿多头拼接后的通道维归一化 concept AV：

\[
\overline O^c_{t,\ell}(k,:)
=\frac{O^c_{t,\ell}(k,:)}
{\|O^c_{t,\ell}(k,:)\|_2+\epsilon},
\qquad \epsilon=10^{-8}.
\]

随后对每个 image 位置 \(p\in\{1,\ldots,HW\}\) 计算：

\[
s_{t,\ell,k}(p)
=\left\langle
O^x_{t,\ell}(p,:),
\overline O^c_{t,\ell}(k,:)
\right\rangle
=\sum_{d=1}^{D_\ell}
O^x_{t,\ell}(p,d)\overline O^c_{t,\ell}(k,d).
\tag{9}
\]

\[
S_{t,\ell}\in\mathbb R^{H\times W\times3}.
\tag{10}
\]

归一化与点积均用 FP32 计算，消除各概念输出范数对类别比较的影响。image AV 不做 L2 归一化，因此该分数不是完整的余弦相似度。归一化只用于分数读出，原始 concept AV 继续参与概念路径的原生状态更新，anchor 替换激活也保持原始 AV。每次比较必须使用**同一层、同一 attention、同一 user 前向**的 image 和 concept 输出；不同层之间不平均特征向量。

如需可视化相对分数，可以在 concept 维计算：

\[
\pi_{t,\ell,k}(p)
=\frac{\exp s_{t,\ell,k}(p)}
{\sum_{j\in\mathcal K}\exp s_{t,\ell,j}(p)}.
\tag{11}
\]

该 softmax 不改变单个位置的类别排序，因此下面的二值判定直接使用 \(s\)。\(\pi\) 是三个配置概念之间的相对分数，不作为校准后的 target 存在概率。

### 5.1 每层 target 掩码

只有 target 严格胜过 anchor 与 background 时，当前位置才标记为 1：

\[
m_{t,\ell}(p)
=\mathbf1\!\left[
s_{t,\ell,\mathrm{tar}}(p)
>\max\left\{
s_{t,\ell,\mathrm{anc}}(p),
s_{t,\ell,\mathrm{bg}}(p)
\right\}\right].
\tag{12}
\]

平局取 0。background 提供第三个竞争类别，使位置可以归到 target、anchor 之外；它仍是一个文本概念，不自动等价于“所有其他语义”的完美拒识类别。

### 5.2 跨层投票比例阈值

令 \(L_q=|\mathcal L_q|\)，先求逐层二值掩码的平均：

\[
v_t(p)=\frac{1}{L_q}
\sum_{\ell\in\mathcal L_q}m_{t,\ell}(p),
\tag{13}
\]

\[
M_t(p)=\mathbf1[v_t(p)>\eta],
\qquad \eta\in[0,1],\quad \eta_{\rm default}=0.
\tag{14}
\]

\(\eta\) 是可消融的超参数。默认 \(\eta=0\) 时，只要任意一个所选层支持 target，\(M_t(p)=1\)，等价于逐层掩码的并集。\(\eta=0.5\) 恢复严格多数投票，恰好一半层支持时取 0；\(\eta=1\) 时掩码恒为 0。每个 block 投一票，默认定位层集是所有主 attention 层。

本方法当前采用的是 **“各层判定后再投票”**，不采用“先平均各层原始分数再 argmax”或“先平均各层 softmax 分数再阈值”。所有汇总均在当前 \(t\) 内完成，不跨采样步平均。

## 6. 按 target 面积触发 anchor 前向

\[
r_t=\frac{1}{HW}\sum_{p=1}^{HW}M_t(p),
\qquad
g_t=\mathbf1[r_t>\rho],
\qquad \rho_{\rm default}=0.
\tag{15}
\]

\(r_t\) 只统计 target 掩码面积。默认 \(\rho=0\) 时，掩码非空就触发，空掩码仍跳过 anchor 前向。设置 \(\rho=0.20\) 时恢复旧的面积门控：\(r_t=0.20\) 不触发，\(r_t>0.20\) 才触发。未触发时，继续完成 user 原生输出。最终阈值通过消融确定。

定义实际应用的掩码：

\[
\widehat M_t(p)=g_tM_t(p).
\tag{16}
\]

面积表示 token 区域比例；这里判定的是 target 的相对定位结果是否足够大，而不是“任意一个概念是否占据足够面积”。

## 7. 从当前 latent 计算 anchor source activations

当 \(g_t=1\) 时，以 **anchor 短语本身**作为原生 prompt，在同一个 \(x_t\) 和 \(\tau_t\) 上计算一次 anchor 条件模型前向：

\[
\left\{O^a_{t,\ell}\right\}_{\ell\in\mathcal L_e}
=\operatorname{CaptureAV}_{\mathcal L_e}
\left[F_\theta(x_t,\tau_t;\kappa_{a,t})\right].
\tag{17}
\]

例如 \(c_{\rm anc}=\texttt{clothed}\) 时，anchor 分支输入就是 `clothed`，按模型原生流程编码完整 anchor prompt 及其 pooled 等条件。**定位时的第一个 token 规则仅用于式 (1)，不截断此处完整 anchor prompt 的原生条件。**

该分支使用模型原生 text–image joint attention。文本条件中不加入 user prompt、target 或 background，也不附加定位概念 token；图像侧仍可读取当前 \(x_t\) 的信息。因此，anchor 条件与 target 文本分离，并不意味着当前 image latent 已经没有 target 信息。

提取的 source 必须与 user 被替换激活具有相同的层、attention 分支、空间 token 顺序和表示位置。该分支只提供中间激活；不推进 scheduler，不用其最终预测单独更新 latent。source 已提取且不需要最终预测时，可以提前结束其余计算。

## 8. 默认在最终层执行 AV 硬替换

默认 \(\mathcal L_e=\{L\}\)。将式 (16) 的空间掩码沿通道维广播：

\[
\widetilde O^x_{t,L}(p,:)
=\begin{cases}
O^x_{t,L}(p,:), & g_t=0,\\
\left(1-M_t(p)\right)O^x_{t,L}(p,:)
+M_t(p)O^a_{t,L}(p,:), & g_t=1.
\end{cases}
\tag{18}
\]

未触发的分支直接返回原 AV，不需要计算或引用 \(O^a_{t,L}\)。

然后使用 \(\widetilde O^x_{t,L}\) 继续 **user 分支**的输出投影、门控、残差、MLP 与最终预测头。原始 user 残差和已计算的其他特征保留在 user 路径中；替换对象只是图像行的 AV。

先得到干预后的 user 条件预测：

\[
\widetilde f_{u,t}
=\operatorname{UserSuffix}_{t,L}
\left(\widetilde O^x_{t,L};\mathcal R^u_{t,L}\right),
\tag{19}
\]

再按模型原生 guidance 方式得到送入 scheduler 的预测。对常规 CFG：

\[
\widetilde y_t
=f^-_t+w\left(\widetilde f_{u,t}-f^-_t\right).
\]

其中 \(f^-_t\) 为相同 \(x_t,\tau_t\) 下的原生负条件预测。没有外部 CFG 时，\(\widetilde y_t=\widetilde f_{u,t}\)；模型内嵌 guidance 已包含在原生网络条件中。最后执行：

\[
x_{t-1}=\operatorname{SchedulerStep}
\left(x_t,\tau_t,\widetilde y_t\right).
\tag{20}
\]

\(\mathcal R^u_{t,L}\) 表示继续 user 计算所需的原生状态，例如残差、调制参数、text 输出，以及 FLUX 单流层并行的 MLP 特征。\(\widetilde y_t\) 按模型定义可以是速度或噪声预测；式 (20) 调用原生 scheduler，不另设采样更新公式。

### 8.1 最终层暂停与前向次数

默认执行顺序为：

1. user＋概念路径前向，逐层计算式 (9)–(12)，累计投票。
2. 在最终层同时得到 image 与 concept AV 后，**在 user image AV 输出投影前暂停**。
3. 利用全部定位层结果计算 \(M_t,r_t,g_t\)。
4. 若触发，从原始当前 \(x_t\) 计算 anchor source；随后应用式 (18)。
5. 恢复待完成的 user suffix，并调用一次 scheduler。

因此，默认每步需要一遍 user 主干计算；触发时增加一遍 anchor 主干计算。概念 query 带来额外 attention 计算，但不需要独立的完整概念模型前向。这里的“两次前向”依赖于暂停、缓存或等价的分段执行；不能在 user 输出已经完成后，仅改一个缓存的 AV 就视为已完成干预。

若干预改为早期层或多层，同时继续使用所有层的最终投票掩码，则必须先完成定位，再从最早受影响的 user 计算位置重放相应网络；详见第 10.3 节。

### 8.2 模型原生接口

- **SD3.5** 使用带有独立 image/text 投影和更新路径的 joint attention；默认替换主 attention 的图像 AV，在其输出投影之前操作。
- **FLUX 双流层** 在 image attention 输出投影之前操作；概念沿用 text 侧路径。
- **FLUX 单流层** 的 AV 与并行 MLP 特征拼接后进入融合投影；只替换图像行中的 AV 通道，随后沿用 user 的 MLP 特征、融合投影、门控和残差。

为明确后续实现，本稿约定：SD3.5 原生 CFG 存在时，仅在正条件 user 路径定位与替换，负条件预测沿用原生结果，再按原生 CFG 组合。anchor source 提取其 anchor 正条件路径的 AV；这些原始 AV 不做 CFG 线性组合。该分支选择是模型接入约定，与六组消融设计分别记录。FLUX 默认沿用模型的内嵌 guidance；具体 guidance 数值与原生采样配置共同记录。

## 9. 单采样步算法

```text
输入：当前 x_t、原生时间 tau_t、完整 user 条件、概念初始表示 C^0
参数：定位层集 L_q、编辑层集 L_e={L}、eta=0、rho=0

重置当前步概念状态 C <- C^0
votes[p] <- 0

执行 user 前向，并在所有携带概念 token 的层执行单向概念读取：
    原始 text/image attention 只能读取原始 text/image
    concept attention 可以读取原始 text/image 和 concept

    对每个定位层 ell in L_q：
        取同层 image raw AV O_x 与 concept raw AV O_c
        O_c_normalized[k,:] <- float32(O_c[k,:]) / (norm(float32(O_c[k,:])) + 1e-8)
        s[p,k] <- dot(float32(O_x[p,:]), O_c_normalized[k,:])
        m[p] <- s[p,target] > max(s[p,anchor], s[p,background])
        votes[p] <- votes[p] + m[p]

    在最终层 image/concept AV 已得到、image 输出投影尚未执行时暂停

M[p] <- votes[p] / |L_q| > eta
r <- sum(M) / (H * W)
g <- r > rho

如果 g：
    以完整 anchor 短语及原生条件，从相同 x_t、tau_t 执行 anchor 前向
    提取最终层同一位置的 image raw AV O_a
    O_x[p,:] <- where(M[p], O_a[p,:], O_x[p,:])

继续 user 的输出投影、门控、残差、MLP 与预测头
如模型启用 CFG，按原生方式组合 user 正条件与负条件预测
使用所得预测执行一次原生 scheduler step，得到 x_{t-1}
```

完整采样对 \(t=T,\ldots,1\) 重复上述过程，最后执行原生图像解码。

## 10. 六组保留的消融选择

以下设计均保留为后续实验选项。默认配置是实验起点，不预先认定其为最优。

### 10.1 替换表示：AV 与残差更新后表示

**默认：**使用式 (18) 的原始 AV。

**备选：**在对应层的 attention 残差更新后替换 image 表示。对具有独立 attention 残差步骤的层，定义：

\[
R^b_{t,\ell}
=H^{b,\mathrm{in}}_{t,\ell}
+G^b_{t,\ell}\odot
\mathcal P^O_\ell\left(O^b_{t,\ell}\right),
\qquad b\in\{u,a\},
\tag{21}
\]

其中 \(H^{b,\mathrm{in}}\) 为该分支 image 残差，\(G^b\) 为原生门控；\(\mathcal P^O_\ell\) 表示从 AV 到残差通道空间的原生输出投影算子，包含其实际偏置等运算。比较时仅在触发分支计算 anchor 表示并更新，未触发时直接保留 user 表示：

\[
\widetilde R^u_{t,\ell}
=\begin{cases}
R^u_{t,\ell}, & g_t=0,\\
(1-M_t)R^u_{t,\ell}+M_tR^a_{t,\ell}, & g_t=1.
\end{cases}
\tag{22}
\]

随后从该位置继续 user 后续计算。两种表示下的定位仍统一使用第 5 节的 raw AV，先控制定位因素。

对 FLUX 单流层，原生结构为 AV 与 MLP 特征的融合投影后再加残差。此时须把备选位置明确记为“融合残差更新后”，并匹配提取两条分支的相同表示；它包含 MLP 贡献，不能写成未经投影的 \(AV+H\) 或与纯 attention 残差位置混为一项结果。

### 10.2 定位层集：全部层与关键层

\[
\mathcal L_q^{\rm all}=\{1,\ldots,L\},
\qquad
\mathcal L_q^{\rm key}\subseteq\{1,\ldots,L\}.
\tag{23}
\]

默认全部层；备选使用明确列出的关键层子集。两者均按式 (13)–(14) 在所选层集内等权统计支持比例，并使用同一 \(\eta\) 判定，分母同步改为 \(|\mathcal L_q|\)。关键层的选择依据和最终索引由实验确定并记录；仅改变读出层集不等于提前删除概念的逐层演化路径。

### 10.3 替换层集：最后一层、全部层与关键层

\[
\mathcal L_e^{\rm last}=\{L\},\qquad
\mathcal L_e^{\rm all}=\{1,\ldots,L\},\qquad
\mathcal L_e^{\rm key}\subseteq\{1,\ldots,L\}.
\tag{24}
\]

定位层集 \(\mathcal L_q\) 与替换层集 \(\mathcal L_e\) 是独立选择。多层编辑时，每个编辑层都使用**当前步固定的同一个 \(\widehat M_t\)**，但 source 必须来自该层自己的 anchor 激活，不能把最终层的 source 直接复制给其他层：

\[
\widetilde O^{x,\mathrm{edit}}_{t,\ell}
=\begin{cases}
O^{x,\mathrm{edit}}_{t,\ell}, & g_t=0,\\
(1-M_t)O^{x,\mathrm{edit}}_{t,\ell}
+M_tO^a_{t,\ell}, & g_t=1,
\end{cases}
\quad \ell\in\mathcal L_e.
\tag{25}
\]

\(O^{x,\mathrm{edit}}\) 是干预执行过程中当前 user 路径的 AV，已反映更早编辑层的影响；不是把第一次只读定位时缓存的 AV 覆盖回去。

若掩码需要后面的层才能确定，则执行：只读 user 定位 → 按需 anchor source 前向 → 从最早编辑点恢复或重放 user 计算。掩码在该步的重放过程中固定，不边编辑边重新计算。一次定位 user 前向、一次 anchor 前向、一次完整 user 重放是直接实现方式；缓存早期状态可以缩短重放范围。比较这些方案时同时报告实际计算开销，默认最终层配置的两分支执行方式不能直接套用到多层编辑。

### 10.4 面积阈值 \(\rho\)

\[
g_t(\rho)=\mathbf1[r_t>\rho],
\qquad \rho\in[0,1].
\tag{26}
\]

默认 \(\rho=0\)。建议起始消融网格为
\(\{0,0.05,0.10,0.20,0.30,0.40\}\)，最终实验网格可根据结果调整；这些值是实验建议，不是已经验证的设置。

在当前严格不等号下，\(\rho=0\) 表示只要存在 target 掩码位置便触发，\(\rho=1\) 永不触发。比较时报告擦除效果、保留效果、触发率与额外 anchor 计算量，尤其观察小目标受面积阈值影响的情况。

### 10.5 更新算子：硬替换、比例融合与缩放相加

统一令 \(X\) 为当前被编辑的 user 表示、\(A\) 为同位置的 anchor 表示、\(M=\widehat M_t\)。以下更新公式用于已触发分支，沿通道广播 \(M\)；未触发时直接令 \(X'=X\)，无需计算 \(A\)：

| 更新算子 | 定义 | 系数 |
| --- | --- | --- |
| 硬替换（默认） | \(X'=X+M\odot(A-X)\) | 不使用 \(\lambda\) |
| 比例融合 | \(X'=X+\lambda M\odot(A-X)\) | \(0\le\lambda\le1\) |
| 缩放相加 | \(X'=X+\lambda M\odot A\) | \(\lambda\ge0\) |

比例融合的 \(\lambda=1\) 与硬替换相同，\(\lambda=0\) 不更新；缩放相加不会移除原有 \(X\) 分量。若“相加”的向量是 \(A-X\)，该方法已经属于上表的比例融合，不另列为独立算子。

融合与相加的 \(\lambda\) 分别调参，并记录各自扫描值；硬替换作为无强度系数的基准。

### 10.6 跨层投票阈值 \(\eta\)

\(\eta\in[0,1]\) 按式 (14) 控制需要多少定位层支持 target，默认 \(\eta=0\)。建议起始消融网格为 \(\{0,0.1,0.25,0.5,0.75\}\)。所有设置都使用严格大于：例如 \(\eta=0.5\) 时，恰好一半层支持仍取 0。

定位层集 \(\mathcal L_q\) 决定哪些层参与统计，\(\eta\) 决定这些层中所需的支持比例；二者分别配置。\(\eta\) 与面积阈值 \(\rho\) 也分别判定：前者决定每个位置是否进入掩码，后者决定当前步是否计算 anchor source 并替换。

## 11. 超参数与实验配置记录

| 配置项 | 符号或字段 | 当前设置 | 属性 |
| --- | --- | --- | --- |
| 目标、anchor、background 原文 | \(c_{\rm tar},c_{\rm anc},c_{\rm bg}\) | 随任务配置；background 起始文本为 `background` | 文本条件 |
| 概念数 | \(N_c\) | 3 | 当前单对概念设计常量 |
| 定位层集 | \(\mathcal L_q\) | 全部主 attention 层 | 消融轴 2 |
| 投票阈值 | \(\eta\) / `--eta` | \(0\)，严格大于 | 数值超参数，消融轴 6 |
| 面积阈值 | \(\rho\) / `--rho` | \(0\)，严格大于 | 数值超参数，消融轴 4 |
| concept 归一化 | 通道 L2，\(\epsilon=10^{-8}\) | FP32，原生 concept AV 除以 `norm + epsilon` | 固定的分数计算规则 |
| 替换表示位置 | `representation_site` | `av_pre_projection` | 消融轴 1 |
| 替换层集 | \(\mathcal L_e\) | 最后一层主 attention | 消融轴 3 |
| 更新算子 | `update_rule` | `hard_replace` | 消融轴 5 |
| 更新强度 | \(\lambda\) | 硬替换不启用 | 融合、相加时的数值超参数 |
| 干预步集 | \(\mathcal T_e\) | \(\{T,\ldots,1\}\) | 本文默认每步定位与判定 |
| 原生 CFG 干预分支 | `cfg_edit_branch` | 正条件路径；负条件预测保持原生 | 本稿明确的模型接入约定 |
| 采样步数、时间表、guidance、seed、分辨率 | 原生采样配置 | 沿用所选模型实验配置 | 与方法超参数分别记录 |

\(1/\sqrt{d_\ell}\) 是原生 attention 缩放，\(D_\ell\)、\(H\)、\(W\) 是模型或输入形状；它们不作为额外的方法调参项。本文也不增加新的 softmax 温度、置信度阈值或学习式门控。

最小实验记录包括：模型与原生采样配置、三个概念原文、概念编码与位置约定、concept 归一化规则、定位层和替换层的完整列表、替换表示、更新算子及系数、\(\eta\)、\(\rho\)，以及逐步的 \(r_t\)、\(g_t\) 和实际额外前向开销。六组消融优先分别改变一个设计项，其他配置保持一致。

## 12. 与 ConceptAttention 的实现对应

本方法沿用 ConceptAttention 的两个具体机制：独立编码概念后取第一个 token，以及在 attention 输出空间计算 image–concept 点积。可对照本仓库中的上游代码：

- [概念编码 `embed_concepts`](../../baseline/ConceptAttention/concept_attention/utils.py)：逐概念编码并提取索引 0 的 token。
- [FLUX 概念与图像输出](../../baseline/ConceptAttention/concept_attention/flux/dit_block.py)：保留多头拼接后的 AV，计算 concept/image 输出点积，并分别更新原始与概念路径。
- [输出空间 saliency 汇总](../../baseline/ConceptAttention/concept_attention/flux/pipeline.py)：在概念维 softmax，再对指定层和采样步求平均。
- [SD3 concept 输出归一化](../../baseline/ConceptAttention/concept_attention/sd3/concept_attention_pipeline.py)：先对 concept 输出做通道 L2 归一化，再与 image 输出点积；本文沿用这一归一化规则。
- [ConceptAttention 论文](https://arxiv.org/abs/2502.04320)。

queryGraft 在此基础上定义了当前的只读扩展：concept query 直接读取原始 user text、image 与 concept；上游 FLUX 概念 attention 默认读取 image 与 concept。queryGraft 的掩码则采用**当前单步内的逐层二值投票**，随后通过在线 anchor 条件 source 执行显式激活替换。上游跨层／跨步 saliency 平均与本文式 (12)–(15) 是不同的汇总算法。
