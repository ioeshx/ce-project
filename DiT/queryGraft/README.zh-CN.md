# queryGraft：只读概念定位与 anchor 激活替换

本文记录 queryGraft 的方法定义与首版实验配置；实现入口与命令见 [运行说明](README.md)。下文核心算法定义一个 target–anchor 对，另加一个 background 概念；现有多 target benchmark 的逐 pair 扩展见运行说明。模型参数冻结，干预发生在推理过程中。

核心流程是：在 user 前向中加入只读概念 token，利用模型自身的 attention 输出定位 target；跨层汇总得到当前采样步的空间掩码；达到面积阈值时，以同一当前 latent 下的 anchor 条件前向激活，替换目标区域内的 image attention 输出。

## 1. 当前默认配置

| 设计项 | 当前默认值 | 后续保留的选择 |
| --- | --- | --- |
| 定位概念 | target、anchor、background，共 3 个 | 本文固定单对概念 |
| 概念初始化 | 每个完整概念短语独立编码，取输出的第一个 token | 沿用当前约定 |
| 概念读取范围 | image 与本组概念 token | `--concept-include-prompt` 显式加入 user prompt；原始 token 不能读取概念 token |
| concept 调制条件 | pooled CLIP 输入置零，保留时间与 guidance | 复用 text 分支参数，单独计算门控、scale 与 shift |
| 定位特征 | 同层原始 image AV 与 concept AV 点积 | 不做 L2 归一化 |
| 区域判据 | 单 target：热图空间均值阈值；多 target：全部概念 argmax | `--my-layer-policy` 开启 `target_max`／`anchor_min` |
| 定位层集 | 全部具有概念读出路径的主 attention 层 | 与关键层子集比较 |
| 层间汇总 | 各层 softmax 软分数图等权平均，再做区域判定 | `--my-layer-aggre` 开启二值掩码投票，使用 $\eta$ |
| 面积阈值 | $\rho=0$，掩码非空即触发 | 对 $\rho$ 做消融 |
| 替换表示 | 原始 AV，位于输出投影之前 | 与 attention 残差更新后、整个 block 最终输出、最终 velocity 比较 |
| 替换层集 | 模型最后一层的主 attention | 与全部层、关键层子集比较 |
| 更新算子 | 硬替换 | 与比例融合、缩放相加比较 |
| 干预采样步 | 每个采样步都重新定位并判定触发 | 采样配置独立记录 |
| anchor 前向条件 | anchor 短语本身作为完整 prompt | 可选联合 user 条件、logits／Value 加权和激活外推 |

这里的 AV 是 softmax attention 权重乘 value 后的输出，不是 value 投影 $V$、attention 权重矩阵，也不是整个 Transformer block 的输出。本文以 `nudity`、`clothed`、`background` 为概念文本示例；实际字符串随实验配置保存。

## 2. 符号与张量定义

下文先定义单样本，批量运行时增加最前面的 batch 维，掩码、面积和触发条件均按样本分别计算。

| 符号 | 含义 |
| --- | --- |
| $\theta$ | 冻结的生成模型参数 |
| $F_\theta$ | 模型原生条件预测网络；内部可通过指定位置捕获 AV |
| $T$ | 采样步数 |
| $t=T,\ldots,1$ | 按执行顺序递减的采样步索引；不是模型原生时间值 |
| $\tau_t$ | scheduler 在第 $t$ 步传给模型的原生 timestep 或等价噪声时间 |
| $x_t$ | 当前采样步的 image latent，已包含之前采样步的干预结果 |
| $c_u,c_{\rm tar},c_{\rm anc},c_{\rm bg}$ | user prompt、target、anchor、background 文本 |
| $P_u$ | 完整 user prompt 的原生序列条件，映射到 Transformer 输入空间后得到的 text token |
| $X_t^0$ | 当前 $x_t$ 经原生图像输入映射得到的 image token |
| $C^0$ | 三个概念 token 的初始表示，形状为 $3\times D_{\rm model}$ |
| $N_p,N_x$ | user text token 数、image token 数；$N_x=H W$ |
| $H,W$ | image token 网格的高、宽，不是最终图像的像素高、宽 |
| $N=N_p+N_x,\;N_c=3$ | 原始 token 总数、附加概念 token 数 |
| $\ell=1,\ldots,L$ | 按模型执行顺序编号的 Transformer 层；每层指定一个主 attention 读出位置 |
| $h_\ell,d_\ell,D_\ell=h_\ell d_\ell$ | 第 $\ell$ 层的 attention 头数、每头通道数、多头拼接后的 AV 通道数 |
| $\kappa_{u,t},\kappa_{a,t},\kappa_{c,t}$ | user、anchor 各自的原生条件；concept 独立使用 pooled CLIP 输入置零、保留原生时间和 guidance 的调制条件 |
| $O^x_{t,\ell},O^c_{t,\ell}$ | user 前向中的 image、concept 原始 AV，形状为 $N_x\times D_\ell$、$3\times D_\ell$ |
| $O^a_{t,\ell}$ | anchor 前向在对应层的 image 原始 AV，形状为 $N_x\times D_\ell$ |
| $\mathcal L_q$ | 用于计算定位掩码的层集，默认 $\{1,\ldots,L\}$ |
| $\mathcal L_e$ | 执行激活替换的层集，默认 $\{L\}$ |
| $m_{t,\ell}(p),M_t(p)$ | 第 $\ell$ 层的 target 二值掩码、跨层汇总后的 target 二值掩码 |
| $r_t,g_t$ | target 掩码面积比例、当前步是否触发替换的二值变量 |
| $\eta$ | 仅 `--my-layer-aggre` 模式使用的跨层支持比例阈值，默认 $0$ |
| $\rho$ | 面积触发阈值，默认 $0$ |
| $\lambda$ | 融合或相加时的强度系数；硬替换不使用该参数 |
| $\widetilde f_{u,t},\widetilde y_t$ | 干预后的 user 条件预测、交给 scheduler 的原生 guidance 组合后预测 |
| $w,f^-_t$ | 原生 CFG 的 guidance scale、同一步负条件预测；仅在模型启用原生 CFG 时使用 |

本文的“层”按模型主干的 Transformer block 计数。SD3.5 若存在额外 image-only `attn2`，不把它作为第二个定位汇总来源。FLUX 的双流层、单流层按实际执行顺序统一编号，保留完整模块路径与原生列表内编号。

## 3. 独立编码 user 与三个概念

user prompt 沿用模型的完整原生文本编码流程，得到 $P_u$ 和原生 pooled 等条件。三个概念分别编码：

$$
e_k=E_{\rm seq}(c_k)_{0,:},
\qquad k\in\mathcal K
=\{\mathrm{tar},\mathrm{anc},\mathrm{bg}\},
\tag{1}
$$

$$
C^0=
\begin{bmatrix}
\Pi_c(e_{\rm tar})\\
\Pi_c(e_{\rm anc})\\
\Pi_c(e_{\rm bg})
\end{bmatrix}
\in\mathbb R^{3\times D_{\rm model}}.
\tag{2}
$$

$E_{\rm seq}$ 表示用于概念初始化的序列文本编码器；按照 ConceptAttention 的 FLUX 实现使用 T5，并由 $\Pi_c$ 复用或适配模型的原生 text 输入投影。迁移到 SD3.5 时，同样将概念序列表示映射到 joint attention 的 text 接口。

“取第一个 token”指：**先编码整个多词短语，再取输出序列索引 0 的向量**，不是只输入短语的第一个词。三个概念独立编码，避免初始化时因拼接成一句话而互相改变文本表示。

概念表示在 DiT 输入处加入；user 的原始序列编码、pooled 条件、image/text 位置编号保持原生定义。概念 token 使用对应模型的 text 位置编码约定；采用 ConceptAttention 的 FLUX 约定时其 position IDs 为零。concept 单独构造 $\kappa_{c,t}$：将 pooled CLIP 输入置零，保留当前前向的原生时间和 guidance，经同一套原生条件嵌入参数计算调制向量，保留投影 bias。concept 复用 text 分支的归一化、调制、投影与 MLP 参数；attention 前的 scale/shift、attention 残差门控和 MLP 的 scale/shift/门控均使用独立 concept 条件。user/image 与 anchor source 的条件保持原生定义。

$C^0$ 可预先编码并复用；**每个采样步从 $C^0$ 开始重新演化概念状态，不把上一步的末层概念状态直接传到下一采样步。**

## 4. 只读概念 attention

### 4.1 单向信息流的块掩码

为定义方便，将原始 token 记为 $B^0_t=[P_u;X_t^0]$，逻辑 token 顺序写为：

$$
Z_t^0=[B_t^0;C^0],
\qquad |B_t^0|=N,\quad |C^0|=N_c=3.
\tag{3}
$$

对每一层、每一个 attention head，在 softmax 前加入下列偏置：

$$
\mathcal B=
\begin{bmatrix}
0_{N_p\times N_p} & 0_{N_p\times N_x} & -\infty_{N_p\times N_c}\\
0_{N_x\times N_p} & 0_{N_x\times N_x} & -\infty_{N_x\times N_c}\\
-\infty_{N_c\times N_p} & 0_{N_c\times N_x} & 0_{N_c\times N_c}
\end{bmatrix}.
\tag{4}
$$

行是 query，列是 key。式 (4) 表示：

- 原始 user text、image query 只能读取原始 text、image 的 key/value。
- concept query 默认只读取 image 和本组 concept 的 key/value，不直接读取 user prompt token。
- target、anchor、background 在定位路径内可以互相读取；这个路径只负责定位。

这一默认读取范围对应 [ConceptAttention 第 4.1 节式 (7)–(9)](https://arxiv.org/html/2502.04320v1#S4.SS1)，包含 image cross-attention 与 concept self-attention。显式设置 `--concept-include-prompt`（配置键 `adapter.query_graft.concept_include_prompt=true`）时，式 (4) 左下角的 $-\infty_{N_c\times N_p}$ 改为零，恢复此前 concept 同时读取 prompt、image 与本组 concept 的行为。默认值为 `false`，也可用 `--no-concept-include-prompt` 显式关闭。这里控制的是直接的 K/V 读取范围；concept 的 pooled CLIP 调制输入始终置零，image 状态仍受 user prompt 条件影响。

如果原生 attention 已有 padding 或其他限制，保留原始 $B\!\to\!B$ 掩码，并对 concept 读取的无效 padding 列施加相同限制。式 (4) 展示的是所有原始 token 均有效时的概念隔离偏置。

真实模型可以使用不同 token 排列；实现时根据 text、image、concept 的实际索引映射设置掩码与切片。只读限制应用于**所有继续携带 concept token 的层**，包括接入概念路径后的 FLUX 单流层。

### 4.2 原始路径与概念路径的等价分解

省略 head 维与 $(t,\ell)$ 下标，在应用原生 Q/K 归一化、位置编码后，式 (4) 等价于：

$$
O_B=\operatorname{Attn}(Q_B,K_B,V_B),
\tag{5}
$$

$$
O_C=
\operatorname{Attn}
\left(Q_C,[K_X;K_C],[V_X;V_C]\right),
\tag{6}
$$

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{ConcatHeads}
\left[
\operatorname{softmax}_{\rm key}
\left(\frac{Q_jK_j^\top}{\sqrt{d_\ell}}+\mathcal B_j\right)V_j
\right]_{j=1}^{h_\ell}.
\tag{7}
$$

式 (7) 中的 $\mathcal B_j$ 是该次 attention 所需的有效位置掩码；只读分解后的两次调用分别使用对应的行列范围。图像与文本使用各自的原生 Q/K/V 投影；在有独立 text 参数的层，概念复用 text 侧投影、归一化和更新路径。

式 (6) 的 $K_X,V_X$ 仅包含 image token；启用 `--concept-include-prompt` 时，将它们替换为包含 image 与 prompt 的 $K_B,V_B$。多 pair 时每组只拼接自己的 $K_C,V_C$。

从 $O_B$ 取出 image 行得到 $O^x_{t,\ell}$，从 $O_C$ 取出三个 concept 行得到 $O^c_{t,\ell}$。这里取的是**输出投影前的原始 AV**。

当前层完成读出后，原始 token 和概念状态各自通过对应的原生投影、门控、残差与 MLP 更新，再进入下一层。可统一记为：

$$
B_t^\ell=\mathcal U^B_\ell
\left(B_t^{\ell-1},O_B;\kappa_{u,t}\right),
\qquad
C_t^\ell=\mathcal U^C_\ell
\left(C_t^{\ell-1},O_C;\kappa_{c,t}\right).
\tag{8}
$$

$\mathcal U$ 保留该层实际存在的其他原生运算；FLUX 单流层中也保留并行 MLP 与 attention 输出的融合。概念只通过自己的 query 路径读取，不写回 $B$ 的更新。

因此，在执行显式激活替换前，原始 token 的计算在数学上与原生 user 前向一致。实现上可以使用式 (5)–(6) 分别计算，无需物化完整的 $(N+3)\times(N+3)$ attention 矩阵；数值对照允许不同 kernel 的浮点误差。

## 5. 同层输出空间的概念定位

对每个定位层 $\ell\in\mathcal L_q$、概念 $k\in\mathcal K$ 和 image 位置 $p\in\{1,\ldots,HW\}$，直接计算原始 AV 点积：

$$
s_{t,\ell,k}(p)
=\left\langle
O^x_{t,\ell}(p,:),
O^c_{t,\ell}(k,:)
\right\rangle
=\sum_{d=1}^{D_\ell}
O^x_{t,\ell}(p,d)O^c_{t,\ell}(k,d).
\tag{9}
$$

$$
S_{t,\ell}\in\mathbb R^{H\times W\times3}.
\tag{10}
$$

点积使用 FP32 计算，image AV 与 concept AV 均不做 L2 归一化；向量范数保留在分数中，与 ConceptAttention 原始 FLUX 输出空间点积一致。concept AV 继续参与概念路径的状态更新，anchor 替换激活也保持原始 AV。每次比较必须使用**同一层、同一 attention、同一 user 前向**的 image 和 concept 输出；不同层之间不平均特征向量。

在 concept 维计算软分数图：

$$
\pi_{t,\ell,k}(p)
=\frac{\exp s_{t,\ell,k}(p)}
{\sum_{j\in\mathcal K}\exp s_{t,\ell,j}(p)}.
\tag{11}
$$

单 pair 时 concept 维包含 target、anchor、background；默认多 pair 时包含全部 $3K$ 个概念通道。开启 `--my-layer-policy` 时恢复各 pair 独立的三概念 softmax。$\pi$ 是配置概念之间的相对分数，不作为校准后的 target 存在概率。

### 5.1 ConceptAttention 的区域判据

令 $\mathcal P$ 表示区域判据。单类／多类按配置中的 target 数（pair 数）区分，anchor 和 background 不计为待编辑目标类。只有一个 target 时，对输入软分数图 $H$ 的 target 通道使用空间均值阈值：

$$
\mathcal P_{\rm CA}(H)(p)
=\mathbf1\!\left[
H_{\rm tar}(p)>\frac{1}{HW}\sum_q H_{\rm tar}(q)
\right].
\tag{12}
$$

等于均值时不选中，空间恒定的 target 热图得到空掩码。多个 target 时，在所有概念通道上选择最高分：

$$
j^*(p)=\operatorname*{argmax}_j H_j(p),
\qquad
\mathcal P_{{\rm CA},k}(H)(p)
=\mathbf1[j^*(p)=\operatorname{index}(\mathrm{tar}_k)].
\tag{12a}
$$

anchor 或 background 获胜的位置不编辑；并列最高时按配置中的概念顺序选择。concept attention 的逐 pair 隔离保持不变，默认区域判定在其读出的全部概念图上进行全局竞争。

### 5.2 默认跨层软图平均

令 $L_q=|\mathcal L_q|$，先对所选定位层的软分数图等权平均：

$$
\overline\pi_t=\frac{1}{L_q}
\sum_{\ell\in\mathcal L_q}\pi_{t,\ell},
\tag{13}
$$

$$
M_t=\mathcal P(\overline\pi_t).
\tag{14}
$$

默认 $\mathcal P=\mathcal P_{\rm CA}$，对应式 (12)／(12a)。这是“各层先 softmax，随后平均软图，再做区域判定”；不是先平均原始 logits 再 softmax，也不先二值化。默认不使用 $\eta$。所有汇总仅在当前采样步 $t$ 内完成。

### 5.3 显式开启此前的汇总与区域判据

`--my-layer-aggre` 显式恢复二值掩码投票。每层先应用所选区域判据：

$$
m_{t,\ell}=\mathcal P(\pi_{t,\ell}),
\qquad
v_t(p)=\frac{1}{L_q}\sum_{\ell\in\mathcal L_q}m_{t,\ell}(p),
\qquad
M_t(p)=\mathbf1[v_t(p)>\eta].
\tag{14a}
$$

$\eta\in[0,1]$，默认 $0$：只要任意所选层支持就选中；$0.5$ 为严格多数，$1$ 产生空掩码。没有 `--my-layer-aggre` 时，`eta` 不参与掩码计算。

`--my-layer-policy` 显式恢复各 pair 的 target/anchor/background 判据，使用 `--mask-rule` 选择：

$$
\mathcal P_{\rm target\_max}(H)(p)
=\mathbf1[H_{\rm tar}(p)>\max\{H_{\rm anc}(p),H_{\rm bg}(p)\}],
\tag{14b}
$$

$$
\mathcal P_{\rm anchor\_min}(H)(p)
=\mathbf1[H_{\rm anc}(p)<\min\{H_{\rm tar}(p),H_{\rm bg}(p)\}].
\tag{14c}
$$

`target_max` 为该开关下的默认规则；其 target 并列最高时不选中。`anchor_min` 在 anchor 并列最低时不选中，target 与 background 并列高于 anchor 时仍选中。仅设置 `--mask-rule` 不会开启该策略。

两个开关独立：仅开 `--my-layer-policy` 时，对平均软图应用式 (14b)／(14c)；仅开 `--my-layer-aggre` 时，各层先应用 ConceptAttention 的区域判据再投票。同时开启两者时，恢复此前“每层按原始点积分数比较 target/anchor/background，再按 `eta` 投票”的行为；单层内 softmax 不改变三个概念的分数顺序。多 pair 候选区域重叠时，投票模式按支持票比例选择，软图平均模式按平均 target 分数选择，平票按配置顺序选择。

## 6. 按 target 面积触发 anchor 前向

$$
r_t=\frac{1}{HW}\sum_{p=1}^{HW}M_t(p),
\qquad
g_t=\mathbf1[r_t>\rho],
\qquad \rho_{\rm default}=0.
\tag{15}
$$

$r_t$ 只统计 target 掩码面积。默认 $\rho=0$ 时，掩码非空就触发，空掩码仍跳过 anchor 前向。设置 $\rho=0.20$ 时恢复旧的面积门控：$r_t=0.20$ 不触发，$r_t>0.20$ 才触发。未触发时，继续完成 user 原生输出。最终阈值通过消融确定。

定义实际应用的掩码：

$$
\widehat M_t(p)=g_tM_t(p).
\tag{16}
$$

面积表示 token 区域比例；这里判定的是 target 的相对定位结果是否足够大，而不是“任意一个概念是否占据足够面积”。

## 7. 从当前 latent 计算 anchor source activations

默认 `--source-mode anchor` 下，当 $g_t=1$ 时，以 **anchor 短语本身**作为原生 prompt，在同一个 $x_t$ 和 $\tau_t$ 上计算一次 anchor 条件模型前向：

$$
\left\{O^a_{t,\ell}\right\}_{\ell\in\mathcal L_e}
=\operatorname{CaptureAV}_{\mathcal L_e}
\left[F_\theta(x_t,\tau_t;\kappa_{a,t})\right].
\tag{17}
$$

例如 $c_{\rm anc}=\texttt{clothed}$ 时，anchor 分支输入就是 `clothed`，按模型原生流程编码完整 anchor prompt 及其 pooled 等条件。**定位时的第一个 token 规则仅用于式 (1)，不截断此处完整 anchor prompt 的原生条件。**

该分支使用模型原生 text–image joint attention。文本条件中不加入 user prompt、target 或 background，也不附加定位概念 token；图像侧仍可读取当前 $x_t$ 的信息。因此，anchor 条件与 target 文本分离，并不意味着当前 image latent 已经没有 target 信息。

提取的 source 必须与 user 被替换激活具有相同的层、attention 分支、空间 token 顺序和表示位置。该分支不推进 scheduler，不单独更新 latent。中间表示模式提取对应激活，`velocity` 则使用完整 source 前向的最终预测；source 已提取且不需要最终预测时，可以提前结束其余计算。

### 7.1 可选的联合条件与 source 增强

四种新模式默认关闭，由 `--source-mode joint/logits/values/extrapolate` 分别启用。原始 user prompt 保持不变，定位概念不变；source 的序列和 pooled 条件由原生 pipeline 对每个样本的 `user prompt, anchor` 联合编码得到。

`joint` 直接使用该联合前向的同层激活。对 `logits`／`values`，令 $U$、$A$ 为联合序列中 user 和新增 anchor 内容 token 的索引集合，$M_i$ 为当前 pair 最终执行区域（同 anchor 的 pair 取并集），则 source 主 attention 的 image query 使用：

$$
\widetilde L_{ij}=L_{ij}+M_i\bigl(\mathbf1[j\in U]\log\alpha
+\mathbf1[j\in A]\log\beta\bigr)
\qquad(\texttt{logits}),
$$

$$
\widetilde O_i=\sum_j P_{ij}
\left[1+M_i\bigl((\alpha-1)\mathbf1[j\in U]+(\beta-1)\mathbf1[j\in A]\bigr)\right]V_j
\qquad(\texttt{values}).
$$

其中 $P$ 为当前层原生 softmax attention，$\alpha$、$\beta$ 分别对应 `--source-user-weight`、`--source-anchor-weight`。`values` 不改变当前层 Q/K 与 attention 概率；两个模式都保留当前层原生 text query 输出。特殊／padding token、图像 keys 和 pooled 条件不单独加权。SD3 的 CLIP 行包含两个编码器的通道，只有两个编码器都归入同一组的位置才归入 $U$ 或 $A$，混合位置保持原权重。

`extrapolate` 在所选表示和编辑层上使用：

$$
H^s_{t,\ell}=H^u_{t,\ell}
+\lambda_s\bigl(H^{u+a}_{t,\ell}-H^u_{t,\ell}\bigr).
$$

$\lambda_s$ 对应 `--source-scale`，独立于最终执行融合的 `--strength`。多层配置的 $H^u$ 从同一步未编辑的定位前向捕获，不能使用已经经过前层替换的重放状态。命令与 I2P 评估方式见 [Source 条件与增强模式](README.md#source-条件与增强模式)。

## 8. 默认在最终层执行 AV 硬替换

默认 $\mathcal L_e=\{L\}$。将式 (16) 的空间掩码沿通道维广播：

$$
\widetilde O^x_{t,L}(p,:)
=\begin{cases}
O^x_{t,L}(p,:), & g_t=0,\\
\left(1-M_t(p)\right)O^x_{t,L}(p,:)
+M_t(p)O^a_{t,L}(p,:), & g_t=1.
\end{cases}
\tag{18}
$$

未触发的分支直接返回原 AV，不需要计算或引用 $O^a_{t,L}$。

然后使用 $\widetilde O^x_{t,L}$ 继续 **user 分支**的输出投影、门控、残差、MLP 与最终预测头。原始 user 残差和已计算的其他特征保留在 user 路径中；替换对象只是图像行的 AV。

先得到干预后的 user 条件预测：

$$
\widetilde f_{u,t}
=\operatorname{UserSuffix}_{t,L}
\left(\widetilde O^x_{t,L};\mathcal R^u_{t,L}\right),
\tag{19}
$$

再按模型原生 guidance 方式得到送入 scheduler 的预测。对常规 CFG：

$$
\widetilde y_t
=f^-_t+w\left(\widetilde f_{u,t}-f^-_t\right).
$$

其中 $f^-_t$ 为相同 $x_t,\tau_t$ 下的原生负条件预测。没有外部 CFG 时，$\widetilde y_t=\widetilde f_{u,t}$；模型内嵌 guidance 已包含在原生网络条件中。最后执行：

$$
x_{t-1}=\operatorname{SchedulerStep}
\left(x_t,\tau_t,\widetilde y_t\right).
\tag{20}
$$

$\mathcal R^u_{t,L}$ 表示继续 user 计算所需的原生状态，例如残差、调制参数、text 输出，以及 FLUX 单流层并行的 MLP 特征。$\widetilde y_t$ 按模型定义可以是速度或噪声预测；式 (20) 调用原生 scheduler，不另设采样更新公式。

### 8.1 最终层暂停与前向次数

默认执行顺序为：

1. user＋概念路径前向，逐层计算式 (9)–(12)，累计投票。
2. 在最终层同时得到 image 与 concept AV 后，**在 user image AV 输出投影前暂停**。
3. 利用全部定位层结果计算 $M_t,r_t,g_t$。
4. 若触发，从原始当前 $x_t$ 计算 anchor source；随后应用式 (18)。
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
参数：定位层集 L_q、编辑层集 L_e={L}、rho=0
      my_layer_aggre=false、my_layer_policy=false；投票模式的 eta=0

重置当前步概念状态 C <- C^0
votes[p] <- 0
soft_sum[p,k] <- 0

执行 user 前向，并在所有携带概念 token 的层执行单向概念读取：
    原始 text/image attention 只能读取原始 text/image
    concept attention 默认只读取 image 和本组 concept
    concept 调制单独使用 pooled CLIP 输入为零的条件，保留时间与 guidance

    对每个定位层 ell in L_q：
        取同层 image raw AV O_x 与 concept raw AV O_c
        s[p,k] <- dot(float32(O_x[p,:]), float32(O_c[k,:]))
        pi[p,k] <- softmax_over_concepts(s[p,k])
        soft_sum[p,k] <- soft_sum[p,k] + pi[p,k]
        m[p] <- selected_policy(pi)；自定义策略在每层比较原始 s
        votes[p] <- votes[p] + m[p]

    在最终层 image/concept AV 已得到、image 输出投影尚未执行时暂停

如果 my_layer_aggre：M[p] <- votes[p] / |L_q| > eta
否则：M[p] <- selected_policy(soft_sum / |L_q|)
默认 selected_policy：单 target 用空间均值阈值，多 target 用全部概念 argmax
my_layer_policy=true 时：selected_policy 使用 target_max 或 anchor_min
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

完整采样对 $t=T,\ldots,1$ 重复上述过程，最后执行原生图像解码。

## 10. 六组保留的消融选择

以下设计均保留为后续实验选项。默认配置是实验起点，不预先认定其为最优。

### 10.1 替换表示：AV、attention 残差更新后、block 最终输出与 velocity

默认：使用式 (18) 的原始 AV。

备选：在对应层的 attention 残差更新后替换 image 表示。对具有独立 attention 残差步骤的层，定义：

$$
R^b_{t,\ell}
=H^{b,\mathrm{in}}_{t,\ell}
+G^b_{t,\ell}\odot
\mathcal P^O_\ell\left(O^b_{t,\ell}\right),
\qquad b\in\{u,a\},
\tag{21}
$$

其中 $H^{b,\mathrm{in}}$ 为该分支 image 残差，$G^b$ 为原生门控；$\mathcal P^O_\ell$ 表示从 AV 到残差通道空间的原生输出投影算子，包含其实际偏置等运算。比较时仅在触发分支计算 anchor 表示并更新，未触发时直接保留 user 表示：

$$
\widetilde R^u_{t,\ell}
=\begin{cases}
R^u_{t,\ell}, & g_t=0,\\
(1-M_t)R^u_{t,\ell}+M_tR^a_{t,\ell}, & g_t=1.
\end{cases}
\tag{22}
$$

随后从该位置继续 user 后续计算。两种表示下的定位仍统一使用第 5 节的 raw AV，先控制定位因素。

对 FLUX 单流层，原生结构为 AV 与 MLP 特征的融合投影后再加残差。此时须把备选位置明确记为“融合残差更新后”，并匹配提取两条分支的相同表示；它包含 MLP 贡献，不能写成未经投影的 $AV+H$ 或与纯 attention 残差位置混为一项结果。

第三种选择 `--representation block_output` 在整个 block 的最终 image hidden state 处替换。对具有独立 attention 和 MLP 残差步骤的层，记 $S^b$ 为进入 MLP 子层前的状态（SD3.5 存在额外 `attn2` 时包含该分支更新），则：

$$
B^b=S^b+G^b_{\mathrm{MLP}}\odot
\mathrm{MLP}_{\ell}\!\left(\mathrm{Mod}^b_{\ell}(\mathrm{LN}_{\ell}(S^b))\right),
\qquad
\widetilde B^u=(1-M_t)\odot B^u+M_t\odot B^a.
$$

替换位置为 MLP、MLP 门控及第二次残差相加之后；未触发时保留 user 输出，text 分支不替换。FLUX 单流层没有独立的第二次残差相加，其 `block_output` 与 `residual` 共享原生融合残差输出位置。三种选择均使用 raw AV 定位，并支持相同的编辑层选择和更新算子。

第四种选择 `--representation velocity` 使用完整 Transformer 的最终输出。SD3.5 两条分支各自执行最终 modulation、linear projection 和 unpatchify，得到 $v^u_t$、$v^s_t$；source 输出头使用自身的 pooled 条件。令 $M^{\mathrm{lat}}_t$ 为 token 掩码向其对应 latent patch 的最近邻扩展，硬替换为：

$$
\widetilde v^u_t=(1-M^{\mathrm{lat}}_t)\odot v^u_t
+M^{\mathrm{lat}}_t\odot v^s_t,
\qquad
v^{\mathrm{CFG}}_t=v^-_t+w(\widetilde v^u_t-v^-_t).
$$

负条件预测不替换，CFG 和 scheduler 继续使用原生实现。`joint`／`logits`／`values` 的 source 构造遵循第 7.1 节；`extrapolate` 在完整输出上使用 $v^s_t=v^u_t+\lambda_s(v^{\mathrm{joint}}_t-v^u_t)$。`blend`／`add` 同样可用于 velocity。最终输出只有一个编辑位置，要求 `--edit-layers last`；定位层仍可单独选择。每步取得 user 最终预测后按需计算 source，再直接替换输出，无需重放 user。FLUX 在原生 packed velocity 布局中执行等价替换，保留内嵌 guidance。运行命令见 [Velocity：I2P 采样与评估](README.md#velocityi2p-采样与评估)。

### 10.2 定位层集：全部层与关键层

$$
\mathcal L_q^{\rm all}=\{1,\ldots,L\},
\qquad
\mathcal L_q^{\rm key}\subseteq\{1,\ldots,L\}.
\tag{23}
$$

默认全部层；备选使用明确列出的关键层子集。两者均按式 (13)–(14) 在所选层集内等权统计支持比例，并使用同一 $\eta$ 判定，分母同步改为 $|\mathcal L_q|$。关键层的选择依据和最终索引由实验确定并记录；仅改变读出层集不等于提前删除概念的逐层演化路径。

### 10.3 替换层集：最后一层、全部层与关键层

$$
\mathcal L_e^{\rm last}=\{L\},\qquad
\mathcal L_e^{\rm all}=\{1,\ldots,L\},\qquad
\mathcal L_e^{\rm key}\subseteq\{1,\ldots,L\}.
\tag{24}
$$

定位层集 $\mathcal L_q$ 与替换层集 $\mathcal L_e$ 是独立选择。多层编辑时，每个编辑层都使用**当前步固定的同一个 $\widehat M_t$**，但 source 必须来自该层自己的 anchor 激活，不能把最终层的 source 直接复制给其他层：

$$
\widetilde O^{x,\mathrm{edit}}_{t,\ell}
=\begin{cases}
O^{x,\mathrm{edit}}_{t,\ell}, & g_t=0,\\
(1-M_t)O^{x,\mathrm{edit}}_{t,\ell}
+M_tO^a_{t,\ell}, & g_t=1,
\end{cases}
\quad \ell\in\mathcal L_e.
\tag{25}
$$

$O^{x,\mathrm{edit}}$ 是干预执行过程中当前 user 路径的 AV，已反映更早编辑层的影响；不是把第一次只读定位时缓存的 AV 覆盖回去。

若掩码需要后面的层才能确定，则执行：只读 user 定位 → 按需 anchor source 前向 → 从最早编辑点恢复或重放 user 计算。掩码在该步的重放过程中固定，不边编辑边重新计算。一次定位 user 前向、一次 anchor 前向、一次完整 user 重放是直接实现方式；缓存早期状态可以缩短重放范围。比较这些方案时同时报告实际计算开销，默认最终层配置的两分支执行方式不能直接套用到多层编辑。

### 10.4 面积阈值 $\rho$

$$
g_t(\rho)=\mathbf1[r_t>\rho],
\qquad \rho\in[0,1].
\tag{26}
$$

默认 $\rho=0$。建议起始消融网格为
$\{0,0.05,0.10,0.20,0.30,0.40\}$，最终实验网格可根据结果调整；这些值是实验建议，不是已经验证的设置。

在当前严格不等号下，$\rho=0$ 表示只要存在 target 掩码位置便触发，$\rho=1$ 永不触发。比较时报告擦除效果、保留效果、触发率与额外 anchor 计算量，尤其观察小目标受面积阈值影响的情况。

### 10.5 更新算子：硬替换、比例融合与缩放相加

统一令 $X$ 为当前被编辑的 user 表示、$A$ 为同位置的 anchor 表示、$M=\widehat M_t$。以下更新公式用于已触发分支，沿通道广播 $M$；未触发时直接令 $X'=X$，无需计算 $A$：

| 更新算子 | 定义 | 系数 |
| --- | --- | --- |
| 硬替换（默认） | $X'=X+M\odot(A-X)$ | 不使用 $\lambda$ |
| 比例融合 | $X'=X+\lambda M\odot(A-X)$ | $0\le\lambda\le1$ |
| 缩放相加 | $X'=X+\lambda M\odot A$ | $\lambda\ge0$ |

比例融合的 $\lambda=1$ 与硬替换相同，$\lambda=0$ 不更新；缩放相加不会移除原有 $X$ 分量。若“相加”的向量是 $A-X$，该方法已经属于上表的比例融合，不另列为独立算子。

融合与相加的 $\lambda$ 分别调参，并记录各自扫描值；硬替换作为无强度系数的基准。

### 10.6 跨层投票阈值 $\eta$

仅开启 `--my-layer-aggre` 后，$\eta\in[0,1]$ 按式 (14a) 控制需要多少定位层支持 target，默认 $\eta=0$。建议起始消融网格为 $\{0,0.1,0.25,0.5,0.75\}$。所有设置都使用严格大于：例如 $\eta=0.5$ 时，恰好一半层支持仍取 0。默认软图平均不使用 $\eta$。

定位层集 $\mathcal L_q$ 决定哪些层参与统计，$\eta$ 决定这些层中所需的支持比例；二者分别配置。$\eta$ 与面积阈值 $\rho$ 也分别判定：前者决定每个位置是否进入掩码，后者决定当前步是否计算 anchor source 并替换。

## 11. 超参数与实验配置记录

| 配置项 | 符号或字段 | 当前设置 | 属性 |
| --- | --- | --- | --- |
| 目标、anchor、background 原文 | $c_{\rm tar},c_{\rm anc},c_{\rm bg}$ | 随任务配置；background 起始文本为 `background` | 文本条件 |
| 概念数 | $N_c$ | 3 | 当前单对概念设计常量 |
| 定位层集 | $\mathcal L_q$ | 全部主 attention 层 | 消融轴 2 |
| 跨层汇总 | `my_layer_aggre` / `--my-layer-aggre` | 默认关闭，软图平均；开启后投票 | 离散汇总规则 |
| 区域判据 | `my_layer_policy` / `--my-layer-policy` | 默认关闭，ConceptAttention 单类／多类规则 | 离散定位规则 |
| 自定义判据 | `mask_rule` / `--mask-rule` | 仅 `my_layer_policy=true` 时生效，`target_max` 或 `anchor_min` | 离散定位规则 |
| 投票阈值 | $\eta$ / `--eta` | 仅 `my_layer_aggre=true` 时生效，$0$，严格大于 | 数值超参数，消融轴 6 |
| 面积阈值 | $\rho$ / `--rho` | $0$，严格大于 | 数值超参数，消融轴 4 |
| concept 调制 | $\kappa_{c,t}$ | pooled CLIP 输入置零，保留时间和 guidance | 复用 text 分支参数 |
| 概念分数 | 原始 AV 点积 | FP32，image 与 concept 均不做 L2 归一化 | 固定的分数计算规则 |
| 替换表示位置 | `representation_site` | `av_pre_projection` | 消融轴 1 |
| 替换层集 | $\mathcal L_e$ | 最后一层主 attention | 消融轴 3 |
| 更新算子 | `update_rule` | `hard_replace` | 消融轴 5 |
| 更新强度 | $\lambda$ | 硬替换不启用 | 融合、相加时的数值超参数 |
| 干预步集 | $\mathcal T_e$ | $\{T,\ldots,1\}$ | 本文默认每步定位与判定 |
| 原生 CFG 干预分支 | `cfg_edit_branch` | 正条件路径；负条件预测保持原生 | 本稿明确的模型接入约定 |
| 采样步数、时间表、guidance、seed、分辨率 | 原生采样配置 | 沿用所选模型实验配置 | 与方法超参数分别记录 |

$1/\sqrt{d_\ell}$ 是原生 attention 缩放，$D_\ell$、$H$、$W$ 是模型或输入形状；它们不作为额外的方法调参项。本文也不增加新的 softmax 温度、置信度阈值或学习式门控。

最小实验记录包括：模型与原生采样配置、三个概念原文、概念编码与位置约定、concept 独立调制与原始 AV 点积规则、`my_layer_aggre`、`my_layer_policy`、`mask_rule`、定位层和替换层的完整列表、替换表示、更新算子及系数、$\eta$、$\rho$，以及逐步的 $r_t$、$g_t$ 和实际额外前向开销。消融优先分别改变一个设计项，其他配置保持一致。

## 12. 与 ConceptAttention 的实现对应

本方法沿用 ConceptAttention 的两个具体机制：独立编码概念后取第一个 token，以及在 attention 输出空间计算 image–concept 点积。可对照本仓库中的上游代码：

- [概念编码 `embed_concepts`](../../baseline/ConceptAttention/concept_attention/utils.py)：逐概念编码并提取索引 0 的 token。
- [FLUX 概念与图像输出](../../baseline/ConceptAttention/concept_attention/flux/dit_block.py)：保留多头拼接后的 AV，计算 concept/image 输出点积，并分别更新原始与概念路径。
- [输出空间 saliency 汇总](../../baseline/ConceptAttention/concept_attention/flux/pipeline.py)：在概念维 softmax，再对指定层和采样步求平均。
- [FLUX 独立概念调制](../../baseline/ConceptAttention/concept_attention/flux/dit.py)：concept 使用 pooled CLIP 输入置零后的独立调制条件，复用 text 分支参数。
- [ConceptAttention 论文](https://arxiv.org/abs/2502.04320)。

queryGraft 与上游 FLUX 的 concept attention 默认均读取 image 与 concept；可通过 `--concept-include-prompt` 显式加入 user text。当前默认的跨层软图平均和单类／多类区域判据与 ConceptAttention 一致，每个当前采样步独立计算，再通过在线 anchor 条件 source 执行显式激活替换。`--my-layer-aggre` 与 `--my-layer-policy` 可独立恢复此前的二值投票与三概念判据；跨采样步的平均仅用于诊断展示。

## 13. 逐采样步诊断与样本汇总的数学定义

诊断默认关闭，显式传入 `--trace` 后开启；`--trace-root` 仅指定保存路径。开启后，每个样本、每个时间步保存一张四联图与一份数值张量。多 pair 在同一张图中按行展示。下面恢复 pair 索引 $k$，并用 $w_t(p)$ 表示经过面积门控与重叠选择后的胜出 pair；没有胜出者时为 −1。

### 13.1 最终执行掩码

$$
A_{t,k}(p)=\mathbf1[w_t(p)=k],
\qquad
A_{t,\ell,k}(p)=\mathbf1[\ell\in\mathcal L_e]A_{t,k}(p).
\tag{27}
$$

第一张热力图展示 $A_{t,k}$。单 pair 时它等于式 (16) 的 $\widehat M_t$；多 pair 时还剔除重叠位置中未胜出的概念。未触发时为零图。这里记录的是算法执行替换的区域；实际数值改变量由第 13.4 节独立统计。

### 13.2 实际决策输入

第二张热力图使用实际所选定位层 $\mathcal L_q$。默认软图平均模式显示 target 通道：

$$
U_{t,k}(p)=\frac1{|\mathcal L_q|}
\sum_{\ell\in\mathcal L_q}\pi_{t,\ell,k,\mathrm{tar}}(p).
\tag{28}
$$

默认单类时，在图下标注本样本的空间均值阈值，按严格大于生成候选掩码；多类时，标明在全部 $3K$ 个概念上 argmax 及当前 target 的通道索引。开启 `--my-layer-policy` 时，标注实际的三概念比较规则。开启 `--my-layer-aggre` 时，第二图改为式 (14a) 的二值支持比例 $v_t$，阈值为 $\eta$。非投票模式不在标题中展示无效的 `eta`。

### 13.3 候选掩码与复算数据

第三张热力图展示经过所选区域策略／投票阈值后的候选掩码，尚未经过面积门控和重叠选择：

$$
V_{t,k}(p)=M_{t,k}(p).
\tag{29}
$$

该图与第一张执行掩码图对照，可区分区域判定、面积门控和重叠选择。`localization_scores_mean` 保存全部所选层的三通道均值；单类默认模式另存实际 `decision_threshold`，投票模式在同名字段记录 $\eta$。默认多类模式另存 `concept_argmax`，其索引顺序为 `[target_0,anchor_0,background_0,target_1,...]`，只有索引 $3k$ 对应第 $k$ 个 target。无标量阈值时，metadata 的 `decision_threshold` 为 `null`。

全执行层辅助统计 `layer_mask_mean`、`concept_scores_mean`、`concept_logits_mean` 继续保存，供额外核查；它们不作为默认决策输入绘图。metadata 另记录实际的 `aggregation`、`region_policy` 和 `softmax_scope`。trace 不改变层选择、掩码计算或模型前向次数。

### 13.4 实际替换位置的激活变化

令 $X_{t,\ell}$、$X'_{t,\ell}$ 为编辑位置更新前、更新后的 user 表示；位置遵循 `av`／`residual`／`block_output`／`velocity` 配置。对 pair $k$，只在其实际执行区域统计，记通道元素集合为：

$$
\Omega_{t,k}=\{(\ell,p,d):\ell\in\mathcal L_e,
A_{t,k}(p)=1,\ 1\le d\le D_\ell\}.
$$

其中 $D_\ell$ 在 `residual`／`block_output` 配置下使用该编辑表示的实际通道数。`velocity` 将一个 token 对应 patch 的全部 latent 位置与通道作为该行元素，诊断编辑索引记为 −1；这仅改变统计布局，不对 velocity 数值做 patch 平均。第四个面板用柱状图展示 $Z\in\{X,X',X'-X\}$ 的：

$$
\operatorname{RMS}_{\Omega}(Z)
=\sqrt{\frac{\sum_{(\ell,p,d)\in\Omega_{t,k}}Z_{t,\ell}(p,d)^2}
{|\Omega_{t,k}|}}.
\tag{30}
$$

同时标注相对 L2 改变量与最大绝对改变量：

$$
R_{t,k}
=\frac{\sqrt{\sum_{\Omega_{t,k}}(X'-X)^2}}
{\sqrt{\sum_{\Omega_{t,k}}X^2}+10^{-8}},
\qquad
B_{t,k}=\max_{\Omega_{t,k}}|X'-X|.
\tag{31}
$$

集合为空时统计值记为零，并显示未执行更新。多层汇总使用各层平方和与通道元素数计算，不直接平均各层 RMS；另存逐编辑层的相同指标与 token 数。默认 AV 模式的统计发生在输出投影之前，不能将它解释为最终速度预测的改变量。

### 13.5 单个样本跨时间步的汇总

令 $\mathcal T_d$ 为本次采样实际执行并记录的全部时间步，$S=|\mathcal T_d|$。采样完成后，将每步的最终执行掩码等权平均：

$$
\overline A_k(p)=\frac1S\sum_{t\in\mathcal T_d}A_{t,k}(p).
\tag{32}
$$

未触发步骤的 $A_{t,k}$ 为零，仍计入分母。$\overline A_k(p)\in[0,1]$ 表示该位置被编辑的时间步比例，不再施加二值阈值。

对实际所选定位层的逐层 softmax、跨层平均分数，再计算跨时间步的均值。保存每组三个概念通道，汇总图展示 target：

$$
\overline P^c_k(p)=\frac1S\sum_{t\in\mathcal T_d}
\frac1{|\mathcal L_q|}\sum_{\ell\in\mathcal L_q}
\pi_{t,\ell,k,c}(p),
\quad c\in\{\mathrm{tar},\mathrm{anc},\mathrm{bg}\}.
\tag{33}
$$

这里按采样步等权平均，不按模型 timestep 的间隔加权，也不先平均原始点积分数再 softmax。未触发替换的步骤仍参与概念分数平均。汇总图不对时间均值重新施加空间阈值或 argmax；执行频率直接由各步已经执行的掩码平均得到。投票模式下，第二张汇总热图标为辅助软分数。

每个样本目录直接保存 `summary.png`（执行频率与所选定位层 target 分数两张热力图）及 `summary.safetensors`（`applied_mask_mean`、`localization_scores_mean`，以及保留的全执行层辅助统计 `concept_scores_mean`）。逐步四联图位于 `timesteps/step_XXXX.png`，逐步数值位于 `tensors/step_XXXX.safetensors`，`metadata.json` 保留在样本根目录。

实现仅增加空间图的 CPU 累加、统计、绘图与 I/O，不增加模型前向。诊断仍默认关闭。输出字段、目录与运行命令见 [运行说明](README.md)。
