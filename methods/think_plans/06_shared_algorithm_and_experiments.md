# 共用算法、符号、执行接口与实验协议

本文定义五套方案共享的基础流程。每篇方案另外给出其独立的校准、供体、选层、约束或控制算法。

## 1. 符号与任务接口

采样索引 `n=0,...,N-1` 按实际执行顺序递增；`sigma_n` 为模型原生 scheduler 的噪声时间。这个索引约定不依赖论文中 `t=0` 表示干净还是噪声的约定。

| 符号 | 定义 |
| --- | --- |
| `p` | 原始 user prompt |
| `e, a` | 待擦除概念查询与安全替代概念查询 |
| `R` | 明确保留／背景概念查询集合 |
| `p^-` | 保持场景语义、将 `e` 改成安全内容的供体 prompt |
| `x_n` | 当前实际采样轨迹上的 latent |
| `F_theta` | 冻结的生成模型，输出原生预测，例如 flow velocity |
| `X_{n,ell}` | 第 `ell` 个 block 的图像残差状态 |
| `O_{n,ell}` | 该 block 图像侧、多头拼接后、输出投影前的 attention 输出 |
| `C_{n,ell,c}` | 概念 `c` 的 attention 输出向量；不是生成模型文本 prompt token |
| `P=H_g W_g` | 空间 patch 数；`D_ell` 为当前干预位置的特征维度 |
| `L_loc` | 用于概念读出的层集合 |
| `L_edit` | 实际替换的层／attention 分支集合 |
| `M_n(p)` | 当前步统一空间掩码，`p` 在此为 patch 索引；上下文中与 prompt 区分 |
| `eta_n` | 干预强度，基础版取 `[0,1]` 内值 |

文中公式也用 `i,j` 表示 patch 位置，避免 prompt 与位置共用字母。多概念时为每个目标 `e_k` 配置安全概念 `a_k`、安全条件 `p_k^-` 和独立的掩码。

部署接口可以是 `generate(p, configured_erasure_set, paired_safe_conditions)`。概念集合由开发者预配置，在线查询所需的目标词与安全条件均属于方法资源。不能同时把“需要在线概念编码”描述成“无需在线概念信息”。

## 2. 概念只读分支

### 2.1 不改变原图文主路径

原图像、prompt 的 joint attention 维持原有计算。新增概念残差流，通过文本侧的参数产生 `Q_c,K_c,V_c`，每个 head 计算：

\[
O_c=\operatorname{softmax}
\left(Q_c[K_c;K_x]^\top/\sqrt{d_h}\right)[V_c;V_x].
\tag{C1}
\]

图像和 prompt 的主路径不读概念的 K/V。概念经过文本侧的输出投影、调制和 MLP，递推到下一层。原有 Q/K normalization、RoPE 和时间条件按当前模型的实现复用。这是 ConceptAttention 提供的基础机制。[原文 §4.1](https://arxiv.org/html/2502.04320v2)

每个采样步均从固定文本编码得到的初始概念 token 重新开始，再沿本次模型的层递推；不把上一步最后一层的概念向量直接作为下一步初始值。本轮默认概念侧使用当前 user 分支的时间与 pooled text 调制条件；复现原仓库时按其模型专用条件处理，二者分别记录配置。

全称概念有多个 token 时，本轮设计保留其有效文本 token，在每层将所属 token 的输出平均成一个 phrase 向量：

\[
C_{n,\ell,c}=|J_c|^{-1}\sum_{j\in J_c}O_{c,j}.
\tag{C2}
\]

`J_c` 排除 pad／特殊 token。这个 phrase 聚合是明确的实现选择；原始 Flux `embed_concepts` 取首 token，不等于已经支持完整多词查询。比较单 token、首 token 与 phrase 聚合，测试集固定同一种规则。

### 2.2 原生图像输出与辅助图像输出的选择

本组候选默认用**原主路径的图像 `O`** 与概念 `C` 计算读出，保持“定位当前实际生成表示”的定义。

本地 `baseline/ConceptAttention/concept_attention/sd3/dit_block.py` 的实现有一个重要细节：先对 image＋prompt 调 attention，再对 image＋concept 调 attention；缓存用于热图的 `image_output_vectors` 来自第二次辅助调用。它与原主路径图像输出并不完全相同。

因此接入时显式保存 `native_image_av` 与 `auxiliary_image_av`，基线复现使用仓库原选择，研究候选使用本文定义，并做二者的定位和擦除对照。不能仅因变量都叫 image vectors 就混用结果。

### 2.3 逐层读出

基础分数、概念维归一化与聚合定义为：

\[
s_{n,\ell,c}(i)=
\langle O_{n,\ell}(i),C_{n,\ell,c}\rangle/T_\ell,
\qquad
H_{n,\ell,c}(i)=
\frac{\exp s_{n,\ell,c}(i)}{\sum_{c'\in\{e,a\}\cup R}\exp s_{n,\ell,c'}(i)}.
\tag{C3}
\]

`T_ell>0` 为在校准集固定的尺度；原始点积版取 `T_ell=1`。若使用概念 L2 归一化，须固定此规则再校准，不能把它当作等价变换。SD3 仓库封装包含这一归一化。

\[
\bar H_{n,e}(i)=
\sum_{\ell\in L_{loc}}\omega_\ell H_{n,\ell,e}(i),
\quad \omega_\ell\ge0,\quad\sum_\ell\omega_\ell=1.
\tag{C4}
\]

基础版 `omega` 同权；特定方案可在独立训练／开发集上选择权重。所有温度、层集合和查询词固定后再做阈值校准。

概念流包含概念间 self-attention，因此增加或替换查询词不仅改变 softmax 分母，也可能改变 `C` 本身。扩大概念 bank 后重新标定；多目标方案不能只沿用单目标阈值并假定读出不变。

### 2.4 两个清楚的掩码基线

\[
M_n^{soft}(i)=\mathbf1[\bar H_{n,e}(i)>\theta],
\tag{C5}
\]

\[
M_n^{vote}(i)=\mathbf1\left[
|L_{loc}|^{-1}\sum_{\ell\in L_{loc}}
\mathbf1[H_{n,\ell,e}(i)>\theta_\ell]>1/2\right].
\tag{C6}
\]

主方案需要拒绝空目标时，使用方案一的图像级 gate 乘在以上区域掩码上。C6 的严格大于处理平票为 0。SD3.5 的 `attn2` 不额外获得一张层票；概念定位默认只读每个 block 的主 joint attention。

## 3. 安全供体与基本 patch

### 3.1 同 latent、同时间的供体

\[
O^-_{n,\ell}=\operatorname{ReadAV}_\ell
F_\theta(x_n,\sigma_n,p^-).
\tag{C7}
\]

这里不先沿 `p^-` 独立采样到另一 latent。供体分支输出 prediction 不推进主采样器。供体的 global pooled text 条件按 `p^-` 计算；噪声、guidance 数值和分辨率保持与原条件相同。

基础替换：

\[
O'_{n,\ell}(i)=O_{n,\ell}(i)
+\eta_nM_n(i)[O^-_{n,\ell}(i)-O_{n,\ell}(i)].
\tag{C8}
\]

`eta=1` 是硬替换；`eta=0` 不变。将 C8 的 `O` 取第二次实际干预前向的当前值，不能把第一次 probe 的旧 `O` 整块写回已被更早层改变的状态。

另设离线原型版 `O^- = mu_anchor[n,ell,i]`，对应现有实现，作为成本较低的基线。本文实时供体版本额外付出在线前向成本。

### 3.2 局部固定图像输入的补丁供体

方案三可使用局部反事实重放：在同一 block 图像输入 `X` 下，替换文本状态和文本调制条件，读取该 block 的安全 attention 写入：

\[
\widetilde O^-_{n,\ell}
=\operatorname{AV}_\ell
(X_{n,\ell}^{p},Y_{n,\ell}^{p^-},m_{n}^{p^-}).
\tag{C9}
\]

`Y^{p^-}` 来自安全条件的同状态前缀，`m` 包含模型实际使用的 pooled text 条件。C9 是局部组合干预；它不是一条完整 safe 轨迹，需与 C7 分别验证。

## 4. 最小基础算法 B0

```text
输入：冻结模型，user prompt p，target e，safe query a，背景查询 R，配对供体 p^-
固定：L_loc、末层 patch 点 ell*、采样时间窗 W、theta、eta
初始化 x_0 为原生采样噪声
for n = 0,...,N-1:
    若 n 不在 W：用原模型输出执行一次原生 scheduler step；continue
    从 x_n 运行 user＋只读 concept probe，保存各层 H、patch 点重放状态、原 prediction
    用 C4/C5 计算 M_n；或明确选择 C6 的多数投票基线
    若 M_n 为空：用原 prediction 更新一次 scheduler；continue
    从同一 x_n 运行安全供体 p^-，只缓存需要的 O^- 与条件
    在 ell* 处用 C8 替换，执行受影响的用户后缀，得到 edited prediction
    沿用原生 CFG 组合与 scheduler，仅推进一次 x_{n+1}
输出：最终图像；掩码、编辑点、强度和成本日志
```

`L_loc` 与 `L_edit` 是不同集合。末层版 `L_edit={ell*}`；全层版和稀疏版在相同定位图上比较。概念流只用于 probe；第二次干预前向若不再读图，可以关闭概念流。

## 5. 精确的前缀复用与成本

### 5.1 何时可以复用

只要 patch 点之前没有发生编辑，第一次用户前向到该点的状态，与重新运行用户模型到该点相同。缓存这一状态，再替换该点的 `AV` 并运行后缀，在确定性 eval 模式下与完整重跑对应。

缓存或 hook 必须覆盖后缀实际依赖的变量：图像／文本状态、skip／残差输入、时间与文本调制、位置编码以及并行 MLP 分量。关闭 dropout，避免将多次前向产生的额外随机数误用为下一步采样噪声。

若所有读出在末层 `AV` 形成时已齐备，且供体已缓存，也可直接在第一次用户前向该 hook 中计算最终掩码并 patch。这种调度不需要额外用户后缀重放。

不同于上述情形，若掩码定义还需要 patch 点之后的某层输出，就必须先完成 probe，再从 patch 点之前缓存的状态重放；不能在信息尚未齐备的 hook 中使用最终掩码。

### 5.2 成本公式

设 `K` 为实际编辑步数，`rho_n` 为重放后缀成本占同 batch 完整 DiT 前向的比例，`C_concept` 为新增概念 attention、MLP 和读出成本。

- 原模型：`N` 次模型前向。
- 末层版：用户 probe `N` 次＋安全供体 `K` 次＋用户后缀重放，等价约 `N+K+sum rho_n` 次前向，再加 `C_concept`。
- 掩码齐备后仍需重放大部分模型的早层／全层版：约 `N+2K` 次完整前向，再加概念成本。
- 使用离线均值供体时：去掉在线 `K` 次供体前向，但保留离线收集和缓存成本。

供体先行、末层当场 patch 的调度在每个启用步使用两次完整前向，但掩码为空时已经为供体付费。按需供体调度先 probe，再决定是否运行供体，并支付后缀重放。以实测时延决定选择。

这里按相同 batch／CFG 设置比较；SD3 CFG 的 negative／positive 双行虽可打包成一次 Python 调用，算力不能按单行计。报告调用数、处理样本行数、实际 FLOPs 或延迟三者，不能只数函数调用。

## 6. SD3.5 与 FLUX 的落点

### SD3.5 Medium

- 主 joint attention 使用独立图像和文本投影；概念借用文本侧 `add_q/k/v_proj`。
- 部分 block 有额外纯图像 `attn2`；它没有天然的文本概念输出。定位读主 `attn`，共享空间掩码能否用于 `attn2` 作为单独消融。
- 使用 CFG 时，基础版只编辑正向分支，保留原负向分支。最终组合 `v = v_u + w(v_c-v_u)` 会将正向改动放大 `w` 倍，强度必须在该 guidance 下校准。
- 最后 block 的 `context_pre_only` 不再递推文本。概念读出在其 attention 输出处仍可计算，但不能继续调用不存在的后续文本 MLP。

### FLUX.1

- 概念流先在 dual-stream blocks 递推；若只在这些层定位，`L_loc` 可固定在此范围。
- 模型最后仍有 single-stream blocks。末层 patch 默认指整个 transformer 的最后一个可用图像 attention 输出，而非最后一个双流 block。
- single-stream 融合实现中，attention 与 MLP 通道会共同进入 `proj_out`；只改图像行的 attention 通道，保留文本行与 MLP 通道。
- 概念流扩展到共享 single-stream 层是额外实现选择，主实验不依赖它即可做末层 patch。

初始分辨率 1024×1024 时，常见配置的图像 token 网格为 64×64。实际由 VAE scale 与 transformer patch size 推导，不硬编码跨模型通用 64。

## 7. 通用实验设计

### 数据分组

1. 目标明确出现的 prompt；
2. 目标隐式描述、别称、组合描述；
3. 与目标相近但应保留的概念；
4. 同图目标与保留主体共现，包含身份、姿态、背景约束；
5. 完全无目标的普通 prompt；
6. 概念词出现在文字排版、引用或否定中的情况，按预定视觉标注规范评价。

开发集选层和超参数，独立校准集定阈值，测试集只报告结果。场景模板和 seed 分组切分，避免同模板改几个词进入两侧。

### 指标

- 擦除：目标出现率、任务对应的内容／身份／对象／风格识别结果，人工复核抽样。
- 保留：近邻概念成功率、非目标人物身份／动作／姿态／布局、区域外图像变化、普通 prompt 遵循度。
- 触发：目标图像召回、无目标图像误触发、编辑区域面积。
- 代价：离线 GPU 时间、在线整体延迟与显存、DiT 等价计算量、供体分支数和缓存。
- 时序：目标风险随噪声水平曲线、首次编辑后的重现、编辑步数与累计修改量。

局部保留可以使用独立标注的目标区域或评估器生成的固定区域。不能用方法自己的掩码定义“区域外保留”，再据此证明掩码正确。评估器不复用方法的概念点积评分，避免同一读出同时当优化目标和成功标准。

### 最低比较矩阵

`原模型`、`原距离门控`、`距离多数投票末层`、`距离多数投票全层`、`CA软图＋末层替换`、`CA投票＋末层替换`、`上下文完整安全prompt生成`、`各方案自己的去组件消融`。

与公开方法比较优先使用当前模型可运行的 UVR／GLoCE／ActErase 等；跨 U-Net 与 DiT 的结果分开报告。NLCE、PER 和 FOMO 有直接机制参考价值，但原论文的架构和训练设置不能当作同模型同预算成绩直接排表。

所有强度在相同开发预算内选择，并报告擦除—保留曲线。仅在某一个固定强度上比较，不能区分算法变化与强度变化。

## 8. 首轮建议规模与决定顺序

建议先固定 SD3.5 Medium，一个局部对象／身份概念和一个属性概念；每类约 20 个开发 prompt、20 个校准 prompt、40 个留出测试 prompt，每 prompt 2–4 个 seed。这是工程与研究筛选预算，不用于稳定估计低概率安全失败率。

先跑无编辑 probe：检查 sidecar 是否影响原预测、主／辅助图像输出读出差异、空目标误触发。再用固定掩码比较供体与 patch 点；最后研究时序和多目标。大规模安全评估需另设更大样本与独立攻击集，并给出计数和置信区间。
