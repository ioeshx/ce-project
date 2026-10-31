# CWC 展开说明：符号、公式与完整流程

CWC 将离线测得的局部条件写入差分，编译为少数 DiT block 上的自动纠正模块。生成模型保持冻结；在线模块读取已有状态，判断是否触发，并输出当前图像 token 所需的低秩纠正。本说明延续 [方法正文](CWC_METHOD.zh-CN.md)，给出具体计算定义。擦除效果尚待实验验证。

先区分两个状态：`z` 是采样器更新的图像 latent；`X` 是 latent 进入 DiT 后，在某个 block 内的图像隐藏状态。CWC 在 `X` 所在的隐藏空间中干预，最终通过 DiT 的输出影响 latent 的采样轨迹。

**生成模型与基本符号。** 设冻结的 flow matching DiT 参数为 `θ`，离线配置的擦除集合为 `C`。

| 符号 | 定义 |
|---|---|
| \(\mathcal C=\{c_1,\ldots,c_K\}\) | 部署前确定的 K 个擦除概念 |
| \(p\) | 用户的普通生成 prompt |
| \(\sigma\) | 采样噪声水平；本文采用从 1 向 0 生成的约定 |
| \(z_\sigma\) | 噪声水平 σ 的图像 latent |
| \(v_\theta(z_\sigma,\sigma,p)\) | 原 DiT 输出的 velocity |
| \(\ell\) | DiT block 编号 |
| \(N,d_\ell\) | 图像 token 数、该 block 的隐藏维度 |
| \(X_\ell\in\mathbb R^{N\times d_\ell}\) | 进入该 block 的图像 token 矩阵 |
| \(Y_\ell\) | 进入该 block 的文本 token 矩阵 |
| \(m_\ell\) | 该 block 的调制条件，包含模型实际使用的时间、文本池化与 guidance 信息 |
| \(\mathcal S,\Omega_\ell\) | 入选 block 集合、block ℓ 的干预噪声时间窗 |

图像 token 在矩阵中按行存储。下文的小写 `x_{ℓj}`、`r_{ℓj}` 均按 `d_ℓ` 维列向量记，`j` 表示第几个图像 token。

设 block 的图像输出函数为 \(\mathcal B_\ell^{\mathrm{img}}\)。定义该 block 的图像写入残差：

\[
\mathcal R_\ell(X,Y;m)
=\mathcal B_\ell^{\mathrm{img}}(X,Y;m)-X,
\qquad
X_{\ell+1}=X_\ell+R_\ell.
\tag{1}
\]

其中 \(R_\ell=\mathcal R_\ell(X_\ell,Y_\ell;m_\ell)\)。对于带多个残差支路的 block，取整个 block 的图像输出减图像输入；双流和单流 block 都按此定义。单流 block 根据 token 索引提取图像部分。

这一定义给出了实际要修改的对象：本次 block 对图像状态增加的内容。

**离线测量目标条件引起的写入变化。** 每个离线样本准备一对 prompt：\(p^+\) 包含目标，\(p^-\) 去除目标并保留其他语义。多概念样本的 \(p^-\) 一次移除其中所有已配置且相关的目标。

例如，\(p^+\) 描述 Pikachu 坐在咖啡馆里，\(p^-\) 将角色换成普通卡通动物，保留咖啡馆、动作与构图。离线可以知道目标及替代条件；这些文本仅用于制作擦除包。

沿 \(p^+\) 的正常采样轨迹，在当前 \(z_\sigma\) 保存图像输入 \(X_\ell\)、原文本状态 \(Y_\ell^+\) 与原调制条件 \(m_\ell^+\)。再从同一个 \(z_\sigma\) 运行 \(p^-\) 的模型前缀，取得 \(Y_\ell^-\) 与 \(m_\ell^-\)。两次调用使用相同噪声水平、guidance 设置与图像位置编码。

在当前 block 做一次受控重放：

\[
\begin{aligned}
R_\ell^+
&=\mathcal R_\ell(X_\ell,Y_\ell^+;m_\ell^+),\\
R_\ell^{\mathrm{cf}}
&=\mathcal R_\ell(X_\ell,Y_\ell^-;m_\ell^-),\\
\Delta R_\ell
&=R_\ell^+-R_\ell^{\mathrm{cf}}.
\end{aligned}
\tag{2}
\]

`cf` 表示这里定义的局部条件替换。两次残差计算都使用相同的 \(X_\ell\)。\(m^-\) 中改变的是文本产生的调制分量，时间和 guidance 设置保持原值。

完整减去差分时，有精确的局部等式：

\[
X_\ell+R_\ell^+-\Delta R_\ell
=X_\ell+R_\ell^{\mathrm{cf}}.
\tag{3}
\]

因此，\(\Delta R_\ell\) 是把当前 block 的图像输出改成局部去目标输出所需的纠正。它描述的是所选配对条件带来的写入变化；语义专一性由配对质量和保留实验检验。

**先选择有效位置，再拟合纠正器。** 对候选 block 和噪声时间窗，先使用式（3）的精确替换，完成整条生成轨迹。该步只把重放得到的图像残差写回主轨迹，文本输出沿用正常前向结果。

根据最终图像的目标出现率与其他内容保留情况，选出 \(\mathcal S\) 和 \(\Omega_\ell\)。起步使用 2–4 个 block。这个实验直接判断：少数局部条件写入是否足以影响目标的最终生成。后续压缩与预测均以它为参照。

**用低秩基压缩差分。** 对选定 block，将不同 prompt、图像 token 和噪声水平的差分向量堆成列矩阵：

\[
D_\ell
=[\delta_{\ell 1}^{(1)},\delta_{\ell 2}^{(1)},\ldots]
\in\mathbb R^{d_\ell\times Q_\ell},
\qquad
D_\ell=A_\ell\Sigma_\ell B_\ell^\top,
\qquad
U_\ell=A_\ell[:,1:r].
\tag{4}
\]

\(\delta_{\ell j}\) 是 \(\Delta R_\ell\) 中第 j 个 token 的差分列向量；\(Q_\ell\) 是汇集的向量数；\(r\) 是纠正秩。差分矩阵不减均值，使平均纠正也参与低秩分解。

\(U_\ell\in\mathbb R^{d_\ell\times r}\) 的列正交，差分的低秩近似为：

\[
a_{\ell j}^*=U_\ell^\top\delta_{\ell j},
\qquad
\delta_{\ell j}\approx U_\ell a_{\ell j}^*.
\tag{5}
\]

在线预测 r 个系数即可构造 `d_ℓ` 维纠正。起步可令 `r=8`，通过精确重放与低秩重放的生成效果比较，判断这个秩是否足够。

**读取当前图像状态与本次写入。** 纠正量需要随当前样本变化。除 \(U_\ell\) 外，对缓存的图像输入做 PCA，取得上下文基 \(V_\ell\in\mathbb R^{d_\ell\times s}\)，起步 `s=8`。定义：

\[
q_{\ell j}
=\begin{bmatrix}
V_\ell^\top(x_{\ell j}-\mu_{x,\ell})\\
U_\ell^\top(r_{\ell j}-\mu_{r,\ell})
\end{bmatrix}
\in\mathbb R^{s+r}.
\tag{6}
\]

\(\mu_{x,\ell}\) 与 \(\mu_{r,\ell}\) 是离线缓存均值。第一部分提供当前图像状态的上下文，第二部分读取本次残差沿纠正子空间的分量。所需低维标准化参数也由离线数据确定，并在训练和部署时一致使用。

使用低维 token 的均值和逐维最大值形成全局读数：

\[
\eta_\ell
=\left[
\frac{1}{N}\sum_{j=1}^{N}q_{\ell j};
\operatorname{max}_{j}q_{\ell j};
\sigma
\right]
\in\mathbb R^{2(s+r)+1}.
\tag{7}
\]

平均值提供全局统计，逐维最大值补充局部较强的响应。门控使用全局读数；纠正头同时使用局部与全局读数。

**门控判断是否涉及已配置目标。** 在每个入选 block 上，用小预测头 \(G_{\phi_\ell}\) 输出 K 个相关性分数：

\[
s_{\ell k}
=\operatorname{sigmoid}\!\left(
[G_{\phi_\ell}(\eta_\ell)]_k
\right),
\qquad
g_\ell
=\mathbf 1\!\left[
\exists k:\ s_{\ell k}\ge\tau_{\ell k}(\sigma)
\right].
\tag{8}
\]

其中 \(\phi_\ell\) 是门控参数，\(\tau_{\ell k}(\sigma)\) 是离线校准阈值，可按少数噪声区间配置。分数表示与擦除任务的相关性，触发规则由目标召回与保留样本误触发的实测关系确定。

训练标签 \(b_k\in\{0,1\}\) 表示离线样本是否涉及第 k 个应阻断的视觉目标。显式目标、间接描述及近邻负例一起参与监督。门控损失可写为：

\[
\mathcal L_{\mathrm{gate}}
=\mathbb E_{\mathcal D}
\left[
\frac{1}{K}\sum_{k=1}^{K}
\operatorname{BCE}(s_{\ell k},b_k)
\right],
\tag{9}
\]

其中 BCE 为二元交叉熵，\(\mathcal D\) 是缓存样本集。目标词出现于文字排版、引用或否定表达时，标签按所定义的视觉任务标注。部署时 K 个输出的含义已经固定在擦除包中。

**纠正头预测当前 token 的差分系数。** 用小型两层 MLP \(F_{\psi_\ell}\) 预测：

\[
\widehat a_{\ell j}
=F_{\psi_\ell}([q_{\ell j};\eta_\ell])
\in\mathbb R^r.
\tag{10}
\]

\(\psi_\ell\) 是纠正头参数。目标样本的监督为式（5）的 \(a^*\)，保留样本的监督为零。设 \(\mathcal D_+\) 是目标样本，\(\mathcal D_0\) 是保留样本，则：

\[
\begin{aligned}
\mathcal L_{\mathrm{corr}}
={}&\mathbb E_{\mathcal D_+}
\left[\frac1N\sum_j
\|\widehat a_{\ell j}-U_\ell^\top\delta_{\ell j}\|_2^2\right]\\
&+\lambda_{\mathrm{keep}}\,
\mathbb E_{\mathcal D_0}
\left[\frac1N\sum_j
\|\widehat a_{\ell j}\|_2^2\right].
\end{aligned}
\tag{11}
\]

\(\lambda_{\mathrm{keep}}\) 控制保留样本在拟合中的权重。样本级均值可以避免高分辨率样本因 token 更多而自动占据更大权重。

门控负责决定是否触发；纠正头负责预测触发后的变化。门控误触发时，保留样本上的零纠正监督仍提供第二层保留约束。所有梯度仅更新 \(\phi_\ell,\psi_\ell\)，\(\theta,U_\ell,V_\ell\) 均固定。拟合过程使用缓存，无需穿过 DiT 或完整采样器反向传播。

**在线只执行一次低秩写回。** 定义时间窗开关 \(w_\ell(\sigma)=\mathbf1[\sigma\in\Omega_\ell]\)。在选定位置，图像输出为：

\[
\boxed{
x'_{\ell+1,j}
=x_{\ell j}+r_{\ell j}
-w_\ell(\sigma)\,g_\ell\,
U_\ell F_{\psi_\ell}([q_{\ell j};\eta_\ell])
}.
\tag{12}
\]

时间窗关闭时直接跳过模块；门控关闭时沿用原 block 输出；门控开启时计算纠正头并写回。门控按当前图像样本判断，纠正量逐 token 预测，因此不同位置可以得到不同更新。

模块使用的全部变量来自当前正常前向、当前噪声水平和预存参数。在线无需构造 \(p^-\)、目标文本 embedding 或额外模型分支。

设插入这些模块后整台 DiT 的 velocity 为 \(v_{\theta,\mathcal P}\)，其中 \(\mathcal P\) 表示擦除包。将 velocity 记为对当前噪声坐标的导数，即 \(dz_\sigma/d\sigma=v_{\theta,\mathcal P}\)，则 Euler 更新为：

\[
z_{\sigma_{n+1}}
=z_{\sigma_n}
+(\sigma_{n+1}-\sigma_n)
v_{\theta,\mathcal P}(z_{\sigma_n},\sigma_n,p),
\qquad \sigma_{n+1}<\sigma_n.
\tag{13}
\]

具体实现沿用 checkpoint 与 scheduler 的原始 velocity 符号和积分规则。FLUX.1-dev 沿用其 guidance 条件；显式 CFG 模型在现有条件分支上纠正，按原公式组合条件与无条件 velocity，并在相同 guidance 下标定。

**多概念使用联合监督与一个总纠正。** 对样本 a，设 \(\mathcal A_a\subseteq\mathcal C\) 为相关目标集合，\(p_a^{-\mathcal A_a}\) 为一次移除这些目标的离线 prompt。联合监督为：

\[
\Delta R_\ell^{\mathrm{joint}}
=\mathcal R_\ell(X_\ell,Y_\ell(p_a^+);m_\ell(p_a^+))
-\mathcal R_\ell(X_\ell,Y_\ell(p_a^{-\mathcal A_a});m_\ell(p_a^{-\mathcal A_a})).
\tag{14}
\]

将单目标与组合样本的差分一起用于建立 \(U_\ell\)，再训练同一个 \(F_{\psi_\ell}\)。门控输出 K 个分数，纠正头输出一个 r 维总纠正。它的在线输入仍是式（10）的状态读数。

例如，\(\mathcal C=\{c_1,c_2\}\) 时，数据至少覆盖仅有 \(c_1\)、仅有 \(c_2\)、两者共同出现、两者均不出现，以及目标与保留概念共现。组合样本直接提供共同移除后的差分，不需要设定单目标差分之间的可加关系。

`K=1` 时，上述过程自然退化为单概念方法。增加目标需要更新离线缓存、检查所需位置与秩，并重新拟合联合模块。

**一次补采覆盖编辑后的状态。** 在拟合模块后运行少量完整采样。对出现偏差的样本，保存前面模块干预后到达的新状态，再在该状态上测量式（2），补入缓存并重新拟合一次。

补采使用该请求原本的擦除标签，纠正系数则由新的局部重放计算。这样监督持续表达“该条件还会写入多少需要纠正的分量”。最终目标成功消失本身不作为将标签改成保留样本的依据。

**在线成本来自低秩读写。** 一个启用位置的主要投影成本为：

\[
O\!\left(Nd_\ell(s+2r)\right),
\tag{15}
\]

分别对应读入 \(V^\top x\)、读入 \(U^\top r\) 和写回 \(U\widehat a\)，小 MLP 与门控另计。时间窗外直接跳过，门控关闭时省去纠正头与写回。采样过程的额外完整 DiT 前向次数为零。

若 `d_ℓ=3072`、`r=s=8`，两组基矩阵共有 `3072×(8+8)=49,152` 个参数，BF16 存储约 96 KiB／block；4 个位置约 384 KiB，小预测头、统计量和临时激活缓冲分别记录。实际延迟用同一硬件与采样设置测量。

**完整执行顺序。** 离线依次完成配对与保留样本准备、精确局部替换定位、差分 SVD 与输入 PCA、小预测头拟合、一次编辑轨迹补采。将位置、时间窗、基矩阵、统计量、门控阈值和预测头保存为擦除包。在线依次执行正常 prompt 编码、原 DiT 前向、选定位置的式（12）、原 velocity 输出与原 scheduler 更新。

实验应依次比较精确局部替换、低秩替换和预测纠正。三者分别回答干预位置是否有效、差分是否能压缩、缓存小模块能否替代局部重放；自动触发和联合概念表现再通过显式／间接表达、近邻负例与组合样本检验。
