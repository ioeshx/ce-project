# Attention Replacement：基于概念原型的注意力输出替换

本方法在冻结的文本生成图像 Diffusion Transformer 中施加推理干预。离线分别构建目标概念与锚概念的注意力输出均值；在线根据图像 token 到两类原型的距离生成空间掩码，将命中位置的注意力输出替换为锚概念原型，以抑制目标概念的生成。

**1. 符号与干预对象**

| 符号 | 含义 |
| --- | --- |
| $F_\theta$ | 参数固定的基础生成模型 |
| $\{(c_i,a_i)\}_{i=1}^{m}$ | $m$ 对目标概念 prompt 与锚概念 prompt |
| $N$ | 每个概念用于离线平均的随机 seed 数 |
| $k\in\{0,\ldots,T-1\}$ | 按实际执行顺序编号的采样步；$T$ 为总步数 |
| $s=(\ell,b)\in\mathcal S$ | 选定的干预位置，由 block 编号与 attention 分支组成 |
| $p$ | 图像 token 的二维网格位置 |
| $X_{k,s}(p)\in\mathbb R^{d_s}$ | 输出投影之前，该位置的多头 attention 输出向量 |

对第 $j$ 个 attention head，记模型实际使用的 query、key、value 为 $Q_j,K_j,V_j$，则干预对象为

$$
A_j=\operatorname{softmax}\!\left(\frac{Q_jK_j^\top}{\sqrt{d_j}}\right),
\qquad
X_{k,s}=\left[\operatorname{Concat}_{j}(A_jV_j)\right]_{\mathrm{image}}.
$$

其中 $Q_j,K_j$ 沿用模型原生的位置编码与归一化，$[\cdot]_{\mathrm{image}}$ 表示选取图像 token 行。$X_{k,s}$ 的完整形状为 $H_s\times W_s\times d_s$，各 head 的输出在通道维连接后参与距离计算。

**2. 离线阶段：构建概念原型**

对每个短概念 prompt $q\in\{c_i,a_i\}$，使用同一组 $N$ 个 seeds 独立运行完整的原生去噪轨迹。目标与锚概念使用对应 seed 的相同初始噪声，各自沿正常生成轨迹前向。

记第 $n$ 个 seed 的注意力输出为 $X^{q,n}_{k,s}(p)$，在相同采样步、干预位置和空间索引上求均值：

$$
\mu^{q}_{k,s}(p)
=\frac{1}{N}\sum_{n=1}^{N}X^{q,n}_{k,s}(p).
$$

由此得到目标原型 $\mu^{c_i}_{k,s}$ 和锚原型 $\mu^{a_i}_{k,s}$。同一配置下，相同锚 prompt 共享一份原型。离线按概念在 CPU 中以 FP32 累计，只保存最终均值和必要元数据。

**3. 在线阶段：距离判定与空间掩码**

输入用户 prompt $y$，在选定的 $(k,s)$ 处取得当前图像侧输出 $x_{k,s}(p)$。该输出来自当前采样轨迹，包含此前干预的影响。以下省略 $(k,s)$ 下标。

对每个目标—锚概念对，计算逐位置的平方 L2 距离：

$$
D_i^{\mathrm{tar}}(p)
=\left\|x(p)-\mu^{c_i}(p)\right\|_2^2,
\qquad
D_i^{\mathrm{anc}}(p)
=\left\|x(p)-\mu^{a_i}(p)\right\|_2^2.
$$

距离按全部通道求和，以 FP32 计算。定义相对接近目标的分数

$$
\Delta_i(p)=D_i^{\mathrm{anc}}(p)-D_i^{\mathrm{tar}}(p).
$$

结合相对距离阈值 $\tau_i$ 和目标距离上限 $u$，构造掩码：

$$
M_i(p)
=\mathbf 1\!\left[\Delta_i(p)>\tau_i\right]
\cdot\mathbf 1\!\left[D_i^{\mathrm{tar}}(p)\le u\right].
$$

当 $\tau_i=0$ 时，第一项表示当前位置相对更接近目标原型；第二项限制它与目标原型的绝对距离。$D_i^{\mathrm{tar}}>u$ 时，该目标的掩码为 0。$u$ 是所有概念共用的上限，未配置时视为 $+\infty$。$\tau_i$ 可按概念指定，未指定的概念使用全局 $\tau$。

**4. 多概念选择与硬替换**

所有概念均在同一个替换前输出 $x$ 上判定。将掩码取并集，并记录当前位置命中的概念集合：

$$
M(p)=\bigvee_{i=1}^{m}M_i(p),
\qquad
\mathcal C(p)=\{i\mid M_i(p)=1\}.
$$

当 $\mathcal C(p)\ne\varnothing$ 时，选取命中概念中分数最大的一个：

$$
w(p)=\underset{i\in\mathcal C(p)}{\arg\max}\;\Delta_i(p).
$$

分数相同则取配置顺序靠前的概念；无概念命中时记 $w(p)=-1$。随后执行完整 token 向量的硬替换：

$$
\widetilde x(p)=
\begin{cases}
\mu^{a_{w(p)}}(p), & M(p)=1,\\
x(p), & M(p)=0.
\end{cases}
$$

将 $\widetilde x$ 转回模型原生精度，送入后续输出投影。掩码外位置保留本次前向的当前输出。SD3 离线与在线均取 CFG 条件分支的图像输出，并在此分支替换；FLUX 单流层保留文本行与并行 MLP 通道，仅替换图像侧 attention 输出。

**5. 干预窗口与完整生成流程**

预先指定 block 范围及其 attention 分支 $\mathcal S$，并指定采样步窗口

$$
\mathcal K=\{k_s,\ldots,k_e-1\}.
$$

按前段比例 $\rho$ 设置窗口时，取 $k_s=0$、$k_e=\lceil\rho T\rceil$。例如 $T=40,\rho=0.5$ 对应前 20 步，即 $[0,20)$。这里的比例指采样执行步数。

在线沿原生采样顺序前向：在 $k\in\mathcal K,s\in\mathcal S$ 时依次计算距离、生成掩码、选择锚原型并替换；其余位置沿用原生计算。完整去噪结束后，由原生 VAE 解码得到图像。在线只运行用户 prompt 的生成过程，原型从离线缓存读取。

主要超参数为 $N$、$\mathcal S$、$\mathcal K$ 或 $\rho$、$\tau_i$ 和 $u$。改变 $\tau_i$ 或 $u$ 可复用已有原型；采样轨迹配置与离线阶段保持一致，在线窗口可取已收集范围的子集。

在线保存 $D_i^{\mathrm{tar}},D_i^{\mathrm{anc}},\Delta_i,M_i,M,w$ 及修改量标量，用于分析空间命中和层—时间分布。完整输出向量在前向中即时使用。模型权重始终固定，目标概念抑制与无关内容保留的程度由生成结果和 benchmark 指标衡量。
