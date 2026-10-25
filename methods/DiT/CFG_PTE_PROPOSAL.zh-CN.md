# 保留约束的 CFG 概率流擦除

**工作名：CFG-PTE — Preservation-constrained Transport Erasure under CFG**  
**日期：2026-10-24；阶段：一种方法的研究提案，尚无真实 DiT 实验结果。**

## 0. 方法概括与研究判断

考虑一个冻结的 flow-matching 文生图 DiT。我们只取得条件与空条件的最终 velocity 输出，构造真实 CFG 速度场。在它实际产生的分布上，定义一个降低目标概念风险、同时保持非目标语义统计的 KL 最小改动分布。随后从连续性方程推导出实现这个分布变化所需的附加 velocity：它是一个带保留约束的概率质量输运，而不是先指定某个负概念方向再调强度。离线用粒子样本和外部语义探针求解输运教师，训练外置控制器，部署时直接输出修正后的 velocity。

一个核心关系是：**在保持所选非目标语义统计时，风险随擦除强度下降的速率，等于风险无法被这些保留语义线性解释的剩余方差。** 因而方法同时提供“如何改 velocity”和“哪些概念／保留要求组合难以兼顾”的分析工具。

建议将论文主线限定在三个相互关联的候选贡献：

1. **保留约束的擦除分布与可行性诊断**：在实际 CFG 分布上定义信息投影，推导风险下降与语义纠缠之间的关系。
2. **满足连续性方程的 velocity 输运构造**：利用基准 CFG 续程语义，将需要的质量变化转化为最小瞬时能量的速度修正，并给出可计算的弱形式教师。
3. **风险预算校准的 CFG 连续族**：利用风险剩余方差求解不同 CFG 所需的擦除强度，再学习相应修正场及其响应，研究未训练 CFG 值上的擦除与保留稳定性。

这三项是**待验证的创新主张**。信息投影、加权 Poisson 方程、Galerkin 近似和蒸馏本身均有成熟先例；不能将这些数学工具重新命名后当成独立创新。第三项尤其需要与“只把 CFG 当输入＋随机采样 CFG”的强消融区分。如果没有额外收益，应收缩该项主张。

### 与本次要求的对应

| 要求 | 本方案的选择 | 需要明确的条件 |
| --- | --- | --- |
| Flow Matching DiT 优先 | 全部主实验仅使用 flow-matching 文生图 DiT | 统一 velocity 与时间方向后接入各自原生求解器 |
| 不依赖 DiT 内部结构 | 仅访问状态、文本条件、时间及最终 velocity；外部模块不读取隐藏层 | 架构无关指算法接口，不代表同一套控制器权重可跨不同 latent 格式直接复用 |
| 在 velocity 上操作并依赖 CFG | 对 `v_uncond + gamma*(v_cond-v_uncond)` 加控制场 | 必须存在可用的真实两分支 CFG，不能把 guidance embedding 当成它 |
| 有工作量与创新 | 分布定义／可行性、输运求解、CFG 连续族学习 | 要证明完整问题的价值，不依靠小模块数量凑贡献 |
| 理论与可视化 | 风险单调性、保留约束、连续性方程、区间稳定性；配套诊断图 | 理论对所选风险与统计成立，不自动升级为视觉语义或对抗鲁棒性的保证 |

**方法类别**：训练外部模块＋推理期 velocity 控制。生成器权重冻结；控制器是部署组成部分。本文所称擦除是该系统的生成行为改变，不声称基模型参数中已永久删除概念。

## 1. 为什么从分布与输运入手

已有方法已经覆盖了很多直接思路：DVE 在输出速度上构造目标与 anchor 的差分方向并进行选择性纠正；GEM 将教师的吸引／排斥信号用于几何擦除；BEM 学习 velocity 回归所用的 Bregman 几何。[R1–R3] 因而“负 CFG”“正交投影”“对目标方向换一个距离”都不足以单独构成这里的研究主线。

本方案首先追问：**如果某个提示还包含场景、物体关系、颜色等应该保留的信息，理想擦除后的生成分布究竟是什么？** 只给一个负方向无法回答这个问题。我们用可测量的保留统计定义它，再推导相应的 velocity。这样，擦除与保留的冲突、求解误差和 CFG 变化都能进入同一个框架。

选择分布级保留意味着不承诺每个 seed 的图像布局完全不变。相同 seed 的内容保持仍需作为独立实验指标；最小速度能量也不等价于最小感知变化。

## 2. 接口与符号：先正确处理 CFG

采用噪声到数据的时间约定：

\[
t\in[0,1],\qquad z_0\sim\mathcal N(0,I),\qquad \dot z_t=b_t^\gamma(z_t,c).
\]

模型原生时间若相反，应在适配层同时变换时间与 velocity 符号。不要直接将本文件的符号代入原生 scheduler 再反转一次。

设冻结模型输出

\[
v_0=v_\theta(z,t,\varnothing),\quad
v_c=v_\theta(z,t,c),\quad
g=v_c-v_0,\quad
b_t^\gamma=v_0+\gamma g. \tag{1}
\]

\(\gamma\) 是**外部真实 CFG 系数**。如果模型还接受原生 guidance embedding，先将其固定为记录在实验 manifest 中的 \(g_{\rm native}\)，式 (1) 两次调用均使用同一设置。FLUX API 的 `guidance_scale` 与 `true_cfg_scale` 不能混用；SD3 系列的 CFG 接口也需通过实际两分支调用核对。[R11–R12]

令 \(p_t^\gamma\) 为式 (1) 的 ODE 从高斯初始分布实际推送出的分布。本方案始终使用这个定义，**不假定**

\[
p_t^\gamma\propto p_t(\cdot\mid c)^\gamma p_t(\cdot)^{1-\gamma}.
\]

线性组合输出并不能一般性地保证该密度等式；近期相关工作也明确讨论了 CFG 与目标幂次分布的差异。[R10] 本方案不需要知道 \(p_t^\gamma\) 的解析密度，只需要从原始 CFG 采样。

限制：某些蒸馏模型的空条件分支没有充分训练。结构上可调用两分支不代表该分支可靠。应先验证 CFG 可用性；不能声称本方法天然适配所有 flow-matching 检查点。

## 3. 定义要擦除什么、要保留什么

### 3.1 终点风险与保留语义

对解码后的终点图像定义：

- \(R_F(x)\in[0,1]\)：目标概念风险分数。它由外部图像评估器或人工标注建立，需明确概念边界。
- \(F_c(x)\in\mathbb R^m\)：与提示有关的非目标语义统计，例如非目标物体、属性、背景和关系。排除目标本身及删除目标必然改变的属性。

可将二阶特征或随机特征加入 \(F_c\)，但保持有限维统计并不等于保持完整非目标分布。训练评估器与最终测试评估器应独立；否则可能只是在利用同一评估器的漏洞。

对于“目标人物在海边、旁有一只狗”，可以保护海边场景、狗及其属性，而不能同时强制保留目标人物身份。“完全保留全部语义再删除其中一项”可能本身不可行。仅包含目标、没有可保护内容的提示，可取空保留向量。

### 3.2 基于原始 CFG 续程的语义坐标

令 \(\Phi_{t\to1}^\gamma\) 表示从当前状态继续运行**原始** CFG 场直至终点的映射，VAE 解码并入 \(R_F,F_c\)。定义

\[
r_t^\gamma(z)=R_F(\Phi_{t\to1}^\gamma(z)),\qquad
f_t^\gamma(z)=F_c(\Phi_{t\to1}^\gamma(z)). \tag{2}
\]

它们描述“从此状态按原始 CFG 继续生成会得到什么”，在原始轨迹上满足

\[
D_b r=0,\quad D_b f=0,\qquad
D_b:=\partial_t+b_t^\gamma\cdot\nabla_z. \tag{3}
\]

这里的 \(r\) 是确定性终点分数，不称为未来事件概率。式 (2) 也不是“用当前一步预测的 \(\hat x_1\) 直接打分”。后者便宜，但一般不满足式 (3)。

设

\[
m_\gamma=\mathbb E_{p_1^\gamma}F_c
           =\mathbb E_{p_t^\gamma}f_t^\gamma.
\]

实际通过外部探针近似 \(r,f\)。探针默认只读 \((z,t,c,\gamma)\) 与外部概念描述，避免任何隐藏层依赖。它的输入梯度可直接计算，无需反传穿过 DiT。训练标签来自完整原始 CFG 续程。探针不是本方案单独的创新，应重点验证其续程一致性与校准误差。

## 4. 候选贡献一：保留约束的信息投影

对每个提示和 CFG 系数，定义中间目标分布

\[
q_t^\gamma(z)=\frac{p_t^\gamma(z)}{Z_t^\gamma}
 \exp\{-\beta(t)r_t^\gamma(z)-\lambda_t^{\gamma\top}f_t^\gamma(z)\}, \tag{4}
\]

满足

\[
\mathbb E_{q_t^\gamma}f_t^\gamma=m_\gamma,\qquad
\beta(0)=0,\quad\lambda_0^\gamma=0. \tag{5}
\]

因此 \(q_0^\gamma=p_0^\gamma=\mathcal N(0,I)\)：初始噪声分布保持不变。\(\beta\) 是擦除强度日程，例如先试 \(\beta(t)=\beta_{\max}t^2\)，但不预先声称这个日程最优。

在可行且有有限对偶解的条件下，式 (4) 是以下约束问题的解：

\[
\min_q\operatorname{KL}(q\|p_t^\gamma),\quad
\text{s.t. }\mathbb E_q r_t^\gamma\le\kappa_t,\quad
\mathbb E_q f_t^\gamma=m_\gamma. \tag{6}
\]

\(\beta\ge0\) 是风险约束乘子。指定 \(\beta\) 时，\(\kappa_t\) 是隐含得到的风险水平；指定风险预算时需另解对偶问题。不能同时任意指定二者。

有限 \(\beta\) 对风险进行软抑制，通常不使目标事件概率严格为零。如果原始分布几乎没有满足保留要求的低风险样本，重加权也不会凭空创造可靠的替代语义。

### 4.1 对偶求解

固定 \(t,\gamma,c\)，求

\[
\lambda^*=\arg\min_\lambda
\left[\log\mathbb E_p e^{-\beta r-\lambda^\top f}
      +\lambda^\top m_\gamma\right]. \tag{7}
\]

梯度为 \(m_\gamma-\mathbb E_qf\)，Hessian 为 \(\operatorname{Cov}_q(f,f)\)。可用低维 Newton 法加线搜索。去除冗余特征后若协方差仍退化，需要诊断不可辨识性；使用岭正则或软保留约束属于实际近似，应报告相应保留残差。

### 4.2 风险下降与语义纠缠：可证明的结论

记

\[
\Sigma_{ff}=\operatorname{Cov}_q(f,f),\quad
C_{fr}=\operatorname{Cov}_q(f,r),
\]

\[
r_\perp=(r-\mathbb E_qr)
 - C_{fr}^{\top}\Sigma_{ff}^{-1}(f-m_\gamma). \tag{8}
\]

这里是**语义随机变量在分布中的回归残差**，不是对 DiT velocity 做欧氏正交投影。若 \(\Sigma_{ff}\succ0\)，则沿保留约束的对偶路径

\[
\frac{d\lambda}{d\beta}=-\Sigma_{ff}^{-1}C_{fr}, \tag{9}
\]

\[
\boxed{
\frac{d\mathbb E_qr}{d\beta}
=-\left[\operatorname{Var}_q(r)
 - C_{fr}^{\top}\Sigma_{ff}^{-1}C_{fr}\right]
=-\mathbb E_q[r_\perp^2]\le0.} \tag{10}
\]

**证明要点**：对式 (4) 求导得到指数族协方差恒等式；对 \(\mathbb E_qf=m_\gamma\) 求导得式 (9)，代回风险导数即可。方括号是协方差矩阵的 Schur 补，非负。

**可分析的问题**：保护哪些语义会明显压缩擦除余地？不同目标概念的这个量有何差异？其随 CFG 的变化是否解释高 CFG 下的保留损伤？可以将剩余方差或归一化解释率画成热图，并与实测 Pareto 曲线关联。

**边界**：它衡量所选统计下的局部风险下降能力；不是所有视觉语义的全局不可擦除定理。若剩余方差为零，说明当前风险在该分布上由所选保留特征的仿射组合解释；错误或过强的保留特征也会制造这种冲突。

## 5. 候选贡献二：由质量变化推导 velocity 修正

### 5.1 连续性方程给出的质量源项

原始 CFG 分布满足

\[
\partial_t p+\nabla\cdot(pb)=0.
\]

在理想续程语义的式 (3) 下，式 (4) 满足

\[
\partial_tq+\nabla\cdot(qb)=q h_t,\qquad
\boxed{h_t=-\dot\beta(t)r_\perp.} \tag{11}
\]

这是将 \(\dot\lambda=-\dot\beta\Sigma_{ff}^{-1}C_{fr}\) 代入归一化指数权重的物质导数得到的。\(\mathbb E_q h=0\)，且 \(\mathbb E_q[f h]=0\)。当风险残差为正时，\(h<0\)，对应降低该处的相对概率质量。

只对样本重加权并不能给单条 ODE 轨迹一个可执行的 velocity。我们寻找 \(u_t\)，使

\[
\partial_tq+\nabla\cdot(q(b+u))=0,
\quad\text{即}\quad
\boxed{\nabla\cdot(q u)=-q h.} \tag{12}
\]

于是部署速度为

\[
\boxed{v_{\rm erase}=v_0+\gamma(v_c-v_0)+u_t^\gamma.} \tag{13}
\]

### 5.2 最小瞬时能量解与理论范围

在给定 \(q_t\) 路径的条件下求

\[
\min_u\frac12\mathbb E_q\|u\|^2,
\quad\text{s.t. }\nabla\cdot(qu)=-qh. \tag{14}
\]

在足够光滑、边界通量消失、相关加权 Poisson 问题可解的条件下，最小范数解为

\[
u=\nabla\psi,\qquad
\nabla\cdot(q\nabla\psi)=-qh,\qquad
\mathbb E_q\psi=0. \tag{15}
\]

**证明要点**：约束的拉格朗日函数对 \(u\) 变分给出 \(u=\nabla\psi\)。任意另一可行场可写为 \(\nabla\psi+w\)，其中 \(\nabla\cdot(qw)=0\)，分部积分给出 \(\mathbb E_q\langle\nabla\psi,w\rangle=0\)，能量因此增加 \(\tfrac12\mathbb E_q\|w\|^2\)。

若该场精确求解，连续性方程唯一，且初始分布相同，则修正 ODE 的边缘分布为式 (4)。在终点，\(f_1=F_c,r_1=R_F\)，从而满足所定义的保留约束与风险变化。

这里的“最小”仅指**给定中间分布路径的瞬时附加速度能量**。它不等于所有可能擦除路径中的全局最优输运，也不保证同 seed 图像的最小感知改动。多峰分布的狭窄连接区域还可能导致大输运能量；这是需要分析的失败模式。

加权 Poisson 与粒子 Galerkin 求解在反馈粒子滤波中已有直接先例。[R8] 本方案应引用并复用其基础，不把式 (15) 本身作为新定理。研究增量在于：保留约束的 CFG 擦除问题产生了什么源项、如何估计与摊销、能否得到更好的擦除／保留折中。

### 5.3 不估计高维密度的弱形式

对测试函数 \(\phi\)，式 (15) 等价于

\[
\mathbb E_q\langle\nabla\phi,\nabla\psi\rangle
=\mathbb E_q[\phi h]. \tag{16}
\]

取 \(\psi_K(z)=\sum_{j=1}^K a_j\phi_j(z)\)，得到

\[
A_{ij}=\mathbb E_q\langle\nabla\phi_i,\nabla\phi_j\rangle,
\quad d_i=\mathbb E_q[\phi_i h],\quad
(A+\epsilon I)a=d, \tag{17}
\]

\[
u_K(z)=\sum_j a_j\nabla\phi_j(z).
\]

从原始 CFG 粒子 \(z_i\sim p_t^\gamma\) 得到权重

\[
w_i=\operatorname{softmax}_i[-\beta r(z_i)-\lambda^\top f(z_i)],
\]

即可近似式 (7)、(17) 的期望，不需要估计 \(\nabla\log p\) 或高维密度。

有限基只保证相应测试函数上的弱约束；不能据此声称已经精确采样式 (4)。基函数可从风险／保留探针输出及其非线性组合开始，初试 \(K=8,16,32\)，通过额外未参与求解的测试函数检查残差。若增加 \(K\) 仍无法改善真实保留与风险，应停止将“小矩阵可解”当作可行性证据。

风险与保留特征是低维的，不代表真实 Poisson 解也处在相同低维函数空间。若固定基不足，可在同一弱求解器内学习共享基函数，最小化 Dirichlet 变分目标

\[
\mathcal J(\psi)=\tfrac12\mathbb E_q\|\nabla\psi\|^2-\mathbb E_q[\psi h].
\]

这仍属于式 (15) 的数值近似，不另列为创新。训练／验证粒子必须分开，避免有限粒子上的虚假低能量解。应与同容量的普通风险梯度控制器比较，确认改善来自输运约束。

### 5.4 学习探针时不能省略的导数

实际 \(\hat r,\hat f\) 不严格满足式 (3)。定义

\[
\ell_t=-\beta(t)\hat r_t-\lambda_t^\top\hat f_t.
\]

对相应的归一化倾斜路径，正确源项为

\[
\boxed{h_t=D_b\ell_t-\mathbb E_q[D_b\ell_t].} \tag{18}
\]

它包含 \(D_b\hat r,D_b\hat f\) 和 \(\dot\lambda\)。只有在续程一致性足够好时，才能近似为式 (11)。直接丢掉这些项却继续宣称精确分布保证，会造成理论与实现不一致。

默认探针只读 \(z,t,c,\gamma\)，式 (18) 的 JVP 仅涉及外部探针，\(b\) 通过 DiT 前向获得，不需要 DiT 的参数梯度或内部钩子。也可用沿原始场的一步有限差分检查物质导数；步长偏差要报告。如果额外把 velocity 作为探针输入，精确求导便还需处理其状态依赖，不能静默 `detach` 后沿用精确结论。

主实验应同时报告“完整源项教师”和“近似运输一致探针的简化教师”的差异，验证式 (11) 的效率近似是否合理。

**初始分布还需要额外核对。** 学习到的 \(\hat f\) 不满足精确续程关系时，若仍把所有时刻的目标矩硬设成真实终点矩 \(m_\gamma\)，可能在 \(t=0\) 就需要 \(\lambda_0\ne0\)，破坏高斯初始条件。实际教师应使用随时间变化的基准矩

\[
\hat m_\gamma(t)=\mathbb E_{p_t^\gamma}\hat f_t,
\]

从而 \(\beta(0)=\lambda_0=0\) 可行，再用完整式 (18) 处理它产生的对偶变化。理想探针下它退化为常数 \(m_\gamma\)。终点真实语义矩偏差至多为两项探针误差之和：

\[
\|\mathbb E_{q_1}F_c-\mathbb E_{p_1}F_c\|
\le\mathbb E_{q_1}\|F_c-\hat f_1\|
   +\mathbb E_{p_1}\|F_c-\hat f_1\|,
\]

其中暂假设学习特征的矩约束被精确求解；有限粒子和求解残差还需另加。它明确说明：守住探针统计不等于守住真实语义，特别要检查受控分布上的误差。

## 6. 候选贡献三：风险预算校准的 CFG 连续族

不同 \(\gamma\) 下，原始场、参考分布、续程风险、保留矩和对偶解均会改变：

\[
(b^\gamma,p^\gamma,r^\gamma,f^\gamma,m_\gamma,\lambda^\gamma,u^\gamma).
\]

因此，不应把在一个 CFG 上求得的修正向量乘以比例后用于所有 CFG。以 \(\gamma\) 为条件训练控制器是必要基线，但本方案进一步研究其**轨迹响应一致性**。

为避免不同 CFG 只是在使用不同的任意擦除强度，可以指定共同的终点风险预算 \(\kappa\)，对每个 CFG 解

\[
\mathcal R(\beta_\gamma^*,\gamma)
:=\mathbb E_{q_1^{\gamma,\beta_\gamma^*}}R_F=\kappa.
\]

若预算可达、约束激活且剩余方差 \(\Omega=\mathbb E_qr_\perp^2>0\)，由式 (10) 与隐函数定理

\[
\boxed{\frac{d\beta_\gamma^*}{d\gamma}
=\frac{\partial_\gamma\mathcal R(\beta,\gamma)|_{\beta=\beta_\gamma^*}}
{\Omega(\beta_\gamma^*,\gamma)}.}
\]

这把 CFG 强度校准与贡献一的擦除／保留可行性直接关联：剩余方差很小时，为维持同一风险预算，所需强度可能对 CFG 极敏感。分子包含参考分布与续程观测随 CFG 的完整变化，不能只对式 (1) 的线性系数求导。离线可用共同噪声的相邻 CFG 估计该响应，利用单调性求根，再将 \(\beta(t)\) 替换为 \(\beta_\gamma^*b(t)\)，其中 \(b(0)=0,b(1)=1\)。

如果原始风险已低于预算，可取零干预；根不存在或强度／ESS 超出可用范围时，标记该提示与 CFG 的预算不可实现，不声称统一保证。无约束的固定 \(\beta_{\max}\) 路径保留为基线。控制器摊销的是这个经校准的教师族，未见提示仍只有经验泛化能力。

令学生的总场为 \(B^\gamma=b^\gamma+u_\varphi^\gamma\)，同一初始噪声下

\[
s_t^\gamma=\partial_\gamma z_t^\gamma,
\]

满足

\[
\dot s_t^\gamma
=J_z B_t^\gamma\,s_t^\gamma
 +(v_c-v_0)+\partial_\gamma u_\varphi^\gamma,
\qquad s_0^\gamma=0. \tag{19}
\]

式 (19) 假设式 (1) 的两分支在改变外部 \(\gamma\) 时固定，包括固定 \(g_{\rm native}\)。若同时改变原生 embedding，这个导数就不再适用。

### 6.1 训练目标

先用教师监督速度修正，再在共享噪声的相邻 CFG 上监督轨迹响应：

\[
\mathcal L=\mathcal L_{\rm field}
 +\alpha\mathcal L_{\rm response}
 +\eta\mathcal L_{\rm benign}. \tag{20}
\]

- \(\mathcal L_{\rm field}=\mathbb E\|u_\varphi-u_K\|^2\)：粒子教师修正回归。
- \(\mathcal L_{\rm response}\)：在短续程／少数检查点上，匹配教师和学生的共同噪声有限差分 \([z^{\gamma+\delta}-z^{\gamma-\delta}]/(2\delta)\)。可补充固定状态的 \(\partial_\gamma u\) 匹配，但不能把二者混为一谈。
- \(\mathcal L_{\rm benign}\)：对无目标提示约束修正与语义漂移；仅在“风险在整个相关支持上为常数”等特殊理想情形才有精确零修正结论。

为保持输出接口独立，响应标签优先由前向有限差分生成；不会要求读取 DiT 的内部 Jacobian。它可能很昂贵，须计入教师成本。学生监督应逐渐加入控制后的轨迹状态，并在这些状态上重新运行**原始 CFG 续程**校准式 (2)，不能仅在原始轨迹上训练就假设没有分布偏移。

### 6.2 有条件的 CFG 区间界

若在考察区域内 \(\|J_zB^\gamma\|\le L\)、\(\|\partial_\gamma B^\gamma\|\le G\)，由 Grönwall 不等式

\[
\|s_t^\gamma\|\le S_t:=G\frac{e^{Lt}-1}{L}, \tag{21}
\]

\(L=0\) 时取 \(Gt\)。若终点风险对 latent 为 \(L_R\)-Lipschitz（包含解码器），CFG 网格最大间距为 \(\Delta\)，则网格覆盖区间内

\[
\mathbb E R_F(z_1^\gamma)
\le\max_j\mathbb E R_F(z_1^{\gamma_j})
 +L_RS_1\Delta/2. \tag{22}
\]

这说明控制 CFG 敏感度与网格覆盖有理论联系。它不是现成的认证：高维生成器的全局常数可能极大，经验最大值不能当成严格上界，硬阈值评估器也未必 Lipschitz。实验应检验局部响应与插值表现，不能由有限网格宣称任意 CFG 或任意提示安全。

## 7. 实际算法与成本

### 7.1 离线阶段

1. **冻结生成器并验证接口**：统一时间与 velocity 定义，检查真实 CFG、固定原生 guidance 设置，记录模型版本、求解器、latent 格式。
2. **建立参考粒子库**：对训练提示、多个 CFG、共享噪声采样原始轨迹；解码终点，产生目标风险及保留语义标签。
3. **训练外部续程探针**：使用同一原始轨迹的多时刻状态与终点标签监督 \(\hat r,\hat f\)，增加相邻状态的物质导数一致性。保留一批完整续程作为探针校准集。
4. **生成输运教师**：对同一提示／CFG 的粒子批，先计算相应学习特征的基准矩，再在各时刻解式 (7)，估计式 (18)，解式 (17)，得到修正标签。检查 ESS、协方差条件数和独立弱残差。
5. **校准与训练输出控制器**：固定论文比较协议的风险预算，对各 CFG 求可行的强度，建立校准教师族；输入 \((z,v_0,v_c,t,\gamma,c,F)\) 的外部描述，输出同形状 \(u_\varphi\)；执行式 (20)，再做少量受控轨迹刷新。

对于未见提示，部署时使用摊销控制器，不在线建立粒子集合。**未见提示的 per-prompt 保留约束没有被精确求解**，只是控制器需要泛化的目标；应单独比较已见与未见提示的矩残差。

教师统计默认按提示计算。若为降低成本按提示族合并粒子，只能主张该提示族平均约束，不能继续声称逐提示精确保留。可以把这一点作为成本／准确性的消融。

### 7.2 推理阶段

```text
z = Gaussian(seed)
for each native solver evaluation:
    t = native_to_noise_to_data_time(t_native)
    v0 = frozen_model_velocity(z, t, empty, fixed_native_guidance)
    vc = frozen_model_velocity(z, t, prompt, fixed_native_guidance)
    u  = controller(z, v0, vc, t, external_cfg, prompt_descriptor, concept)
    v  = v0 + external_cfg * (vc - v0) + u
    return to_native_velocity(v) to native solver
decode final latent
```

控制器可用小型、与基模型分离的 latent 网络实现；具体网络是实现选择，不作为创新点。概念条件化泛化与多概念联合擦除先不作为首篇的额外主张。第一阶段可训练单概念控制器，减少不必要的范围膨胀。

运行时无需读取中间层，无需在线图像评估器或粒子重采样。两次 DiT 分支评估及外部控制器的实际耗时必须全部计算；批处理两分支不意味着计算免费。

### 7.3 必须面对的代价

| 成本／问题 | 如何计量与处理 |
| --- | --- |
| 教师粒子库昂贵 | 报告总 DiT NFE、解码／评估次数、GPU 小时、存储；初试每提示／CFG 32 或 64 粒子，检验是否足够，不能把它当已确定超参 |
| 罕见概念与重要性权重退化 | 报告 \(\mathrm{ESS}=1/\sum_iw_i^2\) 与概念覆盖；随 \(\beta\) 扫描；没有低风险参考支持时标记教师不可用 |
| 保留矩退化 | 标准化特征、去冗余；正则后报告保留偏差，不宣称仍是精确约束 |
| 高维 Poisson 近似不足 | 扫描基函数数目并测独立弱残差、真实指标；\(K\times K\) 求解便宜不等于梯度计算或完整教师便宜 |
| 探针偏移 | 校准基准／控制状态的原始续程误差，加入受控状态重新标注 |
| 摊销误差 | 比较教师与学生的速度、终点风险与保留指标，不仅报告训练 MSE |
| 输出场强导致求解困难 | 记录修正范数与步长敏感性；裁剪若改变理论目标，作为明确的近似变体报告 |

不把硬门控、裁剪、稀疏步干预等默认加进精确理论。它们可作为后续效率消融，每种都须记录目标偏差。

尤其是显式强目标提示：若参考粒子全部具有相同高风险，式 (10) 的剩余方差为零，重加权教师无法提供擦除信号。连续软分数可减少数值饱和，却不能保证解决缺少低风险支持的问题。必须报告此类提示的比例、所需粒子量和失败结果；不通过改写提示或混入 anchor 样本后继续声称参考分布仍是原始 \(p_t^\gamma\)。

## 8. 理论结论与实际证据的分层

| 层次 | 可以主张什么 | 不能直接主张什么 |
| --- | --- | --- |
| 理想分布层 | 可行信息投影满足指定矩约束；风险随强度非增 | 所有语义、每个 seed 都保持不变 |
| 理想连续流层 | 正确源项＋精确 Poisson＋相同初始分布产生指定路径 | 任意数值步数都精确，或全局最优感知修改 |
| 有限粒子／基函数教师 | 经验矩与测试函数上的近似约束 | 高维真实密度已经精确匹配 |
| 学生部署层 | 可实测的风险、保留、效率与 CFG 插值 | 因回归损失低就获得理想定理全部结论 |
| 语义评估层 | 对独立评估器／人工协议下的概念抑制证据 | 任意越狱、任意新概念、永久权重遗忘保证 |

若理想教师总场为 \(B_*\)，学生与教师的场误差在相关区域满足统一界 \(\varepsilon\)，且教师场为 \(L\)-Lipschitz，则同初始噪声下可得

\[
W_2(\mu_1^{\rm student},q_1)
\le \varepsilon\frac{e^L-1}{L}, \tag{23}
\]

此处暂忽略教师本身的近似与积分误差。仅有经验 MSE 时不能直接使用这个统一界。完整误差分析应区分：探针／物质导数、重要性采样／对偶、Galerkin、蒸馏、数值积分五类误差。

## 9. 与相关工作的实质差异与创新风险

以下是方法层面的比较，不重做录用状态清单。完整文献筛选见仓库中的 [DiT 扩展检索报告](../../papers/dit_primary_expanded_20261024/REPORT.zh-CN.md)。

| 相关工作 | 与本方案最接近的部分 | 本方案需要成立的区别 |
| --- | --- | --- |
| DVE [R1] | 输出 velocity 的目标／anchor 差分与推理期纠正 | 从保留约束分布推出输运场；并非再设计一个差分方向的投影系数 |
| GEM [R2] | 教师引导的 flow 几何擦除、吸引与排斥 | 教师来自连续性方程和显式语义矩约束，而非直接给出几何匹配目标 |
| BEM [R3] | 对 velocity 学习更合适的几何，改善擦除与保留 | 本方案改变目标概率路径与纠正场，不以更换回归散度为主线 |
| EraseAnything++ [R4] | 擦除／保留的受约束多目标训练与梯度冲突处理 | 本方案约束的是实际 CFG 分布的语义矩，不依赖其注意力级正则或内部编辑位置 |
| FlowErase-OPD [R5] | 多教师 on-policy 蒸馏与自适应保留 | 摊销与 on-policy 本身不能算新；要验证这里的约束输运教师优于现有教师 |
| Mosaic [R6] | 概念 LoRA 与空间掩码下的多速度场融合 | 本方案从分布矩与连续性方程构造单个修正场，不依赖空间掩码拼接；首阶段不主张多概念组合能力 |
| On the Guidance of Flow Matching [R7] | 基于目标能量的一般流引导，也有学习式引导 | 不能声称首次推导能量目标的 velocity；增量是实际 CFG 分布、保留约束、续程源项与 CFG 族耦合 |
| Feedback Particle Filter [R8] | 加权 Poisson、粒子与 Galerkin 近似 | 数值基础属于已有工作；在此用于保留约束的 CFG 擦除，不包装为新数学工具 |
| Feynman-Kac Correctors [R9] | 校正分布路径中的概率质量项 | 必须承认质量校正已有先例；本方案用连续 velocity 输运、固定原始噪声，另有语义约束，而非仅提出重加权 |
| IWCFM [R10] | 针对 CFG 的目标分布／重要性加权 | 本方案以实际 CFG 生成分布为参考，构造擦除约束；不将 CFG 误视为精确幂次分布，也不声称首次纠正该差异 |
| Source-Guided Flow Matching [R13] | 通过重加权达到指定目标分布 | 本方案固定高斯初始分布并修改输出 velocity；应比较代价，而不能仅以“目标分布可解释”作为创新 |

本构思的可发表性风险主要有三点：第一，完整框架可能只是把通用引导工具用于擦除，必须证明保留可行性诊断和新教师带来实质收益；第二，有限粒子与基函数可能让高维分布理论失去实践联系；第三，外部控制器训练成本可能明显超过简单 velocity 教师。论文必须同时回答这三点，不能仅靠形式较长的推导支撑创新。

相对于仓库已有的 inference-time proposal，本方案不读取内部特征，不以“修复后何时退出”为主问题；相对于 close-form proposal，也不进行参数闭式编辑。这里的核心对象始终是实际 CFG 概率流及其最终输出修正。

## 10. 实验：每一个主张都应有对应证据

### 10.1 模型、任务与协议

- 主实验先选具备可靠显式 CFG 的 SD3 / SD3.5 flow DiT；第二骨干验证 FLUX 的真实 CFG 配置，固定并记录原生 guidance embedding。[R11–R12] 如果空条件分支不可靠，应更换合适检查点，不能把 embedding sweep 冒充 CFG sweep。
- 不做 SD1.4 / SDXL 迁移实验。方法适配的是输出接口；跨结构实验用于验证无需内部钩子，而非要求某个固定控制器权重零样本跨骨干泛化。
- 先做两类差异明显的概念：可独立识别的实体／身份与风格。之后再扩展到安全概念，避免一开始把所有任务的评价问题混在一起。
- 提示分为显式目标、间接描述、目标与非目标组合、目标缺席、语义近邻、对抗提示。按模板和语义切分训练／测试，不能只改 seed。
- 真实 CFG 区间初试 \([1,7]\)，训练若干网格值，测试插值值，并明确标记区间外测试。具体范围根据原始模型有效区间调整，原生 embedding 保持固定。
- 原始、擦除、对比方法使用同一模型版本、噪声、原生 solver、步数、分辨率与解码设置。非 SD1 模型使用明确声明的原生采样配置；将所有设置、数据划分和额外控制计算写入 manifest，通过项目 benchmark 存储真实实验结果。

### 10.2 必须包含的对比

1. 同模型上的 DVE、GEM；可复现时加入 BEM、EraseAnything++、FlowErase-OPD、Mosaic。训练式与推理式同时比较效果与完整成本，不只比较基模型参数是否更新。
2. 相同外置控制器、相同标签量，直接拟合传统负引导教师，检验收益是否仅来自多一个网络。
3. 相同风险评估器的普通梯度引导／基于一步终点估计的 velocity 引导，以及不加保留约束的指数倾斜。
4. 粒子重加权／终点拒绝或可复现的 Feynman-Kac、Source-Guided 方案，报告相同 DiT NFE 下的质量与多样性，检验输运是否值得额外复杂度。

### 10.3 核心消融与图

| 待检验主张 | 消融 | 推荐图／指标 |
| --- | --- | --- |
| 矩约束能降低附带损伤 | 去掉 \(f\)；逐步增加保留特征；改软约束 | 目标风险—保留误差 Pareto；近邻概念保持；目标缺席提示质量 |
| 剩余方差诊断有解释力 | 不同目标、保留集、CFG、\(\beta\) | Schur 补热图与实际风险下降的关系；显示失败而非只显示成功例 |
| 连续性校正优于局部方向 | 负速度方向／纯梯度；不解 Poisson；不同 \(K\) | 独立弱残差、输运能量、最终风险／语义漂移；二维场图辅助解释 |
| 续程语义是必要的 | 一步 \(\hat x_1\) 评分；完整／简化源项 | 续程标签误差、物质导数残差、不同时间段的干预效果 |
| 摊销可用 | 教师与学生；无／有受控状态刷新 | 教师—学生差距、样本效率、未见提示矩残差 |
| CFG 校准与响应学习有额外价值 | 单 CFG；固定强度；仅输入 CFG；随机 CFG 训练；预算校准；加入响应损失 | 未见 CFG 的风险与保留曲线、\(\beta_\gamma^*\) 与剩余方差、相同噪声轨迹敏感度、预算不可达与区间外失败 |
| 成本可接受 | 粒子数／基函数数／教师预算扫描 | 训练 GPU 小时、NFE、延迟、峰值显存、ESS、质量—成本前沿 |

视觉结果应展示同 seed 的目标存在、目标缺席和近邻概念，而不只是目标消失的图片。保留评估既要测定义中的矩，也要测没有进入约束的关系、背景与人工偏好。FID／CLIP 分数不能替代概念专属擦除率或关系保留评估。

### 10.4 分阶段继续／停止标准

1. **解析检验**：先检查源项、符号、矩约束和 CFG 依赖，见下一节。
2. **小规模真实流实验**：验证续程探针可校准，教师弱残差随基函数数／粒子数增加而下降。若矩保留在经验数据上都不可行，先解决统计定义与参考覆盖。
3. **教师效果检验**：同风险水平下，约束输运教师应比无约束教师改善保留；若教师本身没有优势，不进入大规模控制器训练。
4. **摊销与 CFG 检验**：教师收益能够保留到学生；响应损失优于只输入 CFG 的基线。否则缩减贡献或调整学习方案。
5. **完整评测**：才扩展概念、骨干、攻击与计算预算。任何阶段都不以已写出定理替代真实模型可行性证据。

## 11. 随提案提供的二维解析检验

`toy_transport_check.py` 构造一个真正由两分支线性组合得到的二维 CFG 场。它的原始边缘分布为解析高斯；终点风险取第一维，保留统计取第二维，续程观测严格满足式 (3)。因此式 (4) 与最小能量修正均可解析计算。

为便于解析，这里的风险是**无界线性坐标**，不是概念概率；高斯的指数矩有限，倾斜仍有定义。这个实验只验证公式，不代表真实语义风险。

脚本比较：精确约束输运、不加保留约束的倾斜、仅保留修正场第一维的局部删减，以及把 \(\gamma=3\) 的修正冻结后用于其他 CFG。它检查：

- 数值积分终点均值是否与解析目标一致；
- 保留语义矩是否保持；
- 式 (10) 的导数、式 (12) 的散度符号是否正确；
- 固定一个 CFG 的修正能否保持其他 CFG 的矩；
- 相同高斯噪声下修正场是否从一开始就可执行。

运行命令、数值和图分别见 [README](./README.md)、[metrics.json](./toy_results/metrics.json)、[transport_check.png](./toy_results/transport_check.png)。这些数值仅为解析数学检验，**没有训练探针／控制器，没有运行文生图 DiT，没有获得概念擦除率**。

本次 CPU float64 运行的结果：数值积分与解析目标均值最大误差为 \(9.31\times10^{-12}\)，终点保留均值最大误差为 \(1.88\times10^{-13}\)，散度与源项抵消残差为 \(3.11\times10^{-15}\)。另以独立的 \(24\times24\) 高斯求积和 Newton 对偶求解检查指数倾斜，均值误差小于 \(2.45\times10^{-15}\)。

| CFG | 方法 | 终点风险坐标均值变化 | 终点保留坐标均值变化 |
| --- | --- | --- | --- |
| 3 | 约束输运 | -0.844755 | 约 0 |
| 3 | 不加保留约束的倾斜 | -1.054801 | -0.470698 |
| 3 | 只保留修正的第一维 | -0.852927 | -0.068363 |
| 7 | 约束输运 | -0.556718 | 约 0 |
| 7 | 复用 CFG=3 的修正 | -0.851988 | -0.090570 |

这里直接复用修正有时降低更多风险，但同时偏离规定的保留约束；该对比不是等风险性能排名。它只说明局部坐标删减与 CFG 复用一般不实现同一个约束分布。

## 12. 论文可以如何组织

建议标题方向：**Preservation-Constrained Probability Transport for Concept Erasure in Flow-Matching DiTs**。摘要的中心应是“从显式擦除／保留分布推出可执行 velocity，并分析 CFG 变化”，而不是“在已有负引导上加入多个模块”。

方法部分按“目标分布与可行性 → 连续性方程与输出场 → 实际教师与摊销 → CFG 连续族”组织。理论正文保留式 (10)、(15) 及其条件，区间误差界放附录也可以。实验的首要结果应是擦除／保留前沿与约束诊断，再展示 CFG 与跨骨干分析。

目前最有把握的是方法链条在数学上可自洽；最大的不确定性是有限粒子教师和低维弱求解能否在真实 DiT latent 中保留优势。这应作为下一阶段首先回答的问题。

## 参考资料与核对范围

以下均为原论文、作者论文页或官方文档。局部差异判断属于本提案的研究判断，不表示已经证明与所有现有方法不重合。检索截止 2026-10-24。

- **[R1] DVE**：*Differential Vector Erasure: Unified Training-Free Concept Erasure for Flow Matching Models*. [论文](https://arxiv.org/html/2602.01089)。已阅读前轮本地全文及方法部分；直接相关的输出 velocity 基线。
- **[R2] GEM**：*Geometric Erasure by Contrastive Velocity Matching in Rectified Flows*. [ICML 2026 官方论文页](https://proceedings.mlr.press/v306/grebe26a.html)，[论文](https://arxiv.org/abs/2606.00140)。已结合本地全文核对。
- **[R3] BEM**：*Learning Bregman Divergences for Concept Erasure in Flow Matching Models*. [论文](https://arxiv.org/html/2610.24508)。重点核对摘要及方法动机；不能把“改变 velocity 空间的度量”当作未有人研究。
- **[R4] EraseAnything++**：[论文](https://arxiv.org/abs/2603.00978)。沿用前轮全文筛选记录；实际复现前应再次绑定具体版本。
- **[R5] FlowErase-OPD**：[论文](https://arxiv.org/abs/2608.07620)。沿用前轮全文筛选记录；on-policy 蒸馏不作为本方案独立创新。
- **[R6] Mosaic**：[论文](https://arxiv.org/abs/2605.25574)。沿用前轮全文筛选记录；比较以具体实验协议为准。
- **[R7] On the Guidance of Flow Matching**：[论文](https://arxiv.org/html/2502.02150)。本轮核对一般引导、协方差／梯度近似和学习式引导部分。
- **[R8] Taghvaei & Mehta, Gain Function Approximation in the Feedback Particle Filter**：[作者全文](https://mehtagroup.web.engr.illinois.edu/downloads/conferences/2016_CDC_Gain_Function_Approximation_FPF.pdf)。加权 Poisson、Galerkin 和无需显式密度的粒子近似的基础来源。
- **[R9] Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts**：[ICML 2025 官方论文页](https://proceedings.mlr.press/v267/skreta25a.html)。本轮核对官方摘要；与本方案的全面数值比较仍待实现。
- **[R10] Learning ω-Powered Conditional Distributions with Importance-Weighted Flow Matching**：[论文](https://arxiv.org/html/2610.16419)。本轮核对 CFG 密度差异和重要性加权目标，不把其所有近似结果移用于本方案。
- **[R11] Diffusers SD3 官方文档**：[接口](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)。
- **[R12] Diffusers FLUX 官方文档**：[接口](https://huggingface.co/docs/diffusers/api/pipelines/flux)。`true_cfg_scale` 与原生 `guidance_scale` 需区分，并按实际代码版本复核。
- **[R13] Source-Guided Flow Matching**：[ICLR 2026 官方论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/adedc33539eafdf3d2c14a5e0032db6e-Abstract-Conference.html)。本轮核对官方摘要；修改源分布的引导已有先例。
