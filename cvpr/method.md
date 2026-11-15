# Method：实现梳理与写作规划

核对日期：2026-11-15。依据 `DiT/detect_erase/` 的代码、README、训练脚本与数据文件整理；这是方法说明和章节规划，尚未撰写论文正文。

## 1. 方法主线与当前配置

每个目标概念分别学习一个 **detection prompt** 和一个 **suppression prompt**：前者形成文本检测参照，后者作为 CFG 的负向生成条件。两个 prompt 按概念配对，分别训练；所有文本编码器、DiT 和 VAE 权重冻结。推理时先检测输入，再决定使用哪个 suppression prompt，整个采样过程沿用该决定。

代码中的 `detect` 对应 detection prompt，`erase` 对应 suppression prompt。本文档的数学符号统一用上标 `d`、`s`。

### 1.1 区分训练入口默认值与脚本配置

| 项目 | `train.py` 新训练默认值 | `run_dual_gpu.sh` | `run_single_gpu.sh` |
|---|---|---|---|
| 模型 | SD3.5 Medium | SD3.5 Medium | SD3.5 Medium |
| Detection 表示 | `condition`：完整 CLIP 隐藏序列 | `condition` | `condition` |
| Detection loss | 必须显式指定 | `ti-flow-bce` | `ti-flow-supcon` |
| Suppression 表示 | `hidden`：完整 sequence + pooled | `hidden` | `hidden` |
| Suppression 教师 | `cfg` | `cfg` | `cfg` |
| 训练状态来源 | `trajectory` | `trajectory` | `trajectory` |
| 推进训练状态的轨迹 | `unsafe` | `safe` | `safe` |
| Suppression 文本对 | `dataset/erase/` | `dataset/safe-describe/` | `dataset/safe-describe/` |

另有 `train_detect.sh`、`all.sh`、`train_nudity_detect_new.sh` 和 `train_nudity_2.sh` 使用单个输入 soft token；检测目标为 `clip`，即仅 BCE。后两个脚本分别使用 2400 条、1440 条 nudity 合成训练文本。

README 第 374 行仍将两个完整训练脚本都描述为 SupCon；当前双卡脚本第 7 行实际为 BCE。后续论文的主配置应以选定实验的 `training.json` 和 checkpoint 为准。

### 1.2 两类 prompt 的学习对象

| Prompt | 当前完整表示 | 学习目标 | 使用时机 |
|---|---|---|---|
| Detection prompt | 每个 CLIP 一份倒数第二层完整隐藏序列 | 区分目标与非目标文本；可联合图像 flow 重建 | 采样前计算概念匹配分数 |
| Suppression prompt | 一份完整生成 sequence 与 pooled condition | 让原输入配合负条件后的 CFG 预测接近安全参考预测 | 命中后作为整个采样过程的负条件 |

“独立学习”有两层含义：各概念独立训练；同一概念的两类 prompt 也独立训练。Detection 内部的 TI + 分类联合目标只优化 detection 参数，并不与 suppression 联合优化。

## 2. 统一符号与生成基础

| 符号 | 含义 |
|---|---|
| `c`、`q` | 目标概念、用户输入文本 |
| `E(q)=(S(q),P(q))` | 冻结文本编码器产生的生成条件：sequence 与 pooled |
| `C_0=E("")` | 空文本生成条件 |
| `v_theta(z_sigma,sigma;C)` | 冻结 DiT 在给定状态、时间和条件下的 velocity 预测 |
| `Phi_c^d`、`Phi_c^s` | 概念 `c` 的 detection、suppression 可训练参数 |
| `s` | 外层 CFG scale；SD3 使用 `guidance_scale`，FLUX 使用 `true_cfg_scale` |
| `sg` | 停止梯度 |

### 2.1 Flow 状态与重建目标

使用 VAE posterior mode 编码图像，并应用原模型的 shift/scaling，得到 `z_0`。随机加噪时：

$$
\sigma\sim\mathcal U(0,1),\qquad
\epsilon\sim\mathcal N(0,I),\qquad
z_\sigma=(1-\sigma)z_0+\sigma\epsilon,\qquad
v^*=\epsilon-z_0.
\tag{F1}
$$

本文档用归一化时间 `sigma` 表述。实现传入 `t=1000*sigma`；FLUX 的模型调用再除以 1000，并对 latent 做原生 packing。生成沿噪声端向图像端推进，采用模型原生的 FlowMatch Euler 调度。

### 2.2 CFG 与负向条件

$$
\mathcal G_s(z_\sigma,\sigma;C_+,C_-)
=v_\theta(z_\sigma,\sigma;C_-)
+s\left[v_\theta(z_\sigma,\sigma;C_+)
-v_\theta(z_\sigma,\sigma;C_-)\right].
\tag{F2}
$$

普通 CFG 使用 `C_+=E(q)`、`C_-=C_0`；本文在命中目标概念时，用 learned suppression condition 替换 `C_0`。

FLUX 的 embedded guidance 与上述外层 CFG 分开：`guidance_scale` 传入模型内部，`true_cfg_scale` 才是式 F2 的 `s`。训练入口为 FLUX 设置的外层 CFG 默认值为 2。

## 3. Detection prompt：表示、目标与训练

### 3.1 完整隐藏序列表示：`condition`

对第 `k` 个 CLIP，学习：

$$
\Phi_c^d=\{H_{c,k}^d\}_{k=1}^{K},\qquad
H_{c,k}^d\in\mathbb R^{L_k\times D_k},\qquad
H_{c,k}^{d,(0)}
=E_{k,\mathrm{penultimate}}(\text{``a photo of \{c\}''}).
\tag{D1}
$$

代码中的张量形状为 `[1,77,D_k]`，前面的 1 用于 batch 广播，不代表为不同输入分别学习参数。SD3 有两个 CLIP，`K=2`；FLUX 有一个，`K=1`。

检测参照由完整隐藏序列经过冻结的最后一个 CLIP Transformer block、final layer norm、初始化模板的 EOS 位置池化和 text projection 得到：

$$
a_{c,k}=\left[\operatorname{LN}_k
\left(F_{k,\mathrm{last}}(H_{c,k}^d)\right)\right]_{e_{c,k}},
\qquad u_{c,k}=W_k a_{c,k},\qquad
g_{c,k}=\frac{u_{c,k}}{\|u_{c,k}\|_2}.
\tag{D2}
$$

`e_{c,k}` 固定为初始化模板的 EOS 位置。最后一个 block 使用 causal attention；隐藏序列已经包含位置信息，从该位置继续计算时不再添加位置编码。

将用户文本经完整冻结 CLIP 得到的 projected 特征归一化为 `h_k(q)`，匹配分数为：

$$
r_c(q)=\frac1K\sum_{k=1}^{K}h_k(q)^\top g_{c,k}.
\tag{D3}
$$

两个 CLIP 的 cosine 等权平均，不先拼接再计算一个 cosine。`g_{c,k}` 可在加载 checkpoint 时预计算。

用于 flow 重建的 detection 生成条件 `C_c^d` 与分类参照来自同一组 `H_c^d`：

| 模型 | Sequence condition | Pooled condition | 检测特征 |
|---|---|---|---|
| SD3 | 两个 `H_c,k^d` 沿通道拼接、补零后，与固定 `T5(c)` 沿序列拼接 | 两个未归一化 `u_c,k` 拼接 | 分别归一化两个 `u_c,k` |
| FLUX | 固定 `T5(c)` | 未投影的 `a_c,1` | 对 `a_c,1` 另做 text projection、归一化 |

这里的固定 T5 文本是概念 `c`，不是初始化 CLIP 用的完整模板。FLUX 检测用的 projection 来自冻结的原版 CLIP contrastive head；生成条件沿用 FLUX 原生 pooled 表示。

### 3.2 分类目标：BCE

检测训练数据为 `D_c^cls={(q,y_c)}`，标签表示文本是否属于目标概念的正类。原始小数据集每概念含 30 条正例、10 条相近负例、10 条普通负例。

$$
p_c(q)=\operatorname{sigmoid}
\left(\frac{r_c(q)-\tau_c}{T_{\mathrm B}}\right),
\qquad
\mathcal L_{c,\mathrm{BCE}}
=-\mathbb E_{(q,y_c)}
\left[y_c\log p_c(q)+(1-y_c)\log(1-p_c(q))\right].
\tag{D4}
$$

实现直接使用 `BCEWithLogits((score-tau)/temperature,label)`。训练分界与推理的 `r_c(q)>=tau_c` 一致；默认 `tau=0.2`、`T_B=0.1`。

`--detect-loss clip` 只优化式 D4，只需要 CLIP。各概念训练自己的二分类参照，不使用跨概念 softmax 分类目标。

### 3.3 分类目标：prototype-anchor SupCon

对一个概念，分别采样正文本集合 `B_c^+` 和负文本集合 `B_c^-`。可训练的 detection 表示是 anchor，文本特征冻结：

$$
\mathcal L_{c,\mathrm{Con}}
=-\frac1{|\mathcal B_c^+|}
\sum_{q\in\mathcal B_c^+}
\log\frac{\exp(r_c(q)/T_{\mathrm C})}
{\sum_{q'\in\mathcal B_c^+\cup\mathcal B_c^-}
\exp(r_c(q')/T_{\mathrm C})}.
\tag{D5}
$$

对应代码 `logsumexp(logits)-mean(positive_logits)`。它约束正文本相对负文本的匹配程度；推理仍使用原始 cosine 与 `tau`，训练不会自动校准该阈值。默认 `T_C=0.1`。

联合训练的分类 batch 默认 16 条，正负各 8 条。训练前按 `(prompt,label)` 去重，batch 内无放回，后续 batch 独立抽样。原始风格数据的 50 条记录去重后为 37 条分类文本。

### 3.4 TI-flow 重建目标及联合训练

先用目标正文本和冻结基础模型生成固定图像，每条正文本一张，负文本为空；图像不经外部分类器筛选。原始小数据集默认每概念 30 张，可用 `--ti-image-count` 限制数量。

这些图像经 VAE 编码后，按式 F1 随机加噪；detection 生成条件通过 flow 重建获得梯度：

$$
\mathcal L_{c,\mathrm{TI}}
=\mathbb E_{x\in\mathcal I_c^d,\sigma,\epsilon}
\left\|v_\theta(z_\sigma,\sigma;C_c^d)
-(\epsilon-z_0)\right\|_2^2.
\tag{D6}
$$

代码提供两种联合目标：

$$
\mathcal L_c^d=
\begin{cases}
\mathcal L_{c,\mathrm{TI}}+
\lambda_{\mathrm B}\mathcal L_{c,\mathrm{BCE}},
&\texttt{ti-flow-bce},\\
\mathcal L_{c,\mathrm{TI}}+
\lambda_{\mathrm C}\mathcal L_{c,\mathrm{Con}},
&\texttt{ti-flow-supcon}.
\end{cases}
\tag{D7}
$$

TI 权重固定为 1，分类项默认权重也为 1。代码另保留 `ti-flow`，只优化式 D6。

联合训练步骤：

1. 缓存正图像的 VAE latents，以及全部分类文本的 CLIP 特征。
2. 从正图像取一个 batch，重新采样噪声和时间，计算 TI loss。
3. 独立抽取平衡的正负文本 batch，计算 BCE 或 SupCon。
4. 两项相加，只更新同一组 `H_c,k^d`。

图像 batch 与分类文本 batch不要求逐条配对；默认大小分别为 1 和 16。冻结 DiT 仍需保留到 detection 参数的梯度，固定文本特征则预编码后停止梯度。

训练标签决定检测语义：原始数据针对概念相关文本，新的 nudity 合成数据按文本描述的可见性标注。TI 重建提供生成侧约束；推理分数仍是文本与概念参照的相似度。

### 3.5 保留的表示对照

| 选项 | 可训练参数 | 初始化和读出 |
|---|---|---|
| `soft` | 每个 CLIP 一个输入 token embedding | Detection 以概念子 token embedding 的均值初始化；可加固定上下文，经完整 CLIP 得到参照 |
| `hidden` | CLIP pooled 表示 | 从概念文本初始化；需要生成条件时，sequence 和 T5 固定 |
| `condition` | CLIP 倒数第二层完整序列 | 式 D1–D2；两种联合 loss 要求此空间 |

Soft 和 hidden 可用于单项 BCE 或 TI-flow 对照。当前完整联合训练不是仅优化一个词的 Textual Inversion，也不是微调 CLIP 权重。

## 4. Suppression prompt：表示、目标与训练状态

### 4.1 直接学习生成条件

`hidden` 空间为每个概念学习一组完整生成条件：

$$
\Phi_c^s=C_c^s=(S_c^s,P_c^s),\qquad
(S_c^{s,(0)},P_c^{s,(0)})=E(\text{``A photo of ''}).
\tag{S1}
$$

初始化字符串末尾包含一个空格。Sequence 与 pooled 是独立可训练张量，同一概念的所有训练文本、时间步共享这组参数。

- SD3：完整 sequence 包含两路 CLIP 与 T5 部分，pooled 为两路 CLIP projected pooled 的拼接。
- FLUX：sequence 对应 T5 输出，pooled 对应原生 CLIP pooled。
- 梯度经冻结 DiT 回到这两个条件张量；不通过 detection prompt，也不更新文本编码器。
- `soft` 对照只学习每个 CLIP 的单个输入 token，初始化为 EOS embedding，T5 固定为空文本编码。

### 4.2 正文本与安全参考文本

训练对为 `D_c^s={(q^u,q^a)}`：`target_prompt` 是含目标概念的正文本，`anchor_prompt` 是期望保留内容的安全参考文本。教师条件取自每条记录的 `anchor_prompt`。

两个完整训练脚本使用 `dataset/safe-describe/`：

- Nudity：改为明确穿衣的成年人，保留人物动作和场景。
- IP、object：移除目标主体，描述剩余背景或承载表面。
- Style：保留画面内容，改为真实照片描述。

该数据共 11 个概念、700 对文本。`train.py` 默认的 `dataset/erase/` 则多采用概念词删除后的模板；它与 safe-describe 的教师语义不同，应在实验配置中区分。

### 4.3 核心目标：匹配安全参考的 guided velocity

在同一个 `z_sigma,sigma` 上计算四个分支：

$$
v_u=v_\theta(z_\sigma,\sigma;E(q^u)),\quad
v_a=v_\theta(z_\sigma,\sigma;E(q^a)),\quad
v_0=v_\theta(z_\sigma,\sigma;C_0),\quad
v_s=v_\theta(z_\sigma,\sigma;C_c^s).
\tag{S2}
$$

学生保留原输入正条件，将 suppression prompt 放在负分支：

$$
\widehat v_s=v_s+s(v_u-v_s).
\tag{S3}
$$

默认安全教师自身也使用 CFG；另保留仅使用原始 conditional prediction 的对照：

$$
v_{\mathrm{teacher}}=
\begin{cases}
v_0+s(v_a-v_0),&\texttt{erase-target=cfg},\\
v_a,&\texttt{erase-target=conditional},
\end{cases}
\qquad
\mathcal L_c^s=
\mathbb E\left\|\widehat v_s-operatorname{sg}
(v_{\mathrm{teacher}})\right\|_2^2.
\tag{S4}
$$

学生和 CFG 教师使用相同的 `s`。`v_u,v_a,v_0` 在无梯度分支计算，只有 `v_s` 保留到 suppression 参数的梯度。Loss 比较中间 velocity，不比较最终图像。

`s>1` 是训练入口要求：当 `s=1` 时，式 S3 退化为 `v_u`，负条件不再影响学生预测。

### 4.4 训练状态来源一：trajectory

每个文本 batch 从随机噪声开始，沿原生 scheduler 的所有时间步推进。四个分支始终访问同一个当前 latent。可选的推进预测为：

$$
v_{\mathrm{roll}}=
\begin{cases}
v_0+s(v_u-v_0),&\texttt{unsafe},\\
v_0+s(v_a-v_0),&\texttt{safe},\\
\widehat v_s,&\texttt{student}.
\end{cases}
\tag{S5}
$$

训练入口默认 unsafe，两个完整脚本实际使用 safe。每个时间步：

1. 在当前 latent 计算四个预测及式 S4。
2. 保存更新前的 `sg(v_roll)`。
3. 更新一次共享的 suppression condition。
4. 用已保存的预测执行 scheduler step，得到下一状态并停止其梯度。

因此，每个去噪步骤都是一次参数更新，不对完整轨迹反向传播；safe/unsafe 不是两条分别采样后再比较的生成轨迹。即便选择 `conditional` 教师，safe rollout 仍使用式 S5 的 safe CFG。

### 4.5 训练状态来源二：image-noise

先用概念词本身作为正文本、空负文本，从冻结基础模型生成少量固定图像，默认每概念 10 张，并缓存其 VAE latents。每次更新：

1. 读取一个 unsafe/safe 文本对 batch。
2. 独立、有放回地抽取概念图像，按式 F1 加噪。
3. 在同一 noisy latent 上计算式 S2–S4，更新一次 suppression condition。

此模式不推进训练 latent；`erase-trajectory` 不适用。它与 detection TI-flow 使用相同的随机加噪构造，但监督目标仍是安全教师 velocity，**不是 `epsilon-z_0`**。图像仅作为固定训练状态来源，不经最终图像评分筛选或反馈优化。

## 5. 推理流程与公式

### 5.1 阈值与全量 top1

记 `C_det` 为加载的全部 detector 概念，`C_erase` 为本次启用的擦除目标。默认 `--all-detect-prompts` 扫描所有 detection checkpoints，先在全量候选中检测，再与启用目标求交：

$$
\mathcal M(q)=\{c\in\mathcal C_{\mathrm{det}}:r_c(q)\ge\tau_c\},
\qquad
\mathcal A(q)=
\begin{cases}
\{\arg\max_{c\in\mathcal M(q)}r_c(q)\}
\cap\mathcal C_{\mathrm{erase}},&\mathcal M(q)\ne\varnothing,\\
\varnothing,&\mathcal M(q)=\varnothing.
\end{cases}
\tag{I1}
$$

默认选择至多一个 suppression prompt。若最高分概念属于本次擦除目标之外，即使某个启用目标也达到阈值，当前实现也不回退选择它。

阈值默认分别读取每个 detection checkpoint 的 `tau`；采样命令的 `--tau` 可统一覆盖。论文一般部署设定可以令 `C_det=C_erase`，实验则需交代候选集合和启用集合。

### 5.2 构造负条件与采样

$$
C_-(q)=
\begin{cases}
C_0,&\mathcal A(q)=\varnothing,\\
\displaystyle\frac1{|\mathcal A(q)|}
\sum_{c\in\mathcal A(q)}C_c^s,&\mathcal A(q)\ne\varnothing,
\end{cases}
\qquad
\widehat v(q)=\mathcal G_s
(z_\sigma,\sigma;E(q),C_-(q)).
\tag{I2}
$$

默认 top1 下，上式直接取被选概念的条件。关闭全量检测后可用 `mean`：对全部过阈值概念的 sequence 与 pooled 分别取算术平均，随后只计算一个负分支；不按分数加权，也不逐概念计算 velocity 后相加。

SD3 在负条件送入采样器前，将 CLIP 序列补零通道重新清零；FLUX 无此处理。Suppression 的 hidden 训练及 checkpoint 保存未执行这一步，所以式 I2 的 SD3 部署条件应理解为经过该补零处理的 `C_c^s`。这是后续实现细节需要交代的一处训练、推理差异。

推理步骤：

1. 加载每个概念的 prompt pair，预计算 detection 参照。
2. 编码用户原始正文本，计算全部 detection 分数。
3. 按式 I1 决定启用概念，按式 I2 构造负条件。
4. 将正条件、负条件和初始噪声传给原生 pipeline，整个采样过程保持该负条件。
5. 未触发时使用原始正条件和空负条件，与同一协议下的普通 CFG 一致。

Detection 不访问生成图像、VAE 或采样中间状态。当前完整生成代码先计算正生成条件，检测时又调用一次 CLIP 特征编码；“一次检测”指采样前作出一次决定，不表示完整流程只执行一次 CLIP 编码。

同样的双分支 CFG 协议下，加载更多概念增加文本匹配计算，不增加每步 DiT 的分支数。

### 5.3 旧表示的 L2 对照

Soft/hidden checkpoints 另支持完整 generation condition 的 RMS 距离：

$$
d_c(q)=\sqrt{\frac{\|S(q)-S_c^d\|_F^2+
\|P(q)-P_c^d\|_2^2}{\operatorname{numel}(S(q))+
\operatorname{numel}(P(q))}}.
\tag{I3}
$$

全量 L2 模式无距离阈值，直接选最近参照；非全量模式使用 `d_c(q)<=l2_threshold`，默认阈值 0.2。`condition` detector 在采样入口被限定为 cosine，L2 留作旧表示对照。

## 6. 建议的 Method 章节结构

建议四个小节，按“生成基础 → 两类独立学习 → 配对使用”的顺序展开。论文的 detection 默认目标已确定为 **TI-flow + SupCon**；这是论文主配置选择，不改变训练入口和已有脚本的默认参数。

| 小节 | 段落任务 | 核心公式 |
|---|---|---|
| **3.1 Preliminaries** | 潜空间、rectified flow 的训练目标、CFG 与负向提示；沿用当前正文的符号 | 当前 `3_method.tex` 的四个基础公式 |
| **3.2 Learning detection prompts** | 开头单独一段介绍两类 prompt 的目的与推理分工；随后说明 detection 的连续表示、概念参照、SupCon 和 TI-flow 联合目标 | 见下方 6.4 的表示与四组公式 |
| **3.3 Learning suppression prompts** | 从 unsafe/safe 文本对引出期望预测；说明 sequence/pooled 参数化，再给出学生、教师和匹配损失；最后交代训练状态如何取得 | S1；将 S2–S4 合成一组核心公式；主配置对应的 S5 或 F1 |
| **3.4 Selective inference with prompt pairs** | 描述一次检测、阈值筛选、top1 与负条件选择，最后解释未触发分支及计算开销 | I1、I2 |

### 6.1 各节的写作重点

- **方法开场段**：放在 Preliminary 之后的第一个方法小节开头，单独成段，不另设 Overview 小节。说明 detection prompt 学习输入概念匹配，suppression prompt 学习负向生成控制；二者独立训练、按概念配对，推理时先检测，再启用匹配的 suppression prompt，未启用时使用空文本负分支。
- **Detection**：按“同一连续表示 → 两种用途 → SupCon 与 TI-flow → 联合优化”叙述。正文不展开张量尺寸、初始化模板、数据构造、batch、优化器和底层编码操作。
- **Suppression**：重点解释为何优化的是“原输入 + learned negative”的 guided prediction，而不是孤立的负分支或最终图片。安全参考只用于离线训练，推理保留用户原始正文本。
- **Inference**：用明确的分支公式呈现 selective suppression；默认 top1 与可选 mean 分开说明。
- **两类学习的关系**：分别写 `Phi_c^d*=argmin L_c^d` 与 `Phi_c^s*=argmin L_c^s`。可在 detection 内写 TI+分类联合目标，整套 prompt pair 则保持独立训练的表述。

### 6.2 正文、实现细节与对照的分配

| 内容 | 建议位置 |
|---|---|
| 最终选定的表示、detection loss、suppression 教师及训练状态 | Method 主文 |
| 未选中的 BCE/SupCon、soft/condition、conditional/CFG、unsafe/safe/student、image-noise | 实验消融或补充材料 |
| SD3/FLUX 的条件拼接、projection、packing、时间缩放与补零处理 | 实现细节或补充材料 |
| 学习率、batch、更新数、图像数量、阈值和数据构造数量 | Experiments 的 implementation details |
| 检测混淆矩阵、外部图像分类器与生成指标 | Experiments |

正文统一称这些参数为 learned prompt representations 或 prompt embeddings，并说明实际优化位置；完整 condition 包括隐藏序列与 pooled 表示。当前实现没有跨概念联合训练、prompt pair 共享参数或端到端联合损失。

### 6.3 已确定与待确定的配置

1. Detection 主目标已确定为 TI-flow + SupCon，BCE 及单项 loss 保留为对照。
2. 当前联合训练实现要求 `condition` 表示，因此主方法学习连续的中间文本表示；其尺寸和底层读出操作放在实现细节中。
3. Suppression 主状态来源及轨迹：入口默认 unsafe，完整脚本当前是 safe；image-noise 是另一条实现路线。
4. 主实验的训练文本与阈值协议：原始小检测集或新的 nudity 数据；suppression 使用 erase 或 safe-describe；推理采用全量 top1 或其他已实现设置。

剩余设置可从最终采用实验的配置记录确定，不需要改动总体章节结构。

### 6.4 Detection prompt：当前写作规划

本节只规划机制、数学定义和论述顺序，尚不撰写英文正文。主文以 TI-flow + SupCon 为默认目标，省略尺寸、具体模板和数据构造过程。

#### 方法开场段：先交代两类 prompt 的分工

这一段属于 Preliminary 之后的第一个方法小节开头，说明四件事：

1. 为每个目标概念分别学习 detection prompt 与 suppression prompt。
2. Detection 的目的是判断输入文本是否匹配目标概念；suppression 的目的是提供有效的负向生成引导。
3. 两类 prompt 分别优化，通过目标概念关联；基础模型与文本编码器冻结。
4. 推理在采样前完成检测，随后启用所选概念的 suppression prompt 作为负分支的 text embeddings；未启用时使用空文本 embeddings。

这里先交代目的与使用关系，阈值筛选、top1 和多概念选择规则留到最后的 inference 小节。

#### 第一段：表示形式与匹配分数

定义概念 `c` 的 detection prompt 为可学习的连续文本表示 `p_c^d`。它位于文本编码器的中间表示空间，由概念相关的文本表示初始化。正文说明优化发生在表示空间，不展开隐藏层索引、长度、维度或具体初始化文本。

同一组 detection 参数具有两种用途：

- 通过原模型的冻结读出，获得概念参照 `g_c,k`，与输入文本进行匹配。
- 组成生成侧 text embeddings `e_c^d`，送入冻结 DiT 计算 flow 重建目标。

这两种用途共享同一个 `p_c^d`，不是两组独立学习的 detection 参数。读出与组合复用原模型的固定计算。

符号沿用 Preliminary，并新增以下必要记号：

| 符号 | 在本节中的含义 |
|---|---|
| `E`、`mathcal E` | 分别为 VAE 编码器、文本编码器 |
| `z_0,z_t,t,epsilon,v_theta` | 沿用 Preliminary 的潜变量、时间、噪声与 velocity 预测 |
| `p_c^d` | 概念 `c` 的可学习 detection 表示 |
| `g_c,k` | 从 detection 表示读出的第 `k` 路概念参照 |
| `h_k(q)` | 从文本编码器 `mathcal E` 的相应 CLIP 分支获得的输入文本特征 |
| `e_c^d` | 由同一 detection 表示组成、可送入 DiT 的 text embeddings |

通过 cosine similarity 定义匹配分数：

$$
r_c(q)=\frac1K\sum_{k=1}^{K}
\operatorname{cos}\!\left(h_k(q),g_{c,k}\right).
\tag{PD1}
$$

`K` 只表示所用特征分支数，正文不列具体分支配置。此式保留实现中“各分支 cosine 等权平均”的操作。

#### 第二段：SupCon 提供文本判别监督

先解释目标：让同一概念的正文本与概念参照具有更高匹配程度，同时与非目标文本拉开差距。对一个概念，记正、负文本集合分别为 `B_c^pos` 和 `B_c^neg`：

$$
\mathcal L_{\mathrm{SupCon}}^{(c)}
=-\frac1{|\mathcal B_c^{\mathrm{pos}}|}
\sum_{q\in\mathcal B_c^{\mathrm{pos}}}
\log\frac{\exp(r_c(q)/T)}
{\sum_{q'\in\mathcal B_c^{\mathrm{pos}}\cup
\mathcal B_c^{\mathrm{neg}}}\exp(r_c(q')/T)}.
\tag{PD2}
$$

解释两个关键点即可：anchor 是 detection 表示读出的概念参照；正负输入文本的编码器保持冻结，目标更新 detection 表示。`T` 为对比温度，具体数值放入实验配置。

`q^-` 保留为 Preliminary 中 CFG 的负向提示。对比学习的负文本用 `B_c^neg` 集合表示，以区分训练标签与生成控制角色。

#### 第三段：TI-flow 提供图像重建监督

定义概念图像集合 `I_c` 作为训练输入，省略其构造过程。沿用 Preliminary，令 `z_0=E(x)`，按原插值式得到 `z_t`。将同一 detection 表示组成的 `e_c^d` 输入 DiT：

$$
\mathcal L_{\mathrm{TI\text{-}flow}}^{(c)}
=\mathbb E_{x\in\mathcal I_c,\epsilon,t}
\left\|v_\theta(z_t,t;\mathbf e_c^d)
-(\epsilon-z_0)\right\|_2^2.
\tag{PD3}
$$

本段说明与 SupCon 的互补关系：SupCon 监督文本匹配的相对关系，TI-flow 让同一 detection 表示接受目标概念图像的 velocity 重建约束。写作中将后者表述为训练约束；检测收益由实验报告。

两个目标共享 detection 参数，但可以使用独立的文本与图像训练输入；主文无需展开 batch 抽样过程。模型保持冻结，重建梯度通过 DiT 返回 detection 表示。

#### 第四段：联合目标与训练后使用

$$
\mathcal L_{\mathrm{det}}^{(c)}
=\mathcal L_{\mathrm{TI\text{-}flow}}^{(c)}
+\lambda\mathcal L_{\mathrm{SupCon}}^{(c)},
\qquad
\mathbf p_c^{d,*}
=\arg\min_{\mathbf p_c^d}\mathcal L_{\mathrm{det}}^{(c)}.
\tag{PD4}
$$

说明两项共同优化同一个 detection prompt，并按概念分别训练。段末用一句话承接部署：训练后预先获得概念参照，在线只需编码输入文本、计算匹配分数；图像重建仅用于训练。

本小节的主线为：

`连续 detection 表示 → 概念参照与生成 embeddings → 文本判别 + 图像重建 → 一个联合训练目标`。

主文保留：连续表示的实际优化空间、两种用途共享参数、匹配分数、SupCon、TI-flow 和联合目标。张量形状、模板、图像获取方式、数据数量、最后一层读出细节、分支拼接、优化器及超参数数值，放到实现细节或实验部分。

## 7. 代码定位与复现要点

| 内容 | 代码入口 |
|---|---|
| 全部生成模型组件冻结 | `DiT/detect_erase/prompts.py:39`，`PromptModel.__init__` |
| CLIP 冻结、FLUX projection | `DiT/detect_erase/encoding.py:18`，`TextEncoder.__init__`；`:47`，`project` |
| Prompt 初始化、参数提取 | `DiT/detect_erase/encoding.py:133`，`initialize`；`:163`，`parameters` |
| 完整 condition 的最后一层读出及生成条件组合 | `DiT/detect_erase/encoding.py:181`，`condition`；`:118`，`combine` |
| Cosine 与检测选择 | `DiT/detect_erase/encoding.py:9`，`cosine_scores`；`:220`，`prepare_detector`；`:230`，`detect` |
| BCE / SupCon 公式 | `DiT/detect_erase/prompts.py:17`，`classification_loss`；`:21`，`supervised_contrastive_loss` |
| 单独 BCE 训练 | `DiT/detect_erase/train.py:115`，`train_clip` |
| 图像编码、detection TI 图像、suppression 概念图像 | `DiT/detect_erase/train.py:155`，`encode_images`；`:173`，`ti_images`；`:202`，`erase_images` |
| Detection TI / 联合训练 | `DiT/detect_erase/train.py:243`，`train_ti_flow` |
| Suppression 四分支、教师、损失与状态推进 | `DiT/detect_erase/train.py:316`，`train_erase` |
| 原生 velocity 调用、时间与 scheduler | `DiT/detect_erase/prompts.py:65`，`latents`；`:74`，`timesteps`；`:100`，`velocity` |
| 负条件选择、均值和 SD3 padding | `DiT/detect_erase/prompts.py:120`，`negatives` |
| 完整推理入口 | `DiT/detect_erase/prompts.py:146`，`generate` |
| 默认采样模式、checkpoint 候选与匹配约束 | `DiT/detect_erase/sample.py:123`、`:183`、`:197`、`:262` |
| 纯文本检测评测 | `DiT/detect_erase/sample.py:321` |
| 检测统计与可视化，位于方法外部 | `DiT/detect_erase/detection.py`，`similarity.py` |

训练优化器为 AdamW，weight decay=0，默认学习率 `1e-3`、5 epochs、每概念每阶段最多 2000 次更新；参数以 FP32 优化和保存，模型计算 dtype 由配置决定。

更新次数的计数不同：BCE 每个文本 batch 一次；TI 与联合 detection 每个图像 batch 一次；trajectory suppression 每个采样时间步一次；image-noise suppression 每个文本对 batch 一次。Epoch 与更新数上限先到者结束。

Checkpoint 按概念、阶段分别保存 prompt 参数和配置，不保存改写后的基础模型。续训使用新建优化器；旧 suppression checkpoint 未记录教师类型时，入口会沿用旧的 conditional 语义，因此新训练默认值不能直接套到已有实验上。
