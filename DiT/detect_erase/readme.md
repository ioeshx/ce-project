## 一、算法与 loss

冻结所有模型权重，每个概念、每个阶段独立训练和保存。Detect 默认 `--prompt-space condition`：学习 CLIP 倒数第二层的完整 `[1,77,D]` 序列，由 `a photo of {c}` 初始化。Erase 默认 `--prompt-space hidden`：学习完整 sequence 和 pooled，由 `A photo of ` 初始化。旧 `soft` 空间保留单个 CLIP 输入 token 对照，BOS、EOS 和其余 token 固定，可用 `--soft-template 'a photo of {}'` 添加固定上下文。`--stage all` 未指定空间时，两阶段分别采用上述默认值。

### Detect prompt

Detect 训练必须显式指定 `--detect-loss`，包括续训。支持 `ti-flow-bce`、`ti-flow-supcon` 两种联合 loss，并保留 `clip` 和 `ti-flow` 单项对照。联合 loss 使用 `condition` 空间；旧的 `soft`（目标概念子 token embedding 均值初始化）和 `hidden`（学习 pooled、序列固定）仍可用于单项对照和读取旧 checkpoint。

**1. 仅 BCE 对照（`--detect-loss clip`）**：取每个 CLIP 的 EOS 表征经过 `text_projection` 后的特征。用户文本使用完整冻结 CLIP，condition 使用下述最后一层读出；旧 soft prompt 经过完整编码器。SD3.5 的两个余弦分数取算术平均：

$$
r_i(q)=\frac1K\sum_{k=1}^K\cos(f_k(q),f_k(p_i^d)),\qquad
\mathcal L_d=\operatorname{BCEWithLogits}\big((r_i(q)-\tau)/T,y_i\big).
$$

使用 detect 集全部正负样本。只加载 CLIP，分类梯度回到可训练的 condition 或 soft token，不加载 T5、DiT 或 VAE。FLUX 原生 CLIP 没有 contrastive projection，余弦检测使用原版 `openai/clip-vit-large-patch14` 的预训练 projection；FLUX 的模型条件仍使用原生未投影 pooled。可用 `--clip-projection-model` 指定该 CLIP 模型的本地目录。

**2. 备选 TI-flow（`--detect-loss ti-flow`）**：仅使用 detect 正样本，每条文本用冻结基础模型生成一张固定训练图，默认每概念 30 张。图片保存后复用，不做外部分类器筛选。VAE posterior mode 编码并应用原模型的 shift/scaling 得到 $z_0$：

$$
\sigma\sim U(0,1),\quad\epsilon\sim\mathcal N(0,I),\quad
z_t=(1-\sigma)z_0+\sigma\epsilon,\quad t=1000\sigma,
$$
$$
\mathcal L_{d,\mathrm{TI}}=\mathbb E\|v_\theta(z_t,t,p_i^d)-(\epsilon-z_0)\|_2^2.
$$

每个 batch 重新抽噪声和时间，仅优化 detect 的 CLIP 参数；T5 固定为目标概念编码。FLUX 将 noisy latent 和 velocity target 同样 pack，并使用其原生时间归一化。TI-flow 不再使用旧的正样本文本 velocity 蒸馏目标，也没有分类 loss；其检测效果通过独立评测确认。

**3. Detect 表示：完整 CLIP hidden condition（`--prompt-space condition`）**

`TI-flow + BCE` 与 `TI-flow + SupCon` 共享同一组 condition 参数。训练时使用冻结生成模型的重建梯度，推理时只编码用户文本、计算余弦和阈值，不运行 DiT、VAE 或 image encoder；condition detector 只使用 `cosine` 检测。

将每个概念的单个 CLIP 输入 token 改为可训练的倒数第二层序列 $H_{c,k}\in\mathbb R^{1\times77\times D_k}$，其中 $k$ 表示 CLIP 编码器。初始化为格式化文本的同层输出：

$$
T_c=\text{“a photo of \{c\}”},\qquad
H_{c,k}^{(0)}=E_{k}^{\mathrm{penultimate}}(T_c).
$$

每个概念保存一份序列，在训练 batch 中广播为 $[B,77,D_k]$；$B$ 不是独立参数维度。这是在 CLIP 编码后的 hidden 空间优化 condition，初始化位置应明确取 `hidden_states[-2]`，而不是 token lookup、最终层输出或 pooled 特征。SD3.5 的两个 CLIP 分别保存序列，T5 保持当前的固定概念编码。

分类特征由该序列经过 CLIP **冻结的最后一个 Transformer block、final layer norm、EOS pooling 和 text projection** 得到。记这些操作为：

$$
a_{c,k}=\left[\operatorname{LN}_k\!\left(F_{k,\mathrm{last}}(H_{c,k})\right)\right]_{e_{c,k}},\qquad
u_{c,k}=W_{k,\mathrm{proj}}a_{c,k},\qquad
g_{c,k}=\frac{u_{c,k}}{\|u_{c,k}\|_2}.
$$

$e_{c,k}$ 固定为初始化模板在对应 tokenizer 下的 EOS 位置，最后一个 block 沿用原生 CLIP 的 causal attention 和 attention 设置；$H$ 已包含位置编码，不再重复添加。不能仅取倒数第二层 EOS、直接均值池化或展平后，就将结果当作现有的 CLIP contrastive 特征。用户文本 $q$ 通过完整冻结 CLIP 得到相同 projection 空间的归一化特征 $h_k(q)$。

SD3.5 的 TI-flow 条件使用两个 $H_{c,k}$ 拼接、补齐通道后与固定 T5 序列组合，pooled 条件使用未做 L2 归一化的 $u_{c,k}$ 拼接。序列与 pooled 由同一组 $H$ 派生，联合 loss 均更新 $H$，不额外学习一份独立 pooled 参数。FLUX 原生序列条件来自 T5，CLIP 只提供 pooled：若用于 FLUX，$H$ 只经上述读出产生 native pooled $a_{c,1}$，检测另用配对 projection 得到 $g_{c,1}$；不能把 CLIP 的 $[77,D]$ 直接当作其原生 T5 condition。

完整序列增加了生成条件的优化自由度，但是否改善检测需与单 token 对比。检测最终仍是单个 projected 向量与用户文本的比较；原生 causal attention 下，EOS 后的位置不参与该分类读出，在 SD3.5 中这些位置可通过 TI-flow 的序列条件路径获得梯度。

以下两种联合 loss 共用同一检测分数，SD3.5 对两个 CLIP 等权平均，FLUX 使用一个：

$$
r_c(q)=\frac1K\sum_{k=1}^{K}h_k(q)^\top g_{c,k},\qquad
\hat y_c(q)=\mathbf1[r_c(q)\ge\tau_c].
$$

detect prototype $g_{c,k}$ 可在加载 checkpoint 时预计算；每条用户文本只需一次各 CLIP 的文本编码。推理成本不随 TI-flow 的训练图数量或训练采样步骤增加。

**4. TI-flow + BCE（`--detect-loss ti-flow-bce`）**

TI-flow 保持上面的 noisy latent、velocity target 和 MSE，使用正样本图像重建目标概念；其条件替换为由 $H$ 组成的条件。分类项使用全部正负文本，对每条文本直接约束是否超过检测阈值：

$$
\mathcal L_{c,\mathrm{BCE}}
=\mathbb E_{(q,y_c)}\operatorname{BCEWithLogits}
\left(\frac{r_c(q)-\tau_c}{T_{\mathrm{BCE}}},y_c\right),
$$
$$
\mathcal L_c^{\mathrm{TI+BCE}}
=\mathcal L_{c,\mathrm{TI}}+\lambda_{\mathrm{BCE}}\mathcal L_{c,\mathrm{BCE}}.
$$

$T_{\mathrm{BCE}}>0$ 为分类温度，$\lambda_{\mathrm{BCE}}>0$ 控制分类项权重。每次更新分别取正图像 batch 与包含正负样本的文本 batch，计算两项后求和更新同一组 $H$；两种 batch 不要求逐条配对。文本编码器冻结，用户文本特征预先缓存；从 $H$ 到分类特征的读出路径和 TI-flow 的 DiT 路径保留梯度。BCE 的训练分数与推理分数一致，训练显式使用阈值 $\tau_c$。

两种联合 loss 的 TI 权重固定为 1，分类项权重默认 1。Epoch 沿用 TI-flow：每轮遍历一次正图像，每个图像 batch 更新一次 condition。每次更新独立抽取分类文本，默认 16 条（8 正、8 负），**batch 内无放回，跨 batch 可重复**；相同文本和标签先去重，保证 batch 内没有重复文本。正负文本数量均需达到分类 batch 的一半，数量检查在训练入口完成。图像 batch 仍由 `--batch-size` 控制，默认 1。

**5. TI-flow + SupCon（`--detect-loss ti-flow-supcon`）**

采用以可训练 detect 表征为 anchor 的监督对比形式。对概念 $c$，每次抽取正文本集合 $\mathcal P_c$ 和负文本集合 $\mathcal N_c$，令 $\mathcal A_c=\mathcal P_c\cup\mathcal N_c$，分母同时包含正负样本：

$$
\mathcal L_{c,\mathrm{SupCon}}
=-\frac1{|\mathcal P_c|}\sum_{q^+\in\mathcal P_c}
\log\frac{\exp\!\left(r_c(q^+)/T_{\mathrm{Con}}\right)}
{\sum_{q\in\mathcal A_c}\exp\!\left(r_c(q)/T_{\mathrm{Con}}\right)},
$$
$$
\mathcal L_c^{\mathrm{TI+SupCon}}
=\mathcal L_{c,\mathrm{TI}}+\lambda_{\mathrm{Con}}\mathcal L_{c,\mathrm{SupCon}}.
$$

$T_{\mathrm{Con}}>0$ 为对比温度。该式是 SupCon 的 detect prototype anchor 适配：同概念正文本作为多个 positive，其余文本作为 negative，优化同一个 $H$ 派生的 anchor；各概念可独立训练。用户文本特征冻结，所以只在这些固定特征之间计算标准样本两两 SupCon，会得到与 $H$ 无关的常数，不能训练 detect。每个对比 batch 至少包含一个正文本和一个负文本；文本对比 batch 可独立于 TI 图像 batch 设置，不能直接沿用当前单文本的 `--batch-size 1`。

SupCon 约束正文本相对于负文本的匹配程度，训练目标不直接指定绝对阈值 $\tau_c$。按当前实验约定，SupCon 推理同样使用上述余弦分数，默认沿用阈值 0.2，可由采样的 `--tau` 覆盖；当前不自动选择或校准阈值。SupCon 温度默认 0.1。BCE 与 SupCon 的主要差异分别是逐条文本的阈值分类和 batch 内正负文本的相对对比，二者均可与 TI-flow 联合优化。

联合训练的分类含义由文本标签决定：TI-flow 提供图像重建约束，不能自动将“文本包含概念”的标签变成“该生成模型容易生成目标内容”的风险标签。比较实验保留分类项单独训练的基线，用于判断 TI-flow 是否带来额外检测收益。

参考：[SupCon 原论文](https://arxiv.org/abs/2004.11362)；[CLIP 的 pooling 与 projection 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py)；[SD3 的 CLIP condition](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py)；[FLUX 的 CLIP/T5 condition](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py)。

### Erase prompt

Erase 默认学习完整 sequence 和 pooled condition，初始化为 `A photo of `；safe target 默认带 CFG。`--erase-state trajectory`（默认）沿采样轨迹训练，`--erase-state image-noise` 使用概念图像随机加噪训练。`--erase-target` 控制教师目标；轨迹模式另用 `--erase-trajectory` 选择共享轨迹，默认 `unsafe`。

**1. 学习完整 sequence 和 pooled condition**

每个概念学习一组完整模型输入条件 $C_c^e=(H_c^e,p_c^e)$，其中 $H_c^e\in\mathbb R^{1\times\mathrm{Seq}\times\mathrm{Dim}}$、$p_c^e\in\mathbb R^{1\times D_p}$。两者都初始化为冻结文本编码器对 `A photo of `（末尾一个空格）的原生编码输出，再作为独立参数共同优化：

$$
(H_c^{e,(0)},p_c^{e,(0)})=\operatorname{Encode}_{\mathrm{native}}(\text{“A photo of ”}).
$$

| 模型 | Sequence condition | Pooled condition |
|---|---|---|
| SD3.5 | 两路 CLIP 的倒数第二层序列沿通道拼接、补齐通道，再与 T5 输出沿序列拼接 | 两路 CLIP 的 projected pooled 拼接 |
| FLUX | T5 输出序列 | CLIP 的 native pooled，不使用检测用的 contrastive projection |

SD3.5 的 CLIP 与 T5 序列部分均可训练；FLUX 的 T5 序列与 CLIP pooled 均可训练。每个概念保存一份 condition，训练和采样时广播到 batch，所有输入文本与时间步共享该 condition。模型权重冻结，梯度通过 DiT 回到 sequence 和 pooled；erase 直接使用这些模型输入，不需要 detect 的 CLIP 分类读出。

该方案使用 `hidden` 空间的完整条件表示；旧 soft token 保留为对照，仍初始化为 EOS 的输入 embedding，T5 固定为空文本。完整条件增加优化自由度，擦除效果与内容保留能力通过同一实验协议比较。

**推理时的 CLIP padding**：SD3.5 的 erase 条件在构造最终 negative sequence 时，将 CLIP 段的补零区域 `[:, :77, 2048:]` 显式清零，与正常 prompt 的输入布局一致；CLIP 有效通道、T5 段和 pooled 保留训练值。代码从 CLIP 配置读取长度与宽度，支持单概念和多概念平均。该处理仅作用于推理输入，训练及 checkpoint 保存保持完整 condition；FLUX 使用 T5 序列，不执行此清零操作。

**2. Safe target 支持有／无外层 CFG**

在共享的 $z_t,t$ 上，分别计算 unsafe、safe、空文本和 erase 条件的 velocity：

$$
v_u=v_\theta(z_t,t,q^+),\quad
v_a=v_\theta(z_t,t,q^a),\quad
v_0=v_\theta(z_t,t,\varnothing),\quad
v_e=v_\theta(z_t,t,C_c^e).
$$

Erase 作为 unsafe 正条件的负条件，学生输出仍为：

$$
\hat v_e=v_e+s(v_u-v_e).
$$

`--erase-target conditional|cfg` 的两种模式只改变拟合目标：

$$
v_{\mathrm{target}}=
\begin{cases}
v_a, & \text{conditional：safe 分支不加外层 CFG},\\
v_0+s(v_a-v_0), & \text{cfg：safe 分支加外层 CFG}.
\end{cases}
$$
$$
\mathcal L_e=\mathbb E\left\|\hat v_e-\operatorname{stopgrad}(v_{\mathrm{target}})\right\|_2^2.
$$

默认 `cfg`，保留 `conditional` 复现旧的无 CFG 目标。学生与 safe CFG 使用同一个 $s>1$，训练与部署一致：SD3.5 使用 `guidance_scale`，FLUX 使用 `true_cfg_scale`。FLUX 的 embedded guidance 在所有分支保持相同原生设置；`conditional` 仅表示不额外组合外层 CFG。

**3. 共享状态下的三种采样轨迹**

在 `--erase-state trajectory` 下，`--erase-trajectory unsafe|safe|student` 控制轨迹来源。每个训练样本从同一份高斯噪声 $z_T$ 出发，每个时间步始终只有一个 $z_t$；四个条件分支都在该状态上计算，不分别推进 safe 与 unsafe latent。轨迹选项决定用于推进的 velocity：

$$
v_{\mathrm{roll}}=
\begin{cases}
v_0+s(v_u-v_0), & \text{unsafe},\\
v_0+s(v_a-v_0), & \text{safe},\\
\hat v_e, & \text{student}.
\end{cases}
$$

| 轨迹 | 训练访问的状态 | 用途 |
|---|---|---|
| `unsafe` | 基础模型使用 unsafe 正条件、空文本负条件的 CFG 轨迹 | 保留当前状态采样方式，作为默认对照 |
| `safe` | 基础模型使用 safe 正条件、空文本负条件的 CFG 轨迹 | 在目标生成路径上学习匹配 |
| `student` | unsafe 正条件与当前 erase 负条件组成的 CFG 轨迹 | 在当前擦除过程访问的状态上查询 safe 教师 |

默认 `unsafe`，可显式比较 `safe` 与 `student`。Safe target 类型与轨迹来源独立配置：`conditional` 只改变 loss 的教师目标；`safe` 轨迹始终使用正常 safe CFG 推进。

每个时间步按以下顺序操作：

1. 在当前同一个 $z_t,t$ 上计算 $v_u,v_a,v_0$ 和 $v_e$；前三个分支停止梯度，erase 分支保留到 condition 的梯度。
2. 计算学生 $\hat v_e$、所选 safe target 和 loss，并保存本步 $\operatorname{stopgrad}(v_{\mathrm{roll}})$。
3. 反向传播并更新一次共享的 sequence 和 pooled condition。
4. 用保存的 rollout velocity 调用原生 scheduler，得到下一步共享 latent，并停止其梯度。`student` 使用本步更新前已计算的学生 velocity，无需重新前向。

上述过程不反向传播完整轨迹。`student` 在训练中利用当前学生的中间状态，不依赖完整图像或外部评价反馈；`unsafe` 保留原训练轨迹。三种模式的 loss 都比较同一个 $z_t,t$ 上的 velocity。CFG 组合及 MSE 使用 FP32。

**4. 概念图像随机加噪路线（`--erase-state image-noise`）**

每个概念准备少量固定训练图，默认 $N=10$。冻结基础模型直接使用概念词作为生成 prompt，例如 `nudity`、`Pikachu` 或 `Van Gogh`，负文本为空，不使用 erase condition 或外部分类器筛选。图像在首次训练准备阶段生成，之后的优化更新只从这些图像采样，不再运行完整去噪轨迹。

使用 VAE posterior mode 和原模型的 shift/scaling 得到 $z_0$。每次优化更新、每个 batch 样本独立且有放回地随机抽取一张概念图像，重新抽取噪声和时间：

$$
\sigma\sim U(0,1),\quad\epsilon\sim\mathcal N(0,I),\qquad
z_t=(1-\sigma)z_0+\sigma\epsilon,\quad t=1000\sigma.
$$

SD3.5 和 FLUX 都采用上述 flow 加噪；FLUX 另按原生接口 pack latent 并归一化时间。unsafe/safe 文本继续使用当前概念在 `dataset/erase/` 中的文本对，图像抽样与文本对独立。四个分支在同一个 $z_t,t$ 上预测，仍使用第 2 节的 safe-velocity CFG 拟合 loss；这里借鉴 TI/CI 的状态采样方式，不将 erase loss 改为重建真实噪声或 $\epsilon-z_0$。

每轮 epoch 仍遍历一次 unsafe/safe 文本对，每个文本 batch 仅更新一次 condition；默认 batch=1 即每次随机选一张图、一个时间步。此模式不调用 scheduler 推进训练 latent，`--erase-trajectory` 不适用；同时显式指定该选项时在入口报错。`--erase-target conditional|cfg` 均可使用。

图像缓存位于：

```text
dataset/erase/concept_images/<模型>/<概念>/<开始时间>/
├── images.json          # 生成协议、原始概念词、每张图片的 seed 和路径
├── concept_000.png
├── concept_001.png
└── …
```

- `--erase-image-count` 默认 10，控制本次使用的图像数量。
- 默认开启 `--erase-reuse-images`。续训优先复用 checkpoint 指向的兼容批次，否则查找最近的兼容批次；按基础模型、概念词和完整生成协议匹配。没有匹配批次时创建新批次。
- 缓存数量少于请求值时补充新图，更多时取前 N 张并保留其余图；缺失或损坏的已记录图片按原 seed 补齐。
- `--no-erase-reuse-images` 在每次启动训练（包括续训）时新建一批图片，每张使用新的随机 seed，保留旧批次。不会在每个优化步骤重新生成图片。
- 图像生成 seed 独立于训练 `--seed`，实际 seed 均写入 `images.json`；训练 `--seed` 控制文本打乱、图像抽样、加噪和时间抽样。
- `--erase-image-generation-steps` 默认继承 `--sampling-steps`。生成尺寸、CFG、模型 revision 等继承本次模型配置；协议变化会使用其他兼容批次或新建批次。

VAE 编码结果在一次训练中缓存在 CPU，优化时只抽取所需 latent。Checkpoint 额外记录 `erase_image_manifest` 和实际使用的 `erase_image_count`，用于追踪训练图像来源。

新训练的默认配置如下：

| 配置 | 默认值 | 可选项／含义 |
|---|---|---|
| Erase `--prompt-space` | `hidden` | 完整 sequence 与 pooled；`soft` 保留为对照 |
| 初始化文本 | `A photo of ` | 同时初始化 sequence 与 pooled |
| `--erase-target` | `cfg` | `conditional` / `cfg` |
| `--erase-state` | `trajectory` | `trajectory` / `image-noise` |
| `--erase-trajectory` | `unsafe` | 仅轨迹模式：`unsafe` / `safe` / `student` |
| `--erase-image-count` | 10 | 仅 image-noise：每概念图像数量 |
| `--erase-reuse-images` | 开启 | 关闭时新建随机 seed 批次并保留旧图 |

配置保存在 `train_erase/training.json`。Checkpoint 保存 `erase_sequence`、`erase_pooled`、初始化文本 `erase_fixed_text`、`erase_target`、`erase_state`、`erase_trajectory` 和 `cfg_scale`；image-noise 的 `erase_trajectory` 记为 null。`--resume` 沿用已保存的空间、模板、目标类型、状态来源及图像配置；旧实验未记录目标类型时沿用 `conditional`，状态来源为 `trajectory`、轨迹为 `unsafe`，原条件张量与初始化记录保留。显式传入选项可切换续训配置。

**检测与采样**：默认开启 `--all-detect-prompts`，扫描训练运行目录 `train_detect/*.pt`，所有已保存的 detect prompt 共同参与比较。`cosine` 在达到 tau 的候选中选最高分；全部低于阈值时不预测、不触发。命令行 `--tau` 覆盖所有 detector 的阈值，未指定时分别使用 checkpoint 中的训练 tau。`l2` 选完整 CLIP/T5 序列与 pooled 的 RMS 距离最小值，全量 L2 模式仍不使用距离阈值。

候选集合与实验擦除目标分开：`predicted` 是通过门控后的 top1（未达阈值时为空），`selected` 是 `predicted ∩ case.targets`。只有选中的概念属于当前实验的擦除目标时，才使用对应 erase prompt；否则使用空文本负条件。例如 `ip_pikachu` 的输入被预测为 Snoopy 时，记录预测 Snoopy，但不触发 Pikachu 擦除。

使用 `--no-all-detect-prompts` 限定为 case.targets 内的阈值检测：余弦达到 tau／RMS 距离不超过阈值才触发。`top1` 选有效候选的最佳项，`mean` 对所有有效项的 erase 条件取平均。全量模式仅支持 top1。

## 二、超参数

| 参数 | 默认值 / 含义 |
|---|---|
| `--stage` | `all`；当前仅训练检测时显式指定 `detect` |
| `--prompt-space` | Detect 默认 `condition`，Erase 默认 `hidden`；单 token 对照用 `soft`；联合 loss 要求 `condition` |
| `--erase-target` | `cfg`；可选 `conditional`，续训继承已保存设置 |
| `--erase-state` | `trajectory`；可选 `image-noise`，续训继承已保存设置 |
| `--erase-trajectory` | `unsafe`；仅 trajectory，可选 `safe` / `student` |
| `--erase-image-count` | 10；image-noise 每概念使用的图像数 |
| `--erase-image-dir` | `dataset/erase/concept_images`；图像缓存根目录 |
| `--erase-image-generation-steps` | 继承 `--sampling-steps`；仅控制缓存图像生成 |
| `--erase-reuse-images` | 默认开启；`--no-erase-reuse-images` 强制新建随机 seed 图像批次 |
| `--soft-template` | Soft 默认 `{}`；恰好一个占位符；condition 的初始化模板固定为 `a photo of {}` |
| `--detect-loss` | Detect 训练必须显式指定：`ti-flow-bce`、`ti-flow-supcon`、`clip`、`ti-flow` |
| `--learning-rate` | `1e-3`，AdamW，weight decay=0 |
| `--epochs` | 5，每个概念、阶段分别计数 |
| `--max-steps` / `--steps`（训练） | 每概念、阶段最多 2000 次优化更新；0 不设上限 |
| `--batch-size` | 1 |
| `--classification-batch-size` | 16；联合 loss 每次抽取等量正负文本，batch 内无放回 |
| `--classification-weight` | 1；联合 loss 的分类项权重，TI 项权重固定为 1 |
| `--contrastive-temperature` | 0.1；SupCon 温度，与 BCE 的 `--temperature` 独立 |
| `--tau`、`--temperature` | 0.2、0.1，用于训练分类 loss；tau 在全量和非全量 cosine 检测时均用于门控 |
| `--seed` | 0，训练噪声、时间和 shuffle 的随机种子 |
| `--ti-image-count` | 全部正样本，当前每概念 30 张；可限制前 N 条 |
| `--ti-generation-steps` | 继承 `--sampling-steps`，用于固定训练图的生成 |
| `--ti-seed` | 继承 `--seed`；第 j 张图使用 seed+j |
| `--sampling-steps`（训练） | benchmark 默认步数，用于 erase 轨迹和 TI 训练图 |
| `--steps`（采样） | 继承训练采样配置；仅检测模式不使用 |
| `--mscoco-prompt-count`（采样） | 1000，MSCOCO caption 数量 |
| `--all-detect-prompts`（采样） | 默认开启：全部已保存 detector 参与 top1，cosine 使用 tau；`--no-all-detect-prompts` 关闭 |
| `--output-dir`（采样） | 独立结果根目录；不填时写回 `--run-dir` |
| `--similarity`（采样） | `cosine`；旧 soft/hidden checkpoint 可选 `l2` |
| `--l2-threshold` | 0.2，仅关闭全量模式时用于门控 |
| `--multi-concept` | `top1`；关闭全量模式后可选 `mean` |
| CFG | SD3.5 继承 benchmark guidance；FLUX true CFG=2，embedded guidance 继承 benchmark |
| `--dtype` | 继承模型配置，通常 BF16；参数及优化器状态为 FP32 |
| `--gradient-checkpointing` | 默认开启，可用 `--no-gradient-checkpointing` 关闭 |

Epoch 和更新数上限先到者结束。CLIP 分类 50 条、batch=1、5 epochs 是 250 次更新；TI-flow 及联合 loss 以图像计数，30 张、batch=1、5 epochs 是 150 次。Erase trajectory 每个去噪步骤更新一次，50 对 × 40 步已达 2000 次；image-noise 每个文本 batch 更新一次，50 对、batch=1、5 epochs 是 250 次，与缓存图像数 N 无关。`--max-steps` 始终表示优化更新次数上限，0 不设上限。

## 三、实现与运行配置

训练 TI-flow+BCE，并评测文本检测能力：

```bash
RUN_DIR="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
python DiT/detect_erase/train.py --run-dir "$RUN_DIR" --model sd35_medium \
  --stage detect --detect-loss ti-flow-bce --track
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --cases all \
  --detect-only --batch-size 16 --track
```

训练 TI-flow+SupCon 时使用独立目录：

```bash
SUPCON_RUN="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-ti-supcon-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
python DiT/detect_erase/train.py --run-dir "$SUPCON_RUN" --model sd35_medium \
  --stage detect --detect-loss ti-flow-supcon \
  --classification-batch-size 16 --classification-weight 1 --contrastive-temperature 0.1 --track
python DiT/detect_erase/sample.py --run-dir "$SUPCON_RUN" --cases all \
  --detect-only --batch-size 16 --track
```

两条训练命令默认使用完整 condition、图像 batch=1、分类 batch=16、分类权重=1、余弦阈值=0.2。`clip` 和 `ti-flow` 可单独运行作为分类项／重建项对照；需要旧单 token 对照时显式加 `--prompt-space soft`。

`--detect-only` 不需要 erase checkpoint，不加载 DiT/VAE、不生成参考图或实验图、不计算图像指标。余弦只加载 CLIP，L2 额外加载 T5。它每次重算所选文本；指定新的 `--output-dir` 可独立保存此次全部结果，原有训练与实验文件保持原样。即使关闭 `--track` 仍保存检测统计和图；`--track` 额外同步 SwanLab。

快速检查文本检测：

```bash
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --detect-only \
  --i2p-prompt-count 20 --ip-template-count 2 --style-template-count 2 --object-template-count 2
```

旧 soft/hidden checkpoint 仍可加 `--similarity l2`；新的 condition detector 使用 CLIP 余弦。

仅 TI-flow 对照另建实验根目录：

```bash
TI_RUN="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-ti-flow-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
python DiT/detect_erase/train.py --run-dir "$TI_RUN" --stage detect \
  --detect-loss ti-flow --ti-generation-steps 40 --ti-seed 0 --track
python DiT/detect_erase/sample.py --run-dir "$TI_RUN" --detect-only --track
```

`train_detect/ti_images/<概念>/positive_000.png` 等保存训练图，`images.json` 记录文本、种子和生成协议。`--resume` 复用完整图片，只补缺失／损坏的图片；缓存协议改变时另建 run。续训重新创建优化器：

```bash
python DiT/detect_erase/train.py --run-dir "$RUN_DIR" --stage detect \
  --detect-loss ti-flow-bce --targets nudity --resume --epochs 5 --track
```

旧 hidden 输出对照使用 `--prompt-space hidden`，从新 run 开始。`--cases`、`--targets`、`--limit-prompts` 可限制训练范围；联合 loss 的文本截断后仍需有足够的正负文本。快速检查可限制 `--ti-image-count` 和 `--max-steps`，保留完整分类文本。`--model flux1_dev` 切换模型。默认余弦检测的 CLIP 放在 `--device`，不使用生成模型的 `--device-map`。

后续训练 erase 并生成／计算 benchmark 指标：

```bash
python DiT/detect_erase/train.py --run-dir "$RUN_DIR" --stage erase \
  --erase-target cfg --erase-trajectory unsafe --track
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --cases all --track
```

Erase 默认使用完整 condition；将 `--erase-trajectory` 改为 `safe` 或 `student` 可比较共享轨迹，将 `--erase-target` 改为 `conditional` 可比较无 CFG 教师。单 token 对照显式加 `--prompt-space soft`。不同配置使用独立 run；续训同一配置使用 `--resume`，未显式指定的目标与轨迹继承原训练设置。

概念图像随机加噪训练（首次自动生成每概念 10 张，之后默认复用）：

```bash
ERASE_RUN="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-erase-image-noise-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
python DiT/detect_erase/train.py --run-dir "$ERASE_RUN" --model sd35_medium \
  --stage erase --erase-state image-noise --erase-target cfg \
  --erase-image-count 10 --erase-reuse-images --track
```

同样的 image-noise 命令加 `--no-erase-reuse-images` 可强制生成新批次；加 `--erase-image-generation-steps 40` 可指定缓存图像的生成步数。该路线不传 `--erase-trajectory`。只训练 erase 的新 run 尚无 detect checkpoint；完整门控采样需在同一 run 中另行训练 detect。

`--phase sample` 仅生成，`--phase evaluate` 仅评测已生成图片。I2P 使用 NudeNet（Q16 可选），IP/style 使用 CS、FID、VLM，object 使用 ResNet Top1/Top3，MSCOCO 使用 CS、reference CS 和 FID。参考图复用 `benchmark/original/<模型>/` 中完整协议匹配的图片。

完整训练、采样及 benchmark 可使用以下两个脚本，所有阶段均开启 SwanLab 追踪：

```bash
# 单机双卡：GPU 0 训练 detect，GPU 1 训练 erase；完成后分组并行采样及评测。
bash DiT/detect_erase/scripts/run_dual_gpu.sh

# 单卡：detect → erase → 全部采样及评测，依次在 GPU 0 上执行。
bash DiT/detect_erase/scripts/run_single_gpu.sh
```

两者默认 `MODEL="sd35_medium"`、`DETECT_LOSS="ti-flow-supcon"`、`ERASE_TARGET="cfg"`、`ERASE_STATE="trajectory"`、`ERASE_TRAJECTORY="safe"`，detect 使用 `condition`，erase 使用 `hidden` 完整 sequence 和 pooled condition。两个脚本的 erase 训练均通过 `--erase-dir DiT/detect_erase/dataset/safe-describe` 使用正向画面描述数据集。所有配置均在脚本顶部直接赋值，修改后运行脚本即可。例如，要切换 BCE、无 CFG 教师和 student 轨迹，在脚本中修改：

```bash
DETECT_LOSS="ti-flow-bce"
ERASE_TARGET="conditional"
ERASE_STATE="trajectory"
ERASE_TRAJECTORY="student"
```

要切换概念图像加噪并生成新批次，在脚本中修改：

```bash
ERASE_STATE="image-noise"
ERASE_IMAGE_COUNT=10
ERASE_REUSE_IMAGES=false
```

`DETECT_LOSS` 可选 `clip/ti-flow/ti-flow-bce/ti-flow-supcon`；`ERASE_TARGET=conditional` 表示无 CFG 的 safe target。`ERASE_STATE=image-noise` 时脚本不传轨迹参数；`ERASE_REUSE_IMAGES` 默认 `true`，设为 `false` 则生成新批次并保留旧图。双卡编号通过 `DETECT_GPU`、`ERASE_GPU` 指定，单卡通过 `GPU` 指定。`TRACK_MODE` 默认 `local`，可改为 `offline/online`；`SAMPLE_PHASE` 默认 `all`，设为 `sample` 仅采样。每次自动新建带时间的实验目录，也可用 `RUN_ROOT` 指定目录。训练默认学习率 `1e-3`、5 epochs、每概念每阶段最多 2000 次更新、batch=1，可通过 `LEARNING_RATE/EPOCHS/MAX_STEPS/BATCH_SIZE` 修改；采样 batch 使用 `SAMPLE_BATCH_SIZE`。

`--cases all` 包含 MSCOCO；也可单独指定 `--cases mscoco`（case ID 为 `mscoco_nudity`）。它复用 benchmark 中 nudity 的保留能力协议，默认取 `benchmark/dataset/mscoco.csv` 前 1000 条 caption，擦除目标仍为 nudity；默认与全部已训练 detector 比较，无需新增训练数据。仅检测时把这些 caption 作为无关文本对照，统计误触发率；完整采样时同样经过门控，计算 CS/FID，并复用协议一致的参考图。

```bash
python DiT/detect_erase/sample.py --run-dir "$RUN_DIR" --cases mscoco \
  --detect-only --mscoco-prompt-count 1000 --track
```

`--mscoco-prompt-count` 可缩小实验；生成并计算 FID 时至少取 2 条。双卡脚本将 MSCOCO 与 i2p/style 放在同一张卡。

重新测试已有 prompt，结果单独保存（包含 MSCOCO）：

```bash
cd /home/shx/code/ce-new
TRAIN_RUN="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-detect-20261112-202026"
TEST_RUN="/home/shx/code/ce-new/DiT/detect_erase/results/sd35_medium-detect-all-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python DiT/detect_erase/sample.py \
  --run-dir "$TRAIN_RUN" --output-dir "$TEST_RUN" \
  --cases all --detect-only --all-detect-prompts \
  --similarity cosine --multi-concept top1 --batch-size 16 --device cuda:0 \
  --track --swanlab-mode local --swanlab-name detect-all-benchmark
```

所有结果位于 `$TEST_RUN/exps/<case>/`，汇总表、日志和 SwanLab 在 `$TEST_RUN` 下。无需重新训练或复制 checkpoint，`--run-dir` 始终指向原训练目录。完整图片采样的续跑／单独评测也需保留同一组 `--run-dir` 和 `--output-dir`。

**保存结构**：模型名和开始时间仅在根目录出现一次。训练目录和各总汇表留在根目录，各 case 的检测、图片、指标和耗时统一放在 `exps/`。每完成一个概念立即单独保存 checkpoint。

```text
results/sd35_medium-<开始时间>/
├── run.json                       # 配置与逐概念 checkpoint 索引
├── train_detect/
│   ├── nudity.pt, pikachu.pt, …    # 每概念独立 CPU FP32 参数，不保存模型权重
│   ├── training.json
│   ├── loss.csv, epochs.csv, training_summary.json  # --track 时
│   └── ti_images/nudity/           # TI-flow 及两种联合 loss 创建
├── train_erase/                    # 训练 erase 后才创建
├── detect_summary.csv/json         # 全部 case 的检测汇总
├── exps/
│   ├── i2p_nudity/
│   │   ├── detect_manifest.json
│   │   ├── detect_gates.jsonl      # scores、matched、selected
│   │   ├── detect_records.csv     # 每条文本及其真实标签、检测结果
│   │   ├── detect_summary.json
│   │   ├── detect_by_prediction.csv  # 全量 top1 分布
│   │   ├── detect_by_concept.csv
│   │   ├── detect_by_detect_prompt.csv
│   │   ├── detect_plots/           # 混淆矩阵和比率图，各有 PNG/PDF
│   │   ├── similarity_plot/       # 每个 detector 的正负文本分布图和分数明细
│   │   │   ├── nudity.png, nudity.pdf
│   │   │   ├── records.csv        # 按唯一文本 × detector 保存分数、阈值和选中情况
│   │   │   ├── summary.csv, summary.json
│   │   │   └── histogram.csv      # 每个区间的数量与组内百分比
│   │   └── images/, manifest.json, metrics.json, sampling_time.json, …
│   ├── mscoco_nudity/             # MSCOCO 保留能力及误触发对照
│   ├── ip_pikachu/
│   └── …                          # 其他 case 同级
├── logs/
└── swanlog/
```

Condition detect checkpoint 格式为 `detect-erase-concept-v3`，保存 `detect_clip_conditions`（每个 CLIP 一个 `[1,77,D]` 张量）、`detect_eos_positions`、初始化模板、loss、阈值、分类权重／温度／batch、图像数和更新数，参数为 CPU FP32。加载后从这些序列读出并缓存检测 prototype。

旧 `detect-erase-concept-v2` 仍可读取。Soft 文件保存 `<stage>_soft_tokens`（每个 CLIP 一个 `[1, hidden_size]` 张量）、空间、固定文本和模板。Hidden detect 保存 `detect_pooled`，完整序列在需要 L2/TI 时由固定概念生成；hidden erase 保存 `erase_sequence` 和 `erase_pooled`。Detect 与 erase 可采用不同空间，运行时分别编码。

**检测统计**以实际触发的 selected 为准。日志和逐条 CSV 同时记录全部候选分数、`predicted`（门控后的 top1）和 `selected`（实际触发）；原始日志另保存实际阈值 `thresholds`、是否应用阈值和达到阈值的 `matched` 列表。

- `detect_by_prediction.csv` 统计各候选被预测的次数和比率；该表覆盖全部已加载概念。JSON 同时记录 candidate_targets 和 threshold_applied。
- 每个 case 擦除目标一对其余的 TP/TN/FP/FN，以及 precision、recall、F1、FPR、FNR、binary accuracy。判成其他目标同时产生正确目标的 FN 和错误目标的 FP；汇总计数的分母是“已记录文本数 × case 目标数”。预测为 case 外概念不触发擦除：目标输入计漏检，无关输入计正确不触发。
- 保留正确触发率、其他目标触发率、完全未触发率、无关文本误触发率和 exact-set accuracy。分母为 0 记录 null；缺少日志的输入单独计为 missing，不充当 TN。
- I2P 按 `benchmark/dataset/real_nude_mapping.csv` 的 sample_id＋seed 对齐，125 条 `is_real_nude=1` 为正，其余为无关文本；标签只用于评测。IP/style/object 使用 benchmark concept 标签；MSCOCO 按保留能力对照记为非目标输入。
- 完整采样始终保存同样统计及 `trigger_plots/`，文件前缀为 `trigger_`，原始门控日志为 `gates.jsonl`。分数在检测完成后、图像生成前写入，不依赖 `--track`。缓存图片对应日志缺失时显示 coverage；重复日志取最后一次。

**相似度分布**保存在各 case 的 `exps/<case>/similarity_plot/`，检测和完整采样都会输出，不依赖 `--track`。非全量模式只画 case.targets 中的 detector；全量模式为所有已加载 detector 分别画一张图。每张图只分为“该 detector 对应概念文本”和“其余文本”两组；多目标 case 的其他目标文本属于当前 detector 的负类。分组使用上述评测标签，不使用预测结果。

横轴为 cosine similarity，纵轴为每个区间占该组文本总数的百分比，两组分别归一化到 100%。每个 detector 的横轴范围取两组实际分数的最小值和最大值，下界向下、上界向上取到最近的 0.05 倍数；所有分数相同时至少保留 0.05 的宽度。cosine 分箱仍使用 [-1, 1]、宽度 0.02 的固定网格，画出实际 tau 竖线，分别标注两组达到阈值（`score >= tau`）的数量和比例；另标注原始分数 top1 与实际 selected 数量。L2 图使用 RMS 距离，非全量模式标记距离阈值。没有正例或负例时显示 n=0，对应比例为空。

原始门控日志保留每个 sample/seed 的记录，分布图和 `similarity_plot/records.csv` 按 case 内完整 prompt 文本去重，取最后一条已记录样本，`sample_count` 标记对应的样本数量。`summary.json` 同时记录缺失样本数、唯一文本数、分箱和归一化口径。已有全量日志若未使用阈值，补画时仍展示 checkpoint tau，但标注为参考阈值，不改写历史检测结果。

**训练追踪**由 `--track` 开启：loss、学习率、梯度／参数范数、batch 大小、更新数；CLIP 分类额外记录训练批次准确率和正／负平均余弦；flow 记录时间步。联合 loss 分别记录 `ti_loss`、`classification_loss`、分类 batch 大小及正负余弦，`batch_size`／epoch examples 表示 TI 图像数量。每轮 epoch 同步 GPU 后计时，记录完整／部分 epoch 和总时间，TI 图生成、模型加载、预编码及保存计入总时间的 outside-epochs 部分。

SwanLab 默认 local，需要 `pip install 'swanlab[dashboard]'`；可选 `--swanlab-mode offline/online`。看板命令：`swanlab watch "$RUN_DIR/swanlog"`。未开启追踪时不导入 SwanLab。

仍需实验选择的配置：学习率、温度、训练时长、阈值、soft template、TI 图数量和生成步数，以及 top1/mean。当前采用上表默认值，不通过最终图像指标自动调参。

### 训练数据集来源与构造

目标取自 `benchmark/configs/dit_cases.json`，共 11 个。数据分别在 `dataset/detect/`、`dataset/erase/`，每个目录按 i2p、ip、style、object 分四个 JSON。

| 类别 | 目标 | Detect 每概念 | Erase 每概念 |
|---|---|---:|---:|
| i2p | nudity | 50 条 | 50 对 |
| ip | Pikachu、Snoopy | 50 条 | 80 对 |
| style | Van Gogh、Monet、Picasso | 50 条 | 30 对 |
| object | church、golf ball、parachute、cassette player、banana | 50 条 | 80 对 |

Detect 每概念 30 正样本＋10 相近负样本＋10 普通负样本。I2P 由 AI 直接编写成人裸体、穿衣成人和无关场景；IP/object 使用 Textual Inversion 对象模板 `dataset/personalized.py`，style 使用 `dataset/personalized_style.py`，填入目标／负概念。去重后不足则循环，style 17 种模板因此有重复；负概念保存在 JSON 的 `similar_concepts`、`ordinary_concepts`。CLIP loss 使用全部 50 条，TI-flow 只取 30 条正样本生成图像。联合 loss 复用这些正图像，分类分支从全部正负文本中先去重再按 batch 抽样。

Erase 的 I2P 为 50 对共享上下文的成人场景，仅将 nude 换成 clothed。IP/object 使用 benchmark 全部 80 个 ImageNet 模板，style 使用全部 30 个 painting 模板；分别填入 target 与 `anchor=""`，空 anchor 保留模板上下文。按约定直接复用测试模板。

共 550 条 detect 文本、700 对 erase 文本。固定场景和负概念均保存在生成脚本，复现：`python DiT/detect_erase/dataset/generate.py`。
