# EraseAnything 的 DiT 实验适配

本页记录 FLUX.1-dev 适配。SD3.5 Medium 使用独立入口 `scripts/benchmark_sd35_all.sh`，详见 [SD35_ADAPTATION.md](SD35_ADAPTATION.md)。

执行层更新：训练图生成、训练、采样和评测均使用独立子进程，总控隐藏 GPU；阶段结束时退出进程以释放 CUDA 上下文。公共实现为 `benchmark/execution.py`。全部 CUDA 组件仅能使用 `--device` 指定的一张卡，其他组件可选 CPU；辅助 GPU 和内存预算不能指向其他卡，自动放置也限制在单卡内。训练适配只新增设备编号映射，原四项损失未改。执行层升级通过精确源码摘要映射兼容已有训练数据、权重和计划，不修改历史 JSON。

当前工作区原先没有 EraseAnything，已从[作者仓库](https://github.com/tomguluson92/EraseAnything)下载源码，上游提交为 `67d8c0bd5b153ff0e394b45e893b6671bc40e30b`。

**本页入口实现 FLUX.1-dev 的四类实验。** 作者发布的训练入口、latent 打包、时间步处理、注意力图输出和四项损失都针对 FLUX，因此本页 `benchmark_dit.py` 默认且仅运行 FLUX.1-dev。SD3.5 的模型调用、注意力及损失接入另由新增文件实现，请使用上述独立入口。

FLUX 方法适配由本目录中的两个 Python 文件、四个单项脚本和一个总入口组成。公共执行层及 Object 配置与其他 baseline 共用并同步更新，没有编写新的测试文件。使用当前安装的 Diffusers/PEFT，不安装仓库内的旧 fork。

## 修改后可行的实验

实际读取 `benchmark/configs/dit_cases.json`、`dit_sampling.json`、`dit_evaluation.json`。训练的 guide 读取 `dit_target_anchor.json`，作为作者 `calculate_upper_loss` 的 `neg_prompts`，可用本地配置覆盖。

2026-10-31 Object 更新：保留 church 单目标与五目标联合训练，联合目标改为 church、golf ball、parachute、cassette player、banana。评测十类和 anchor 对应关系见[公共类别表](../../benchmark/exp_plan/dit_experiment_plan.md#43-objectimagenet-10)，FLUX 与 SD3.5 入口共用新的 `dit_cases.json` 和既有 `dit_target_anchor.json`。此次同步修改公共 `dataset/imagenet10.csv`、`object_classifier.IMAGENET10` 及 Object 脚本注释；数据准备和四项损失代码未改。新 Object 配置会生成新的评测计划，需要重新 prepare（`--phase all` 自动包含此步骤）；旧五目标训练数据与检查点需使用新的 `--run-root` 重新生成和训练，历史记录保持原样。

| 实验组 | case | 擦除目标 | 评测内容 | FLUX.1-dev |
|---|---|---|---|---|
| I2P | `i2p_nudity` | nudity | sexual 931 张、NudeNet；同一权重额外评测 COCO 1K 的 CS/FID | 已接入 |
| IP | `ip_pikachu` | Pikachu | Snoopy/Pikachu/Mickey，960 张，CS/FID、Qwen 单标签分类 | 已接入 |
| IP | `ip_snoopy_pikachu` | Snoopy、Pikachu | 同上，960 张 | 已接入，联合训练 |
| style | `style_van_gogh` | Van Gogh | 三位画家，270 张，CS/FID、Qwen 单标签分类 | 已接入 |
| style | `style_monet` | Monet | 同上，270 张 | 已接入 |
| style | `style_picasso` | Picasso | 同上，270 张 | 已接入 |
| Object | `object_church` | church | ImageNet-10，共 500 张，官方 ResNet-50 V2 Top-1/3 | 已接入 |
| Object | `object_multi5` | church、golf ball、parachute、cassette player、banana | 同上，500 张 | 已接入，五目标联合训练 |

共 **8 个训练 case、9 项评测任务、5,661 张方法图**。I2P 和 COCO 使用同一份 nudity LoRA；其他 case 不附带 COCO。默认 I2P 不启用 Q16，`--with-q16` 改为全量 4,703 条，NudeNet 仍检测 sexual 子集。这里不使用旧 SD1.4 的 CIFAR-50→COCO 实验。

正式采样使用公共原生 FLUX pipeline：**BF16、512×512、28 步、guidance 3.5、T5 长度 512**，保留基础模型自身 scheduler，以及公共的逐记录 CPU 初始噪声、latent 打包和原图共享机制。

## 启动入口

```bash
# 一次依次执行四项实验，默认 FLUX.1-dev
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_dit_all.sh \
  --device cuda:0 --offline

# 只查看 case 和命令，不创建数据或加载权重
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_dit_all.sh --dry-run

# 只运行多目标 IP case
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_dit_ip.sh \
  --cases ip_snoopy_pikachu --device cuda:0 --offline
```

| 脚本 | 执行内容 |
|---|---|
| `scripts/benchmark_dit_i2p.sh` | nudity 训练，I2P＋COCO |
| `scripts/benchmark_dit_ip.sh` | 两个 IP case |
| `scripts/benchmark_dit_style.sh` | 三个 style case |
| `scripts/benchmark_dit_object.sh` | 两个 Object case |
| `scripts/benchmark_dit_all.sh` | 上述全部，顺序执行 |

脚本只转发参数，可在任意目录调用；显式路径相对于调用目录，`PYTHON` 可选择解释器。`--base-model /本地目录` 可指定 FLUX.1-dev 的本地 Diffusers 权重。

`--phase all` 默认执行：生成训练图片→训练→准备评测清单→采样→评测。也可用 `data`、`train`、`prepare`、`sample`、`evaluate` 分阶段执行。train 要求已有训练数据；prepare 要求已有匹配的检查点，sample/evaluate 使用匹配的已准备计划。保持相同参数及目录即可复用已完成训练或补齐评测图片。

`--offline` 限制资源读取为本地或缓存，尚未下载完整模型时去掉它。依赖为 `requirements-benchmark.txt`，指标权重准备见 `benchmark/DIT.md`。支持 `--vlm-model`、`--resnet-weights-path`、`--skip-vlm` 等公共评测选项。

`--cpu-offload`、`--device-map balanced/auto`、`--max-memory` 控制训练图片生成及后续评测采样；训练 denoiser 仍位于单个 `--device`，这些选项不启用训练 offload。文本编码器默认使用 CPU，可用 `--text-device cuda:0` 改为 GPU；`--gradient-checkpointing` 作用于训练。完整模型、注意力图和梯度仍需要足够显存。

## 保留的算法与必要核心修改

直接调用作者 `utils/calc_loss.py` 的两项上层损失与两项下层损失：

```text
上层：λ1 · 擦除预测 MSE + λ2 · 目标 token 的注意力抑制
下层：λ3 · Flow Matching MSE + λ4 · 作者原实现的 InfoNCE 项
```

保留作者 InfoNCE 的符号、温度和计算函数，也保留当前模型 `no_grad` 分支产生教师信号的方式。四项权重默认均为正，未删除任一损失。LoRA 仍只插入 `attn.add_k_proj`、`attn.add_q_proj`，其余基础参数冻结。

主要接入改动：

1. **现代 Diffusers 的注意力输出。** 作者 fork 在 transformer 输出中额外返回最后一个双流 block 的完整注意力矩阵。新入口用 hook 按相同 Q/K 投影、归一化、RoPE、缩放和 text-first 顺序计算该矩阵，并附加到训练输出；保留正常模型前向，评测时不安装 hook。这样不用把旧 Diffusers/PEFT 全局安装到工作环境。数学定义相同，但不宣称与旧 fork 在不同 kernel 下逐位一致。
2. **设备、精度与分辨率。** `utils/esd_utils.py`、`utils/calc_loss.py` 将写死的 CUDA 设备、`.cuda()`、BF16 和 512 改为实际模型设备、输入 dtype、配置分辨率；修正 VAE 下采样倍率为 `2**(层数-1)`，使 VAE latent、采样和 unpack 使用一致几何。正式协议仍为 512/BF16，小模型验证才显式缩小。
3. **训练 scheduler 状态。** `tools/scheduler_process.py` 的 `set_train_timesteps` 同步设置 `sigmas=t/1000`、末尾零 sigma，并重置 step/begin index。原函数只更换 timesteps，连续 rollout 会沿用旧 index/sigma 表；此修复影响训练轨迹，已作为核心计算修复记录，正式评测仍使用原模型 scheduler。
4. **多概念训练。** 每个 case 使用一个 LoRA 和一个 AdamW；按目标洗牌循环，每个目标执行一对上层／下层更新，下层沿用该对上层选取的时间。不是分别训练后合并，也不是把多个名字拼成一个目标字符串。

## 数据准备和与作者流程的差异

作者原入口依赖本地训练图片、WordNet 同义词和在线大模型返回的无关概念，仓库没有提供本次 case 的这些材料。新入口默认：

- 原模型为每个目标生成 4 张独立训练图，固定 seed，保存 prompt、seed、图片校验值；不读取 I2P/COCO 测试 prompt 或 IP/style 的评测模板。
- 通用概念训练 prompt 为 `an image depicting <目标>`，风格为 `an artwork painted in the style of <艺术家>`；guide 来自现有 DiT anchor 映射。
- 默认同义词列表为目标本身，三个无关概念为 `mountain`、`bicycle`、`fruit`。不自动调用外部 API、不下载 WordNet，不把这些固定词宣称为作者大模型输出。
- 保留 90% 概率的词序打乱、50% 概率的同义词替换和随机裁剪。目标 token 按作者的 token-ID membership 规则定位，并排除 EOS/padding。

**默认运行属于具有明确本地数据准备设置的 EraseAnything 适配实验，不是作者原始训练数据和提示扩展流程的逐项复现。** 这些设置会影响结果，正式论文比较时应报告。可用 `--training-config /路径/concepts.json` 显式设置每个目标的训练数据描述：

```json
{
  "Pikachu": {
    "prompt": "an image depicting Pikachu",
    "anchor": "Cartoon Character",
    "synonyms": ["Pikachu"],
    "irrelevant": ["mountain", "bicycle", "fruit"]
  },
  "Van Gogh": {
    "synonyms": ["Van Gogh", "Vincent van Gogh"],
    "irrelevant": ["airplane", "apple", "chair"]
  }
}
```

未写字段沿用默认值；每个目标必须有三个无关概念。最终完整配置保存到 dataset 和 checkpoint metadata，不依赖之后重新查询外部服务。

训练默认值沿用作者配置：rank=8、学习率 1e-3、AdamW、负向引导 1、四项权重 `[1, 0.001, 0.1, 0.1]`、BF16、512 分辨率、28 个 rollout 时间步、T5 长度 256。rollout guidance 为作者实现中的 3，直接预测 guidance 为 3.5；正式评测 T5 长度仍为公共配置的 512。

每目标默认 200 次优化器更新，即 100 对上层／下层更新；双 IP 为 400 步，五 Object 为 1,000 步。可用 `--iterations` 指定总更新数，或 `--iterations-per-concept` 改变每目标预算；须为偶数且至少为每目标提供一对更新。支持 `--rank`、`--lr`、`--negative-guidance`、`--loss-weights`、`--training-resolution`、`--training-steps`、`--training-sequence-length`、`--training-dtype`。

## 文件记录与保存位置

| 新增或修改文件 | 内容和目的 |
|---|---|
| `benchmark_dit.py` | 新命令入口、配置编译、阶段编排、checkpoint 对应检查 |
| `benchmark_training.py` | 训练图准备、注意力兼容 hook、调用原生四项损失的联合 LoRA 训练 |
| `utils/esd_utils.py` | 设备、dtype、latent 几何接入 |
| `utils/calc_loss.py` | VAE 倍率、设备和分辨率接入，保留损失公式 |
| `tools/scheduler_process.py` | sigma 表和 rollout index 重置 |
| `scripts/benchmark_dit_{i2p,ip,style,object,all}.sh` | 四个独立入口和一个总入口 |
| `requirements-benchmark.txt`、`.gitignore` | 公共现代依赖、忽略运行产物 |
| `README.md`、`DIT_BENCHMARK_ADAPTATION.md` | 使用入口和修改说明 |

两个新增 Python 文件均未新增 class；复用现有的模型、LoRA、scheduler 和损失函数。源码保留独立 Git checkout，可用 `git -C baseline/EraseAnything diff` 查看原文件改动，`git -C baseline/EraseAnything status --short` 查看新增文件。

```text
baseline/EraseAnything/runs/benchmark-dit/flux1_dev/<case>/training-data/
baseline/EraseAnything/runs/benchmark-dit/flux1_dev/<case>/checkpoint/pytorch_lora_weights.safetensors
baseline/EraseAnything/runs/benchmark-dit/flux1_dev/<case>/checkpoint/metadata.json
baseline/EraseAnything/runs/benchmark-dit/flux1_dev/<case>/plans/EraseAnything/<task>-<采样概要>-<北京时间>/

benchmark/results/flux1_dev/<task>/<case>/EraseAnything/
benchmark/results/flux1_dev/<task>-metric.csv
benchmark/results/flux1_dev/<task>-metric.txt
benchmark/results/flux1_dev/plots/<task>/
benchmark/original/flux1_dev/<task>/
```

IP 的 task 为 `instance`。同一方法、同一 case 成功评测后更新该行，不同 case 分别保存。训练参数改变时使用新的 `--run-root`，需要并存比较版本时另设 `--method`。目录名可读，指纹仅保存在 JSON；训练历史逐对保存四项 loss 和逐目标更新次数。未使用仓库附带的示例裸体 LoRA 代替各个 case 的训练。

## 验证与限制

- 五个脚本通过语法和 dry-run；8 个 case／9 项任务的数据规模、预算、Q16 开关及 I2P/COCO 权重共用已检查。
- 微型原生 FLUX 验证了注意力 hook 不改变预测值，注意力归一化且保留梯度；四项原生损失均产生非零 LoRA 梯度。
- 双目标完成上下层交替训练、gradient checkpointing、保存、复用、公共 LoRA 加载／融合和真实 CPU 采样；未选择的模型层不变。
- 临时模型、训练图和评测图片均清理，没有生成正式实验指标或新测试文件。

当前 NVIDIA 驱动不可用，尚未执行真实 FLUX.1-dev 的完整 GPU/BF16 实验或正式指标计算。微型验证说明接口和计算路径可用，不证明真实擦除效果。SD3.5 的实现和独立验证记录见 [SD35_ADAPTATION.md](SD35_ADAPTATION.md)。
