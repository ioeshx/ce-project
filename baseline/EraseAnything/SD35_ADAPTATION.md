# EraseAnything 的 SD3.5 Medium 适配

执行层更新：训练图生成、训练、采样和评测均使用独立子进程，总控隐藏 GPU，各阶段退出后释放 CUDA 上下文。公共实现为 `benchmark/execution.py`；全部 CUDA 模型仅使用 `--device` 指定的一张卡，辅助组件可选 CPU，其他 GPU 选项在入口报错。子进程内该卡映射为 `cuda:0`，保存设置保留原编号。四项损失和交替更新方式未改；精确的源码摘要兼容映射允许复用本次执行层升级之前的训练数据、权重及计划。

SD3.5 Medium 独立入口接入 I2P、IP、style、Object 四类实验，由本目录中的两个 Python 文件、五个 Shell 脚本组成。模型调用和训练目标由新增 SD3.5 文件实现，公共执行层及 Object 配置与 FLUX 和其他 baseline 共用。

这是作者 FLUX 实现的 **SD3.5 架构迁移版本**，保留四项损失的数学目标和上下层交替更新方式；不是作者发布的 SD3.5 原生实现。上游源码提交及已有 FLUX 适配说明见 [DIT_BENCHMARK_ADAPTATION.md](DIT_BENCHMARK_ADAPTATION.md)。沿用 `requirements-benchmark.txt` 中的现代 Diffusers/PEFT，无需安装仓库内的旧 fork。

## 支持的实验

实际读取公共 `benchmark/configs/dit_cases.json`、`dit_sampling.json`、`dit_evaluation.json` 和 `dit_target_anchor.json`，没有在新增文件中复制实验目标或评测模板。

2026-10-31 Object 更新：保留 church 单目标与五目标联合训练，联合目标改为 church、golf ball、parachute、cassette player、banana。评测十类和 anchor 对应关系见[公共类别表](../../benchmark/exp_plan/dit_experiment_plan.md#43-objectimagenet-10)。本次只调整公共 case 配置、ImageNet-10 数据与索引映射、脚本注释和文档；SD3.5 数据准备、训练及四项损失代码未改。新配置需重新 prepare（`--phase all` 自动包含此步骤）；旧五目标训练数据与检查点需使用新的 `--run-root` 重新生成和训练，历史记录保持原样。

| 实验组 | case | 擦除目标 | 评测规模 |
|---|---|---|---|
| I2P | `i2p_nudity` | nudity | sexual 931，另用同一 LoRA 评测 COCO 1K |
| IP | `ip_pikachu` | Pikachu | Snoopy/Pikachu/Mickey，共 960 |
| IP | `ip_snoopy_pikachu` | Snoopy、Pikachu | 同上，960；联合训练 |
| style | `style_van_gogh` | Van Gogh | 三位画家，共 270 |
| style | `style_monet` | Monet | 同上，270 |
| style | `style_picasso` | Picasso | 同上，270 |
| Object | `object_church` | church | ImageNet-10，共 500 |
| Object | `object_multi5` | church、golf ball、parachute、cassette player、banana | 同上，500；联合训练 |

共 **8 个训练 case、9 项评测任务、5,661 张方法图**。I2P 默认 NudeNet sexual 子集；`--with-q16` 使用全量 4,703 条并开启 Q16。IP/style 使用公共 Qwen 分类，Object 使用公共 ResNet-50 V2，CS/FID 等设置沿用公共协议。

正式 SD3.5 采样为 **BF16、512×512、40 步、guidance 4.5、T5 长度 256**，使用基础模型原生 FlowMatchEulerDiscreteScheduler。正式采样不安装训练注意力 hook。

## 启动

```bash
# 默认 SD3.5 Medium；一次依次执行全部实验
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_sd35_all.sh \
  --device cuda:0 --text-device cpu --gradient-checkpointing --offline

# 四类独立入口
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_sd35_i2p.sh --device cuda:0 --offline
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_sd35_ip.sh --device cuda:0 --offline
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_sd35_style.sh --device cuda:0 --offline
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_sd35_object.sh --device cuda:0 --offline

# 仅查看计划，不创建数据、不加载模型
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_sd35_all.sh --dry-run

# 指定本地模型，只执行双目标 IP
bash /home/shx/code/ce-project/baseline/EraseAnything/scripts/benchmark_sd35_ip.sh \
  --base-model /path/to/stable-diffusion-3.5-medium --cases ip_snoopy_pikachu --offline
```

脚本可从任意目录调用，只转发参数，不执行额外参数检查。`PYTHON` 可指定解释器；用户提供的相对路径相对于调用目录。

默认 `--phase all` 执行训练图生成→训练→评测清单准备→采样→评测。支持单独的 `data`、`train`、`prepare`、`sample`、`evaluate` 阶段。分阶段调用时保持相同设置和目录；`sample/evaluate` 要求已有匹配的 prepare 计划。可用 `--tasks mscoco` 单独准备或评测 nudity 权重的 COCO 任务。

`--cpu-offload`、`--device-map balanced`、`--max-memory` 控制训练图生成和评测采样；训练 transformer 仍放在 `--device`，不启用训练 offload。文本编码器默认放在 CPU，可用 `--text-device` 指定其他设备。`--gradient-checkpointing` 只控制训练 transformer。`--offline` 要求模型和评测资源已在本地，未准备时去掉该选项。

## 新增代码与算法对应

| 文件 | 内容和目的 |
|---|---|
| `benchmark_sd35.py` | 复用已有参数定义和 case 配置编译，新增 SD3.5 阶段编排、一次入口校验和迁移设置记录 |
| `benchmark_sd35_training.py` | SD3.5 前向、注意力 hook、训练数据批次、四项损失、联合 LoRA 训练与保存 |
| `scripts/benchmark_sd35_{all,i2p,ip,style,object}.sh` | 一个总入口、四个单项入口 |
| `SD35_ADAPTATION.md` | 支持范围、迁移细节和验证记录 |

没有新增 class，没有拆出独立 backend、loss、dataset 模块。训练图生成复用 `benchmark_training.prepare_images`，采样和指标仍复用公共 benchmark。

### 模型调用与注意力

- 使用 `StableDiffusion3Pipeline` 的两个 CLIP 和一个 T5 编码结果。两个 CLIP 的 token 特征按通道拼接、补齐后与 T5 按序列拼接；pooled 条件使用两个 CLIP pooled 特征。
- SD3 transformer 接收四维 latent 和原始尺度的 timestep；不用 FLUX 的 latent pack/unpack、`txt_ids/img_ids`、RoPE 或 distilled-guidance 输入。
- 原 scheduler 的加噪计算会将 BF16 latent 提升为 float32；新增统一前向函数在输入 transformer 前转回模型 dtype，兼容默认 BF16 训练。
- 训练 rollout 使用 SD3 CFG，默认 scale=3，可通过 `--rollout-guidance` 修改。直接用于四项损失的预测为单个条件分支；anchor 分支与目标分支仍来自当前 LoRA 模型的 `no_grad` 计算，没有引入固定基础模型教师。
- 最后一个联合 block 的主 `attn` 用于注意力损失；SD3.5 的额外图像自注意力 `attn2` 不纳入该图。hook 使用实际 Q/K 投影、可选 Q/K norm 和 `1/sqrt(head_dim)`，正常 transformer 前向继续使用原生 attention processor。
- SD3 原生 attention 为 image-first；hook 同时重排 query/key 为 text-first，使完整矩阵具有原损失的索引约定。
- **目标列只选择 T5 段**，前置偏移读取实际 CLIP 序列长度，正式模型通常为 77；不把 CLIP token 列另外加入抑制项。这样延续 FLUX 原实现以 T5 token 定义目标列的方式。这是迁移选择，应在论文中说明。

### 四项损失

```text
上层：λ1 · MSE(en, stopgrad(e0 - negative_guidance · (ep - e0)))
    + λ2 · 目标 token 列的完整注意力矩阵范数

下层：λ3 · MSE(velocity, noise - clean_latent)
    + λ4 · 原 utils.infoNCE.calculate_steer_loss
```

原 `utils/calc_loss.py` 的函数内嵌 FLUX 特有调用，因此新文件分别实现 `upper_loss`、`lower_loss`，保留公式、当前模型教师和停止梯度范围。注意力范数仅计算目标列，与原完整零掩码矩阵的范数等价。MSE 和范数归约使用 float32。

训练时间分布复用已有 `CustomFlowMatchEulerDiscreteScheduler`：随机 logit-normal sigma 排序，t=1000·sigma；每次完整 rollout 重置 scheduler。上层保留作者“先完整生成 latent，再在独立抽取的 t 上构造消除目标”的过程，下层沿用该对上层抽取的 t。**训练 rollout 的随机时间表不是正式评测的原生 40 步时间表。** SD3.5 的 CFG、张量布局和注意力来源是此次架构迁移部分。

InfoNCE 直接调用原函数，保留 `nominator - denominator` 的符号、temperature=0.07、一个同义词和三个无关概念、第一 attention head，以及在各参考图上使用当前 batch 相同目标列索引的方式。参考图在 `no_grad` 下生成，center 图保留梯度。仅保留最终会使用的随机时间步 attention，避免存储整个 rollout 的所有完整矩阵。

LoRA 仍仅作用于文本侧 `attn.add_q_proj`、`attn.add_k_proj`，其他模型参数冻结。每个 case 一个 LoRA、一个 AdamW；按目标洗牌循环，各目标先上层更新、再下层更新，不分别训练后合并。

### 数据准备和训练参数

沿用已有适配的本地数据设置：基础 SD3.5 每目标生成 4 张独立训练图，不使用评测 prompt；默认同义词为目标本身，无关概念为 `mountain/bicycle/fruit`。未调用外部大模型、未下载 WordNet，不宣称复现作者的提示扩展与训练数据。`--training-config` 可覆盖各目标的 `prompt/anchor/synonyms/irrelevant`，格式见原 DiT 说明。

保留词序打乱、同义词替换和随机裁剪。SD3.5 新批次函数在增强完成后，根据实际 prompt 和实际替换词定位 T5 token，排除 special token、padding 和被截断部分；原 FLUX 批次函数未修改。

默认训练设置：batch=1、rank=8、lr=1e-3、AdamW、negative_guidance=1、loss_weights=`[1,0.001,0.1,0.1]`、BF16、512×512、28 步训练 rollout、T5 长度 256、rollout CFG=3。训练 rollout 步数保留原适配默认值；正式采样仍为公共 SD3.5 的 40 步和 CFG=4.5。

每目标 200 次优化器更新，即 100 对上层/下层更新；双目标默认 400 次，五目标默认 1,000 次。可用 `--iterations` 指定 case 总更新数，或 `--iterations-per-concept` 设置每目标预算；总数必须为偶数且至少每目标一对更新。还支持 `--rank`、`--lr`、`--negative-guidance`、`--loss-weights`、`--training-resolution`、`--training-steps`、`--training-sequence-length`、`--training-dtype`。`--steps/--dtype/--height/--width` 控制训练图及评测采样，不替代对应训练参数。

## 输出位置

```text
baseline/EraseAnything/runs/benchmark-dit/sd35_medium/<case>/training-data/
baseline/EraseAnything/runs/benchmark-dit/sd35_medium/<case>/checkpoint/pytorch_lora_weights.safetensors
baseline/EraseAnything/runs/benchmark-dit/sd35_medium/<case>/checkpoint/metadata.json
baseline/EraseAnything/runs/benchmark-dit/sd35_medium/<case>/plans/

benchmark/results/sd35_medium/<task>/<case>/EraseAnything/
benchmark/results/sd35_medium/<task>-metric.csv
benchmark/results/sd35_medium/<task>-metric.txt
benchmark/original/sd35_medium/<task>/
```

源码、脚本、训练数据、权重及计划保存在方法目录；图片和指标按公共 benchmark 结果协议保存。`--run-root/--output-dir/--original-dir/--cache-dir` 可覆盖目录。训练设置、迁移选项、源码摘要、训练图片身份、四项 loss 和逐目标更新次数写入 JSON；权重使用标准 SD3 LoRA 格式。配置改变时选择新的 run-root，需要并存比较版本时修改 `--method`。

## 验证与限制

- 五个脚本通过语法检查及从其他目录调用的 dry-run；8 个 case、9 项任务、5,661 张图、训练预算、Q16 全量开关和 I2P/COCO 共用权重均已核对；原 FLUX dry-run 仍通过。
- 临时微型原生 SD3 使用两个 CLIP、T5、VAE、joint attention、额外图像 attention 和 Q/K norm，验证 hook 不改变原生预测，attention 归一化且可以反传，矩阵重排与原生 image-first 计算一致。
- 检查 T5 的 CLIP 偏移、同义词替换后的 token 索引，以及 raw timestep、SD3 CFG、完整 Euler rollout 和停止梯度。
- 开启 gradient checkpointing，四项损失分别产生有限且非零的 LoRA 梯度。
- float32 和 BF16 微型双目标训练均完成上下层交替更新、保存和复用、公共 benchmark prepare/sample；标准 SD3 LoRA 加载与融合后，仅所选投影权重发生改变。各轮真实 CPU float32 采样生成 3 张方法图和 3 张参考图，采样时没有训练 hook。
- 临时模型和产物已清理，未创建工程测试文件或正式实验指标。

当前 NVIDIA 驱动不可用，尚未执行完整 SD3.5 Medium 的 GPU/BF16 训练或正式擦除指标计算。微型验证覆盖接口和计算路径，真实擦除效果及默认超参数需要实际实验确认。
