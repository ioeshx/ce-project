# GEM 的 DiT benchmark 适配

默认 **SD3.5 Medium**，另支持 **FLUX.1-dev**。直接调用本仓库的 `GEM` operator、`GEMSD3Config/GEMConfig` 和原生 PEFT 参数选择，不另写训练损失或采样算法。

## 实验与配置

运行时读取公共 `benchmark/configs/dit_cases.json`、`dit_target_anchor.json`、`dit_sampling.json` 和 `dit_evaluation.json`。

| 分组 | case | 擦除目标 | 评测 |
|---|---|---|---|
| I2P | `i2p_nudity` | nudity | sexual 931 张＋COCO 1K，同一 LoRA |
| IP | `ip_pikachu` | Pikachu | Snoopy/Pikachu/Mickey，960 张 |
| IP | `ip_snoopy_pikachu` | Snoopy、Pikachu | 同上，联合训练 |
| style | `style_van_gogh` | Van Gogh | 三位画家，270 张 |
| style | `style_monet` | Monet | 同上 |
| style | `style_picasso` | Picasso | 同上 |
| Object | `object_church` | church | 十类各 50 张，共 500 张 |
| Object | `object_multi5` | church、golf ball、parachute、cassette player、banana | 同上，联合训练 |

每模型 8 个训练 case、9 项评测、5,661 张方法图。Object 十类为 church、golf ball、English springer、goose、chain saw、cassette player、French horn、garbage truck、banana、parachute；类别及官方索引与公共配置一致。Anchor 严格读取现有表，无统一默认值。

正式采样为公共协议：512×512、BF16；SD3.5 为 40 步／CFG 4.5／T5 256，FLUX 为 28 步／guidance 3.5／T5 512。I2P 使用 NudeNet，COCO 与 IP/style 使用公共 CLIP/FID，IP/style 另使用 Qwen，Object 使用 ResNet-50 V2 的完整千类 Top-1/3。`--with-q16` 显式切换 I2P 全量协议。

## 启动

在合适的现有 PyTorch 环境中安装本方法依赖：

```bash
python -m pip install -r /home/shx/code/ce-project/baseline/GEM/requirements-benchmark.txt

# 默认 SD3.5，依次运行全部四组
bash /home/shx/code/ce-project/baseline/GEM/scripts/benchmark_dit_all.sh \
  --device cuda:0 --text-device cpu --gradient-checkpointing

# 四个分组可单独启动
bash /home/shx/code/ce-project/baseline/GEM/scripts/benchmark_dit_i2p.sh --device cuda:0
bash /home/shx/code/ce-project/baseline/GEM/scripts/benchmark_dit_ip.sh --device cuda:0
bash /home/shx/code/ce-project/baseline/GEM/scripts/benchmark_dit_style.sh --device cuda:0
bash /home/shx/code/ce-project/baseline/GEM/scripts/benchmark_dit_object.sh --device cuda:0

# FLUX 使用相同入口
bash /home/shx/code/ce-project/baseline/GEM/scripts/benchmark_dit_all.sh \
  --model flux1_dev --device cuda:1 --gradient-checkpointing

# 两张卡并行，各卡内顺序执行所选分组；分别在两个终端启动
bash /home/shx/code/ce-project/baseline/GEM/scripts/benchmark_dit_all.sh --cases i2p,object --device cuda:0
bash /home/shx/code/ce-project/baseline/GEM/scripts/benchmark_dit_all.sh --cases style,ip --device cuda:1

# 仅查看计划，不创建输出或加载模型
bash /home/shx/code/ce-project/baseline/GEM/scripts/benchmark_dit_all.sh --dry-run
```

总脚本顺序执行，避免在同一张卡上同时加载多个模型。各脚本可从任意目录调用，只转发参数；`PYTHON` 可指定解释器。`--cases` 接受分组或具体 case ID 的逗号组合，`--tasks mscoco` 可单独选择 nudity LoRA 的 COCO 任务。

默认 `--phase all` 为训练→prepare→采样→评测；可单独选择 `train/prepare/sample/evaluate`。相同配置复用已完成的检查点，分阶段运行须保持相同训练设置和目录。`--offline` 仅使用已有模型及指标缓存；默认通过模型 ID 自动下载。`--base-model` 接受本地 Diffusers 目录或对应架构的 Hub ID，原模型参考图使用同一来源。

总控隐藏 GPU；训练、采样、评测在独立子进程运行，结束后释放 CUDA 上下文。仅暴露 `--device` 对应 GPU，文本编码器和指标可选 CPU，其他 GPU 参数在入口拒绝。若外部设置了 `CUDA_VISIBLE_DEVICES`，`--device` 使用其逻辑编号。

## 训练目标与参数

保留作者代码的轨迹生成、教师 adapter 禁用、AdamW 和对比距离损失：对目标条件学生速度，拉近 anchor 教师速度，推远 target 教师速度，使用 `relu(dist_pos - eta * dist_neg)`，按原实现乘 `gamma` 并除以轨迹长度。没有把 GEM 换成 ESD 或其他擦除损失。

| 参数 | 默认值 |
|---|---|
| LoRA | rank=16、alpha=16、dropout=0.1、`qk_dual`；选中的 Q/K 投影沿用原生参数选择器 |
| 学习率 | SD3.5：2e-4；FLUX：1e-3，分别对应原生配置 |
| `eta` / `gamma` | 1 / 100 |
| 预算 | 每目标 250 次速度预测，总预算乘目标数；`--budget` 可覆盖总数 |
| 部分轨迹 | 学生轨迹，guidance=3，结束 callback 索引 10（含该步） |
| 内部时间表 | 默认读取模型 profile 的 40／28 步 |
| 并行时间步 | 最多 6，`--max-parallel-timesteps` 可降低激活占用 |
| 训练精度 | BF16；LoRA 参数按作者 `prepare_adaptation` 转为 FP32 |
| 文本设备 | CPU；输出条件转到训练设备，T5 长度读取模型 profile |
| 随机种子 | 0，沿用原生随机数设置与目标抽样函数 |

预算不是 AdamW 更新次数：作者按一次部分轨迹更新参数，速度预测次数耗尽时停止。多目标共享一个 LoRA；原生名为 `stratified_sampling` 的函数实际使用有放回 multinomial 抽样，本次保持不变，并在 metadata 记录每目标实际参与的轨迹数。不会将多个目标拼为一个字符串或分别训练再合并。

训练默认 BF16 是本地适配设置；作者 SD3.5 默认 FP32、FLUX 默认 FP16。可通过 `--training-dtype` 恢复。`--training-resolution/--training-steps/--training-sequence-length` 只影响训练；`--dtype/--steps/--height/--width` 只影响正式采样。`--cpu-offload/--device-map/--max-memory` 只控制采样；训练 transformer 与 VAE 保留在单卡，文本编码器默认 CPU。FLUX 完整 denoiser 及训练激活仍需要足够显存。

## 文件及核心代码修改记录

| 文件 | 修改及目的 |
|---|---|
| `benchmark_dit.py`（新增） | 一个入口包含参数、case 配置、训练 worker、原生 LoRA 导出和 benchmark 编排；只在入口校验参数 |
| `scripts/benchmark_dit_{all,i2p,ip,style,object}.sh`（新增） | 一个总入口、四个分组入口，无额外检查 |
| `requirements-benchmark.txt`（新增） | 公共评测依赖和 GEM 训练实际使用的 PEFT／wandb |
| `gem/wrappers/pipeline_wrapper_sd3.py` | 动态 scheduler 按图像 patch 数计算 `mu`；timestep 改为一维，条件扩展到时间步 batch；可配置文本设备、T5 长度及几何；SD3 CFG 预编码同长度的空负条件，避免 rollout 重新调用位于 CPU 的编码器 |
| `gem/wrappers/pipeline_wrapper_flux.py` | 条件扩展到时间步 batch；使用实际训练几何，可配置文本设备与 T5 长度 |
| `gem/operators/erasure/erasure_gem.py` | 两处 `set_timesteps` 传真实图像 token 数，修复原来误传 latent 通道数的问题；记录目标轨迹次数；损失与抽样、优化算法未改 |
| `gem/utils/memory.py` | CPU 验证时跳过 CUDA synchronize，GPU 行为保持不变 |
| `.gitignore`、`README.md`、本文档 | 忽略本方法运行产物、链接入口并记录适配 |

没有新增 class，也未另建 backend/loss/dataset 模块。未调用原 `gem.train.run_operator` 中固定 CUDA、随机目录和完整模型保存流程；仍直接调用相同 `GEM` operator 与原生适配器。仅将训练后的 transformer LoRA 提取为标准 Diffusers 格式，公共 benchmark 原生加载／融合，不需要自定义采样 hook。未修改 `benchmark/`。

## 输出

```text
baseline/GEM/runs/benchmark-dit/<model>/<case>/checkpoint/pytorch_lora_weights.safetensors
baseline/GEM/runs/benchmark-dit/<model>/<case>/checkpoint/metadata.json
baseline/GEM/runs/benchmark-dit/<model>/<case>/checkpoint/train.log
baseline/GEM/runs/benchmark-dit/<model>/<case>/plans/GEM/<task>-<采样概要>-<起始时间>/
benchmark/results/<model>/<task>/<case>/GEM/
benchmark/results/<model>/<task>-metric.csv
benchmark/results/<model>/time/<task>-sampling-time.csv
```

目录名不包含哈希；参数、源码和权重摘要只写入 JSON。相同配置复用检查点和计划；改变训练设置时使用新的可读 `--run-root`，例如 `runs/gem-sd35-20261101`。公共计划和图片分配器用时间／run 序号区分不同身份，历史 JSON 不改写。

### 2026-11-02：修复未初始化检查点的重试

原 `train()` 仅根据 checkpoint 目录是否存在判断训练完成。模型加载或初始化中断后，
目录可能只留下空 `train.log`；再次运行就会因缺少 `metadata.json` 而退出。
现仅在 LoRA 权重和 metadata 同时存在、设置及权重摘要匹配时复用。
空目录或仅包含空 `train.log` 的目录允许重新启动；其他不完整目录保留文件并提示使用新的 `--run-root`。
修改仅涉及 `benchmark_dit.py` 的训练产物状态判断，未改变训练算法，也未自动启动正式训练。

## 验证

已完成以下验证，没有新增工程测试文件：

- 两种模型的五个 Shell 入口均通过语法及从其他目录调用的 dry-run；展开为每模型 8 个 case、9 项任务、5,661 张方法图，Object 使用新十类和 banana 联合目标。
- 总控 dry-run 不导入 Torch、不创建 CUDA 状态；训练由真实独立 worker 执行。
- 临时原生 SD3 模型包含 Q/K norm、额外图像 self-attention 和动态 scheduler；临时 FLUX 模型包含 joint／single block。两者均完成原 `GEM` operator 的双目标、多时间步批处理、gradient checkpointing、训练与标准 LoRA 导出，两个目标实际参与更新。
- 两种模型的 CPU FP32 和 BF16 训练均通过；LoRA 参数按原生逻辑保留 FP32，检查点可复用，原生加载／融合后仅 Q/K 投影 weight 改变。
- 两种模型均完成公共 prepare/sample，实际生成各 3 张方法图和 3 张原模型图；采样时间落入公共 `results/<model>/time/`，临时模型、权重、清单和图片均已清理。
- 现有 benchmark DiT／case 回归测试 23 项通过。

这些是临时微型模型的接口与计算验证，尚未执行完整 SD3.5 Medium／FLUX.1-dev 的正式 GPU 训练、全量图片或擦除质量评测。
