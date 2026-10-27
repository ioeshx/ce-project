# Renoise Self Check

冻结 DiT，在输出 velocity 上构造两个候选；原提示在候选新状态上重新预测，再用共享重噪和 target/anchor 条件的速度重建误差选择当前一步。支持 `sd35_medium`、`flux1_dev`，默认先运行 SD3.5-medium。

模型加载、采样 profile、四类 case、用户提示、种子和指标均复用 `benchmark/`。代码只保存最终图像和指标/配置元数据，不保存 velocity、latent、候选图像或完整 rollout。

## 阅读代码

工程包含三个 Python 文件：

| 文件 | 职责 |
| --- | --- |
| `method.py` | 核心 `denoise()` 循环、`score()` 重噪评分、target/anchor 提示构造、标量统计 |
| `model.py` | 两个模型的编码、velocity 查询、原生采样时间表与解码；复用 benchmark 的模型加载和设备分配 |
| `run.py` | 命令行参数、四类实验遍历、benchmark 采样和评价入口 |

建议先阅读 `method.py` 的 `denoise()`，再看 `score()` 和 `generate()`。`RenoiseSelfCheck` 是 benchmark 插件，继承处理模型差异的 `FlowModel`；参数和提示对均使用普通字典。方法参数在插件初始化时检查一次，启动脚本负责解析参数。

## 已确定的配置

| 参数 | 当前值 |
| --- | --- |
| 模型、分辨率、步数、前向精度、guidance、设备与 batch | `benchmark/configs/dit_sampling.json` 及 benchmark CLI 配置 |
| 检查窗口 | 连续中间 40% 步，按实际采样步数量取最近整数；40 步为 **第 13–28 步**，28 步为第 9–19 步，均为 1-based |
| `lambda_ref` | **0.01** |
| `rho` | **0.3** |
| `theta`、`margin` | **0、0** |
| Probe | 每个目标一组提示对，每个检查步一份新高斯噪声，两候选共享 |
| 自检随机源 | 根种子 **0**，结合 sample ID、生成种子、检查步及目标稳定派生；独立于初始生成噪声 |
| Guidance | **所有分支沿用 benchmark，包括 SD3 的 CFG 与 FLUX-dev 的内嵌 guidance** |
| 多目标 case | 在检查步之间按 case 中的目标顺序轮流检查，每个检查步处理一个目标 |
| 数值运算 | 模型精度沿用 benchmark；差分、clean estimate、重噪构造、MSE 使用 float32；模型输入和正式 sampler velocity 转回模型精度 |

`lambda_ref` 后续候选为 `1, 0.1, 0.05, 0.001`，当前不自动扫描。`rho`、窗口、阈值同样通过命令行控制。

## 运行

在项目根目录运行。默认 `--model sd35_medium --cases all`。`all` 使用 benchmark 的全部四类 case，包含其配置的 COCO 保留任务。

完整运行四类实验（采样和指标计算）：

```bash
bash DiT/renoise_self_check/benchmark.sh

# FLUX-dev 使用相同实验，按现有两卡环境分配模型。
MODEL=flux1_dev bash DiT/renoise_self_check/benchmark.sh --device-map balanced
```

脚本依次执行 `i2p`、`ip`、`style`、`object` 四组，展开为 8 个 case、9 个采样/评价任务，共 5,661 张方法图像。`--mode both` 复用已有参考图，并补齐缺失参考图；`--offline` 使用本地模型和指标权重。I2P 计算 NudeNet，COCO 计算 CLIP Score/FID，IP/Style 计算 CLIP Score/FID 和 benchmark 配置的 VLM 分类指标，Object 计算 ResNet 分类指标。重新执行同一命令可继续生成缺失图像。

```bash
# 检查所有 case 的样本量、检查步与参数，不加载模型或生成图片。
python DiT/renoise_self_check/run.py --phase prepare --dry-run

# 准备配置、清单和输出目录。
python DiT/renoise_self_check/run.py --phase prepare

# 生成四类 case 的方法图像，沿用 benchmark 的采样设置。
python DiT/renoise_self_check/run.py --phase sample

# 使用 benchmark 原有指标评价；需要的原模型 reference 使用已有 benchmark 图片。
python DiT/renoise_self_check/run.py --phase evaluate

# FLUX-dev 使用同一入口和它自己的 benchmark profile。
python DiT/renoise_self_check/run.py --model flux1_dev --phase sample

# 后续超参数实验；默认方法名包含参数，不覆盖其他 λ/ρ 的指标行。
python DiT/renoise_self_check/run.py --cases style --phase sample --lambda-ref 0.05 --rho 0.5
```

可继续传入 benchmark 的 `--cases`、`--tasks`、`--max-records`、`--limit-prompts`、`--seeds`、`--device`、`--batch-size`、`--dtype`、`--cpu-offload`、`--device-map`、`--offline` 等参数。`--strength` 是 `--lambda-ref` 的别名，`--m` 是 `--margin` 的别名。

默认采样模式为 `image`，不额外生成原模型对照。原生 CS/FID、对象/角色准确率等指标需要匹配的 benchmark reference；这些 reference 缺失时，可显式使用 `--mode both` 补齐。不会额外执行候选完整 rollout 或诊断实验。

首次小规模验证示例，保留正常的 40 步、512×512、guidance 4.5：

```bash
python DiT/renoise_self_check/run.py --cases i2p_nudity --tasks mscoco --phase sample --max-records 1 --offline
```

也可直接使用原 benchmark 插件入口：

```bash
python -m benchmark dit --model sd35_medium --cases all --method-config DiT/renoise_self_check/method.json --phase prepare
python -m benchmark dit --model sd35_medium --cases all --method-config DiT/renoise_self_check/method.json --phase sample --mode image
```

## 提示构造

读取 `benchmark/configs/dit_cases.json` 的 targets 和 `dit_target_anchor.json` 的 anchor。用户提示保持 benchmark 原文。每个概念缓存两组不同措辞的方向/probe 条件：

| 组别 | 方向 target / anchor | Probe target / anchor |
| --- | --- | --- |
| Nudity | `nudity` / `dressed` | `a photo of a nude adult person` / `a photo of a dressed adult person` |
| IP | `an image of {target}` / `an image of a cartoon character` | `a picture depicting {target}` / `a picture depicting a cartoon character` |
| Style | `a painting by {target}` / `a painting in a general artistic style` | `an artwork in the style of {target}` / `an artwork in a general artistic style` |
| Object | `a photo of a/an {target}` / `a photo of a/an {anchor}` | `an image depicting a/an {target}` / `an image depicting a/an {anchor}` |

模板和最终展开文本写入标量统计文件，便于解释每个实验的评分。

## 算法与节省计算

检查步执行：

```text
v0 = G(z, t, user)
z0 = EulerPreview(z, v0)
x0 = z0 - sigma_next * G(z0, t_next, user)
R0 = ReNoisedScore(x0, shared_noise, rho)

if R0 > theta:
    d  = G(z, t, anchor) - G(z, t, target)
    v1 = v0 + lambda_ref * d
    z1 = EulerPreview(z, v1)
    x1 = z1 - sigma_next * G(z1, t_next, user)
    R1 = ReNoisedScore(x1, shared_noise, rho)
    choose v1 only if R1 < R0 - margin
else:
    choose v0

commit exactly one real Euler step
```

`ReNoisedScore = MSE(anchor_prediction, epsilon-x) - MSE(target_prediction, epsilon-x)`。原候选未超过 θ 时跳过差分与纠正候选的额外前向；已计算的选中候选下一状态用户 velocity 在下一正常步精确复用。两种优化保持原接受规则。

不纠正和纠正两候选顺序计算以控制显存。SD3 的 target/anchor 成对查询共享同状态的无条件分支；所有 guidance 数值仍与 benchmark 一致。统计中分别记录 transformer 调用次数和按输入样本展开的计算量。

## 输出

最终方法图像：

```text
DiT/renoise_self_check/results/<model>/<task>/<case>/<method>/images/<run>/*.png
```

benchmark 指标、采样耗时和方法评分统计：

```text
benchmark/results/<model>/<task>-metric.csv
benchmark/results/<model>/<task>/<case>/<method>/evaluations/
benchmark/results/<model>/<task>/<case>/<method>/self_check/<run>/statistics.json
```

`statistics.json` 保存按样本和目标汇总的 `R0`、`R1`、`R0-R1` 分布统计、检查/接受次数以及查询量。它们用于后续选择 θ、m；运行时不自动校准阈值。未执行的纠正候选在统计中不计为零分。继续未完成运行时按样本 ID 更新，避免重复计数。

## 验证记录

2026-10-27 首次实现验证：两个模型的真实本地权重均完成一次 COCO 采样。SD3.5-medium 另外完成一次 I2P 采样及 benchmark NudeNet 评价。均沿用正常步数、512×512 和默认 guidance，使用当前 λ=0.01、ρ=0.3、θ=m=0。

| 真实权重验证 | 生成耗时（含自检，不含模型加载） | 检查 / 触发纠正候选 / 接受 |
| --- | --- | --- |
| SD3.5-medium / COCO 单样本 | 15.39 s | 16 / 14 / 7 |
| SD3.5-medium / I2P 单样本 | 10.51 s | 16 / 4 / 1 |
| FLUX-dev / COCO 单样本 | 28.65 s | 11 / 3 / 1 |

以上在两张 RTX 3090 上串行验证，FLUX 使用 `--device-map balanced`。单样本记录用于验证实现和指标流程；完整 benchmark 与参数消融尚未执行。

本次代码收敛为三个文件，并移除方法目录中的测试代码。沿用 benchmark 现有的 31 项相关回归测试，均通过。重构后分别复跑两个模型的 COCO 单样本：生成图像与重构前逐像素一致，自检评分及接受次数也一致；两个模型的全部四类实验配置预览通过。
