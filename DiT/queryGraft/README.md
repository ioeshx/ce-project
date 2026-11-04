# queryGraft

方法公式、符号与设计依据见 [方法定义](README.zh-CN.md)。代码集中于 `method.py`（模型接入与算法）和 `run.py`（命令行与 benchmark），`run.sh` 提供默认实验入口。

默认设置是：独立编码 target、anchor、background 并取各自第一个序列 token；在同层原始 AV 空间定位；所有主 attention 层的二值掩码严格多数投票；target 区域严格超过 20% 时，使用相同当前 latent 的 anchor 条件激活，在最后一层原始 AV 处硬替换。

## 运行

在仓库根目录执行，使用安装了 `benchmark/requirements.txt` 的环境。支持现有 `sd35_medium` 和 `flux1_dev` 模型配置；直接使用原生 pipeline、scheduler、seed 与 guidance 设置。

```bash
# 预览四类 benchmark，打印配置和图片数量；不加载模型、不创建实验产物。
python DiT/queryGraft/run.py benchmark --cases all --dry-run

# 默认全部实验：I2P 擦除与 COCO 保留，以及 IP、style、object。
bash DiT/queryGraft/run.sh

# 单独选择实验组、模型；通用参数沿用 benchmark。
python DiT/queryGraft/run.py benchmark --model flux1_dev --cases ip --offline
python DiT/queryGraft/run.py benchmark --cases i2p --max-records 4 --phase sample

# 独立 prompt；可多次指定 --prompt 进行批量生成。
python DiT/queryGraft/run.py sample --model sd35_medium \
  --pairs '[{"target":"nudity","anchor":"clothed"}]' \
  --prompt 'a person in a garden' --seed 42
```

`--cases` 接受 `all/i2p/ip/style/object` 或逗号分隔的具体 case ID；`--tasks` 可进一步选择任务，例如 `--cases i2p --tasks mscoco`。

现有 benchmark 包含多 target 用例，因此实现提供逐 pair 扩展：每对 target/anchor/background 独立读取同一原始 user/image 路径与本组概念，不跨 pair 读取；每对分别进行严格多数投票与面积门控。区域重叠时，使用支持票比例最高的 pair，平票按配置顺序选择。每个触发的不同 anchor 从同一个当前 latent 单独前向；相同 anchor 复用 source。单 pair 的默认算法与方法文档一致。

## 五组消融参数

| 设计项 | 参数 | 默认值 | 备选 |
| --- | --- | --- | --- |
| 替换表示 | `--representation` | `av` | `residual` |
| 定位层集 | `--localization-layers` | `all` | 逗号分隔的 block 索引，例如 `0,3,6` |
| 替换层集 | `--edit-layers` | `last` | `all` 或逗号分隔的 block 索引 |
| target 面积阈值 | `--rho` | `0.2` | `[0,1]` 内数值，始终使用严格大于 |
| 更新算子 | `--update-rule` | `replace` | `blend`、`add`，用 `--strength` 控制系数 |

层索引从 0 开始；FLUX 的双流 blocks 在前、单流 blocks 在后。每个 block 的主 attention 投一票，SD3.5 额外的 image-only `attn2` 不增加投票。

`blend` 的更新为 `X + strength * M * (A - X)`；`add` 为 `X + strength * M * A`；`replace` 不使用 `strength`。最后一层 AV 替换可在当前 user 前向内取得掩码后按需计算 anchor source；多层替换使用固定的当前步掩码并重放 user 路径，实际前向成本随配置变化。

`residual` 在 SD3.5／FLUX 双流层指主 attention 投影、门控、残差更新后的表示。SD3.5 的额外 `attn2` 更新保留在 user 路径；FLUX 单流层则使用 AV 与 MLP 融合后的原生残差输出。定位分数在这些配置下仍统一从 raw AV 计算。

```bash
# 定位仍使用全部层，只改变编辑层与融合强度。
python DiT/queryGraft/run.py benchmark --cases ip \
  --edit-layers all --update-rule blend --strength 0.5

# 使用关键定位层、关键编辑层与残差后表示。
python DiT/queryGraft/run.py benchmark --cases style \
  --localization-layers 0,3,6 --edit-layers 6 \
  --representation residual --rho 0.1

# 默认脚本可通过环境变量或额外 CLI 参数修改。
MODEL=flux1_dev CASES=object RHO=0.1 UPDATE_RULE=blend STRENGTH=0.5 \
  bash DiT/queryGraft/run.sh --phase sample
```

额外参数包括 `--background`、`--pairs`、`--anchor-overrides`、`--trace/--no-trace`、`--trace-root`。target/anchor 默认来自 benchmark 的概念配置；可用 `--anchor-overrides '{"nudity":"clothed"}'` 显式覆盖。anchor 分支使用完整 anchor 短语作为原生 prompt。

## benchmark 阶段与结果

```bash
# 准备计划，终端输出 manifest 路径。
python DiT/queryGraft/run.py benchmark --cases ip_pikachu --phase prepare --offline

# 复用上述 manifest；现有图片会由 benchmark 自动复用。
python DiT/queryGraft/run.py benchmark --manifest /path/to/manifest.json --phase sample

# 对已经采样的结果运行原有指标。
python DiT/queryGraft/run.py evaluate --manifest /path/to/manifest.json --device cuda:0
```

指定现有 manifest 时，方法、概念和采样科学配置以 manifest 为准，修改配置应准备新计划。运行时设备、batch size、指标选项等沿用 benchmark 原生规则。

方法图片和诊断默认位于 `DiT/queryGraft/results/<model>/`；manifest、指标及统计使用现有 `benchmark/results/<model>/` 流程。实验名称包含替换表示、更新算子、面积阈值、层选择和起始时间，目录名不使用哈希。`--result-root` 修改图片与诊断根目录，`--output-dir` 修改 benchmark 结果根目录；独立 prompt 可用 `--run-dir` 指定输出目录。

默认 trace 记录每步各 pair 的 target 面积、触发结果与逐层面积，并保存掩码、投票比例和最终 pair 选择图；不会保存完整 AV。`metadata.json` 另记配置、层位置及实际前向次数。可用 `--no-trace` 关闭这些诊断。

## 验证范围

已在本地 Diffusers 0.41.0 下使用真实小型 SD3／FLUX Transformer 验证只读前向一致性、三类判定与严格多数票、面积阈值、两种替换表示、三种更新算子、单层／多层执行和多 pair source 复用。真实小型 pipeline、VAE 与 scheduler 的两步采样也已通过；其中语义文本编码使用确定性替代输入，用于验证计算与接口，不衡量擦除效果。

现有 DiT／case／plugin 相关检查共运行 27 项，25 项通过、2 项按原检查条件跳过。两模型全部 benchmark 配置预览与离线 manifest 准备通过。当前环境无法使用 CUDA，尚未运行正式权重的图像生成与擦除指标实验。
