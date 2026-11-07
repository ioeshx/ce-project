# detectFix

detectFix 在去噪过程中先检测 target，再将命中样本的文本条件完整切换为对应 anchor。默认采用 `detect once`：首次命中后，剩余步骤一直使用该 anchor。

代码集中在 `method.py`（检测、条件切换与模型接入）、`run.py`（命令行与 benchmark）和 `diagnostics.py`（诊断记录、可视化、VAE 解码与统计）。检测复用 queryGraft 的 `QueryAttention`、`ConceptModulation`；支持 `sd35_medium` 和 `flux1_dev`。

## 检测与修正

每个检测步骤从当前 latent 出发，使用用户原始 prompt 运行带只读 concept 分支的 Transformer：

1. 独立编码每组 target、anchor、background，取各自第一个 T5 序列 token。concept 使用 pooled CLIP 输入为零的独立调制条件，默认只读取 image 与本组 concept 的 K/V。
2. 计算同层原始 image AV 与 concept AV 的 FP32 点积；每组在三个概念通道上 softmax，再对所选定位层等权平均，得到分数图 `P`。
3. 按下式取交集，所有比较均为严格大于：

   ```text
   mask = (P_target > spatial_mean(P_target))
        & (P_target > P_anchor)
        & (P_target > P_background)
   detected = mean(mask) > rho
   ```

命中后，在同一当前 latent 上重新运行完整 Transformer，正条件的序列嵌入和 pooled 嵌入均来自 anchor 文本。该步骤使用 anchor 预测推进 scheduler，整个去噪步骤只推进一次。SD3.5 保留原生负条件分支与 CFG；FLUX 保留原生内嵌 guidance。

掩码只负责判断是否触发修正。修正作用于整张图像的文本条件，不进行局部 activation 替换。

| 模式 | 行为 |
| --- | --- |
| `once`（默认） | 每个样本首次命中时锁定 anchor，该步及后续步骤均使用它；该样本不再检测。 |
| `every_step` | 每步用原始 prompt 检测当前 latent；命中则该步使用 anchor，未命中则该步使用原始 prompt。 |

多 target 时，每组独立计算掩码与面积。若同一步命中多组，选择掩码面积最大的组；面积相同按配置顺序选择。`once` 会锁定首次选中的 anchor，批内各样本独立决策。

## 运行

在仓库根目录执行，环境依赖沿用 `benchmark/requirements.txt`。

```bash
# 预览配置与图片数量，不加载模型。
python DiT/detectFix/run.py benchmark --cases all --dry-run

# 单个 prompt：先完成 5 步去噪，之后检测；面积超过 10% 时切换为 clothed。
python DiT/detectFix/run.py sample --model sd35_medium \
  --pairs '[{"target":"nudity","anchor":"clothed"}]' \
  --prompt 'a person in a garden' --seed 42 \
  --detection-start-step 5 --rho 0.1

# 每步重新检测，运行 I2P 采样。
python DiT/detectFix/run.py benchmark --cases i2p --tasks i2p \
  --model flux1_dev --detection-mode every_step \
  --detection-start-step 5 --rho 0.1 --phase sample --device cuda:0

# 完整采样与评测，包含各 case 的默认任务。
python DiT/detectFix/run.py benchmark --cases all --phase run

# 使用已有清单恢复采样或执行评测。
python DiT/detectFix/run.py benchmark --manifest /path/to/manifest.json --phase sample
python DiT/detectFix/run.py evaluate --manifest /path/to/manifest.json
```

`sample` 可以重复传入 `--prompt`。`--cases` 支持 `all/i2p/ip/style/object` 或逗号分隔的具体 case ID；`--tasks` 可进一步选择 `i2p`、`mscoco` 等任务。`--steps`、`--batch-size`、`--device`、`--offline` 等通用参数沿用 benchmark。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--detection-mode` | `once` | `once` 或 `every_step`。 |
| `--detection-start-step` | `0` | 开始检测前跳过的去噪步数；索引从 0 开始。 |
| `--rho` | `0` | target 掩码面积比例阈值，范围 `[0,1]`；默认值表示非空掩码即可触发。 |
| `--localization-layers` | `all` | 定位层，或逗号分隔的 0-based block 索引。FLUX 双流层在前、单流层在后。 |
| `--background` | `background` | 背景概念文本。 |
| `--pairs` | case 的 target/anchor | JSON target/anchor 列表。 |
| `--anchor-overrides` | 无 | JSON 映射，例如 `'{"nudity":"clothed"}'`。 |
| `--concept-include-prompt` | 关闭 | 将用户 prompt K/V 也纳入 concept 分支。 |
| `--result-root` | `DiT/detectFix/results` | 图片输出根目录。 |
| `--trace` / `--no-trace` | 关闭 | 显式传入 `--trace` 后保存掩码、首次检出及后续中间图像与统计。 |
| `--trace-root` | 随图片输出目录 | 自定义诊断根目录。 |

`detection_start_step=5` 表示前五步不检测，从第六次去噪计算开始检测；`0` 表示从第一步开始。`rho=1` 不会触发修正。

方法配置存放在 `adapter.detect_fix`，插件入口为 `DiT.detectFix.method:DetectFix`。实验目录使用可读参数与起始时间区分，配置和清单保存实际设置；恢复已有清单时沿用其设置。

每个图片输出目录下使用独立的 `record/` 子目录保存同名 `.detectFix.json`。例如 `images/<setting>/sample.png` 的记录位于 `images/<setting>/record/sample.detectFix.json`，独立采样的 `images/sample.png` 对应 `images/record/sample.detectFix.json`。记录包含样本、方法设置、采样配置、`localization_sites` 和整个生成 batch 的 `batch_forward_calls`。逐步决策位于 `steps`，记录 `step_index`、`timestep`、`checked`、`target_area`、`selected_pair` 和实际使用的 `condition`。跳过检测时 `target_area` 为 `null`；`selected_pair=-1` 表示使用原始 prompt。可用它确认首次触发位置、anchor 持续使用情况，以及 `every_step` 是否切回原始 prompt。

## 诊断记录

诊断默认关闭，显式传入 `--trace` 才开启。以下两条命令均开启诊断，跳过前五步，从第六步开始检测，面积严格超过 20% 时修正：

```bash
# Detect once：首次命中后锁定 anchor。
python DiT/detectFix/run.py benchmark --cases i2p --tasks i2p --phase sample \
  --rho 0.2 --detection-start-step 5 --detection-mode once --trace

# Detect every step：每步重新检测。
python DiT/detectFix/run.py benchmark --cases i2p --tasks i2p --phase sample \
  --rho 0.2 --detection-start-step 5 --detection-mode every_step --trace
```

目录沿用 queryGraft 的布局：独立采样的 `run/images/sample.png` 对应 `run/traces/<method>/sample/`；benchmark 的 `<method>/images/<setting>/sample.png` 对应 `<method>/traces/<setting>/sample/`。`setting` 为含采样设置及起始时间的可读实验名称。指定 `--trace-root` 后，按模型、任务、case、方法和 setting 分组，再保存各样本。

```text
traces/<setting>/
  summary.json
  <sample>/
    metadata.json
    first_detection_step_0005.png
    mask/
      step_0005.png
      step_0005.safetensors
    mid/
      step_0005.png
      step_0006.png
      ...
    mask_mean.png               # every_step 检出掩码的时间均值
    mask_mean.safetensors      # every_step 的均值数值
```

- `metadata.json` 的 `detection_steps` 记录所有检出步的 **0-based 索引**；`detection_step_numbers` 同时提供从 1 开始的步数。`once` 只记录首次命中，`every_step` 记录全部命中；未命中样本保留空列表，首次检出步为 `null`。
- `mask/step_XXXX.png` 展示该步所有触发 target 的掩码，数值保存在同名 `.safetensors` 中。多 target 同时触发时，即使只有最大面积对应的 anchor 用于修正，其他触发 target 的掩码仍被记录。
- `first_detection_step_XXXX.png` 将首次检出时的当前 latent `x_t` 直接经 VAE 解码，保存的是该步修正和 scheduler 更新**之前**的状态。
- `mid/step_XXXX.png` 保存 scheduler 完成该步更新后的 latent 解码结果。从首次检出的该步开始，一直保存到生成结束；`every_step` 后续未命中的步骤也会保存，便于比较整个变化过程。两类图都直接解码当时的 latent，不使用额外的去噪预测。
- `every_step` 的 `mask_mean.png` 展示所有检出步骤的掩码平均：整体图先对同一步触发 target 的掩码取并集，再除以样本的检出步数；各 target 分图分别以该 target 的触发步数为分母。对应数值为 `mask_mean.safetensors` 中的 `union_mask_mean`、`target_mask_mean` 和 `target_detection_counts`。未检出时不生成均值图。

`summary.json` 汇总整个 setting，跨 batch 累积；恢复采样时根据各样本记录重新汇总，同一样本不会重复计数：

| 字段 | 统计口径 |
| --- | --- |
| `sample_count` / `detected_sample_count` / `undetected_sample_count` | 已记录样本数、曾检出样本数和从未检出样本数。 |
| `mean_first_detection_step` | 曾检出样本的首次检出索引均值，索引从 0 开始；无检出样本时为 `null`。 |
| `mean_first_detection_step_number` | 同一均值改用从 1 开始的步数表示。 |
| `mask_count` / `mean_mask_area` | 所有实际计算的 `(样本, 检测步, target)` 掩码数量及其面积占比均值，包含空掩码与未通过 `rho` 的掩码；未执行检测的步骤不参与。 |
| `pairs` | 各 target/anchor 的掩码数量和平均面积占比。 |

每个样本的 `metadata.json` 另保留 `checked_steps`、`mask_count`、`mask_area_sum` 和 `mask_area_mean`，便于复算。诊断不增加 Transformer 前向或 scheduler 更新次数，会增加 VAE 解码与图像写盘时间。不传 `--trace` 时关闭这组诊断，`record/` 下的 `.detectFix.json` 决策记录继续保留。
