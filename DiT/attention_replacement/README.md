# Attention Replacement

实现文件只有两个：`method.py` 包含核心算法、原生 pipeline hooks 与离线缓存；`run.py` 提供收集、独立采样、benchmark 和结果分析。复用仓库现有 benchmark 的数据、原图缓存和指标，无新增测试目录。

## 方法与默认值

捕获的是图像侧 attention 的 **AV，即输出投影之前的结果**，不是 `V` 或 attention map。每个采样步、attention 分支、空间位置分别对离线 seeds 求 FP32 均值。

```python
d_tar = (I_user.float() - target_mean).square().sum(-1)
d_anc = (I_user.float() - anchor_mean).square().sum(-1)
mask = d_anc - d_tar > tau
I_new = torch.where(mask[..., None], anchor_mean.to(I_user.dtype), I_user)
```

距离不归一化，也不除以通道数；没有半径条件。多概念都比较同一个替换前 `I_user`，取掩码并集。不同 anchor 在同一位置命中时，使用 `d_anc-d_tar` 最大者；相同分数取配置中靠前者。`winner=-1` 表示未命中。

默认 `sd35_medium`，同时支持 `flux1_dev`；全部 blocks/可用 attention 分支、前 `ceil(0.3*T)` 个采样步、16 个离线 seeds（10000–10015）、`tau=0`、完整 trace。层和步都是从 0 开始的左闭右开范围。SD3.5 默认 40 步，干预 `[0,12)`；FLUX 默认 28 步，干预 `[0,9)`。

SD3 只观察/修改 CFG 的正向分支，包含主 attention 和存在的 `attn2`。FLUX 双流层在图像输出投影前替换；单流层只替换 `proj_out` 输入中图像 token 的 attention 通道，保留文本行与 MLP 通道。FLUX block 编号先双流、后单流。沿用原生采样器，不增加在线 target/anchor 模型前向。首版使用 FLUX 内嵌 guidance，`true_cfg_scale=1`。

## 运行四类实验

在仓库根目录执行，使用安装了 `benchmark/requirements.txt` 依赖的 Python 环境。

```bash
# 预览全部实验，无模型加载或文件写入
python DiT/attention_replacement/run.py benchmark --cases all --dry-run

# 离线收集全部需要的概念；相同 anchor 复用同一份原型
python DiT/attention_replacement/run.py collect --cases all --model sd35_medium

# 原模型对照 + 方法采样 + benchmark 指标
python DiT/attention_replacement/run.py benchmark --cases all --model sd35_medium
```

| `--cases` | case | 评估任务 |
| --- | --- | --- |
| `i2p` | `i2p_nudity` | I2P 擦除、MS-COCO 1K 保留 |
| `ip` | `ip_pikachu`、`ip_snoopy_pikachu` | IP 单概念、多概念擦除与保留 |
| `style` | `style_van_gogh`、`style_monet`、`style_picasso` | 风格擦除与保留 |
| `object` | `object_church`、`object_multi5` | 对象单概念、多概念擦除与保留 |

`all` 展开为 8 个 case、9 项任务，默认共 5661 张方法图片。指标、阈值和评估概念沿用 `benchmark`，不另写评价协议。可以分别运行四组，也可以直接指定 case，例如 `--cases ip_snoopy_pikachu,object_multi5`。

默认概念对来自 `benchmark/configs/dit_target_anchor.json`，因此 nudity 的 anchor 是 `dressed`。共享表缺少 `gas pump`，本方法补充 `gas pump → machine`。可通过 `--anchor-overrides '{"nudity":"clothed","gas pump":"fuel dispenser"}'` 覆盖；**收集和采样使用相同覆盖**。单概念也可用 benchmark 原有 `--anchor clothed`。离线输入保持概念原文，不添加测试 prompt 模板。

将上述命令中的 `--model sd35_medium` 换成 `--model flux1_dev` 即运行 FLUX。显存配置复用 `--cpu-offload`、`--device-map`、`--max-memory` 等 benchmark 参数。模型已缓存时可加 `--offline` 禁止下载。`--model-path /path/to/model` 可指定本地原生 Diffusers 模型目录。

完整特征默认全部保存，磁盘用量随 seeds、空间 token 数、通道、层和时间窗口线性增长。建议先用较少的层和离线 seeds 完成小规模实验：

```bash
python DiT/attention_replacement/run.py collect --cases ip \
  --offline-seed-count 2 --layer-end 2 --step-end 2

# max-records 会在评估概念之间均衡取样；不改变离线概念 prompt
python DiT/attention_replacement/run.py benchmark --cases ip \
  --offline-seed-count 2 --layer-end 2 --step-end 2 \
  --max-records 6 --phase sample
```

## 调参、独立采样和分阶段评测

方法参数：`--layer-start/end`、`--step-start/end` 或 `--step-fraction`、`--offline-seeds` 或 `--offline-seed-count/start`、`--tau`、`--tau-by-concept '{"Pikachu":10}'`。`--step-end` 与 `--step-fraction` 二选一；比例相对于全部采样步。采样设置使用 benchmark 原有的 `--steps`、`--height`、`--width`、`--guidance-scale`、`--dtype` 等参数。

调整 `tau` 可复用同一缓存；模型、总采样步数、分辨率、guidance 等轨迹参数变化时需要重新收集。`--prototype-dir` 可显式指定已有 collection，并从中使用更小的层/时间窗口；此时实际离线 seed 列表以该缓存为准。`collect` 的显式 collection 则必须匹配原有完整窗口与 seed 列表。

```bash
# 指定离线 concept 对；之后两条命令保持 --pairs 相同
python DiT/attention_replacement/run.py collect --cases i2p \
  --pairs '[{"target":"nudity","anchor":"clothed"}]'
python DiT/attention_replacement/run.py sample --cases i2p \
  --pairs '[{"target":"nudity","anchor":"clothed"}]' \
  --prompt 'a person in a garden' --seed 42 --tau 0

# 观察模式：计算并保存距离/掩码，继续使用原始激活
python DiT/attention_replacement/run.py sample --cases i2p \
  --pairs '[{"target":"nudity","anchor":"clothed"}]' \
  --prompt 'a person in a garden' --seed 42 --intervention observe

# 可独立准备、采样或评测；重复相同配置会复用完整图片
python DiT/attention_replacement/run.py benchmark --cases style --phase prepare
python DiT/attention_replacement/run.py benchmark --cases style --phase sample
python DiT/attention_replacement/run.py benchmark --cases style --phase evaluate

# 直接评测已保存清单，不需要加载生成模型或离线缓存
python DiT/attention_replacement/run.py evaluate --manifest /path/to/manifest.json
```

`sample` 可重复传 `--prompt`，以同一个 `--seed` 生成不同 prompt，并按 `--batch-size` 分批。自定义 target 可用 `--pairs` 显式提供。benchmark 自定义擦除目标时还应传 `--targets`，以同步评估对象。

benchmark 默认采样 `--mode both`，自动填充并复用原模型对照图。仅生成方法图可用 `--mode image`，评测时仍需原图。`--tasks mscoco` 可只运行保留任务。缩小 benchmark 建议用 `--max-records`；`--limit-prompts` 按原数据顺序截断，可能截掉部分评估概念。评测权重和具体指标说明见 `benchmark/DIT.md`。

方法配置也可通过 benchmark 的 `--method-config` 提供，字段为 `adapter.attention_replacement`，CLI 方法参数优先。未指定 `--method` 时名称包含参数/原型摘要，观察与替换、不同阈值或原型分别记录；显式指定方法名时请为不同实验使用不同名称。

## 保存内容与分析

默认缓存位于 `DiT/attention_replacement/cache/<model>/<collection>/`：

```text
collection.json                         # 采样身份、实际时间表、sites/通道/网格
prompts/<prompt-id>/
  seed_10000/step_0000/<site>.safetensors # I: [H,W,D]，原生精度
  seed_10000/final.png                   # 完整离线轨迹的最终生成图
  seed_10000/metadata.json
  seed_10000/complete.json
  means/step_0000/<site>.safetensors      # mean: [H,W,D]，FP32
  means.json                            # 原始 prompt、seeds、均值文件摘要
```

benchmark 方法图在 `DiT/attention_replacement/results/<model>/<task>/<case>/<method>/images/<run>/`；对应 trace 在同级 `traces/<run>/<sample_id>/`。清单、评测指标和 benchmark 采样耗时仍在 `benchmark/results/<model>/`。`--result-root` 修改方法图目录，`--output-dir` 修改 benchmark 指标目录，`--prototype-root` 修改离线缓存目录。

在线每个被处理的 site/step 保存：

- `I_user`、`I_new`：`[H,W,D]`，替换前/后实际激活。
- `d_tar_i`、`d_anc_i`、`margin_i`：各概念的 `[H,W]` FP32 距离/差值。
- `mask_i`、`mask_union`：各概念及并集的布尔掩码。
- `winner`：`[H,W]` 概念索引，用于还原不同 anchor 的重叠选择。
- `metadata.json`：prompt、seed、概念对、参数、site 路径、实际 timestep/sigma、原型摘要、模型调用次数及批量耗时。

`I_user` 是当前实际轨迹的激活：替换模式下包含之前干预造成的轨迹变化。观察模式提供未干预轨迹的诊断。`--no-trace` 关闭在线张量保存，仍记录运行元数据；原型收集始终保存原始特征。耗时中完整 trace 与无 trace 应分开比较；hook 时间为 host 耗时，I/O 时间包含张量向 CPU 的传输。

```bash
python DiT/attention_replacement/run.py analyze \
  --run-dir /path/to/diagnostic-run-or-traces --heatmaps 8
```

分析只读取已保存张量，在 CPU 输出 `analysis/summary.csv`、各 attention 分支的时间—层覆盖率图，以及前若干个原生 token 网格的 mask/margin 图；`--heatmaps` 控制空间图片数量。

## 本次验证范围

已在 CPU 的小型随机 SD3/FLUX 模型上运行原生 pipeline，覆盖离线收集、均值、批量在线替换、观察模式像素一致、trace 与模型调用次数。四类 benchmark 另用小型 SD3 模型完成 8 个 case/9 项任务的对照图与方法图采样，并检查清单、恢复运行和参数身份。随机模型使用固定 prompt embeddings，验证工程流程；不代表真实模型的概念擦除效果。当前机器 CUDA 驱动不可用，尚未运行完整权重的效果评测。
