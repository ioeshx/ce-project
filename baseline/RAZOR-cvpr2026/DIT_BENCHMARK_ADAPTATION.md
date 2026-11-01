# RAZOR 的 DiT benchmark 适配评估

结论：**本轮跳过 SD3.5 Medium／FLUX.1-dev 的四组实验实现**。本目录的扩散脚本是单目标 SD3 Medium style 实验，存在会影响训练停止与选择的核心问题；不能仅修改模型名、配置或命令行就形成可用比较方法。按本次任务第 5 条，不为修复这些问题重新设计核心算法。

## 当前可行范围

| 模型 | I2P | IP | style | Object |
|---|---|---|---|---|
| SD3.5 Medium | 本轮跳过 | 本轮跳过 | 原 SD3 类可更换模型，但核心问题未修复，未接入 benchmark | 本轮跳过 |
| FLUX.1-dev | 本轮跳过 | 本轮跳过 | 无原生 FLUX 生成路径，未接入 benchmark | 本轮跳过 |

SD3.5 与 SD3 共用 `StableDiffusion3Pipeline`，因此加载模型本身不是主要障碍。当前代码尚不能提供有效、可复用的四类擦除模型；也没有将 CLIP／VLM 的其他任务脚本当作生成模型擦除实现。

## 源码与核心问题

检查对象为实际扩散代码 `Razor-SD.py`，而不是仅依据 `Readme.md` 的支持声明。

1. **训练停止信号恒定。** `clip_presence_score` 用 `logits.argmax` 得到索引，再把所有合法索引都计为命中。单标签时索引总为 0，多标签时也总在合法范围；所以任意非空图片集均返回 1。`style_unlearn_metrics` 于是恒有 `UA=1-presence=0`，IRA／CRA 同样恒为 1。`TARGET_UA` 直接控制层扩展的停止，不能仅跳过原评测并替换为 benchmark 的最终指标来修复训练流程。
2. **损失符号与代码注释的作用不一致。** `RazorTextLoss` 的 GA 返回 `-mean(cosine)`，GD 返回 `+mean(cosine)`；在 AdamW 最小化下，分别推高 forget 对齐、压低 retain 对齐。将符号反转会改变训练目标，不能作为普通参数适配静默进行。
3. **层选择与描述不一致。** 实际排序使用梯度范数／权重范数，再除以 `1+max(0, alignment)`；README 描述的是平方范数比与 `(1-cos)^alpha`。应先明确要复现的作者定义，而不是在适配中任意替换公式。
4. **只有单风格提示流程。** `CFG` 固定一个 `forget_style` 和一个 `retain_style`，提示模板均为 style-conditioned。不能把 nudity／IP／Object 字符串直接塞入 style 字段并声称完成对应实验；联合 IP 和五目标 Object 也没有现成实现。
5. **无法直接作为权重插件导出。** 整个脚本在模块顶层创建模型、生成图、计算梯度并修改第一个 CLIP 文本编码器；没有函数入口／参数解析，也没有模型、修改后的 text encoder 或 LoRA 的保存调用。公共 benchmark 若只加载 transformer／LoRA，将漏掉实际改动的文本编码器。
6. **没有 FLUX 路径。** 生成函数固定 SD3 的 `negative_prompt`／CFG 接口和两个 CLIP 编码器结构，没有处理 FLUX 的生成调用及不同文本条件结构。

完整迁移需要先修正并确认损失与层选择、重新定义有辨别力的停止指标、增加概念类型和多目标训练，再实现函数化、文本编码器导出／加载及 FLUX 模型路径。这涉及核心优化与评估定义，超出本轮允许的小范围模型适配。

## 已完成检查

- 对 `Razor-SD.py` 做 AST 检查，避免导入其顶层模型加载、目录创建和图片生成代码。
- 仅提取原 `clip_presence_score` 函数，用固定张量执行实际相似度与 argmax 计算：一标签和三标签两种情形均得到 presence=1、UA=0，即使选中的类别不是目标也被计为命中。
- 确认扩散脚本没有 `save_pretrained`、`save_lora_weights` 或 safetensors `save_file` 导出调用。
- 未下载模型、未启动原训练、未生成正式实验结果或工程测试文件。

## 文件记录

| 文件 | 修改与目的 |
|---|---|
| `DIT_BENCHMARK_ADAPTATION.md`（新增） | 记录各模型／实验范围、跳过原因和代码核查证据 |
| `Readme.md` | 添加本文入口 |

原 `Razor-SD.py` 及 CLIP／VLM 代码保持原样。未生成宣称已支持新实验的启动脚本，也未写入 benchmark 指标。后续若提供修正后的原生实现，可在明确算法定义后重新评估接入。
