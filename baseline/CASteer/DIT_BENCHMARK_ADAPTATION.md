# CASteer 的 DiT benchmark 适配评估

结论：**本轮不实现 SD3.5 Medium／FLUX.1-dev 移植**。当前实现依赖独立 cross-attention 输出；将其接到联合注意力需要定义新的干预位置并修改核心采集和 steering 路径。按本次任务第 5 条，跳过需要大幅修改核心算法的组合。

## 当前可行范围

| 模型 | I2P | IP | style | Object |
|---|---|---|---|---|
| SD3.5 Medium | 本轮跳过 | 本轮跳过 | 本轮跳过 | 本轮跳过 |
| FLUX.1-dev | 本轮跳过 | 本轮跳过 | 本轮跳过 | 本轮跳过 |

这不是概念类型限制：原仓库已有 concrete、style 和 human-related 配对提示及 I2P 脚本。限制来自本次指定模型的注意力结构。原仓库 SD1.4／SD2.1／SDXL／SANA 入口保留，但不属于本次 DiT 模型要求。

## 源码核查

| 位置 | 当前行为与缺口 |
|---|---|
| `core/utils.py:SUPPORTED_DIFFUSION_MODELS`、`init_pipeline_for_image_model` | 只有 SD／SDXL／SANA 系列，没有 SD3.5 或 FLUX loader |
| `core/diffusion_steering.py:DiffusionModelType.from_model` | 仅能归类 SD、SANA；直接拒绝 `sd35_medium`／`flux1_dev` |
| `HookManager._register_hooks_recursive` | SD 分支按 U-Net 的 down/up/mid 遍历，SANA 分支寻找 Sana block；未注册 SD3／FLUX 联合 block |
| `_register_hooks_for_sd_block` | 对 `BasicTransformerBlock.attn2` 的独立 cross-attention 输出注册 hook |
| `core/vector_dump.py:CrossAttentionOutputStatsCollector` | 按该 hook 的每层／每时间步输出收集正负提示统计，形成 steering vector |
| `core/controller.py:CrossAttentionOutputSteering` | 复用上述统计和位置索引做投影；接受 `joint/single` 字符串本身不表示已有对应模型 hook |

SD3.5 的 joint attention 同时包含图像和文本 token，部分 block 另有图像 self-attention `attn2`；这个 `attn2` 不能作为原 U-Net cross-attention 的替代。FLUX 有 joint／single 两类 block，也没有可直接复用的 U-Net `attn2`。把联合注意力返回的图像分支直接当成原始 CA 输出，会同时改变统计来源和干预含义。

若继续移植，需要明确联合注意力中用于采集／修改的张量、图像／文本 token 范围、输出投影位置、CFG 分支与步数对应，并为两种架构实现一致的统计和推理干预。该工作超出仅更换 loader、配置或启动参数的范围，本轮不将其作为原论文 CASteer 的已适配结果。

## 已完成检查

使用临时随机初始化的原生 `SD3Transformer2DModel` 和 `FluxTransformer2DModel`，未下载权重：

- 原生模型类型解析均拒绝新模型名。
- 强制按 SD 类型注册时，两种模型均返回 **0 个 hook**，collector 的 `num_attn_layers=0`。
- 因此仅添加模型名称或调用公共 benchmark，会出现没有采集／没有实际擦除的问题，不能作为有效 baseline。

未运行完整模型、未生成正式实验图片或指标，没有新增工程测试文件。

## 文件记录

| 文件 | 修改与目的 |
|---|---|
| `DIT_BENCHMARK_ADAPTATION.md`（新增） | 记录可行范围、核心缺口与检查证据 |
| `README.md` | 添加本文入口 |

原算法代码保持原样。未生成会误报支持的 DiT 启动脚本；公共 benchmark、已有方法和历史实验产物均未改动。
