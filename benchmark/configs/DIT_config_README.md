# DiT 模型采样设置

[dit_sampling.json](dit_sampling.json) 是四个文生图DiT模型实际读取的基础采样配置。完整运行与结果目录说明见[../DIT.md](../DIT.md)。官方来源核对日期：2026-10-25。

| 配置名 | 模型 | 采样步数 | guidance_scale | T5 最大序列长度 |
|---|---|---:|---:|---:|
| `flux1_dev` | FLUX.1-dev | 28 | 3.5 | 512 |
| `flux1_schnell` | FLUX.1-schnell | 4 | 0.0 | 256 |
| `sd30_medium` | SD3 Medium | 28 | 7.0 | 256 |
| `sd35_medium` | SD3.5 Medium | 40 | 4.5 | 256 |

全部配置按本项目要求使用 **BF16（`bfloat16`）和 512×512**；FLUX.1-dev 的采样步数由官方示例的 50 改为 **28**。这些是项目覆盖值，其余基础设置依据官方模型卡的普通、未量化 Diffusers 示例及 pipeline 默认值记录。SD3 Medium 官方示例使用 FP16，此处也统一覆盖为 BF16。

`steps` 对应 pipeline 调用参数 `num_inference_steps`；`dtype` 对应加载模型时的 `torch.bfloat16`。`pipeline`、`architecture`、`scheduler`、`scheduler_config_source` 是配置元数据，不应作为生成调用参数直接传入。

所有模型使用 `FlowMatchEulerDiscreteScheduler`。`scheduler_config_source: "pretrained"` 表示随对应模型 pipeline 加载该仓库的 `scheduler/scheduler_config.json`，保留其时间偏移、动态偏移及其他原生设置；不同模型不共享一份手工指定的 scheduler 配置。512×512 的时间步由相应 pipeline 按该分辨率计算。

FLUX 的 `guidance_scale` 按官方原生输入记录；`true_cfg_scale=1.0`、`negative_prompt=null` 表示基础配置未启用额外的双分支真实 CFG。SD3 系列的 `guidance_scale` 则是常规 CFG 系数，负提示词为空字符串。

FLUX的文本长度取模型卡示例值；SD3系列取`StableDiffusion3Pipeline`的T5默认值256。该字段不改变CLIP指标模型的原生77-token上限。

FLUX的`gpu_memory_reserve_gib=2.0`是多卡加载的每卡激活预留，CLI可通过`--gpu-memory-reserve`覆盖。`--device-map balanced/auto`切分Transformer内部block：先放置文本编码器与VAE，再按剩余显存减预留量计算预算；`--max-memory`可进一步限制Transformer权重预算。它属于执行设置，不改变采样协议和既有参考图缓存。有效预算和实际层分布保存到图片目录的`sampling-runtime.json`。SD3的balanced仍分配完整pipeline组件。

`benchmark.model_profiles`读取这些profile；`benchmark.dit_adapter`选择原生FLUX/SD3 pipeline。`prepare`读取并记录基础模型的完整scheduler配置，原模型与擦除版本共享；不会用SD1的DPM覆盖Flow Matching。`sd3_medium`仍可在JSON的`model_name`中作为旧别名，标准目录名和CLI选择统一为`sd30_medium`。

官方依据：

- [FLUX.1-dev 模型卡](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [FLUX.1-schnell 模型卡](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [SD3 Medium 模型卡](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [SD3.5 Medium 模型卡](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- [Diffusers FLUX 文档](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- [Diffusers SD3 文档](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)


## DiT v2 数据与评测配置

采样profile保持本文件设置。新增实际读取的[dit_cases.json](dit_cases.json)定义I2P、IP、style和Object的八个case，[dit_evaluation.json](dit_evaluation.json)定义NudeNet/Q16开关、Qwen单标签和ResNet-50 V2。默认I2P只生成sexual 931条并使用NudeNet；`--with-q16`恢复全量4,703条。`--cases all`运行完整套件，默认不传cases仍为nudity的I2P＋COCO。具体运行和方法配置见[DIT.md](../DIT.md)。

Object 保留 `object_church` 单目标和 `object_multi5` 五目标联合实验；后者擦除 church、golf ball、parachute、cassette player、banana。两者的 `dataset.concepts` 均显式列出 church、golf ball、English springer、goose、chain saw、cassette player、French horn、garbage truck、banana、parachute，每类 seed 0–49，共 500 张。类别及千类索引来自 [imagenet10.csv](../dataset/imagenet10.csv)，UCE／EraseAnything 的 anchor 来自 [dit_target_anchor.json](dit_target_anchor.json)；ESD 不使用 anchor。
