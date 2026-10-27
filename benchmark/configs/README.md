# 实验配置及实际读取关系

DiT的四个模型及双实验入口见[../DIT.md](../DIT.md)。`dit_sampling.json`是实际读取的DiT采样profile；`dit/<model>_{i2p,mscoco}.json`是八个单任务配置。DiT的COCO任务评测同一个NSFW擦除模型，不使用下表SD1.4的CIFAR-50 case。所有结果存入`results/<model_name>/`，SD1.4为`results/sd14/`。

这里的JSON是可执行配置。`instance.json`只表示第一组Snoopy实验，不能用一个`targets`数组表达三次独立擦除；数组中的多个概念表示在同一个模型中联合擦除。

JSON中的本地路径以`benchmark/`为基准，例如`dataset/mscoco.csv`、`weights/Q16/ViT-L-14/prompts.p`、`results/flux1_dev`；本地模型/检查点可写`./weights/my-model`或`../baseline/...`。准备后的配置会带`path_base: "benchmark"`标记，不保存机器目录；Hub模型ID保持原样。CLI显式传入的路径仍相对于命令调用目录。

## 当前七组正式实验

| case | JSON | 擦除目标 | 正式评测 |
|---|---|---|---|
| instance_snoopy | [instance.json](instance.json) | Snoopy | instance 4,000图 |
| instance_snoopy_mickey | [instance_snoopy_mickey.json](instance_snoopy_mickey.json) | Snoopy、Mickey联合擦除 | instance 4,000图 |
| instance_snoopy_mickey_spongebob | [instance_snoopy_mickey_spongebob.json](instance_snoopy_mickey_spongebob.json) | Snoopy、Mickey、SpongeBob联合擦除 | instance 4,000图 |
| style_van_gogh | [style.json](style.json) | Van Gogh | style 900图 |
| style_monet | [style_monet.json](style_monet.json) | Monet | style 900图 |
| style_picasso | [style_picasso.json](style_picasso.json) | Picasso | style 900图 |
| mscoco_cifar50 | [mscoco.json](mscoco.json) | CIFAR-50全部50个概念联合擦除 | MS-COCO前1,000条caption |

instance评测固定使用全部5个角色的400条提示；style使用全部3个画家的90条提示；每提示seed 0–9。`targets`用于区分目标和非目标，不会把评测集缩小为目标提示。

MS-COCO只有一组独立目标配置，每个比较版本评测一次。`targets_from: "CIFAR50_classes"`由`benchmark.common.configuration()`实际解析为`benchmark/dataset/prompt_templates.py`中的50个概念，展开后的完整列表写入清单。正式COCO caption保持原始前1K，不按目标筛选；seed为行号0–999。这里不加载CIFAR图片。

[i2p.json](i2p.json)是额外的I2P独立评测入口，当前两个方法的默认七组套件不执行I2P。[sd14_sampling.json](sd14_sampling.json)是所有SD1.4实验共用的采样协议，由`benchmark.sampling_profile`读取；任务JSON中未重复定义采样器、步数和精度。

## 哪些命令会读取这些文件

**直接使用benchmark：**

```bash
# 在仓库根目录执行；prepare只生成清单，不加载模型或计算分数
python -m benchmark prepare --config benchmark/configs/instance_snoopy_mickey.json
python -m benchmark prepare --config benchmark/configs/style_monet.json
python -m benchmark prepare --config benchmark/configs/mscoco.json
```

`benchmark.cli.get_config()`调用`benchmark.common.configuration()`读取指定JSON，再应用CLI覆盖。单独`--task instance`使用内置单任务默认值，不会自动执行三组；运行全部七组请使用方法的suite/rerun入口。

**retention_repair和output_response完整套件：**

```bash
bash retention_repair/script/cfrr_rerun_sd14.sh --device cuda:0 --offline
bash output_response/script/output_response_rerun_sd14.sh --device cuda:0 --offline
```

两个入口都经过`benchmark.experiment_cases`读取上述七个JSON的目标；CFRR还读取`erasure_anchor`，默认恰为一个空格`" "`。Output Response保留原生无anchor的优化方式。

评测阶段，两种方法的`benchmark_configs()`调用`evaluation_configuration()`读取匹配case的JSON，将实际检查点、方法名、插件及CLI设置合并进去，再交给统一benchmark。实际运行配置保存在：

```text
benchmark/results/sd14/cfrr-plans/<method>/<task-采样概要-时间>/<task>.json
benchmark/results/sd14/output-response-plans/<method>/<task-采样概要-时间>/<task>.json
```

其中`configuration_source`记录本次使用了哪个任务JSON。运行清单及其`config`记录最终生效的全部参数；历史清单不会因修改模板而改变。

## 字段含义与覆盖规则

- `targets`：suite的擦除目标，以及通用评测的目标/非目标划分。显式列表优先于`targets_from`；已有检查点的评测以检查点中的实际目标为准。
- `erasure_anchor`：供CFRR suite构造擦除命令使用；通用benchmark只采样/评测，不执行擦除。
- `dataset`、`evaluation`：提供评测默认项，例如seed、COCO数量、CLIP模型和FID计算方式。方法CLI提供的同名参数（包括其默认值）优先；正式采样参数由共享SD1.4 profile和方法CLI确定。
- `adapter.kind: "original"`：直接运行这些JSON时采样原模型基线；不会自动执行概念擦除。评测其他方法时，通过CLI/自定义JSON指定adapter、检查点或已有图片。方法suite会自动换成各自插件和实际检查点。
- `method`：直接运行配置时使用的结果名称。七组名称不同，避免按`original`同名更新时互相覆盖；方法suite会使用包含case和比较版本的名称。
- `description`：中文说明，保留在配置中，不参与算法计算。

修改配置后应启动新的实验目录；续跑仍使用已有清单及原参数。汇总指标统一写入`benchmark/results/sd14/{instance,style,mscoco}-metric.csv`，同时生成文本表格和PNG/PDF图。

## DiT v2 实验配置

- [dit_cases.json](dit_cases.json)：八个实际读取的擦除case，包含I2P＋COCO、两组IP、三组style、两组Object。
- [dit_evaluation.json](dit_evaluation.json)：默认NudeNet-only、Qwen单标签分类及固定ResNet-50 V2配置。
- [dit_sampling.json](dit_sampling.json)：四模型采样参数，继续由所有case共享。

`python -m benchmark dit --model flux1_dev --cases all --dry-run`可查看完整计划；默认不传cases仍运行nudity的I2P＋COCO。单任务入口可用`--case ip_pikachu --task instance`。IP种子0–3、style种子0–2、Object种子0–49；SD1.4保持原有设置。

默认I2P为sexual 931条且不加载Q16，`--with-q16`在prepare时切换全量4,703条。需要anchor的方法声明`adapter.requires_anchor=true`并显式设置`erasure_anchor`，没有默认空格。多个权重case通过`--method-config`按case映射检查点。运行、权重准备及输出规则见[DiT说明](../DIT.md)。
