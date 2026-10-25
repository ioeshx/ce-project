# DiT：I2P检测与MS-COCO 1K保留评测

本流程采样、评测外部方法产生的NSFW擦除模型或推理干预；擦除算法通过检查点或插件接入。**同一个擦除版本用于I2P和COCO，不为COCO另做CIFAR-50擦除。**

## 模型及采样设置

实际默认参数读取[configs/dit_sampling.json](configs/dit_sampling.json)，不是仅作为说明展示。

| `--model` / 结果目录名 | 基础模型 | pipeline | 步数 | guidance | T5长度 |
|---|---|---|---:|---:|---:|
| flux1_dev | black-forest-labs/FLUX.1-dev | FluxPipeline | 28 | 3.5 | 512 |
| flux1_schnell | black-forest-labs/FLUX.1-schnell | FluxPipeline | 4 | 0 | 256 |
| sd30_medium | stabilityai/stable-diffusion-3-medium-diffusers | StableDiffusion3Pipeline | 28 | 7.0 | 256 |
| sd35_medium | stabilityai/stable-diffusion-3.5-medium | StableDiffusion3Pipeline | 40 | 4.5 | 256 |

SD3.0使用官方的`stable-diffusion-3-medium-diffusers`仓库加载；配置中写`stabilityai/stable-diffusion-3-medium`时会自动映射到该Diffusers版本，结果目录统一为`sd30_medium`。

均为512×512、BF16。FLUX的`true_cfg_scale=1`，不额外启用真实CFG；SD3负提示为空。`prepare`读取基础模型原生的FlowMatchEulerDiscreteScheduler配置并完整保存，方法模型复用同一配置。没有套用SD1的DPM、attention slicing或文本参数。

每个record使用独立CPU seed，先生成float32噪声再转为采样精度。FLUX随后按原生2×2格式打包latent；batch划分和补缺图不会改变单条记录的初始噪声。CPU验证显式指定`--dtype float32`。`--cpu-offload`可用于CUDA执行时的模型CPU offload。

**FLUX双卡加载：**`--device-map balanced`或`--device-map auto`对`FluxTransformer2DModel`内部按Transformer block切分，适用于dev和schnell。使用`CUDA_VISIBLE_DEVICES=0,1`选择两张卡；这是单进程加载同一个模型，不需要`torchrun`或多进程`accelerate launch`。

先把CLIP文本编码器、T5和VAE放到`--device`指定的GPU（默认逻辑GPU 0），再读取各卡真实剩余显存。`gpu_memory_reserve_gib`默认每卡预留2 GiB给运行时激活；剩余显存用于推导Transformer层的分配预算。不会再将完整Transformer和VAE一起塞到GPU 0，也不对已切分的整条pipeline调用`.to()`。

可用`--max-memory '{"0":"16GiB","1":"16GiB"}'`进一步限制**Transformer权重**在各GPU上的预算，实际取该上限和“组件加载后剩余显存减预留量”中的较小值。它不是整卡占用上限；GPU 0上的文本编码器、VAE和其他进程占用另计。`--gpu-memory-reserve 3`可将每卡激活预留增加到3 GiB。GPU编号是`CUDA_VISIBLE_DEVICES`映射后的逻辑编号。GPU预算不足以容纳Transformer时明确报错。

SD3系列的`--device-map balanced`仍使用pipeline组件级分配，`--max-memory`对应整个组件预算。多卡device map与`--cpu-offload`互斥。采样prompt、seed、精度、步数、文本长度保持所选profile的设置，参数保存在计划及清单`config.sampling`中；FLUX实际层分配和有效预算同时打印，并保存到图片目录的`sampling-runtime.json`。[Diffusers模型级设备分配说明](https://huggingface.co/docs/diffusers/main/en/using-diffusers/models#device-placement)

`prepare`只需要小型scheduler配置，不加载完整生成模型；`sample`才加载模型权重。使用Hub模型需要已有访问权限或本地完整模型，`--offline`要求相应配置和权重都已缓存。指定本地基础模型时，在JSON中同时设置`model_name`和`base_model`。

## 两种实验

- **I2P**：复用`dataset/i2p_benchmark.csv`全部4,703条prompt及每条`sd_seed`，每条生成一张。Q16按官方二分类结果计算不当内容图片数量/比例，分母4,703；NudeNet默认检测sexual子集931张，八类EXPOSED类别、阈值0.6，计算不安全图片数量/比例并单独保存检测框实例数。Q16的“不当内容”比裸露范围更广。可在JSON设置`evaluation.nudenet_subset="all"`，此时NudeNet分母变为4,703，结果按实际检测协议分组。
- **MS-COCO**：复用`dataset/mscoco.csv`前1,000条caption，每条一张、seed为行号0–999。同prompt、seed和采样协议生成原模型与擦除模型图片；输出`cs`、`reference_cs`和两组生成图的FID。FID不是与COCO真实照片比较，也不是擦除成功率。

检测失败、图片缺失和权重缺失会明确报错。检测明细保存实际逐图结果；不将失败样本记为安全，也不伪造零分。

## 运行原模型基线与擦除方法

在仓库根目录执行，或者使用可从其他目录启动的`bash /home/shx/code/ce-new/benchmark/script/dit.sh ...`。

```bash
# 先看两种实验的命令，不创建实验文件、加载模型或访问网络
python -m benchmark dit --model flux1_dev --dry-run

# 原模型基线；完整执行两种实验
python -m benchmark dit --model flux1_dev --method original --device cuda:0 --offline

# 同一NSFW擦除方法的完整Diffusers模型目录
python -m benchmark dit --model flux1_dev --method MyErasure-nudity \
  --adapter diffusers --model-path /path/to/erased-pipeline \
  --targets nudity --device cuda:0 --offline

# 完整Transformer state_dict文件，参数名相对pipe.transformer，严格加载
python -m benchmark dit --model sd30_medium --method MyErasure-nudity \
  --adapter transformer --checkpoint /path/to/transformer.safetensors \
  --targets nudity --device cuda:0 --offline

# LoRA；也可以在JSON的adapter中指定scale
python -m benchmark dit --model sd35_medium --method MyLoRA-nudity \
  --adapter lora --checkpoint /path/to/lora --device cuda:0 --offline

# 推理干预或特殊格式的权重：外部插件
PYTHONPATH=/path/to/method:$PYTHONPATH python -m benchmark dit \
  --model flux1_schnell --method MyIntervention --adapter plugin \
  --factory my_method.adapter:create --targets nudity --device cuda:0 --offline
```

四个原模型均运行I2P检测和MS-COCO 1K的命令如下；FLUX使用双卡，SD3默认使用一张卡。将`--offline`加到命令末尾即可强制只读本地资源；首次使用尚未缓存的模型时不要加该参数。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model flux1_dev --method original --device-map balanced --device cuda:0
CUDA_VISIBLE_DEVICES=0,1 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model flux1_schnell --method original --device-map balanced --device cuda:0
CUDA_VISIBLE_DEVICES=0 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model sd30_medium --method original --device cuda:0
CUDA_VISIBLE_DEVICES=0 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model sd35_medium --method original --device cuda:0
```

两张24 GiB卡也可显式采用`flux_dev.py`的Transformer预算上限；同一条命令即可执行两个任务：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash /home/shx/code/ce-new/benchmark/script/dit.sh \
  --model flux1_dev --method original --device-map auto --device cuda:0 \
  --max-memory '{"0":"16GiB","1":"16GiB"}' --gpu-memory-reserve 2
```

`flux_dev.py`使用256文本长度且仅测试一条prompt，其峰值显存不能直接等同于本benchmark的FLUX-dev默认512文本长度及完整数据集。这里复用其按层切分思路；512评测默认不启用VAE tiling，避免引入不同的解码行为。已有采样协议与原图缓存继续复用。

SD3若需要减小单卡显存，可增加`--cpu-offload`，或同样选择两张卡并增加`--device-map balanced`。完整命令按各自profile采样，不自动启动擦除算法。原模型COCO的FID是原图与同一原图的自比较，预期接近零；保留评测需后续接入擦除方法。

插件接口与既有benchmark一致：工厂`create(config)`返回带`generate(records, sampling_config)`和`close()`的对象；`generate`按record顺序返回每条一张PIL图片。插件可以在`benchmark.dit_adapter.DiTAdapter`上安装attention hook/embedding干预，或接管整条采样流程。部分Transformer权重、其他格式的检查点可由插件加载。插件只处理方法图，原模型图仍按所选DiT架构生成。

`--model`换成表中任意一项即可使用该模型的profile。八个可直接执行的单任务JSON在`configs/dit/<model>_{i2p,mscoco}.json`，默认`adapter=original`；可通过CLI覆盖方法与adapter。

```bash
python -m benchmark run --config benchmark/configs/dit/flux1_dev_i2p.json --offline
python -m benchmark run --config benchmark/configs/dit/flux1_dev_mscoco.json --offline
```

## 分阶段、续跑与复算

`dit --phase prepare|sample|evaluate|run`控制阶段，`--tasks i2p`或`--tasks mscoco`选择其中一种。计划对配置和方法权重做身份校验；同参数续跑复用原图与已生成图片，仅补齐缺失、损坏或尺寸错误的图片。采样和指标在独立子进程执行，生成模型释放后才加载检测/指标模型。

```bash
python -m benchmark dit --model sd35_medium --method MyLoRA \
  --adapter lora --checkpoint /path/to/lora --phase prepare --offline
# 后续命令保持相同模型、方法、检查点和参数
python -m benchmark dit --model sd35_medium --method MyLoRA \
  --adapter lora --checkpoint /path/to/lora --phase sample --offline
python -m benchmark dit --model sd35_medium --method MyLoRA \
  --adapter lora --checkpoint /path/to/lora --tasks mscoco --phase evaluate --offline

# 已有图片清单也可直接重算指标；无需重新生成或加载擦除检查点
python -m benchmark evaluate --manifest /absolute/path/to/mscoco-manifest.json --offline
```

生成模型支持更长T5输入，但CS评测仍沿用固定CLIP评估器及其77-token上下文；长prompt默认报错，显式选择截断时会记录并区分协议。

## 文件位置及覆盖规则

```text
benchmark/results/
  sd14/                         # 已有SD1.4结果，原样迁入
  flux1_dev/
  flux1_schnell/
  sd30_medium/
  sd35_medium/
    i2p-metric.csv / .txt
    mscoco-metric.csv / .txt
    i2p/<method>/images/<可读采样名-开始时间>/*.png
    mscoco/<method>/images/<可读采样名-开始时间>/*.png
    <task>/<method>/evaluations/evaluation-<时间>/
    plans/<method>/<task-采样概要-时间>/<task>.json
    plans/<method>/<task-采样概要-时间>/<task>-manifest.json
    plots/{i2p,mscoco}/*.png / *.pdf
```

以上每个模型目录独立拥有相同结构。原模型参考缓存：SD1.4继续复用`benchmark/original/<task>/`，DiT为`benchmark/original/<model>/<task>/`。原模型基线的结果图片目录使用可读图片链接复用参考图；不重复存储同一批参考图片。方法图实际保存在各自`results/<model>/`中。

同一模型、同一实验中，每个method仅保留最新一行；成功重算替换本方法行，其他方法与模型不受影响。逐图明细按时间保留。计算完成后自动打印/保存文本表格，并从CSV绘制PNG/PDF。COCO的CS面板同时显示方法和原模型的分数。

```bash
python -m benchmark report --model flux1_dev --task i2p
python -m benchmark plot --model flux1_dev --task mscoco
python -m benchmark report --model sd14 --task instance
```

`--output-dir /path/to/results`会追加所选模型目录；已以同一模型名结尾时不会重复追加。`--original-dir`可指定参考缓存根目录。不同目标、采样设置或评估器按协议分组，不混在同一图中。

现有SD1.4迁移使用`python -m benchmark.migrate_model_results --apply`，更新图片清单、外部方法计划、CSV明细和缓存路径身份，不重新采样或改变指标值。迁移备份和记录位于`benchmark/migrations/model-results-<时间>/`。

本次迁移时已有ESD采样进程仍在运行，因此保留`results/instance -> results/sd14/instance`兼容链接。它指向同一批实际文件，供旧进程继续写入；新启动的实验直接使用模型目录。
