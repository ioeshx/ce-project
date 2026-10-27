# 实现验收记录

2026-10-22，在当前CPU环境完成。没有可用CUDA GPU；使用已有本地模型缓存及新下载的Q16官方参数。所有下列指标均由真实模型计算，没有用模拟分数填充实验结果。

## 自动检查

`python -m unittest discover -s benchmark/tests -v`：14项通过。

覆盖默认数据规模与931条sexual筛选、固定seed/ID、自定义模板、共享原图缓存与协议变更、清单校验、FID自比较近零及低秩/标准算法一致、非整批CS逐图平均、全目标N/A、8进程并发CSV更新、同方法覆盖及失败保留、独立CSV绘图、CASU旧协议导入、损坏图片补采样、NudeNet图片数与框数区分、Q16官方二分类逻辑、UNet/safetensors加载及LoRA/Diffusers接入分派。

额外检查默认CLIP提示长度：instance最长13 token、style最长16 token、COCO前1,000条最长28 token，均未超过77 token。

单元测试替身只用于核验逻辑，运行于临时目录，测试后删除。

## 真实模型链路

| 链路 | 实际输入 | 已验证输出 |
|---|---|---|
| CASU旧图instance导入 | Mickey/Snoopy各3张原图与方法图，512×512、原20步协议 | 真实CLIP、逐概念FID、目标/非目标汇总、CSV/文本/PNG/PDF |
| CASU旧图style导入 | Van Gogh/Monet各2张原图与方法图，512×512、原20步协议 | 真实CLIP/FID、按概念及角色汇总、CSV/文本/PNG/PDF |
| SD1.4原模型采样 | I2P中3条sexual提示，64×64、DDIM 2步、CPU | 逐行seed采样，真实ViT-L/14 Q16与NudeNet推理、逐图/检测明细、CSV/文本/PNG/PDF |
| CASU-2权重修改采样 | Snoopy两条模板，64×64、DDIM 2步、CPU | 原图缓存、真实CASU权重加载及生成、CLIP/FID、全目标非目标N/A |
| 推理干预插件采样 | 与上一行相同提示/seed | 插件负向条件介入、共享原图缓存、真实CLIP/FID、同协议合并图表 |
| MS-COCO原模型 | 前2条真实caption，64×64、DDIM 2步、CPU | 真实生成、CLIP、原图自比较FID接近零、CSV/文本/双面板图 |

NudeNet小样本中未检出通过阈值的框时，零值来自真实检测结果。另验证过I2P子集完全没有sexual记录时明确报错，不发布结果。

2026-10-23目录重构后，历史smoke图片、结果CSV/图表及测试专用缓存已按用户要求删除；上表仅保留已完成验收的文字记录。单元测试继续使用临时目录，不在正式结果目录留下测试图。原有正式instance/COCO实验及其真实采样协议保留，可在当前 `results/`、`original/` 中查看。当前默认采样以 `configs/sd14_sampling.json` 为准。

新增路径检查覆盖可读名称碰撞、同协议复用与不同参数隔离，以及实际文件迁移后的图片/清单完整性和重新prepare复用。迁移日志位于 `migrations/readable-layout-<时间>/migration.json`，包含旧到新路径映射及历史文档备份索引。

## 2026-10-26：DiT v2 扩展验收

已实现八个擦除case／九组评测：I2P nudity及COCO保留、两组IP、三组style、两组Object。配置从`dit_cases.json`、`dit_evaluation.json`和既有`dit_sampling.json`实际读取。默认DiT I2P为sexual 931条、NudeNet-only；SD1.4默认协议保持原样。

### 自动检查

`python -m unittest discover -s benchmark/tests -v`：63项测试，59项通过、4项沿用既有跳过设置（当前工作区缺少旧Retention Repair／Output Response包）。另运行五个baseline的benchmark接入回归：SPEED 7项、ESD 6项、RECE 7项、AdaVD 8项、UCE 6项，全部通过。

新增覆盖实验数量与均衡种子、sexual筛选保留全量原行号／重复记录、同方法不同case的CSV与耗时隔离、原模型跨case共享、不同擦除目标的方法图隔离、显式anchor校验、按case映射权重、旧I2P实际图片复用、原模型image-only写入original且保留相对软链接、关闭Q16不构造Q16模型、VLM多标签／非法输出拒绝、完整千类Top-3排序、分类计数／混淆矩阵的CSV独立绘图，以及跨case预测缓存复用。

### 真实生成和指标链路

使用本机GPU的SD3.5 Medium原模型，按当前profile（512×512、BF16、Flow Matching 40步、guidance 4.5）生成24张普通内容的临时图片。生成结束并释放模型后，运行实际CLIP、FID、Qwen3-VL、ResNet-50和NudeNet。所有临时图片、指标CSV与图表均放在临时目录并已清理，正式results/original未写入测试结果。

| 链路 | 实际输入 | 验收观察 |
|---|---|---|
| IP | 三个角色各2张，共6张 | 真实CS 29.9376；Qwen单标签识别6/6；原模型自比较FID及表格、PNG/PDF、混淆矩阵生成成功 |
| style | 三位画家各2张，共6张 | 真实CS 32.8384；Qwen单标签识别2/6；CLIP/FID及分类完整链路成功 |
| Object | 十类各1张，共10张 | 官方V2 ResNet-50完整千类Top-1为9/10、Top-3为10/10；逐类统计和混淆矩阵成功 |
| COCO | 前2条caption，共2张 | 真实CS 26.0510，原模型自比较FID为0；CSV、表格和双面板图成功 |
| NudeNet-only | 上述普通内容图片中的2张 | 真实八类、阈值0.6检测，检出0张／0个框；没有Q16结果字段 |

以上是原模型、小样本的连通性验收，不代表擦除方法的效果。NudeNet输入为普通内容验证图片，没有在本次验收生成完整I2P sexual图集；正式I2P检测由运行命令执行。

style的6张原模型图片中，VLM仅识别正确2张。完整实验需要同时查看原模型风格识别率与方法识别率，不能直接把未识别的比例解释为擦除成功率；这一小样本也不代表完整数据集上的识别水平。

### 已准备资源

- 官方ResNet-50 ImageNet-1K V2：`weights/resnet50_v2.pth`，加载时验证官方权重SHA256前缀`11ad3fa6`，评测保存完整SHA256。
- Qwen3-VL-2B-Instruct：`weights/Qwen3-VL-2B-Instruct/`，来源revision为`89644892e4d85e24eaac8bacfd4f463576704203`，旁存`model-source.json`。
- 实测环境：torch 2.13.0、torchvision 0.28.0、transformers 5.16.1、diffusers 0.41.0。

完整运行与方法接入命令见[DIT.md](DIT.md)。全量数千张采样未作为实现验收步骤执行。
