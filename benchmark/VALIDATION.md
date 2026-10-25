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
