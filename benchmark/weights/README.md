Q16官方学习参数由 `python -m benchmark download --resource q16` 获取。

默认路径 `Q16/ViT-L-14/prompts.p`，来源及固定commit/SHA256保存在相邻 `prompts.source.json`。
OpenAI CLIP、Transformers CLIP及FID分别使用各自标准缓存，NudeNet 3.4.2自带320n.onnx。
可通过配置指定全部本地权重；详见上级README。
