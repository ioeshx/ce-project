Q16官方学习参数由 `python -m benchmark download --resource q16` 获取。

默认路径 `Q16/ViT-L-14/prompts.p`，来源及固定commit/SHA256保存在相邻 `prompts.source.json`。
OpenAI CLIP、Transformers CLIP及FID分别使用各自标准缓存，NudeNet 3.4.2自带320n.onnx。
可通过配置指定全部本地权重；详见上级README。


DiT v2新增资源：

```bash
python -m benchmark download --resource resnet50
python -m benchmark download --resource vlm
```

分别保存到`resnet50_v2.pth`和`Qwen3-VL-2B-Instruct/`，由`configs/dit_evaluation.json`实际读取；Qwen来源revision记录在模型目录的`model-source.json`。ResNet强制校验官方ImageNet-1K V2权重。支持`--destination`，改位置后相应设置`--resnet-weights-path`或`--vlm-model`。默认DiT sexual I2P不再需要Q16，只有`--with-q16`时使用。
