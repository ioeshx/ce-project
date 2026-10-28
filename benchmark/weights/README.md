Q16官方学习参数由 `python -m benchmark download --resource q16` 获取。

默认路径 `Q16/ViT-L-14/prompts.p`，来源及固定commit/SHA256保存在相邻 `prompts.source.json`。
OpenAI CLIP、Transformers CLIP及FID分别使用各自标准缓存，NudeNet 3.4.2自带320n.onnx。
可通过配置指定全部本地权重；详见上级README。


DiT v2新增资源：

```bash
python -m benchmark download --resource resnet50
python -m benchmark download --resource vlm
```

默认评测通过模型名加载：Qwen 使用 `Qwen/Qwen3-VL-2B-Instruct` 和 Hugging Face Hub 缓存，ResNet 使用 `ResNet50_Weights.IMAGENET1K_V2` 和 Torch 缓存。联网时缺失权重会自动下载，无需本目录中存在对应文件；上述命令用于提前填充缓存，供 `--offline` 使用。

指定 `--destination` 可另存到本地位置，评测时相应设置 `--resnet-weights-path` 或 `--vlm-model`。本地 VLM 下载旁存 `model-source.json`；ResNet 仍校验官方 ImageNet-1K V2 权重。默认 DiT sexual I2P 不需要 Q16，只有 `--with-q16` 时使用。
