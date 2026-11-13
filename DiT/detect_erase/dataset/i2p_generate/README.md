# Nudity detect：3000 条合成文本

按 NudeNet 八类可见性标注：裸体，以及腋下、腹部、臀部、脚部、女性胸部、男性胸部、成年女性/男性生殖部位暴露。男士赤膊、露腹、露腋下和赤脚入镜均为正例；人物均为成年人。标签描述文本要求呈现的内容，不是某个 seed 的生成图检测结果。

| 文件 | 正例 | 负例 | 合计 | 正负近邻对 |
|---|---:|---:|---:|---:|
| `train.jsonl` | 960 | 1440 | 2400 | 864 |
| `val_id.jsonl` | 120 | 180 | 300 | 108 |
| `val_ood.jsonl` | 120 | 180 | 300 | 108 |
| 合计 | 1200 | 1800 | 3000 | 1080 |

**构建**：AI 编写可见性/衣着表达、场景和语言骨架，`generate.py` 固定 seed=42 组合、筛选并标注。它是可复现的合成文本集，不是 3000 条独立采集的真实用户提示词。

- 正例：明确裸体 240 条；日常非性化暴露 480 条；明确局部暴露 240 条；不含 nude/naked 等裸体词、但由衣着和取景确定暴露的描述 240 条。
- 负例：安全穿衣人物 600 条；相近身体/动作但目标被遮盖 480 条；相关无人环境/物体 360 条；关键词干扰、遮挡与裁剪 240 条；普通无关场景 120 条。
- 每条含 `prompt`、`label`、`target`、`stratum`、`group_id`、`exposed_classes`、`label_evidence` 等字段。正负近邻保留相同场景、画风、表达骨架和长度桶，改变决定暴露的衣着或取景。

**划分**：先固定场景和表达素材的集合归属，再生成文本。正负配对只出现在同一 split。Val-ID 使用同类语域的新场景/表达；Val-OOD 保留多人衣着对照、反射取景、跨从句衣着说明和间接可见性描述的组合。无人对照使用相应的物体/反射构图。八类在两个验证集中均至少有 10 条正例，允许多部位标签重叠。

**检查**：全部文本唯一；训练/验证之间、以及对隔离来源的下列冲突均为 0：

1. NFKC、大小写、空白和标点规范化后的精确重复。
2. 连续 8-token 公共片段。
3. 词集合 Jaccard ≥0.80。
4. 规范化字符 `SequenceMatcher(autojunk=False)` 相似度 ≥0.90。

隔离来源包含完整 I2P、完整 30000 条 MSCOCO、其他 benchmark prompt CSV、旧 detect/erase i2p 数据。仅用它们排除重复，不作生成示例。近邻正负对在集合内部允许相似；这些文本检查不等同于证明所有语义改写都无重合。

两个 SD3.5 CLIP tokenizer 均逐条检查，内容不超过 75 token，留出 BOS/EOS。2550 条为 8～40 token，450 条为 41～75 token，正负类的长文本比例均为 15%。`manifest.json` 保存实际计数、部位分布、来源文件校验值、排除统计和数据文件校验值。

审核采用可见性/遮盖素材复核、分层输出抽查和全量程序检查；尚未经过独立双人人工复标。未生成图片或使用外部分类器筛选。

从仓库根目录重新生成或检查落盘数据：

```bash
python DiT/detect_erase/dataset/i2p_generate/generate.py
python DiT/detect_erase/dataset/i2p_generate/generate.py --check-only
```

依赖现有环境的 NumPy、Transformers，以及本地缓存的 SD3.5 两个 CLIP tokenizer。原始设计保留在 [PLAN.md](PLAN.md)。

`i2p.json` 是 `train.jsonl` 的训练入口兼容导出，使用 `datasets → target/samples` 结构，保留全部 2400 条训练样本及其字段，不包含验证集。训练时指定 `--dataset-dir DiT/detect_erase/dataset/i2p_generate --cases i2p --targets nudity --stage detect` 即可读取，无需修改训练代码。两个验证集仍单独保存在 JSONL 中，当前训练入口不会自动评估它们。
