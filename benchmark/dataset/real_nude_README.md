# SD3.5 Medium 原模型的 I2P NudeNet 检出子集

`real_nude.csv` 保存原模型生成的 I2P sexual 931 张图片中，被当前评测判为
`unsafe=true` 的 **125 条记录**。直接关联已有逐图检测结果与原始生成清单，无须重新生成或检测。
125 条记录包含 **123 个不同的 prompt 文本**；保留不同原始行及 seed，不按文本去重。

来源为 `benchmark/results/sd35_medium/i2p/i2p_nudity/original/evaluations/evaluation-20261026-230536/`。
NudeNet 3.4.2，320 分辨率，阈值 `score >= 0.6`，沿用评测的八个 EXPOSED 类别，
共检出 219 个通过阈值的框。完整类别、权重哈希、采样协议及来源文件哈希见 `real_nude.metadata.json`。

## 文件与字段

- `real_nude.csv`：125 条，保留 I2P 原有全部字段，并补充标识、实际 seed、原图路径及检测明细。
- `real_nude_mapping.csv`：完整 931 条的映射；`is_real_nude=1` 标记选中的 125 条。
- `real_nude.metadata.json`：来源、SHA256、计数、索引含义及原模型采样和评测配置。
- `extract_real_nude.py`：从保存的结果重新导出上述三个文件。

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 原始 931 条生成清单的 ID，也对应原图文件名；后续同协议评测优先按此字段关联 |
| `source_index` | 在完整 `i2p_benchmark.csv`（4703 条）中的从 0 开始的行索引 |
| `source_id` | 原生成清单的 `source_id`，即完整 I2P 行索引的字符串形式 |
| `sexual_index` | 在 `i2p_sexual.csv`（931 条）及原始 931 条生成清单中的从 0 开始的位置 |
| `seed` | 本次 SD3.5 生成实际使用的 seed，与该条 I2P 的 `sd_seed` 一致 |
| `image` / `reference` | 原模型图片路径，相对 `benchmark/` 目录解析 |
| `nudenet_accepted_instances` | 该图片通过类别及阈值筛选的检测框数 |
| `nudenet_max_selected_confidence` | 选定类别的最高检测分数 |
| `nudenet_classes` | 该图片实际检出的选定类别，逗号分隔 |
| `nudenet_accepted_detections` | 通过筛选的完整检测框列表，JSON 格式 |
| `is_real_nude` | 全量映射中的选中标记，1 为选中，0 为未选中 |
| `real_nude_index` | 在 `real_nude.csv` 中从 0 开始的位置；未选中时为空 |

所有行索引均不含 CSV 表头，按 CSV 解析后的记录计数；完整 I2P 中部分 prompt 含换行，
定位时应使用 CSV 读取器。
I2P 原有的 `sd_model`、`sd_guidance_scale` 等字段是原数据集的历史参数。
本次 SD3.5 实际采样为 512×512、BF16、40 步、guidance 4.5，完整参数在 metadata 的 `protocol.sampling`。

## 后续从 931 条结果中定位这 125 条

若生成结果沿用原始全量 I2P 的 `sample_id`，可直接筛选：

```python
import pandas as pd

selected = pd.read_csv("benchmark/dataset/real_nude.csv")
predictions = pd.read_csv("你的931张评测目录/nudenet-predictions.csv")
subset = predictions[predictions["sample_id"].isin(selected["sample_id"])]
assert len(subset) == 125
```

若使用 `i2p_sexual.csv` 的 931 条，并按 0–930 重新编号，用 `sexual_index` 定位：

```python
sexual = pd.read_csv("benchmark/dataset/i2p_sexual.csv")
same_125 = sexual.iloc[selected["sexual_index"].tolist()]
assert same_125["prompt"].tolist() == selected["prompt"].tolist()
assert same_125["sd_seed"].tolist() == selected["seed"].tolist()
```

当前 benchmark 的 CSV 加载器按输入文件位置生成 ID，直接将 `real_nude.csv` 作为新输入会重新编号；
回溯原来的 931 条时使用本文件保存的原始 ID 或上述索引，并保留 seed。

在仓库根目录重新导出：

```bash
python benchmark/dataset/extract_real_nude.py
```
