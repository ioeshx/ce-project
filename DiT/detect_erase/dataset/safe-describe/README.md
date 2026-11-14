# Safe-describe erase 数据集

基于相邻的 `../erase/` 数据集构造，使用正向画面描述作为 safe anchor。原数据集保持不变。四个 JSON 可直接通过训练参数 `--erase-dir DiT/detect_erase/dataset/safe-describe` 使用。

| 文件 | 概念 | 每概念文本对 | 总数 |
|---|---|---:|---:|
| `i2p.json` | nudity | 50 | 50 |
| `ip.json` | Pikachu、Snoopy | 80 | 160 |
| `style.json` | Van Gogh、Monet、Picasso | 30 | 90 |
| `object.json` | church、golf ball、parachute、cassette player、banana | 80 | 400 |

共 11 个概念、700 对文本。保留每条原始 `target_prompt`、`id` 和 `template_index`，仅改写 `anchor_prompt` 及相关 anchor 元数据。`templates` 和 `template_source` 仍记录原始 target 文本的模板；新 anchor 逐条编写，不再通过给模板填入空字符串构造。

## 改写规则

- **Nudity**：明确描述穿着长袖上衣和长裤的成年人，保留原人物动作、场景、物品、天气和光照。
- **IP**：删除整个角色，包括雕塑、绘画、玩具等形式中呈现的角色；描述剩余背景或露出的承载表面。
- **Object**：删除整个目标物体及其描绘形式；描述剩余背景或露出的承载表面。
- **Style**：保留人物、物体、地点、动作和自然现象，改为真实照片描述；移除画家名称、笔触和绘画表现方式。情绪、波浪、雾、晨光等仍属于画面内容，予以保留。

IP/object 的原始模板大多没有具体地点，能够保留的上下文主要是成像质量、光照、视角、表现媒介和材料。新 anchor 保留这些已有信息，不添加公园、街道等具体场景。没有背景细节的普通照片使用空背景；雕塑使用空展示底座，涂鸦使用素墙，纹身使用自然皮肤，刺绣使用素布，折纸使用普通折纸纸张，玩具使用空展示表面，塑料和毛绒形式使用对应的普通材料表面。这些是背景缺失时选用的中性描述，并不表示源文本指定了这些背景。

目标主体自身的数量、大小、归属和 clean/dirty 等属性随主体删除，不转移到背景上。低分辨率、模糊、裁剪、明暗、黑白和 JPEG 压缩等成像信息则保留。

## 示例

```text
nudity
target: A nude adult standing beside a wooden chair in a studio lit by a skylight.
anchor: A fully clothed adult wearing a long-sleeved shirt and long trousers, standing beside a wooden chair in a studio lit by a skylight.

IP
target: a black and white photo of the Pikachu.
anchor: A black-and-white photo of an empty background.

object
target: a cropped photo of the church.
anchor: A cropped photo of an empty background.

style
target: Van Gogh style painting of a field with mountains in the background.
anchor: A realistic photograph of a field with mountains in the background, with natural terrain colors and lighting.
```

## 使用

训练入口读取的是每条文本对的 `anchor_prompt`；概念级的 `anchor` 字段仅概括改写目标。两种 erase 状态采样路线都可以使用本数据集：

```bash
python DiT/detect_erase/train.py --stage erase --model sd35_medium \
  --erase-dir DiT/detect_erase/dataset/safe-describe \
  --erase-target cfg --erase-state trajectory --erase-trajectory safe --track
```

单卡和双卡启动脚本的 `ERASE_ARGS` 数组已指定本数据集：

```bash
--erase-dir DiT/detect_erase/dataset/safe-describe
```

训练入口 `train.py` 的默认 erase 目录仍为 `dataset/erase`；两个启动脚本通过上述显式参数选择本数据集。`dataset/generate.py` 仍生成原来的 detect/erase 数据，不会重写本目录。
