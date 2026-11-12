# Detect / Erase 数据

从仓库根目录重新生成：

```bash
python DiT/detect_erase/dataset/generate.py
```

每组保存一个 JSON，`datasets` 中按 target 保存数据。目标来自 `benchmark/configs/dit_cases.json`。

| 组别 | target | Detect：每个 target | Erase：每个 target |
|---|---|---:|---:|
| i2p | nudity | 30 正例＋10 穿衣负例＋10 无关负例 | 50 对 |
| ip | Pikachu、Snoopy | 30 正例＋10 相近负例＋10 普通负例 | 80 对 |
| style | Van Gogh、Monet、Picasso | 30 正例＋10 相近负例＋10 普通负例 | 30 对 |
| object | church、golf ball、parachute、cassette player、banana | 30 正例＋10 相近负例＋10 普通负例 | 80 对 |

- **Detect**：保存于本目录的 `i2p.json`、`ip.json`、`style.json`、`object.json`。i2p 为 AI 直接编写的独立场景；其他组使用本目录 `personalized.py` 的对象模板、`personalized_style.py` 的风格模板替换概念。模板去重后按顺序循环；风格模板有 17 条不同文本，因此 30 条正例包含重复。相近／普通负概念及每条替换概念均保存在 JSON 中。
- **Erase**：保存于 `../erase/`。i2p 为 AI 编写的 50 个独立成人场景，每对仅将 `nude` 替换成 `clothed`。ip/object 使用 `benchmark/dataset/prompt_templates.py` 的全部 80 条 `imagenet_templates`，style 使用全部 30 条 `painting_templates`，直接复用测试模板。
- **空 anchor**：非 i2p 的 `anchor=""`；`anchor_prompt=template.format("")`，保留模板文字和空格，整条 prompt 不为空文本。
- **字段**：Detect 的 `label` 为 1/0，`kind` 区分三类样本；Erase 使用 `target_prompt`、`anchor_prompt`。`template_index` 对应所属 JSON 的 `templates` 列表。
