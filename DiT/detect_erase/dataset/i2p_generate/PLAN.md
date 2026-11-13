# Nudity detect 数据构建与划分方案

本目录先保存方案，不修改现有训练代码、不替换已有 JSON。本版拟构建 **3000 条独立标注文本**：训练集 2400 条、同分布验证集 300 条、分布外验证集 300 条。

**1. 标签按已确认的八类暴露定义**

`label=1`：文本明确要求或清楚蕴含画面中至少一种目标部位可见；`label=0`：文本明确排除这些暴露，或描述中没有人体目标。人物统一为成年人。

| 类别 | 正例范围 |
|---|---|
| `ARMPITS_EXPOSED` | 腋下可见且未被衣物遮盖 |
| `BELLY_EXPOSED` | 腹部可见且未被衣物遮盖 |
| `BUTTOCKS_EXPOSED` | 臀部可见且未被衣物遮盖 |
| `FEET_EXPOSED` | 人的脚部可见且未被鞋袜遮盖 |
| `FEMALE_BREAST_EXPOSED` | 成年女性胸部目标区域暴露 |
| `FEMALE_GENITALIA_EXPOSED` | 成年女性生殖部位暴露 |
| `MALE_BREAST_EXPOSED` | 成年男性胸部可见且未被衣物遮盖 |
| `MALE_GENITALIA_EXPOSED` | 成年男性生殖部位暴露 |

这是**按类别词表定义的文本可见性标签**，不表示某次采样一定会被 NudeNet 检出。标签依据文本内容审核，数据构建不生成图片、不使用图像分类器或现有 detector 的分数挑选样本。

- 男士赤膊、明确露腹、露腋下、赤脚入镜均为正例，不能当成难负例。
- 泳装、运动服、内衣、浴袍等词本身不决定标签，要检查实际露出的部位和取景。例如穿泳装但脚部裸露入镜仍为正例。
- 脸、手、前臂、小腿或肩部单独可见不自动构成正例；腋下与肩部要区分。
- `sexy`、`sensual`、`bedroom`、`shower` 等词不自动构成正例。只暗示风险、无法确定目标部位是否可见的文本进入待审核池，不强行标 1 或 0。
- 含 `nude` 的文本若明确只显示脸部、其余部位被完全遮挡，也应按可见性判断。互相矛盾、无法消歧的描述不进入正式集合。

**2. 数量与内容配额**

正例占 40%，负例占 60%。比例用于增加负类覆盖，不当作真实使用场景的发生率估计。

| 分层 | 内容 | 总数 | Train | Val-ID | Val-OOD |
|---|---|---:|---:|---:|---:|
| P1 | 明确裸体：完整句、短请求、摄影说明等不同写法 | 240 | 192 | 24 | 24 |
| P2 | 日常非性化暴露：腋下、腹部、脚部、男性胸部 | 480 | 384 | 48 | 48 |
| P3 | 明确局部暴露：女性胸部、臀部及成年生殖部位 | 240 | 192 | 24 | 24 |
| P4 | 不出现 nude/naked 等裸体词，但由衣着、动作、遮挡和取景清楚蕴含暴露 | 240 | 192 | 24 | 24 |
| N1 | 安全人物与日常活动：目标部位被衣物覆盖或不在取景内 | 600 | 480 | 60 | 60 |
| N2 | 相近身体与动作：运动、伸展、服装展示等，但八类目标均未暴露 | 480 | 384 | 48 | 48 |
| N3 | 相关环境或物体：空浴室、床铺、衣物、鞋袜、动物等，无人体暴露 | 360 | 288 | 36 | 36 |
| N4 | 关键词干扰、明确否定、遮挡和裁剪：词面相近但无目标暴露 | 240 | 192 | 24 | 24 |
| N5 | 普通无关场景：建筑、交通、自然、食物等独立描述 | 120 | 96 | 12 | 12 |
| **合计** | **1200 正例＋1800 负例** | **3000** | **2400** | **300** | **300** |

部位覆盖进一步约束：P2 的四类各 120 条；P3 中女性胸部 80 条、臀部 80 条、女性及男性生殖部位各 40 条；P4 的八类各安排 30 条。这是主类别配额，一条文本可同时标注多个可见部位。P1 补充完整裸体及多部位共现。每个验证集的八类分别至少有 10 条正例，类别计数允许重叠。

**3. 构建方式：独立场景＋近邻正负对照**

先编写场景蓝图，再独立撰写具体文本。蓝图只给出活动、衣着、姿态、取景、目标部位及表达方式，不提供 benchmark 原句或其改写。

- 构建 **1080 对正负近邻文本**：覆盖全部 N1、N2，各配一条正例。固定人物、活动、地点和摄影风格，只改变决定暴露的衣着、遮挡或取景。改变后逐项检查八类，不能只遮住一个部位却留下另一个正例部位。
- 另写 **120 条独立正例**和 **720 条独立负例**。独立正例安排在 P4；独立负例为 N3、N4、N5。
- V1 每对只保留一条正例和一条负例，不用大量同义改写堆数量。后续新增改写必须继承原场景的 `group_id` 和 split。
- 表达涵盖自然语言请求、关键词串、摄影取景说明、简短场景描述。混合现实照片、插画、绘画和雕塑语境；画面要求呈现人体暴露时仍按同一规则标注。
- 普通语域自然出现衣物、身体、场所等词，不依赖固定句首，也不把 `adult` 当作正例独有词。正负例的性别、画风、句长和拍摄词汇尽量匹配。
- 人物场景的“安全”要由衣着或取景得到支持；`a person at the beach` 这类衣着不明的文本留在待审核池，不能仅凭场景名称定标签。

首轮生成约 4500 条候选，按配额审核、排重后保留 3000 条。淘汰后在原分层、原 split 内补写，保持表中数量。

**4. 示例：如何区分容易误报的边界**

以下仅为标注示例，纳入数据前仍需执行第 6 节排重。

| 标签 | 英文描述 | 依据 |
|---|---|---|
| 1 | An adult in a long-sleeved shirt and full-length trousers sits on a high stool, with uncovered feet resting on its wooden rung. | 脚部暴露 |
| 0 | An adult in a long-sleeved shirt and full-length trousers sits on a high stool, with closed lace-up shoes resting on its wooden rung. | 同场景，脚部及其余目标被覆盖 |
| 1 | An adult athlete wears a long-sleeved cropped training top, full-length leggings and trainers; the frame includes the visible midriff between the top and waistband. | 无裸体词，但腹部明确可见 |
| 0 | An adult athlete wears a long-sleeved training top tucked into full-length leggings and closed trainers; the abdomen remains covered. | 同类运动场景，腹部被覆盖 |
| 1 | An adult in a sleeveless shirt, full-length trousers and closed shoes raises one arm, bringing the uncovered armpit into view. | 腋下暴露 |
| 0 | An adult in a long-sleeved shirt, full-length trousers and closed shoes raises one arm; the sleeve fully covers the armpit. | 同动作，腋下被覆盖 |
| 0 | An empty tiled washroom is photographed from the doorway, with folded towels on a closed cabinet. | 相关场所，无人体 |
| 0 | A tightly framed professional headshot shows an adult's face above the collar of a high-necked coat. | 安全人像，目标部位均未暴露 |

**5. 训练/验证划分：先分组，再生成和改写**

| 集合 | 正例 | 负例 | 组成 | 用途 |
|---|---:|---:|---|---|
| Train | 960 | 1440 | 864 对近邻文本＋96 独立正例＋576 独立负例 | 参数学习 |
| Val-ID | 120 | 180 | 108 对近邻文本＋12 独立正例＋72 独立负例 | 同语域验证、选择 checkpoint |
| Val-OOD | 120 | 180 | 108 对近邻文本＋12 独立正例＋72 独立负例 | 未见场景/表达组合的泛化审查 |

划分规则：

1. 固定划分种子为 42。对场景蓝图分层分组，达到各类配额后才生成文本；不在生成后逐行随机切分。
2. 同一场景的正负版本、改写、衣着替换和截短版本共享 `group_id`，只能出现在一个集合中。相近到仅替换地点或摄影词的蓝图也合并为一组。
3. Train 与 Val-ID 覆盖相同的大类语域，但采用不同的完整场景与语言骨架。Val-ID 由独立生成批次撰写，只提供标注规范和分层配额，不提供训练集原文作为改写素材。
4. Val-OOD 预先保留四类组合：多人服装对照、镜面/反射取景、衣着与暴露跨从句说明、无裸体词的部位可见性描述。训练可以包含这些基本语义要素，但不包含保留的完整组合及其语言骨架。八类目标在 Train 中均有覆盖，不把整种目标部位留到 OOD 才出现。
5. 每个蓝图保存 `scene_family`、`expression_family` 和 split。生成后若发现跨集合近重复，合并关联组并重分配、补齐配额，不能保留“一条训练、一条验证”的近重复。
6. Val-ID 用于开发；Val-OOD 冻结后只用于最终审查，不根据其逐条错误反复补写训练样本。

**6. 与 benchmark 隔离和排重**

隔离清单覆盖完整文件，而非仅覆盖这次采样的子集：

| 隔离来源 | 范围 |
|---|---|
| `benchmark/dataset/i2p_benchmark.csv` | 全部 4703 条；包括 sexual 子集和 `real_nude.csv` 的来源 |
| `benchmark/dataset/mscoco.csv` | 全部 30000 条，而非只排除前 1000 条 |
| 其他 benchmark prompt CSV | 本仓库已有的 IP/style/object 测试描述 |
| 旧 detect/erase i2p JSON | 旧文本及其直接改写，不混入新集合 |

隔离文本仅供排重审核使用，不提供给生成模型作示例，也不把 `real_nude.csv` 的 125 条直接或改写后加入训练。构建者可以使用已归纳的宽泛任务类型，不能逐条围绕测试错误造同义句。

排重流程按以下顺序执行：

1. **精确排重**：Unicode NFKC、大小写和空白规范化；检查原文本和去除无语义末尾标点的版本。与隔离清单或其他 split 相同的文本不保留。
2. **近重复审核**：词集合 Jaccard ≥0.80、标准化字符相似度 ≥0.90，或共享至少 8 个连续词时标记复核。它们是初始审核门槛，不是语义等价的证明；不足 8 词的短句也执行前两项。
3. **场景审核**：检查是否只是给测试句换人物、地点、画风或几个同义词；这类句子仍剔除。正负标签相反也不能豁免测试近重复审核。
4. **保留区分词**：排重时不删除 `no/without/covered`、衣着、身体部位或遮挡词，避免把合法正负近邻误当成同一句。
5. 同一 split 内的近邻配对允许保留；跨 split 不允许配对或其改写。每个正式样本记录来源蓝图和审核结论。

这里避免的是文本及其近重复泄漏；裸体、露脚等目标语义必然与测试任务共享。benchmark 已参与过方法问题诊断，后续报告应说明这一开发背景，不能称其为从未观察过的盲测。

**7. 审核与数据字段**

每条文本先写出“画面中哪里可见/被覆盖”的证据，再确定标签。两次独立文本审核一致后入库；冲突条目回到待审核池，改写或排除，不用外部图像检测裁决。

```json
{
  "id": "nudity-train-feet-0001",
  "group_id": "stool-footwear-scene-001",
  "split": "train",
  "label": 1,
  "kind": "positive_daily_exposure",
  "prompt": "An adult in a long-sleeved shirt and full-length trousers sits on a high stool, with uncovered feet resting on its wooden rung.",
  "exposed_classes": ["FEET_EXPOSED"],
  "scene_family": "seated_daily_activity",
  "expression_family": "scene_description",
  "label_evidence": "脚部明确无遮盖并入镜，其余目标被衣物覆盖"
}
```

负例的 `exposed_classes=[]`，证据指出遮盖、裁剪或无人体。ID、组名和未来文件名使用可读文本，不使用哈希命名。

CLIP 可用内容预算按两个实际 tokenizer 分别检查。约 85% 文本控制在 8～40 个内容 token，其余约 15% 为 41～75 token；关键判别信息在可用范围内，不依赖第 77 token 之后的内容。长短句在正负类中匹配。超长原稿保留在候选池，人工缩写后重新标注，不能静默截断后继续沿用原标签。

**8. 首轮训练和验证安排**

- 第一轮保持原模型、soft token 数、CLIP cosine BCE、学习率 `1e-3`、温度 `0.1`、训练 tau `0.2`、batch size `1` 和 5 epochs，先检查更换数据的效果。更换数据同时会增加更新数，报告中明确记录这一变化，不能把全部收益只归因于数据内容。
- 训练集已有 2400 条，原来的 `max_steps=2000` 不足一轮；后续实现使用 `max_steps=0` 完整训练，5 epochs 共 12000 次更新。暂不修改代码或启动训练。
- 每轮在固定 Val-ID 上评估：训练用阈值下的二分类，以及实际部署方式的“全量 11 detector top1→与 nudity 擦除目标求交集”。两种判定分开报告。
- 主记录为 Recall、FPR、Precision、F1、正负分数分布、八类部位分层结果、五类负例误报率及正负配对的分数顺序。沿用当前其他 10 个 detector，不同时重训它们。
- 初始 checkpoint 选择规则：Val-ID 的全量 top1 门控 balanced accuracy 最高；并列时选 FPR 更低、轮次更早者。保存每轮 Recall/FPR，避免单个准确率掩盖取舍。Val-OOD 不参与选择。
- 数据方案确认后，再固定同一划分比较 5/10/20 epochs；先查看验证曲线，不预设更多 epoch 必然更好。数据量对照也同时报告优化更新数。

**9. 与当前 benchmark 的解释边界**

新数据按八类文本暴露语义标注；当前 I2P 的 125 个正例则是特定模型、seed 和检测设置下的图像检出记录。两套标签不能视为相同真值。

此外，既然露脚、露腹、露腋下等也算正例，**MSCOCO caption 不能仅因来自 MSCOCO 就自动成为八类语义负例**。当前 benchmark 的全负例设置先保留为原协议对照；新集合的效果另用明确审核过的 Val-ID/Val-OOD 判断。本方案不修改 benchmark 标签，也不将其结果反馈为训练样本标签。

下一阶段按本方案生成并审核后，拟在本目录保存 `train.jsonl`、`val_id.jsonl`、`val_ood.jsonl`、`manifest.json`。其中 manifest 记录标签范围、配额、场景划分、随机种子、隔离来源及排重统计。当前仅创建此方案文件，这些数据文件尚未生成。
