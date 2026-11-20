# 论文检查清单

- [ ] **V01｜重要**：第 4 页图 3，引用位置为 `cvpr/sec/3_method.tex:6`，图片为 `cvpr/figures/method-overview.pdf`。最终 PDF 中图内多数文字仅约 3–6 pt，包含 anchor、图例、分支标签和缓存说明，明显小于正文及模板建议的标注字号。建议减少图内完整公式和次要注释、扩大节点与标签；图片是矢量图，单纯提高导出分辨率不能解决字号问题。

- [ ] **M01｜重要，待说明**：`cvpr/sec/3_method.tex:17、61–72、89–96、125–129、193–207`，第 3–5 页。检测部分将 `p_c^d` 定义为 CLIP 输出空间中的表示，随后直接用于 DiT 条件；抑制部分又说明条件编码管线不限于 CLIP，但各处继续使用同一个文本编码符号。现稿未交代检测特征、生成条件各自的形状、所包含的编码器输出及二者的对应方式。建议补一段表示定义或小型对应表，让读者知道实际学习、送入模型和缓存的分别是什么；这里仅要求讲清接口，不判断机制是否正确。


检查日期：2026-11-17。覆盖规划第 1–11 项，不检查补充材料、方法正确性或原始实验结果。

检查对象：`cvpr/main.tex` 实际引入的正文、摘要、结论、LLM 声明，4 张图、5 张表、36 条参考文献，以及现有 PDF 全部 10 页。源码位置均相对仓库根目录，页码对应当前 PDF。

优先级：**重要**表示影响理解、数值一致性或引用准确性；**一般**表示语言或展示问题；**可选**表示风格优化。“待说明”表示现稿信息不足，不据此判定方法或结果有误。

初次检查未修改论文文件；编译、文本提取和页面预览均在临时副本中完成。后续修复状态以勾选项记录。

## 1. 模板与格式合规

已对照 `cvpr/template_backup`：Letter 纸张、双栏、正文 10 pt、审稿模式、行号、页码及图表标题位置符合模板设置，未发现手动压缩字号或修改版心。正文结束于第 8 页，第 9–10 页仅包含 LLM 声明和参考文献，符合 [CVPR 2027 官方作者指南](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)的页数计算规则；规则核验日期为 2026-11-17。

- [x] **F01｜一般，待说明**：`cvpr/main.tex:77–78`，第 9 页。LLM 声明中的 “GPT models” 未具体指明模型，生成 “portions of the training and evaluation data” 也未说明涉及哪些数据和环节。建议按实际使用情况补充工具身份、所涉数据及使用程度，满足官方指南对披露内容的要求；当前声明标题和位置正确。

## 2. 匿名性

在本次范围内未发现明确身份泄露：PDF 显示匿名作者，Author 元数据为空，正文无机构、邮箱、致谢或基金信息，未出现将既有文献直接指认为作者工作的措辞。PDF 中的外部链接仅指向第三方 FLUX 和 NudeNet 仓库。

## 3. 英语语法与自然表达

- [x] **L01｜一般**：`cvpr/sec/1_intro.tex:54`，第 2 页。贡献列表中的 “our method achieve” 应为 “our method achieves”；“NSFW content, IP character, style” 应采用平行的复数表达并补上连接词。该列表还漏列全文已经报告的 object erasure，建议统一为四类任务。
- [x] **L02｜一般**：`cvpr/sec/3_method.tex:58、62`，第 3 页。“retain null-text negative prompt” 缺少限定词，可写为 “retain the null-text negative prompt”；“detection prompt's initial value” 前缺少 “the”，也可直接写 “We initialize the detection prompt as”。
- [x] **L03｜一般**：`cvpr/sec/4_exp.tex:89`，第 6 页。“2400 generated images for 3 concept per setting” 中 concept 应为复数。建议写成 “2,400 images in total across the three characters per setting”，并与该句的模板数量说明保持一致。
- [x] **L04｜一般**：`cvpr/sec/4_exp.tex:158`，第 7 页。“we generate total 900 images” 应为 “we generate a total of 900 images”。
- [x] **L05｜一般**：`cvpr/sec/4_exp.tex:20`，第 5 页。“AdamW Optimizer” 的 optimizer 无须大写，引用前还缺少空格，PDF 显示为 “Optimizer[17]”。可使用 “AdamW optimizer~\cite{...}”；引用对象另见 R01。
- [x] **L06｜一般**：`cvpr/main.tex:77`，第 9 页。LLM 声明的并列列表末尾缺少 and，建议将结尾改为 “figure design, and literature organization”。
- [x] **L07｜一般**：`cvpr/sec/1_intro.tex:24、42`，第 1–2 页。“An illustration of our Detect and Suppress” 搭配不自然，可改为 “An illustration of Detect and Suppress (DAS)”；图 2 图注第二处 Pikachu 使用直双引号，PDF 显示为两个右引号，应统一 LaTeX 左右引号写法。

## 4. 全文叙事与章节衔接

标题、摘要、引言、方法和结论的核心主题一致，章节顺序连贯。以下属于表达优化。

- [x] **S01｜可选**：`cvpr/main.tex:57`，第 1 页。摘要结果句仅以 “outperforms existing methods” 概括四类任务，缺少直观的结果尺度。可在数值口径统一后，选一组关键对比作为量化锚点，避免继续堆叠泛化评价。
- [x] **S02｜可选**：`cvpr/sec/1_intro.tex:29–36`，第 2 页。引入 DAS 的首句将动机、两种角色解释和方法名称嵌入同一长句，随后又用 “specialize without compromise”“forging a clear link” 重复强调优势。建议拆开动机与方法定义，压缩这些未增加具体信息的短语。
- [x] **S03｜一般**：`cvpr/sec/4_exp.tex:5、10`，第 5 页。实验概述的并列项混合了数据集组合与任务名称；下一段又重复使用 datasets 构成长插入语。建议先平行列举四类任务，再说明数据集与细节位置，使读者更容易把握实验结构。

## 5. 方法讲解清晰度

- [ ] **M01｜重要，待说明**：`cvpr/sec/3_method.tex:17、61–72、89–96、125–129、193–207`，第 3–5 页。检测部分将 `p_c^d` 定义为 CLIP 输出空间中的表示，随后直接用于 DiT 条件；抑制部分又说明条件编码管线不限于 CLIP，但各处继续使用同一个文本编码符号。现稿未交代检测特征、生成条件各自的形状、所包含的编码器输出及二者的对应方式。建议补一段表示定义或小型对应表，让读者知道实际学习、送入模型和缓存的分别是什么；这里仅要求讲清接口，不判断机制是否正确。
- [x] **M02｜可选**：`cvpr/sec/3_method.tex:62、127、134`，第 3–4 页。概念描述 `q_c^d`、通用初始化文本 `q^s` 和安全参考 `q^a` 均只给出抽象定义。建议用一个实际训练样例串联原始输入、概念描述和安全参考，并给出实际使用的初始化文本，帮助读者理解这些对象的区别。

## 6. 术语、缩写与数学记号

- [x] **N01｜一般**：`cvpr/main.tex:57`、`cvpr/sec/1_intro.tex:7`，第 1 页。摘要中的 NSFW、IP 未展开；正文虽解释了 NSFW，但 IP 首次出现时没有正式引入缩写。建议在摘要和正文各自首次出现时解释必要缩写。
- [x] **N02｜一般**：`cvpr/sec/3_method.tex:7、17、87–97`，第 3–4 页。方法图及图注使用 “TI-flow”，正文主要称其为 “flow reconstruction objective”，未明确建立名称对应；VAE 首次出现时也未展开。建议在首次介绍时说明 TI-flow 对应的重建目标，并补充 VAE 全称。
- [x] **N03｜一般**：`cvpr/sec/4_exp.tex:92、159`，第 6–7 页。相同的 `Acc_e`、`Acc_ir` 在 IP 实验中表示目标与非目标识别准确率，在风格实验中却分别称为 “erasure efficacy” 和 “specificity”。建议统一为目标识别准确率、非目标识别准确率，并解释方向；specificity 容易被理解为另一种统计指标。
- [x] **N04｜一般**：`cvpr/main.tex:57` 对照 `cvpr/sec/1_intro.tex:46` 和 `cvpr/main.tex:70`。摘要只写 Stable Diffusion 3.5，正文与结论写 Stable Diffusion 3.5 Medium。建议摘要也明确 Medium，并删去信息量较低的 “modern”。

## 7. 图片与视觉表达

- [ ] **V01｜重要**：第 4 页图 3，引用位置为 `cvpr/sec/3_method.tex:6`，图片为 `cvpr/figures/method-overview.pdf`。最终 PDF 中图内多数文字仅约 3–6 pt，包含 anchor、图例、分支标签和缓存说明，明显小于正文及模板建议的标注字号。建议减少图内完整公式和次要注释、扩大节点与标签；图片是矢量图，单纯提高导出分辨率不能解决字号问题。
- [x] **V02｜一般**：第 6 页图 4，引用位置为 `cvpr/sec/4_exp.tex:14`，图片为 `cvpr/figures/exp_result.pdf`。EraseAnything 的末尾 g 被折到下一行并挤在图像上缘，视觉上近似 “EraseAnythin”；“Ip:” 应为 “IP:”，“Ours(DAS)” 建议补空格并与其他图中的方法名称统一。应在原图中调整标签宽度与排版。

## 8. 表格与结果展示

- [x] **E01｜重要**：`cvpr/sec/4_exp.tex:70、313`，第 6 页表 1 与第 8 页表 5。同一完整 DAS 方法在相同 I2P/MS-COCO 设置下，CS 分别为 **29.76** 和 **30.76**，而 NudeNet count、FID 均相同。需要确认实际数值并统一两表及相关文字；本次不选择其中一个作为正确值。
- [x] **E02｜重要，待说明**：`cvpr/sec/4_exp.tex:89、120、224、254`，第 6–8 页。文中 IP 设置对应每角色 800 张图，单次计数准确率的步长应为 0.125 个百分点，但表 2 的 26.60% 无法由这一分母直接计数并四舍五入得到；物体实验明确每类 50 张，单目标 church 的准确率步长应为 2 个百分点，表 4 却报告 42.84%。若数值是跨轮次、子集或其他单位的平均值，需要写明统计口径；否则需核对样本数与表格。此处标记文字和数字的对应问题，不判定实验结果真伪。
- [x] **E03｜一般，待说明**：`cvpr/sec/4_exp.tex:26、38`，第 5–6 页。CS 被描述为 CLIP 相似度，但未说明产生约 20–32 数值的缩放或具体计算方式。建议给出实际使用的定义、是否缩放及聚合方式，避免读者将其与其他 CLIPScore 实现混淆。
- [x] **E04｜一般**：`cvpr/sec/4_exp.tex:54、100、166、233`，表 1–4。表注仅说 best/second-best，实际标记排除了原始 SD3.5 Medium 行，例如表 1 原模型 CS 为 31.86，高于被加粗的 29.76。建议说明标记范围是“擦除方法之间”；按这一范围核对，现有加粗和下划线排序正确。
- [x] **E05｜一般**：`cvpr/sec/4_exp.tex:83–84、152、322`，第 6、8 页。“does not compromise general generation quality”“without degrading unrelated identities” 容易被理解为相对原始模型完全无损，但表 1 的 CS 从 31.86 降至 29.76，表 2 的非目标准确率也低于原模型；“only…optimal balance” 同样超出了表内有限比较的措辞范围。建议明确相对于所比较擦除方法的优势，直接描述表内结果，避免绝对化表述。
- [x] **E06｜一般，待说明**：`cvpr/sec/4_exp.tex:158`，第 7 页。30 个 painting templates、三种风格与总计 900 张图之间缺少重复采样次数或其他乘数说明。建议补明每种风格、每模板的图像数量或实际计数方式，与准确率分母一起交代。

## 9. 图注、表注与交叉引用

已核对 47 个标签：无重复标签、未定义引用或错失引用的主图/主表，子表均包含在已引用的父表中。

- [x] **C01｜一般**：`cvpr/sec/1_intro.tex:42`，第 2 页图 2。图注解释了三列输出，但未给出正向输入 prompt，也未说明比较时是否固定 seed 等条件。建议补充实际输入和关键比较条件，让读者能理解这个 negative prompting 例子的具体语境；不要求重新生成图片。
- [x] **C02｜一般**：`cvpr/sec/4_exp.tex:100、166`，第 7 页表 2–3。表注解释了准确率，却没有说明 FID 的参考分布，以及为什么目标列越大越好、非目标列越小越好。正文已有相关解释，建议在表注用一句话概括，提高表格独立可读性。

## 10. 引文与参考文献呈现

36 个正文引用键均可解析，未发现重复输出的参考文献条目。以下检查针对引用对象和书目信息呈现，不构成逐条文献内容或真实性审计。

- [x] **R01｜重要**：`cvpr/sec/4_exp.tex:20`、`cvpr/main.bib:67`，第 5 页及参考文献 [17]。AdamW 当前引用的是 Kingma 与 Ba 的 Adam 论文。应为 AdamW 补充或改用 Loshchilov 与 Hutter 的 [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)；这项修正涉及引用归属，不涉及优化器实现检查。
- [x] **R02｜一般**：`cvpr/main.bib:137、237、292、400、412、456`，第 9–10 页参考文献 [1]、[5]、[6]、[15]、[23]、[34]。这些条目虽在 BibTeX 中含 arXiv 编号或 URL，但当前样式没有将其输出，成稿通常只剩作者、标题和年份。建议将可定位的预印本编号、DOI 或正式出处放入该样式能够显示的字段，并复核最终 PDF。
- [x] **R03｜一般**：`cvpr/main.bib:60、184、250、260、389、400、412、456`，第 9–10 页。专有名称大小写没有被保护，PDF 中出现 Speed、Flux、Safecfg、Reneg、Razor、Casteer、Microsoft coco、Qwen3-vl。建议对 SPEED、FLUX、SafeCFG、ReNeg、RAZOR、CASteer、COCO、Qwen3-VL 等名称局部加花括号保护，保留其余标题的正常句式大小写。
- [x] **R04｜一般**：`cvpr/main.bib:185`，第 9 页参考文献 [19]。机构作者 `Black Forest Labs` 被按个人姓名解析，并按 Labs 排在 Kumari 之后。建议以机构作者形式整体保护，避免错误的姓氏解析与排序。
- [x] **R05｜一般**：`cvpr/main.bib:304–308、468–473`，参考文献 [36]、[10]。CoOp 的期刊条目缺少卷、期和页码，ResNet 的会议条目缺少页码。建议根据正式出版记录补齐可获得的书目信息。

## 11. 最终 PDF 与编译质量

已从移除旧编译产物的临时副本执行完整 LaTeX/BibTeX 构建，构建成功。重编译 PDF 与现有 PDF 的逐页提取文本一致；全部字体已嵌入，无 Type 3 字体、PDF 附件或加密。已逐页查看全部 10 页，未发现文字重叠、明显裁切、公式或表格超出版心、缺失引用及可见占位文本。图内字号与标签问题见 V01–V02。

- [x] **B01｜可选**：`cvpr/main.bib:493–497`，第 10 页参考文献 [26]。干净构建产生一条 BibTeX 警告：`Warning--empty year in nudenet`。该网页条目已有访问日期；若能确认发布年份可补充，否则应保留明确的无日期/访问日期表达，不应将访问年份冒充发布日期。这不影响当前编译成功。
