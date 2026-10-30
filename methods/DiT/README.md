# 面向 Flow Matching DiT 的概念擦除方法构思

2026-10-29 状态更新：**SCRM 已被用户否决，不再作为候选主线。** 原因是在线计算过重：单概念每个完整触发点在复用下一步输出后仍净增六次样本前向；按逐概念条件分支扩展时，额外前向还会随概念数增长。停止推进该方案的实现与实验，以下保留设计记录。

- [否决决定与历史设计入口](./continuation_response_20261027/README.zh-CN.md)
- [完整方法、公式、伪代码与成本](./continuation_response_20261027/METHOD.zh-CN.md)
- [本地论文依据、表征研究与创新性边界](./continuation_response_20261027/LITERATURE.zh-CN.md)
- [同预算对照、机制实验与停止条件](./continuation_response_20261027/EXPERIMENTS.zh-CN.md)

后续方案优先复用原生生成前向，将概念专属计算放到离线预计算或少量训练中；多概念设计避免逐概念追加完整 DiT 推理分支，并明确报告随概念数变化的在线开销。

以下保留此前两条并列路线：**SIDV：语义索引的局部 velocity 差分擦除**，以及 **FRS：流响应校准的安全语义注入**。这两个历史方案的构造和推理均只使用生成模型自身，不增加在线采样分支。均尚未开展真实 DiT 实验。

- [SIDV：参考 DVE 的新切入点、速度修正公式、创新边界与验证计划](./SIDV_PROPOSAL.zh-CN.md)

SIDV 根据 2026-10-26 的讨论提出：离线记录内部图像表示与局部反事实 velocity 差分的对应关系，在线按当前内容查询，并用替代／保留样本的零响应约束同一次插值。无网络训练，但需要无梯度的字典构建。核心待验证问题是：内部表示相近时，所需的局部速度修正是否也相近。DVE 本身已有平均差分缓存，不能将缓存或不增加 NFE 单独视为新意。

- [FRS：方法、适量数学建模、相关工作边界与实验设计](./FRS_PROPOSAL.zh-CN.md)

FRS 用 DiT 自身的两步采样响应校准候选 steering 方向，并联合分配各层的注入强度。它不训练控制器、不依赖外部判断模型，但包含无梯度的小规模系数拟合，不能称为零校准。部署时只使用缓存向量和内部原型相似度；原生 NFE 不变。尚未进行真实 DiT 实验。

以下是此前的 **CFTR：反事实轨迹回填擦除**，因外部判断器依赖和在线搜索成本，已不作为当前优先方案。

- [CFTR：方法、创新边界、实现流程与实验设计](./CFTR_PROPOSAL.zh-CN.md)

CFTR 先生成不含目标概念的反事实参考，再逐块测试能否从原始中间状态恢复非目标内容。每次候选都完成正常 flow 续程，以目标是否重现、保留是否改善来决定接受或拒绝。无需训练控制器或探针，使用冻结评估器；主要代价是多次候选续程。目前尚未进行真实 DiT 实验。

以下是历史方案 **CFG-PTE**（Preservation-constrained Transport Erasure under CFG）的材料，不代表当前优先方向。

- [完整方法、推导、相关工作与实验设计](./CFG_PTE_PROPOSAL.zh-CN.md)
- [二维解析检验脚本](./toy_transport_check.py)
- [解析检验结果](./toy_results/metrics.json)
- [解析检验图](./toy_results/transport_check.png)

方法直接修正 `v_uncond + gamma * (v_cond - v_uncond)` 的输出。先定义降低目标概念风险、保留非目标语义统计的目标分布，再由连续性方程求最小瞬时能量的 velocity 修正，最后训练外置控制器摊销该计算。

方法属性：**冻结生成器，训练外部探针与控制器，推理时进行 velocity 干预**。不读取 attention、RoPE 或中间层；不要求 single-stream / double-stream。需要可用的真实 CFG 两分支。模型原生的 guidance embedding 与这里的外部 CFG 系数必须区分。

这是研究提案，不是已经得到真实 DiT 实验支持的方法。文档明确区分精确理论、有限样本近似和待验证的创新主张。二维实验仅检查数学构造，不代表图像概念擦除效果。

复现解析检验（CPU；不加载生成模型）：

```bash
python methods/DiT/toy_transport_check.py
```

依赖：Python、NumPy、SciPy、Matplotlib。脚本仅向本目录的 `toy_results/` 写入结果。未运行任何 SD1.4 / SDXL / DiT 图像采样任务。

构思日期：2026-10-24。相关论文范围包含已发表论文与预印本。
