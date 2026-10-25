# 面向 Flow Matching DiT 的概念擦除方法构思

当前只展开一种方法：**保留约束的 CFG 概率流擦除**，工作名 **CFG-PTE**（Preservation-constrained Transport Erasure under CFG）。

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
