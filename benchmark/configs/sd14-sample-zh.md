# SD1.4 公共采样协议（2026-10-23）

本项目所有 SD1.4 正式采样默认读取
[`benchmark/configs/sd14_sampling.json`](benchmark/configs/sd14_sampling.json)，
公共实现为 [`benchmark/sampling_profile.py`](benchmark/sampling_profile.py)。
对 Codex/其他代理的持续约束保存在 [`AGENTS.md`](AGENTS.md)。

| 项目 | 默认值 |
| --- | --- |
| 模型 | CompVis/stable-diffusion-v1-4 |
| scheduler | DPMSolverMultistepScheduler |
| 算法 | dpmsolver++，二阶，midpoint；完整 scheduler_config 见 JSON |
| steps / CFG | 20 / 7.5 |
| 尺寸 / 模型精度 | 512×512 / CUDA float16 |
| negative prompt | 空字符串 |
| 初始噪声 | 逐 seed 在 CPU 生成 float32，然后转换至模型精度/设备 |
| attention | enable_attention_slicing()，匹配旧 SD1 的 sliced attention |
| safety checker | 关闭 |
| instance/style seed | 每条提示 0–9 |
| COCO | 前 1000 条 caption，seed 为原始行号 |

同一实验的原模型、擦除模型和修复模型必须使用相同采样协议。显式缩小步数、
尺寸、数据集或更改精度属于独立实验，必须在清单记录。旧 DDIM/50步/float32
图片和历史清单保持原样，不迁移文件冒充新协议，也不删除。

## 计算精度例外

矩阵闭式解使用 float64、Jacobian/修复校准使用 float32。这是计算精度，
不决定正式图片采样精度。CFRR 的校准轨迹已改用 DPM 20 步。

Output Response 的可微 DDIM 长/短轨迹暂保留为明确声明的 float32 求导代理，
默认也为 20 步；其方法元数据 `linearization_protocol` 记录这一例外。
该方法的正式 benchmark 采样统一为 DPM 20 步 float16。CPU smoke 使用 float32。
其他架构保持其特别声明的原生采样配置。

## 已有检查点如何重新评测

不用重做擦除/修复即可对已有 CFRR 检查点采用统一协议：

```bash
bash retention_repair/script/cfrr_sd14.sh \
  --phase benchmark --cases instance_snoopy \
  --run-root retention_repair/runs/cfrr-sd14 \
  --device cuda:1 --offline
```

默认新方法名前缀为 `CFRR-dpm20`，保留之前 `CFRR-exp` 的历史结果。
以上命令仅重采样/评测，检查点自身的编辑/校准历史保持原样。
若要重新做整个方法，新默认输出为 `retention_repair/runs/cfrr-sd14-dpm20`，
避免旧配置检查点的续跑校验冲突。

对于任意已有检查点，也可使用 `python -m cfrr benchmark --checkpoint ...
--tasks instance,mscoco --method <新名字> --device cuda:1 --offline`。

## COCO 原图复用与验证

统一默认协议的 COCO 原图缓存仍是：

```text
benchmark/original/mscoco/mscoco-1000-captions-dpm20-512x512-float16-20261022-164247/
```

默认配置生成的缓存标识与旧标识完全一致；1000 张已有原图可直接复用。
2026-10-23 在真实 GPU 上逐 seed 重采样前两张：两张的像素最大误差和平均误差
均为 0。记录见
[`shared-profile-20261023.json`](benchmark/results/smoke/shared-profile-20261023.json)。

新增测试核验缓存标识、float32→float16 噪声、scheduler 设置，以及显式修改
steps/dtype/scheduler/noise/revision 时不错误复用原缓存。

只准备清单、不生成图片，可执行 `--phase prepare`，查看清单各记录的
`reference` 路径是否指向以上目录。采样会校验图片并只补缺失/损坏文件。
