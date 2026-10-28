# Early Semantic

测试早期使用 anchor prompt、后期改用 target prompt 时，生成图像的语义是否改变。代码集中在 `run.py`：复用 `benchmark` 的模型加载和采样默认值，实验自身实现 prompt 切换、velocity 查询、Euler 更新和逐步图像保存。

默认读取 `benchmark/configs/dit_target_anchor.json` 的全部 16 组概念，随机生成 **10 条可复现的上下文模板**。同一模板的 target / anchor 两条 prompt 仅替换 `{concept}`；风格使用 `A painting in the style of ...`，nudity 使用明确包含成人的前缀，其余使用 `An image of ...`。原配置里的概念原样填入，完整 prompt 写入结果元数据。

## 运行

在仓库根目录使用现有 GPU 环境运行，依赖与 `benchmark/requirements.txt` 相同。

```bash
# 预览全部 prompt 和采样步分配，不加载模型、不创建输出。
python DiT/early_semantic/run.py --dry-run

# 默认实验：SD3.5 Medium、40 步、512x512、guidance 4.5、p=10%。
python DiT/early_semantic/run.py --offline

# 先生成一条 Snoopy prompt 的完整逐步结果。
python DiT/early_semantic/run.py --targets Snoopy --prompt-count 1 --offline

# p 可自行指定；0 = 全程 target，100 = 全程 anchor。
python DiT/early_semantic/run.py --percent 30 --offline
python DiT/early_semantic/run.py --percent 0 --offline
python DiT/early_semantic/run.py --percent 100 --offline

# FLUX.1-dev：沿用 benchmark 的 28 步和 guidance 3.5，两卡分配。
# 使用独立输出根目录，保留上面的 SD3.5 结果。
python DiT/early_semantic/run.py --model flux1_dev --device-map balanced \
  --output-dir DiT/early_semantic/results/flux1_dev --offline
```

没有本地权重时去掉 `--offline`，由 Hugging Face 下载模型。单卡显存不足可以使用 `--cpu-offload`；它与 `--device-map` 二选一。支持 `--steps`、`--height`、`--width`、`--guidance-scale`、`--dtype`、`--device`、`--seed`、`--template-seed`、`--targets` 和 `--output-dir`。

`--percent` 是**采样步数量的比例**，不是 sigma 区间比例。anchor 步数为 `ceil(T * percent / 100)`：40 步、10% 时，第 1–4 步使用 anchor，第 5–40 步使用 target；FLUX 的 28 步、10% 对应前 3 步使用 anchor。两种模型保留各自原生的 shifted sigma schedule；条件切换不重置 latent 或 scheduler，也不改变 guidance。

相同 `--template-seed` 保持模板一致。第 N 条 prompt 的初始噪声种子为 `seed + N - 1`，跨概念和 p 值一致，便于比较。更改模型、模板种子、图像种子或采样配置时，使用新的 `--output-dir`；同路径重跑会覆盖同名结果。

## 每一步保存什么

以模型使用的时间约定 `sigma: 1 -> 0` 为准：

```python
condition = anchor if step_index < anchor_steps else target
velocity = transformer(z, timestep, condition)  # SD3 包含 CFG；FLUX 使用内嵌 guidance
clean = z - sigma * velocity
z = scheduler.step(velocity, timestep, z).prev_sample
```

每一步保存两种图像：

- `*_clean.png`：`Decode(z_i - sigma_i * v_i)`，是当前条件和 velocity 对干净图像的预测，使用 **更新前** 的 latent 与 sigma。
- `*_latent.png`：`Decode(z_{i+1})`，是完成这一步 Euler 更新后的实际状态；前期直接解码仍会有噪声。
- `final.png`：最终 `sigma=0` 的 latent 解码，与最后一张 `*_latent.png` 相同。

先完成采样，将每步快照暂存在 CPU，再逐张 VAE 解码保存。这样每步图像保存不会在采样过程中触发 transformer / VAE 反复 CPU offload。不会额外调用 transformer 来生成预览。

## 输出

```text
DiT/early_semantic/results/
└── 10/
    ├── experiment.json
    └── Snoopy_Cartoon_Character/
        └── 01/
            ├── metadata.json
            ├── step_001_anchor_clean.png
            ├── step_001_anchor_latent.png
            ├── ...
            ├── step_004_anchor_clean.png
            ├── step_004_anchor_latent.png
            ├── step_005_target_clean.png
            ├── step_005_target_latent.png
            ├── ...
            ├── step_040_target_clean.png
            ├── step_040_target_latent.png
            └── final.png
```

目录中的空格转为下划线，prompt 编号从 `01` 到 `10`。`experiment.json` 保存模型、scheduler、采样配置、随机模板与全部 prompt 对；每条 prompt 的 `metadata.json` 记录种子、实际 anchor 步数、每步条件、timestep、更新前后 sigma 与文件名。

默认完整 SD3.5 实验共 **160 条轨迹、12,960 张 PNG**（每条 40x2+1 张）。初步观察时可以先限定一个 target、一条 prompt，并用 p=0、10、100 对比。逐步 clean 预测反映当前模型的判断；最终语义是否保留需要看 `final.png`。
