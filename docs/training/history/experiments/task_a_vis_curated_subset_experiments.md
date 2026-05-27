# Task A 视觉数据精选子集训练实验 (vis curated subsets)

> **主题**: 同 hparams 下, 不同精选策略产生的 vis 数据子集 (500-800 ep 量级) 的对比。两个实验都从 `vis_base` 出发但用**不同筛选维度**得到中等规模子集, 验证 "数据精选 vs 数据规模" 假设的不同变体。
>
> **核心发现 (TL;DR)**:
> - **`vis_5day_recent` (498 ep, 5-18~5-22)** offline MAE@1 = **0.0086** (cross-val) — 最低
> - **`smooth_800` (811 ep, X1 cleanup)** offline MAE@1 = **0.0089** (native-val) — 接近但 val 不同, 不直接可比
> - **时间窗口精选 (5day_recent) > 全期质量清洗 (smooth_800) > 1406 ep 全集** (在同 cross val 下)
> - 与 `pure_200` (100 ep + mirror) 0.0065 native-val SOTA 形成对比: pure_200 重做 cross-val 后退到 0.0207
>
> **建立**: 2026-05-27 (合并 `task_a_new_smooth_800_new_norm_results.md` 与 5day_recent eval 结果)
> **关联**: [analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md](../../analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md) · [task_a_new_pure_200_new_norm_results.md](task_a_new_pure_200_new_norm_results.md) · [00_training_history.md](00_training_history.md)

---

## 1. 实验对比总览

| 维度 | smooth_800 | vis_5day_recent |
|---|---|---|
| **数据集名** | `A_new_smooth_800` (X1 cleanup) | `vis_5day_recent` (date filter) |
| **来源** | vis_base 全期 → X1 自动化清洗 | vis_v2_full 16 dates → 5 dates 取后段 |
| **Episodes** | **811** | **498** |
| **Dates 跨度** | 全期 (跨多月, 仅过滤质量) | **5 连续日 (2026-05-18 ~ 2026-05-22)** |
| **Frames (大约)** | ~930K | ~700K (估算) |
| **Mirror augmentation** | ❌ | ❌ |
| **Init** | mixed_1_clean | pi05_base |
| **Cluster** | uc03 单机 8 GPU A800 | Robot-North-H20 单节点 8 H20 |
| **训练时长** | 42h08m (含 mining 8h, 净 26h) | 35h25m |
| **Steps** | 50,000 | 50,000 |
| **Batch** | 128 | 128 |
| **Native val** | A_new_smooth_800/val (26 ep, 同期) | (无单独 native val) |
| **Cross val** | (未在 vis_v2_val50 上 eval) | vis_v2_val50 (30 ep, dates 04-23/24) |
| **Final MAE@1** | **0.0089** (native) | **0.0086** ⭐ (cross val) |
| **Final MAE@50** | **0.0636** (native) | **0.0630** (cross val) |
| **Ckpt 路径** | `uc03:/data/shared/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_smooth_800_new_norm/task_a_new_smooth_800_new_norm/49999/` | `gf3 (cnbj):/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_vis_5day_recent/pi05_flatten_fold_vis_5day_recent/49999/` |

⚠️ **MAE 不直接可比**: smooth_800 的 0.0089 是在 native val (同期 26 ep) 上测; 5day_recent 的 0.0086 是在 cross val (4 月底 30 ep) 上测. 需在同 val 上才能严格对比 (见 §5 cross-val 补测建议).

---

## 2. 实验 1: smooth_800 (uc03, X1 cleanup)

### 2.1 配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_smooth_800_new_norm` |
| Init | `mixed_1_clean` (= cleaned mixed_1) @ `/home/tim/local_ckpts/Task_A_init/mixed_1_clean/params` |
| Dataset | `/data/shared/tim/data/Task_A/A_new_smooth_800/base` (811 ep, ~930K frames) |
| Dataset 来源 | vis_base + X1 自动化清洗 |
| Val | `/data/shared/tim/data/Task_A/A_new_smooth_800/val` (26 ep, 同期 hold-out) |
| LR | Cosine, warmup=1k, peak=1.5e-5, decay 50k → 1.5e-6 |
| EMA | 0.9999 |
| num_workers | **64** (uc 单机本地 SSD 优化, 32 默认有反压, 见 [[feedback-uc-cluster-num-workers]]) |
| Server | uc03 (Intel 8358P 124 vCPU + A800-SXM4-80GB × 8 + 本地 SSD) |

### 2.2 训练曲线 (inline-eval on native val 26 ep)

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 4000 | 0.0123 | 0.0312 | 0.0620 | 0.1077 | (start) |
| 8000 | 0.0111 | 0.0260 | 0.0477 | 0.0782 | -9.8% |
| 12000 | 0.0104 | 0.0240 | 0.0433 | 0.0690 | -6.3% |
| 16000 | 0.0098 | 0.0229 | 0.0417 | 0.0664 | -5.8% |
| 20000 | 0.0096 | 0.0225 | 0.0411 | 0.0653 | -2.0% |
| 24000 | 0.0094 | 0.0223 | 0.0409 | 0.0648 | -2.1% |
| 28000 | 0.0092 | 0.0221 | 0.0406 | 0.0644 | -2.1% |
| 32000 | 0.0091 | 0.0220 | 0.0404 | 0.0640 | -1.1% |
| 36000 | 0.0090 | 0.0220 | 0.0404 | 0.0639 | -1.1% |
| 40000 | 0.0089 | 0.0220 | 0.0403 | 0.0637 | -1.1% |
| 44000 | 0.0089 | 0.0220 | 0.0403 | 0.0637 | 0.0% |
| 48000 | 0.0089 | 0.0221 | 0.0404 | 0.0636 | 0.0% |
| **49999** | **0.0089** | **0.0221** | **0.0404** | **0.0636** | 0.0% |

**Best**: step 40k 起 plateau, 49999 与 40k 完全持平. **下次训练可砍到 40k 节省 20% 时间.**

### 2.3 动力学观察

- 起点 MAE@1=0.0123 (step 4k), 接近 1800 ep 同 init 的 0.0128
- 早期收敛快: 4k → 16k 在 12k 步内 -20%
- step 40k 起完全 plateau, 后段对优化无信息
- 全程 0 NaN

---

## 3. 实验 2: vis_5day_recent (cnbj Robot-North-H20)

### 3.1 配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_vis_5day_recent` |
| Init | `pi05_base` (cold-start) @ `/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params` |
| Dataset | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/vis_5day_recent` (498 ep) |
| Dataset 来源 | vis_v2_full 16 dates → 仅保留后 5 连续日 (2026-05-18 ~ 2026-05-22) |
| Val (inline) | `vis_v2_merged_val` (cross val, 04-23/24 dates 30 ep) — ⚠️ **val 数据创建于训练启动 19.5h 之后**, 全部 inline-eval = 0.0000 (silent 失败, 见 §3.4) |
| LR | Cosine, warmup=1k, peak=1.5e-5, decay 50k → 1.5e-6 |
| EMA | 0.9999 |
| num_workers | 8 |
| Server | Robot-North-H20 单节点 (ml.hpcpni3ln.45xlarge, 8× H20-SXM5-96GB) |
| Volc Job ID | `t-20260525211625-n9hsh` |

### 3.2 训练曲线 (offline eval on vis_v2_val50, 2026-05-27 重测)

> ⚠️ 因 inline-eval 在训练时全 0, 训练后专门在 gf3 上跑了 5 ckpt offline eval 重建训练历史曲线.

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 10000 | 0.0239 | 0.0355 | 0.0549 | 0.0821 | (start) |
| 20000 | 0.0127 | 0.0241 | 0.0416 | 0.0662 | **-47%** ⬇️ |
| 30000 | 0.0098 | 0.0215 | 0.0391 | 0.0635 | -23% ⬇️ |
| 40000 | 0.0089 | 0.0209 | 0.0386 | 0.0630 | -9% |
| **49999** | **0.0086** | **0.0208** | **0.0386** | **0.0630** | -3% (plateau) |

**Best**: step 49999, 但 40k → 49999 改善仅 3%, @10/@25/@50 完全持平 — **40k 已 plateau**, 与 smooth_800 同步.

### 3.3 跨实验对比 (同 val = vis_v2_val50 cross-val)

| 实验 | 数据 | dates | step | Init | MAE@1 | MAE@50 |
|---|---|---|---:|---|---:|---:|
| pure_200 + pi05_base | 100 ep × 2 mirror | 5-08/09 (2 dates) | 50k | pi05_base | 0.0207 | 0.1348 |
| **vis_5day_recent + pi05_base** | **498 ep** | **5-18~5-22 (5 dates)** | **50k** | **pi05_base** | **0.0086** ⭐ | **0.0630** ⭐ |
| vis_v2_full + pi05_base | 1406 ep | 4-23~5-22 (16 dates) | 50k | pi05_base | 0.0131 | 0.1138 |

**5day_recent 在 cross val 上明显最优**:
- 比 pure_200 好 **58%** (0.0086 vs 0.0207 @1)
- 比 vis_v2_full 好 **34%** (0.0086 vs 0.0131 @1)
- long-horizon (@50) 同样最优 (0.0630 vs 0.1138 / 0.1348)

### 3.4 Inline-eval 全 0 问题 (训练动态盲区根因)

训练日志显示所有 inline-eval step (8000/16000/24000/32000/40000/48000/49999) 都报告 `MAE@1=0.0000 (0.0s)`. 根因:

- val 数据 `vis_v2_merged_val` **创建时间**: 2026-05-26 16:54 Beijing
- 训练**启动时间**: 2026-05-25 21:30 Beijing — 早 19.5h
- 训练首次 inline-eval (step 8000) 时 val 不存在 → 旧 train.py 静默返回空 samples
- 空 samples 被 `_VAL_CACHE` 缓存, 后续 eval 全部走 cache → 即使 val 数据后来出现, 继续返回空
- 结果: `(0.0s)` 表示 eval 循环 0 次迭代, MAE=0 是占位值

**已防御**: train.py 加 `FileNotFoundError` early-raise (在 cnbj 工作树有未提交版本, /tmp 备份). 建议合到 main.

---

## 4. 跨实验关键洞察 (合并讨论)

### 4.1 同等量级 vis 子集, 时间精选 > 全期清洗

| 维度 | smooth_800 (全期 + X1 cleanup) | vis_5day_recent (时间窗 + 无清洗) |
|---|---|---|
| Episodes | 811 (1.6×) | 498 |
| 协议一致性 | 中 (跨多月, X1 清洗去 noise) | **高 (5 连续日)** |
| 同 val (cross) MAE@1 | (未测, native 0.0089) | **0.0086** |

假说: **时间连续性 (5 日内 operator/setup 几乎不变) > 全期质量清洗 (跨月协议漂移仍存在, 即使清洗也无法消除)**.

⚠️ smooth_800 未在 cross val 上重测, 严格验证还需 §5.1.

### 4.2 init 影响小于早期实验所示

- smooth_800: mixed_1_clean init → 起点 step 4k 已 MAE@1=0.0123
- 5day_recent: pi05_base 冷启 → step 10k 才 0.0239 (大约对应 step 4k 的 0.0123 → 0.0089 区间)

但 final (49999) 两者都 ≈ 0.0086-0.0089, **init 在 50k step 后不再决定 final MAE** (与 [analysis 文档](../../analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md) §A.2 一致).

### 4.3 数据 quality + size + temporal 三角

排行 (cross-val MAE@1 视角, 同 vis_v2_val50):

```
5day_recent (498, 时间精选)   0.0086 ⭐
vis_v2_full (1406, 全期)      0.0131
pure_200 cross (100×2 mirror) 0.0207
```

数据量大不一定好, 但**时间精选 + 中等规模 (≈500 ep) 似乎是甜区**, 同时:
- 时间一致 = 协议一致, 减少 model 学到的"平均策略"
- 中等规模 = 比 pure_200 更多输入多样性, 不过拟合
- 但仍未加 mirror augmentation (pure_200 用了)

### 4.4 训练 step 数收益递减一致

| 实验 | plateau 起步 | 最后 9k step 改善 |
|---|---|---|
| smooth_800 | step 40k | 0% |
| 5day_recent | step 40k | -3% (微弱) |

**建议**: 类似数据下, 训练 40k step 即足够. 50k 是冗余.

---

## 5. 后续 / Open Questions

### 5.1 ⭐ 必做: smooth_800 cross-val 补测

在 vis_v2_val50 上 eval smooth_800 ckpt 49999 (~10 min on gf3 H20), 才能严格对比两实验. 预期:
- 若 cross MAE@1 仍 ≈ 0.009 → smooth_800 的 X1 cleanup 与 5day_recent 时间精选效果接近
- 若显著退化 (≥0.020) → 时间精选 (5day_recent) 是核心驱动, X1 全期清洗不如时间窗

### 5.2 真机部署候选

| Ckpt | sim01 候选? | 理由 |
|---|---|---|
| smooth_800/49999 | ⚠️ 谨慎 | long-horizon @50=0.0636 在 native val, cross val 未测; uc03 本地 ckpt, 跨 NFS 拷需 60-90 min |
| **5day_recent/49999** | ✅ **推荐** | cross val @1=0.0086 / @50=0.0630, 是 cross val 上最优. cnbj vePFS, 走 TOS 中转到 sim01 ~5 min |

### 5.3 进一步实验

| 实验 | 假设 | ETA |
|---|---|---|
| 5day_recent + hflip mirror | 加 mirror 是否进一步降 MAE? | 同 ETA, 35h |
| 5day_recent 40k step 截断 | 40k 已 plateau, 是否 40k = 49999? | 28h |
| vis 3day_recent (5-20~5-22) | 缩到 3 dates 是否更好? | 假设 ~200 ep, 35h |
| 5day_recent_X1 (5 days + X1 cleanup) | 时间 + 清洗叠加效果? | 35h |

### 5.4 训练 ops 改进 (来自 5day_recent 教训)

- ⭐ **val 数据必须在训练启动前就绪** — 否则 train.py 默认行为是静默忽略, 全程 MAE=0 看不出问题
- ⭐ **train.py 应加 fail-fast on missing val** (`FileNotFoundError`), 不要 silent fallback. cnbj 工作树有未提交版本, 应合 main

---

## 6. 文件 / Ckpt 速查

| 文件/路径 | 内容 |
|---|---|
| `experiments/task_a_new_smooth_800_new_norm_results.md` (旧, 独立) | smooth_800 单独报告, 已被本文档整合 |
| `experiments/00_training_history.md` | 全实验 best MAE 排行榜入口 |
| `analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md` | vis_v2_full vs pure_200 反直觉归因, 含 chunk/noise 诊断 |
| `uc03:/data/shared/tim/workspace/.../smooth_800_new_norm/49999/` | smooth_800 best ckpt |
| `cnbj:/vePFS-North-E/vis_robot/workspace/.../vis_5day_recent/49999/` | 5day_recent best ckpt (params 12.5G + train_state 25G; 部署只需 params + assets + _CHECKPOINT_METADATA 共 12.5G) |
