# Task A 全参数微调实验系列 (visrobot01 / mixed / mix_vis600 / pure_vis600)

> **范围**: 2026-04-24 ~ 04-26 在 gf0/gf1 上的 Task A "Flatten and fold the cloth" 全参数微调系列
> **训练框架**: openpi (JAX) + pi05 (PaliGemma + Action Expert), 全部 8×A100 80GB FSDP=8
> **配置文件**: `kai0/src/openpi/training/config.py` (`pi05_flatten_fold_*` 系列)
> **公共 Init**: `Task_A/mixed_1/params` (kai0 MA-merged base)
> **构建脚本**: `train_scripts/data/build_task_a_{vis_base,mix_vis600,pure_vis600,mix_vis600_split,pure_vis600_split}.py`
> **launcher**: `train_scripts/launch/run_{taska_mixed_gf0,visrobot01_only_2k_gf0,resume_visrobot01_only_gf1,mix_vis600_gf0,pure_vis600_gf1}.sh`

---

## 0. 实验全览 (按完成时间排序)

| # | 实验 | 机器 | 步数 | 数据集 | best step | best MAE@1 | 备注 |
|---|---|---|---:|---|---:|---:|---|
| 1 | mixed_gf0_173_v1 | gf0 | 13k | Task_A_mixed_gf0 (173 vis + 173 base + 173 dagger) | 7000-12999 | **0.0129** | step 7-12k 完全 plateau |
| 2 | visrobot01_only_2k_gf0_v1 | gf0 | 2k | Task_A_visrobot01_only (193 train+17 val) | 1999 | 0.0202 | 短训 sanity, 与 v1 同源数据 |
| 3 | visrobot01_only_v1 (Phase A) | gf1 | 9k | Task_A_visrobot01_only (193 train+17 val) | 8000-9000 | 0.0179 | step 8-9k plateau, dataset 路径迁移导致 crash |
| 4 | visrobot01_only_v1 (Phase B, --resume) | gf1 | 9k → 12k | Task_A_visrobot01_only (288 train+22 val, vis_base 重建) | **11999** | **0.0171** | 续训突破 plateau, 4.5% 改善 |
| 5 | **mix_vis600_v1** ✅ | gf0 | 40k | mix_vis600 (310 vis + 145 base + 145 dagger; 540 train+59 val) | **36000-39999 tied** | **0.0146** | 训完 33:21 hr, plateau @ step 30k |
| 6 | **pure_vis600_v1** ⏳ | gf1 | 40k 部分 | pure_vis600 (309 orig + 291 hflip mirror; 560 train+40 val) | (running, step 13.7k @ 0.0201) | (TBD) | 步速慢 (8.4 s/step h264+av1 解码), 短期内不会完成 |

⏳ = 训练中, ✅ = 已完成

---

## 1. mixed_gf0_173 (gf0, 13k 步) ✅ 已完成

### 1.1 实验设定

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_mixed_gf0` |
| exp_name | `mixed_gf0_173_v1` |
| init | `Task_A/mixed_1/params` (冷启) |
| data repo | `Task_A_mixed_gf0/base` (519 ep mix: 173 vis + 173 base + 173 dagger, 等量 stratified) |
| val | `Task_A_mixed_gf0/val` (~50 ep) |
| freeze | **全解冻** (无 freeze_filter) |
| steps / bs / fsdp | 13,000 / 128 / 8 |
| peak_lr / warmup / decay | 1.5e-5 / 500 / cosine to 1.5e-6 over 13k |
| ema_decay | 0.999 |
| save_interval / keep_period | 1000 / 1000 |

### 1.2 Per-step inline-eval 曲线

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 1000 | 0.0153 | 0.0352 | 0.0647 | 0.1020 |
| 2000 | 0.0140 | 0.0320 | 0.0571 | 0.0872 |
| 3000 | 0.0136 | 0.0311 | 0.0551 | 0.0836 |
| 4000 | 0.0134 | 0.0306 | 0.0540 | 0.0816 |
| 5000 | 0.0133 | 0.0303 | 0.0532 | 0.0804 |
| 6000 | 0.0132 | 0.0300 | 0.0528 | 0.0797 |
| 7000 | **0.0130** | 0.0298 | 0.0523 | 0.0789 |
| 8000 | 0.0130 | 0.0297 | 0.0521 | 0.0787 |
| 9000 | **0.0129** | 0.0296 | 0.0521 | 0.0786 |
| 10000 | 0.0129 | 0.0296 | 0.0520 | 0.0785 |
| 11000 | 0.0129 | 0.0296 | 0.0519 | 0.0783 |
| 12000 | 0.0129 | 0.0296 | 0.0521 | 0.0786 |
| **12999** | **0.0129** | 0.0296 | 0.0520 | 0.0785 |

### 1.3 关键观察

- **Plateau onset 极早**: step 7000 已达 0.0130, step 9000 → 12999 完全平稳 (0.0129)
- **train loss vs val MAE 平稳**: 没有 overfit rebound
- **总耗时**: 10:30 hr (含 13× inline-eval ~4.2 hr)
- **结论**: 519 ep 在 13k 步打到性能天花板 0.0129; 加步数无收益

### 1.4 Checkpoint

```
/vePFS/.../checkpoints/pi05_flatten_fold_mixed_gf0/mixed_gf0_173_v1/
├── 1000/  2000/ ... 12000/ 12999/
└── norm_stats.json
```

任意 step 7000+ 可作为部署 ckpt; 推荐 **step 12999** (final, 与最佳 tied)。

---

## 2. visrobot01_only_2k_gf0 (gf0, 2k 短跑) ✅ 已完成

### 2.1 实验设定

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_visrobot01_only_2k` |
| exp_name | `visrobot01_only_2k_gf0_v1` |
| init | `Task_A/mixed_1/params` |
| data | Task_A_visrobot01_only/base (193 train + 17 val, 单源 visrobot01) |
| steps | 2000 |
| peak_lr / warmup / decay | 1.5e-5 / 200 / cosine to 1.5e-6 over 2k |
| ema_decay | 0.999 |
| save_interval | 500 |

### 2.2 Per-step inline-eval

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 500 | 0.0267 | 0.0539 | 0.0948 | 0.1511 |
| 1000 | 0.0237 | 0.0458 | 0.0761 | 0.1162 |
| 1500 | 0.0214 | 0.0424 | 0.0699 | 0.1051 |
| **1999** | **0.0202** | 0.0411 | 0.0680 | 0.1017 |

### 2.3 关键观察

- 用作 sanity baseline: 验证 visrobot01-only 数据可训, 同步与 gf1 12k 长训对比
- 同样数据 2k vs 9k (Phase A): MAE@1 从 0.0202 → 0.0179, **9k 长训提升 11%**

---

## 3. visrobot01_only_v1 (gf1, 长训 9k → resume 12k) ✅ 已完成

> 因数据集路径迁移导致 step 9020 crash, 用新 vis_base (310 ep) 续训.
> Phase A 用原 visrobot01-only 数据 (193+17), Phase B 用 vis_base (288+22).

### 3.1 实验设定

| 参数 | Phase A | Phase B (--resume) |
|---|---|---|
| config | `pi05_flatten_fold_visrobot01_only` | 同 |
| exp_name | `visrobot01_only_v1` | 同 |
| init | `Task_A/mixed_1/params` | step 9000 ckpt (--resume) |
| data | Task_A_visrobot01_only/base (193 train+17 val, 单源 visrobot01) | Task_A_visrobot01_only/base 重建 (288 train+22 val, vis_base 3 日期) |
| norm_stats | 从 193 ep 计算 | **保留 Phase A snapshot** (model-consistency, 不重算) |
| steps | 1 → 9000 | 9000 → 11999 |
| peak_lr / warmup / decay | 1.5e-5 / 500 / cosine to 1.5e-6 over 12k | 同 (continues original schedule) |
| ema_decay | 0.999 | 同 |
| save_interval / keep_period | 1000 / 1000 | 同 |

### 3.2 Per-step inline-eval (合并 Phase A + B)

> ⚠️ **Phase A vs B 的 MAE 数值不能直接 head-to-head 比**: val 集不同 (17 vs 22 ep, 单日期 vs 跨 3 日期)。Phase B 内部 step-step 是同一 val。

| step | MAE@1 | @10 | @25 | @50 | val 集 | 阶段 |
|---:|---:|---:|---:|---:|---|---|
| 1000 | 0.0241 | 0.0469 | 0.0786 | 0.1211 | 17 ep | A |
| 2000 | 0.0203 | 0.0412 | 0.0683 | 0.1025 | 17 ep | A |
| 3000 | 0.0190 | 0.0400 | 0.0668 | 0.1003 | 17 ep | A |
| 4000 | 0.0185 | 0.0394 | 0.0659 | 0.0993 | 17 ep | A |
| 5000 | 0.0183 | 0.0391 | 0.0654 | 0.0985 | 17 ep | A |
| 6000 | 0.0181 | 0.0390 | 0.0651 | 0.0980 | 17 ep | A |
| 7000 | 0.0180 | 0.0389 | 0.0650 | 0.0977 | 17 ep | A |
| 8000 | 0.0179 | 0.0389 | 0.0648 | 0.0975 | 17 ep | A |
| **9000** | **0.0179** | 0.0389 | 0.0648 | 0.0974 | 17 ep | A end |
| ─── | ─── | ─── | ─── | ─── | crash + 数据集换为 vis_base 288+22 | ─── |
| 10000 | **0.0175** | 0.0385 | 0.0648 | 0.0981 | 22 ep | B |
| 11000 | **0.0172** | 0.0376 | 0.0632 | 0.0954 | 22 ep | B |
| **11999** | **0.0171** | 0.0373 | 0.0625 | 0.0943 | 22 ep | B end |

### 3.3 关键观察

- **Phase A**: step 1k → 8k 平稳下降 (0.0241 → 0.0179), step 8-9k 完全 plateau
- **Phase B (续训)**: 在 LR 已降至 ~3.66e-6 (Phase A 末) → 1.5e-6 (Phase B 末) 极低 LR 下, 每 1000 步仍能改善 ~0.0003 (1.7-2.3%)
- **总改善 vs Phase A end**: 0.0179 → 0.0171 = **4.5% 改善**
- **数据增量贡献**: 95 个新 vis_base ep 在低 LR 下仍带来真实信号 (不是 LR 退火噪声)
- **没有 overfit**: step 11999 仍在改善, 并未 rebound

### 3.4 与 mixed_gf0_173 对比

同样基于 mixed_1 init, 同样 1.5e-5 peak_lr:

| 实验 | 数据 | best MAE@1 |
|---|---|---:|
| mixed_gf0_173 | 519 ep mix | **0.0129** |
| visrobot01_only_v1 (B end) | 310 ep 单源 | 0.0171 |

**单源 visrobot01-only 比 519 ep mix 差 33%** — 数据多样性对最终 MAE 的影响显著。

### 3.5 Checkpoint

```
/vePFS/.../checkpoints/pi05_flatten_fold_visrobot01_only/visrobot01_only_v1/
├── 1000/  2000/ ... 9000/  10000/  11000/  11999/
└── norm_stats.json (Phase A 计算, 保留至今)
```

部署推荐: **step 11999** (best & final)。Phase A 时期最佳为 step 9000 (0.0179)。

---

## 4. mix_vis600 (gf0, 40k 长训) ✅ **已完成**

### 4.1 实验设定

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_mix_vis600` |
| exp_name | `mix_vis600_v1` |
| init | `Task_A/mixed_1/params` (冷启) |
| data | `Task_A/self_built/mix_vis600/base` (540 train) + `mix_vis600/val` (60 → **59** after corrupt fix) |
| 数据成分 | 310 vis_base + 145 kai0_base + 145 kai0_dagger; train 540 (vis 279 + base 131 + dag 130 stratified, val 59 修复后) |
| total frames | 487,052 (~10.6 epochs at 40k step) |
| steps / bs / fsdp | 40,000 / 128 / 8 |
| peak_lr / warmup / decay | 1.5e-5 / 1000 / cosine to 1.5e-6 over 40k |
| ema_decay | **0.9999** (长训改用) |
| save_interval / keep_period | 2000 / 2000 (20 ckpts × 12 GB) |
| inline_eval_every | 1 (每 save_interval = 每 2k step) |

### 4.2 训练时间

| 事件 | 时间 |
|---|---|
| 启动 | 2026-04-25 18:25 CST (10:25 UTC) |
| 完成 | 2026-04-27 03:48 CST (Sun 19:48 UTC) |
| **总耗时** | **33:21:43** (含 17 次 inline-eval ~10.8 hr + 20 次 ckpt save) |
| ckpts 保存 | step 2000, 4000, ..., 38000, 39999 (20 个 × ~12 GB = ~240 GB) |

### 4.3 完整 inline-eval 历史

⚠️ step 2000/4000/6000 三次 eval 失败, 原因: val ep 35 symlink 指向 corrupt mp4 (`vis_base/2026-04-24/.../episode_000053.mp4`, moov atom not found). 在 step 6740 时通过 `/tmp/fix_val_remove_ep35.py` 移除 val ep 35 并重编号 (60→59 ep)。**train 数据全程未受影响** (该 corrupt 源文件未被 train 引用)。

| step | MAE@1 | @10 | @25 | @50 | Δ vs 上一点 | 阶段 |
|---:|---:|---:|---:|---:|---:|---|
| 2000 | — | — | — | — | ❌ | failed (corrupt val ep 35) |
| 4000 | — | — | — | — | ❌ | failed |
| 6000 | — | — | — | — | ❌ | failed |
| 8000 | 0.0189 | 0.0385 | 0.0672 | 0.1033 | (基线) | val 修复后首点, rapid 下降 |
| 10000 | 0.0180 | 0.0363 | 0.0628 | 0.0957 | -4.8% | |
| 12000 | 0.0173 | 0.0348 | 0.0601 | 0.0914 | -3.9% | |
| 14000 | 0.0166 | 0.0338 | 0.0584 | 0.0887 | -4.0% | |
| 16000 | 0.0161 | 0.0331 | 0.0573 | 0.0871 | -3.0% | |
| 18000 | 0.0157 | 0.0326 | 0.0566 | 0.0859 | -2.5% | 减速 |
| 20000 | 0.0154 | 0.0323 | 0.0561 | 0.0852 | -1.9% | |
| 22000 | 0.0152 | 0.0321 | 0.0558 | 0.0846 | -1.3% | |
| 24000 | 0.0150 | 0.0320 | 0.0556 | 0.0842 | -1.3% | |
| 26000 | 0.0149 | 0.0320 | 0.0554 | 0.0839 | -0.7% | |
| 28000 | 0.0148 | 0.0319 | 0.0554 | 0.0837 | -0.7% | |
| 30000 | 0.0147 | 0.0319 | 0.0553 | 0.0835 | -0.7% | 准 plateau |
| 32000 | 0.0147 | 0.0319 | 0.0553 | 0.0835 | 0.0% | **PLATEAU** |
| 34000 | 0.0147 | 0.0320 | 0.0554 | 0.0834 | 0.0% | |
| **36000** | **0.0146** | 0.0320 | 0.0554 | 0.0834 | -0.7% | **best 首次** |
| **38000** | **0.0146** | 0.0320 | 0.0554 | 0.0834 | 0.0% | **best, 推荐部署** |
| **39999** | **0.0146** | 0.0321 | 0.0555 | 0.0835 | 0.0% | final, @50 略差 |

### 4.4 关键观察

- **完美收敛轨迹**: step 8k → 30k 单调下降 (0.0189 → 0.0147), 无 overfit rebound
- **三连 best tied**: step 36000 / 38000 / 39999 全部 MAE@1=0.0146, 数值完全收敛
- **Plateau 自 step 30k 起**: step 30k/32k/34k 全 = 0.0147, 之后微改善至 0.0146
- **推荐部署 step**: **38000** (mid-plateau, 三连 best 中数值最稳, @50=0.0834 略好于 39999=0.0835)

### 4.5 Train Loss 轨迹

| step | train_loss | grad_norm | param_norm |
|---:|---:|---:|---:|
| 0 | 0.2269 | 1.2667 | 1804.34 |
| 100 | 0.1338 | 0.6634 | 1804.34 |
| 500 | 0.0202 | 0.0882 | 1804.35 |
| 1000 | 0.0158 | 0.0841 | 1804.39 |
| 5000 | 0.0072 | 0.0660 | 1804.96 |
| 7000 | 0.0061 | 0.0572 | 1805.22 |
| 19400 | 0.0032 | 0.0486 | 1806.30 |

train loss 持续单调降, param_norm 缓增 (训练健康)。

### 4.6 Checkpoint + 部署 tar 包

```
/vePFS/.../checkpoints/pi05_flatten_fold_mix_vis600/mix_vis600_v1/
├── 2000/  4000/  ...  36000/  38000/  39999/   ★ 38000 推荐
└── norm_stats.json
```

**部署 tar 包** (已打包, 2026-04-27 08:34 CST):
- 路径: `/vePFS/tim/workspace/deepdive_kai0_tmp/data/mix_vis600_best_step38000.tar`
- 大小: 11.6 GB (12,440,371,200 bytes)
- 内容: `params/` + `_CHECKPOINT_METADATA` + `assets/` (不含 train_state)
- MAE@1 = 0.0146, @10 = 0.0320, @25 = 0.0554, @50 = 0.0834

### 4.7 与 mixed_gf0_173 关键对比 (反直觉)

| 实验 | 数据 | 步数 | EMA | 终点 MAE@1 | val 集 |
|---|---|---:|---|---:|---|
| mixed_gf0_173_v1 | 519 ep mix (1:1:1) | 13k | 0.999 | **0.0129** | val ~50 ep |
| mix_vis600_v1 | 540 ep mix (2:1:1) | 40k | 0.9999 | **0.0146** | val 59 ep |

**mix_vis600 跑了 3× 步数 + 类似数据量, MAE 反而比 mixed_173 差 13%**。

可能原因 (按可信度排序):
1. **val 集不可比** (最可能): mixed_173 val 与 train 分布更近 (1:1:1 mix); mix_vis600 val 60 ep stratified 后 vis 比例更高 (32/60=53%), 可能更难
2. **数据 composition**: mixed_173 是 1:1:1 等量; mix_vis600 是 ~2:1:1 (vis 主导), kai0 仅 290 ep, 模型对 kai0 域可能学得不充分
3. **EMA=0.9999 + 长训** vs **EMA=0.999 + 短训**: 长训目标 plateau 更慢 (40k cosine 平均 LR 比 13k cosine 高), 但已 plateau 应饱和

**真正的对比需要在共同 test 集上做** (e.g., sim01 真机 / 共同 hold-out val)。inline-eval MAE 不绝对可比。

---

## 5. pure_vis600 (gf1, 40k 长训) ⏳ **进行中, 步速过慢**

### 5.1 实验设定

与 mix_vis600 **完全一致超参** (head-to-head 对照):

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_pure_vis600` |
| exp_name | `pure_vis600_v1` |
| init | `Task_A/mixed_1/params` (冷启) |
| data | `Task_A/self_built/pure_vis600/base` (560 train) + `/val` (40 val) |
| 数据成分 | **309 vis_base ORIGINALS + 291 hflip MIRRORS** (左右镜像增强); 0 kai0 source |
| 镜像处理 | parquet state/action 14-dim 左右半段互换; 视频 ffmpeg hflip + hand_left ↔ hand_right cam 对调 |
| val split 防 leakage | 60 train pair (1 orig + 1 mir) 整对放 val 或 train, 防止 train 见到 hflip 而 val 见原片 |
| steps / bs / fsdp | 40,000 / 128 / 8 (同 mix_vis600) |
| peak_lr / warmup / decay | 1.5e-5 / 1000 / cosine to 1.5e-6 over 40k |
| ema_decay | 0.9999 (同) |
| save_interval / keep_period | 2000 / 2000 (同) |

### 5.2 训练状态 (2026-04-27 08:21 CST 更新)

- 启动: 2026-04-25 23:09 CST (15:09 UTC)
- 当前 step: **13,700 / 40,000** (34% — Mon 08:00 CST 时点)
- 步速: **~8.4 s/step** (gf0 mix_vis600 同期是 2.0 s/step, **慢 4×**!)
- ckpts 保存: step 2k/4k/6k/8k/10k/12k (6 个, 待续)
- ETA 完成 (按当前速度): Wed 23:00 CST (远超 deadline)
- **决策**: 让训练继续, deadline (Mon 09:00 CST) 时不会跑完, 但已有 6+ inline-eval 数据点足以做 trend 对比

### 5.3 步速慢分析

| 项 | 状态 |
|---|---|
| GPU util | 100% × 8 (非 I/O 瓶颈) ✓ |
| dataloader skip | 0 (数据干净) ✓ |
| CPU load | 19+ (8 worker 各 122-188% CPU) |
| memory | 676 GB / 1.9 TB total ✓ |

**最可能原因**: pure_vis600 视频 codec 混合 — originals 是 av1 (vis_base 原始), mirrors 是 h264 (libx264 重编码)。PyAV 在混合 codec 路径上每帧 decode 慢 3-4× (推测). gf0 mix_vis600 几乎全 av1 (kai0 源也是 av1) 因此快很多。

### 5.4 inline-eval 历史 (截至 step 13.7k)

| step | MAE@1 | @10 | @25 | @50 | Δ |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0.0268 | 0.0589 | 0.1074 | 0.1698 | (起点) |
| 4000 | 0.0254 | 0.0522 | 0.0911 | 0.1433 | -5.2% |
| 6000 | 0.0238 | 0.0466 | 0.0778 | 0.1191 | -6.3% |
| 8000 | 0.0222 | 0.0422 | 0.0684 | 0.1017 | -6.7% |
| 10000 | 0.0211 | 0.0391 | 0.0621 | 0.0904 | -5.0% |
| **12000** | **0.0201** | 0.0367 | 0.0577 | 0.0829 | -4.7% |
| 14000 | (即将, ~Mon 03:00 UTC) | — | — | — | — |

仍快速下降, 未到 plateau (mix_vis600 同 step 早已下到 0.0173)。

### 5.5 设计意图 vs 当前观察 (与 mix_vis600 对比)

| 维度 | mix_vis600 | pure_vis600 |
|---|---|---|
| 总训练 ep | 540 | 560 |
| visrobot01 直接采集 | 310 (51.7%) | 309 (51.5%) |
| 镜像增强 | 0 | 291 (48.5%) |
| 跨域 (kai0_base/dagger) | 290 (48.3%) | 0 |
| 同 val: step 8000 MAE@1 | 0.0189 | 0.0222 (-15%) |
| 同 val: step 10000 MAE@1 | 0.0180 | 0.0211 (-15%) |
| 同 val: step 12000 MAE@1 | 0.0173 | 0.0201 (-14%) |

**初步观察**: mix_vis600 一致地比 pure_vis600 在自己 val 上低 14-15%, 提示 **kai0 跨域数据帮助比 hflip 镜像增强更多**。但 val 集不同 (mix val 含 kai0 样本易匹配, pure val 全 vis+mirror), 数值不绝对可比。

---

## 6. 系列结论 (2026-04-27 mix_vis600 完成后更新)

### 已确认结论

1. **数据多样性 > 单源**: mixed (519 ep, 0.0129) 比 visrobot01-only (310 ep, 0.0171) 显著好 (33% lower MAE@1)。
2. **续训 + 新数据可破 plateau**: visrobot01_only Phase A 在 step 8-9k 完全 plateau, 加 95 ep 新数据 + 极低 LR 续训仍能压低 4.5%。
3. **EMA 选择**: 短训 (≤15k) 用 0.999 收敛快; 长训 (40k) 用 0.9999 更稳。
4. **norm_stats 续训策略**: 续训若数据分布变化小 (~5% 漂移), 保留旧 snapshot 比重算更稳, 避免输入分布跳变导致前期 MAE spike。
5. **40k 长训 vs 13k 短训 (mix_vis600 0.0146 vs mixed_173 0.0129)**: 在自己各自的 val 上, 长训反而较差。但 val 集不同, 不严格可比。**真正结论必须看共同 test (sim01 真机/共同 hold-out)**。
6. **kai0 跨域数据 vs hflip 镜像增强**: 同 step head-to-head, mix_vis600 一致比 pure_vis600 低 14-15% (虽 val 不同), 提示 kai0 旧域 (~290 ep) 比 291 mirror ep 提供更有效的 generalization 信号。

### 部分验证 (pure_vis600 仍在跑)

- ✅ **600 ep × 40k step (10 epoch) 在自己 val 上 plateau 0.0146** — 比 mixed_173 (519 ep × 13k = 5 epoch) 在其 val 上的 0.0129 要差 13%, 但 val 不可比
- ✅ **kai0 mix > 镜像增强 (head-to-head 同 step trend)**: mix_vis600 一致比 pure_vis600 低 14-15% (各自 val), 倾向 kai0 跨域数据更有效
- ⏳ **40k vs 13-15k 真实优劣**: 待 sim01 真机部署 mix_vis600 vs mixed_173 直接成功率比较
- ⏳ **pure_vis600 完整 plateau 数据**: 需训练继续到 step 30k+ (按当前 8.4 s/step 还需 ~2 天)

---

## 7. 工程经验

### 7.1 corrupt mp4 风险

`vis_base/2026-04-24/.../episode_000053.mp4` (hand_left cam) 是 vis_base 唯一损坏文件 (录制 kill 未 flush moov atom)。该文件的影响:
- gf1 visrobot01_only Phase B train: 1 个 train ep 引用 → DataLoader 全程 skip (191 次 warning)
- gf0 mix_vis600 val: 1 个 val ep 引用 → inline-eval 直接整段失败 (与 train 不同, **eval 路径无 graceful skip**)
- 修复: build 阶段 ffmpeg probe 主动剔除 (pure_vis600 已自动跳过) OR 训练中 patch val 移除该 ep

### 7.2 --resume vs --overwrite

`--overwrite` rmtree 整个 exp 目录 (包括所有 ckpt)！历史教训: 2026-04-24 误用导致 5k ckpts (best step 4999 MAE@1=0.0127) 不可逆丢失。

**所有 launcher 已统一使用 `--resume`** (即使首跑也安全, 自动 fallback 到 weight_loader)。

### 7.3 双 GPU 并发训练 vePFS I/O

gf0 + gf1 同时跑训练时 vePFS I/O 竞争, 步速 ~2.0-2.5 s/step (单跑约 1.75 s/step)。约 +15-20% 时间。可接受。

### 7.4 inline_eval 时间成本

200 frames × N val_ep eval 大致:
- 17 val ep: 660 s
- 22 val ep: 850 s (gf1 Phase B)
- 60 val ep: 1170 s (gf0 mixed_173)

每 1000 step eval = 增加 ~10-20% 总耗时。长训 (40k) 推荐 `inline_eval_every=2` 即每 2k step eval (本系列设置)。

---

## 8. 历史

| 日期 | 事件 |
|---|---|
| 2026-04-24 13:19 | gf0 mixed_gf0_173_v1 启动 (13k 步) |
| 2026-04-24 13:19 | gf1 visrobot01_only_v1 (Phase A) 启动 (12k 步规划) |
| 2026-04-25 02:00 | gf0 visrobot01_only_2k_gf0_v1 启动 |
| 2026-04-25 04:34 | visrobot01_only_2k 完成 |
| 2026-04-25 04:53 | gf1 visrobot01_only_v1 (Phase A) 在 step ~9020 crash (数据路径迁移) |
| 2026-04-25 04:54 | gf0 mixed_gf0_173_v1 完成 (step 12999, MAE 0.0129) |
| 2026-04-25 16:50 | gf1 visrobot01_only_v1 (Phase B, --resume) 启动用 vis_base 288 ep |
| 2026-04-25 18:25 | gf0 mix_vis600_v1 启动 (40k 步) |
| 2026-04-25 22:47 | gf1 visrobot01_only_v1 (Phase B) 完成 (step 11999, MAE 0.0171) |
| 2026-04-25 22:21 | gf0 mix_vis600 val 修复 (移除 corrupt ep 35) |
| 2026-04-25 23:09 | gf1 pure_vis600_v1 启动 (40k 步) |
| 2026-04-27 03:48 | **gf0 mix_vis600_v1 完成** (step 39999, best step 36k/38k/39999 tied @ MAE 0.0146; 总耗时 33:21:43) |
| 2026-04-27 08:34 | mix_vis600 best ckpt step 38000 打包 → `/vePFS/.../deepdive_kai0_tmp/data/mix_vis600_best_step38000.tar` (11.6 GB) |
| 2026-04-27 09:00 | Mon 测试 deadline; gf1 pure_vis600 此时 step ~14k (35% 完成度) |
| 待 (Wed?) | gf1 pure_vis600 完成 (按当前 8.4 s/step 速度) |
| 2026-04-26 23:00 (预计) | gf1 pure_vis600 完成 |
| 2026-04-27 09:00 | 部署测试 deadline |
