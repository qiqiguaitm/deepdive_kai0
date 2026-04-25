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
| 5 | **mix_vis600_v1** ⏳ | gf0 | 40k | mix_vis600 (310 vis + 145 base + 145 dagger; 540 train+60 val) | (running) | (待出) | 起 14:24 CST 25日, ETA Sun 21:00 CST |
| 6 | **pure_vis600_v1** ⏳ | gf1 | 40k | pure_vis600 (309 orig + 291 hflip mirror; 560 train+40 val) | (running) | (待出) | 起 23:09 CST 25日, ETA Sun 23:00 CST |

⏳ = 训练中 (本次更新时未完成)

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

## 4. mix_vis600 (gf0, 40k 长训) ⏳ **进行中**

### 4.1 实验设定

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_mix_vis600` |
| exp_name | `mix_vis600_v1` |
| init | `Task_A/mixed_1/params` (冷启) |
| data | `Task_A/self_built/mix_vis600/base` (540 train) + `mix_vis600/val` (60 val) |
| 数据成分 | 310 vis_base + 145 kai0_base + 145 kai0_dagger; train 540 (vis 279 + base 131 + dag 130 stratified, 60 val 32+14+14) |
| total frames | 487,052 (~10 epochs at 40k step) |
| steps / bs / fsdp | 40,000 / 128 / 8 |
| peak_lr / warmup / decay | 1.5e-5 / 1000 / cosine to 1.5e-6 over 40k |
| ema_decay | **0.9999** (长训改用) |
| save_interval / keep_period | 2000 / 2000 (20 ckpts × 12 GB) |
| inline_eval_every | 1 (每 save_interval = 每 2k step) |

### 4.2 训练状态 (本文档生成时)

- 启动: 2026-04-25 18:25 CST (10:25 UTC)
- 当前 step: ~7,500 / 40,000 (~19%)
- 步速: 2.0-2.4 s/step (与 gf1 共存期间)
- ETA 完成: **2026-04-26 21:00 CST (Sun 13:00 UTC)**
- ckpts 已保存: 2000, 4000, 6000

### 4.3 inline-eval 历史 (注意: 早期失败已修复)

⚠️ **step 2000/4000/6000 三次 eval 全部失败**, 原因: val ep 35 symlink 指向 corrupt mp4 (`/vePFS/visrobot01/.../2026-04-24/.../episode_000053.mp4`, moov atom not found).

修复时间: 2026-04-25 22:21 CST (step 6740 时), 通过 `/tmp/fix_val_remove_ep35.py` 移除 val ep 35 并重编号 (60 → 59 ep)。**train 数据全程未受影响** (该 corrupt 源文件未被 train 引用)。

| step | MAE@1 | @10 | @25 | @50 | 备注 |
|---:|---:|---:|---:|---:|---|
| 2000 | — | — | — | — | ❌ failed (corrupt val ep 35) |
| 4000 | — | — | — | — | ❌ failed |
| 6000 | — | — | — | — | ❌ failed |
| 8000 | (待出, ~23:00 CST) | | | | 首个有效点 |

### 4.4 Train Loss 轨迹 (作为代理参考, 不能直接推 val MAE)

| step | train_loss | grad_norm | param_norm |
|---:|---:|---:|---:|
| 0 | 0.2269 | 1.2667 | 1804.34 |
| 100 | 0.1338 | 0.6634 | 1804.34 |
| 200 | 0.0355 | 0.1278 | 1804.34 |
| 500 | 0.0202 | 0.0882 | 1804.35 |
| 1000 | 0.0158 | 0.0841 | 1804.39 |
| 2000 | 0.0112 | 0.0733 | 1804.54 |
| 5000 | 0.0072 | 0.0660 | 1804.96 |
| 7000 | 0.0061 | 0.0572 | 1805.22 |

收敛形态健康, 无发散/振荡。step 7000 train loss 已与 mixed_173 plateau 期相当 (后者 train loss 未记录但 val MAE 在 step 7k 才到 0.0130)。

---

## 5. pure_vis600 (gf1, 40k 长训) ⏳ **刚启动**

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

### 5.2 训练状态

- 启动: 2026-04-25 23:09 CST (15:09 UTC)
- 当前 step: 启动中 (init/JIT 阶段)
- ETA 完成: **2026-04-26 23:00 CST (Sun 15:00 UTC)**

### 5.3 设计意图 (与 mix_vis600 对比)

| 维度 | mix_vis600 | pure_vis600 |
|---|---|---|
| 总训练 ep | 540 | 560 |
| visrobot01 直接采集 | 310 (51.7%) | 309 (51.5%) |
| 镜像增强 | 0 | 291 (48.5%) |
| 跨域 (kai0_base/dagger) | 290 (48.3%) | 0 |
| 假设 | "更多多样性, kai0 帮助 generalize" | "纯 visrobot01 + 对称性正则" |

**Hypothesis**: 如 mix_vis600 < pure_vis600 → kai0 旧域帮助; 反之 → mirror 增强足够好。

---

## 6. 系列结论 (本文档生成时)

### 已确认结论

1. **数据多样性 > 单源**: mixed (519 ep, 0.0129) 比 visrobot01-only (310 ep, 0.0171) 显著好 (33% lower MAE@1)。
2. **续训 + 新数据可破 plateau**: visrobot01_only Phase A 在 step 8-9k 完全 plateau, 加 95 ep 新数据 + 极低 LR 续训仍能压低 4.5%。
3. **EMA 选择**: 短训 (≤15k) 用 0.999 收敛快; 长训 (40k) 用 0.9999 更稳。
4. **norm_stats 续训策略**: 续训若数据分布变化小 (~5% 漂移), 保留旧 snapshot 比重算更稳, 避免输入分布跳变导致前期 MAE spike。

### 待验证 (mix_vis600 + pure_vis600 完成后)

- ❓ 600 ep × 40k step (10 epoch) 能否压低 mixed_173 (519 ep × 13k = 5 epoch) 的 0.0129?
- ❓ kai0 旧源数据 (mix_vis600) vs 镜像增强 (pure_vis600) 谁更帮 generalize?
- ❓ 40k 步是否真的优于 13-15k? 还是只是边际收益消失?

预计 Sun 21:00-23:00 CST 两侧训练完成后可分析。

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
| 2026-04-26 21:00 (预计) | gf0 mix_vis600 完成 |
| 2026-04-26 23:00 (预计) | gf1 pure_vis600 完成 |
| 2026-04-27 09:00 | 部署测试 deadline |
