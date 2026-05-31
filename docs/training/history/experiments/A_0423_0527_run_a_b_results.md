# A_0423_0527 双 init 对照 (Run-A pi05_base + Run-B mixed_1, JAX) ⏳ 进行中

> **状态**: ✅ **两 Run 均完成 (50k)**。**Run-A (pi05_base) final=49999 @1=0.0073 @50=0.0402 完胜 Run-B (mixed_1) @1=0.0086 @50=0.0694** (@1 -15%, @50 -42%) — Run-A 49999 = A_0423_0527 全任务 SOTA。详见 §3。
>
> **实验目的**: 在排除 5-16~5-21 校准漂移段后的 13 dates (1,083 ep) 上, 用 `pi05_base` (Run-A) 和 `mixed_1` (Run-B) 双 init 训练 JAX pi05, 验证 (1) 排除校准漂移段后真机回到 smooth 水平; (2) 大数据集上双 init 的 final 差异 (历史经验: 50k step 后趋同)。
>
> **结论 (interim, 2026-05-29 更新)**: 早期 mixed_1 (Run-B) 靠 warmed init 领先, 但 **pi05_base (Run-A) 后期已追平并反超** (Run-A 32k @1=0.0082 ≤ Run-B 40k @1=0.0086, @50 优势更大)。**进度: Run-A 32k/50k (ETA ~14h), Run-B 48k/50k (~1h 完)**。final 对照待两 Run 到 50k 统一 offline-eval。

关联:
- 实验设计: [`../../future_plans/plans/A_0423_0527_excl_calibration_drift.md`](../../future_plans/plans/A_0423_0527_excl_calibration_drift.md)
- 数据集存放规范: [`../../../deployment/training_ops/storage_and_env.md`](../../../deployment/training_ops/storage_and_env.md) §2.3
- v7/v8 校准漂移 audit: `../../analysis/vis_v2_full_data_audit.md` §0.NEXT-v7/v8

---

## 1. 实验配置 (Run-A / Run-B 共享 base config)

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_A_0423_0527` (两 Run 共用, 仅 init override) |
| Model | pi05 (`Pi0Config(pi05=True)`) |
| 训练框架 | JAX/Flax NNX (`scripts/train.py`) |
| **Dataset** | `kai0/data/Task_A/self_built/A_0423_0527` (1,083 ep, 13 dates 排 5-16~5-21 校准漂移 + Class C + End-snap 截尾) |
| Val | `self_built/vis_v2_merged_val` (30 ep cross val, 与 vis_v2_full / 5day_recent 同 val) |
| Prompt | "Flatten and fold the cloth." |
| `use_delta_joint_actions` | False (absolute) |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50k, decay_lr=1.5e-6 |
| EMA decay | 0.9999 |
| Steps / Batch | 50,000 / 128, fsdp_devices=8 |
| Save / keep | every 2,000 step, keep_period=10,000 |
| inline_eval | 每 4k step, 200 frames, val=`vis_v2_merged_val` |
| WandB | offline (`--no-wandb-enabled`) |

### 1.1 两 Run 差异

| | Run-A | Run-B |
|---|---|---|
| Exp name | `A_0423_0527_pi05_JAX` | `A_0423_0527_mixed1_JAX` |
| **Init** | `pi05_base` (原始) | `mixed_1` (Task_A warmed) |
| Cluster | **cnbj Robot-North-H20** (8× H20) | **cnsh robot-task** (8× A100) |
| 数据集 | A_0423_0527 (cnbj, **TOS deref-copy**, bit-identical) | A_0423_0527 (cnsh 原始) |
| 对照锚点 | vs vis_v2_full pi05_base (1406 ep, broken) | vs smooth_800 mixed_1_clean (811 ep, work) |
| Live log | `logs/A_0423_0527_pi05_JAX_cnbj_*.log` (cnbj) | `logs/A_0423_0527_mixed1_JAX_20260529_020158.log` (cnsh) |

> ⚠️ Run-A 数据是 Run-B 数据集的 TOS dereferenced copy (非重新 build), train/val **bit-identical**, 因此双 init 对照直观, 排除数据差异干扰。

---

## 2. MAE@{1,10,25,50} (cross val vis_v2_merged_val 30 ep)

### 2.1 Run-A (pi05_base, cnbj)

| step | MAE@1 | @10 | @25 | @50 | 来源 | 备注 |
|---:|---:|---:|---:|---:|---|---|
| 8000 | 0.0272 | 0.0387 | 0.0584 | 0.0859 | inline (200f) | ⬇ |
| 16000 | 0.0141 | 0.0236 | 0.0370 | 0.0549 | inline (200f) | ⬇ |
| 24000 | 0.0098 | 0.0186 | 0.0303 | 0.0455 | inline (200f) | ⬇ |
| 32000 | 0.0082 | 0.0167 | 0.0277 | 0.0421 | inline (200f) | ⬇ |
| 40000 | 0.0075 | 0.0159 | 0.0267 | 0.0408 | inline (200f) | ⬇ |
| 48000 | 0.0073 | 0.0156 | 0.0263 | 0.0403 | inline (200f) | ⬇ |
| **49999** | **0.0073** | **0.0156** | **0.0263** | **0.0402** | **inline (200f)** ⭐ | **final, BEST** |

**Run-A 已完成 (50k, 2026-05-30 ~22:31 UTC)**。**Best = step 49999, MAE@1 = 0.0073 @50 = 0.0402** ⭐ — **A_0423_0527 全任务最低 MAE** (全 horizon 单调下降到 final, 无 plateau/反弹)。inline-eval 每 8k step 一次, 盘上 kept ckpts: **10k/20k/30k/40k/49999**。

### 2.2 Run-B (mixed_1, cnsh)

| step | MAE@1 | @10 | @25 | @50 | 来源 | 备注 |
|---:|---:|---:|---:|---:|---|---|
| 10000 | 0.0106 | 0.0279 | 0.0506 | 0.0801 | offline backfill | ⬇ |
| 20000 | 0.0094 | 0.0255 | 0.0460 | 0.0720 | offline backfill | ⬇ |
| 30000 | 0.0089 | 0.0247 | 0.0447 | 0.0706 | offline backfill | ⬇ |
| 40000 | 0.0086 | 0.0244 | 0.0445 | 0.0702 | inline (200f) | resumed ✓ (offline 同 step=0.0087) |
| 48000 | 0.0086 | 0.0243 | 0.0443 | 0.0698 | inline (200f) | ✓ (ckpt 已被 keep_period 清) |
| **49999** | **0.0086** | **0.0242** | **0.0439** | **0.0694** | **offline full** ⭐ | **final, BEST, ckpt 已保存** |

**Run-B 已完成 (50k, 2026-05-29 ~10:18 UTC)**。**Best = step 49999** (offline @1=**0.0086** @50=**0.0694**, 全 horizon 最低)。offline 同方法对照: 40k @1=0.0087/@50=0.0699 → 49999 全面微胜。盘上 kept ckpts: **10k/20k/30k/40k/49999** (48k 已被 keep_period 清)。

### 2.3 ⚠️ Run-B inline-eval 失败 + offline 补救

Run-B 原始 run 的 inline-eval 在 step **8000/16000/24000/32000** 全部失败 (无 MAE):
- **根因**: 提交时 config `inline_eval_val_root` 路径 stale — 指向迁移前 `data/Task_A/vis_v2_merged_val`, 数据迁移后实际位于 `self_built/vis_v2_merged_val` → `FileNotFoundError`。
- **现象**: train.py FileNotFoundError 防御 (train.py:62-78) 捕获为 warning, 训练未崩继续跑, 但这些 step 无 MAE (silent eval=0)。
- **补救**: 对保留的 ckpt 10k/20k/30k 做 offline full eval (`eval_val_action_mse.py`, 同 val 同 200 frames) → §2.2 前三行。Backfill 记录: `logs/A_0423_0527_mixed1_JAX_offline_backfill.log`。
- **修复**: kill 原 run + 修正 config val 路径后 resume from 36k, 续跑 inline-eval (40k/48k/49999) 已正常。

> 教训: inline-eval=0 silent failure 需提交前 verify val 路径 (参考 5day_recent inline-eval=0 事故)。Run-A 的 YAML 已加 pre-flight `[ -f "$VAL/meta/episodes.jsonl" ]` 硬检查 (FATAL exit 13)。

---

## 3. Run-A vs Run-B 双 init 对照 ⏳ interim

| step | A @1 | A @10 | A @25 | A @50 | B @1 | B @10 | B @25 | B @50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 0.0272 | 0.0387 | 0.0584 | 0.0859 | — | — | — | — |
| 10k | — | — | — | — | 0.0106 | 0.0279 | 0.0506 | 0.0801 |
| 16k | 0.0141 | 0.0236 | 0.0370 | 0.0549 | — | — | — | — |
| 20k | — | — | — | — | 0.0094 | 0.0255 | 0.0460 | 0.0720 |
| 24k | 0.0098 | 0.0186 | 0.0303 | 0.0455 | — | — | — | — |
| 30k | — | — | — | — | 0.0089 | 0.0247 | 0.0447 | 0.0706 |
| 32k | **0.0082** | 0.0167 | 0.0277 | **0.0421** | — | — | — | — |
| 40k | — | — | — | — | 0.0086 | 0.0244 | 0.0445 | 0.0702 |
| 48k | 0.0073 | 0.0156 | 0.0263 | 0.0403 | 0.0086 | 0.0243 | 0.0443 | 0.0698 |
| **49999** | **0.0073** | **0.0156** | **0.0263** | **0.0402** | **0.0086** | **0.0242** | **0.0439** | **0.0694** |

> A = Run-A pi05_base (全 inline, 200f 子采样); B = Run-B mixed1 (10/20/30k + 49999 offline full, 40/48k inline 200f)。
> **Final (49999): Run-A @1=0.0073 / @50=0.0402 完胜 Run-B @1=0.0086 / @50=0.0694** (@1 -15%, @50 -42%)。

⚠️ **残留方法学差异**: Run-A 全 inline (200f 子采样), Run-B final 49999 是 offline full — 但 Run-B 40k 处 inline(0.0086) ≈ offline(0.0087) 已证两法在本 val 上几乎等价, 故 final 对比可信。**严格化仍建议两 Run 同 offline full 在对齐 step 重测** (见 §5)。

### 观察 (final)
- **早期 mixed1 (Run-B) 领先**: 10k @1=0.0106 优于 Run-A 16k 的 0.0141 — warmed init 起跑优势明显。
- **pi05_base (Run-A) 后期完全反超并持续拉开**: 8k=0.0272 → 49999=**0.0073** (@1 单调降, 无 plateau); Run-B 30k 后即进入 plateau (0.0089→0.0086 几乎不动)。同 step 48k: Run-A @1=0.0073 vs Run-B 0.0086 (**-15%**), @50=0.0402 vs 0.0698 (**-42%**, 长 horizon 差距尤大)。
- **结论**: 大数据集 (1083 ep) + 50k step 下, **pi05_base (冷启但充分收敛) 显著优于 mixed_1 (warmed 但早 plateau)**。与历史"50k 后趋同"经验不同 —— 本数据集 pi05_base 末端仍在降, mixed_1 已饱和, 故 pi05_base final 更优。Run-A 49999 = A_0423_0527 全任务 SOTA。
- **方法一致性已验证**: Run-A 全 inline, Run-B 40k inline=0.0086 ≈ offline=0.0087 → 两法在本 val 上等价, final 跨方法对比 (A inline 49999 vs B offline 49999) 可信。

---

## 4. Ckpt 位置

| Run | step | 路径 |
|---|---|---|
| Run-A (cnbj) | 49999 | `/vePFS-North-E/vis_robot/.../checkpoints/pi05_flatten_fold_A_0423_0527/A_0423_0527_pi05_JAX/49999/` |
| Run-B (cnsh) | 10k/20k/30k(+40k/48k/49999) | `/vePFS/.../checkpoints/pi05_flatten_fold_A_0423_0527/A_0423_0527_mixed1_JAX/{step}/` |

---

## 5. 待办

- [x] **Run-B (cnsh) 已到 50k** (best=49999 @1=0.0086); 最佳 ckpt 已打包: `/vePFS/tim/ckpt_pkg/A_0423_0527_mixed1_JAX_step49999.tar`
- [x] **Run-A (cnbj) 已到 50k** (best=49999 @1=**0.0073** ⭐ 全任务 SOTA)
- [x] 双 init final 对照 — **结论: pi05_base 完胜 mixed_1** (非"趋同", 见 §3 观察)
- [ ] (可选) 统一 offline full eval 对齐 step 严格化 — 当前 inline≈offline 已足够支撑结论
- [ ] **真机测试** (验证排除 5-16~5-21 校准漂移段效果, 见 plan §6 决策树); **Run-A 49999 优先** (MAE 最优); Run-B 49999 备选
- [ ] 打包 Run-A 49999 最佳 ckpt 待真机
- [ ] 更新 master `00_training_history.md`
