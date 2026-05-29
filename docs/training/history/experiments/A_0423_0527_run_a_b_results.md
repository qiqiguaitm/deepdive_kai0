# A_0423_0527 双 init 对照 (Run-A pi05_base + Run-B mixed_1, JAX) ⏳ 进行中

> **状态**: ⏳ 两 Run 训练进行中。本文档同时记录 Run-A + Run-B, 训练完成后补齐 final + 严格 offline 对比 + 真机结果。
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
| 40k+ | — | — | — | — | (inline, pending) | **当前 32k/50k, rate 2.8s/it, ETA ~14h** |

**Run-A best (截至目前)**: step 32000, MAE@1 = **0.0082** (全 horizon 持续单调下降, 未到 final)。inline-eval 每 8k step 一次。

### 2.2 Run-B (mixed_1, cnsh)

| step | MAE@1 | @10 | @25 | @50 | 来源 | 备注 |
|---:|---:|---:|---:|---:|---|---|
| 10000 | 0.0106 | 0.0279 | 0.0506 | 0.0801 | offline backfill | ⬇ |
| 20000 | 0.0094 | 0.0255 | 0.0460 | 0.0720 | offline backfill | ⬇ |
| 30000 | 0.0089 | 0.0247 | 0.0447 | 0.0706 | offline backfill | ⬇ |
| 40000 | 0.0086 | 0.0244 | 0.0445 | 0.0702 | inline (200f) | resumed run ✓ (修复后正常) |
| 48000 | — | — | — | — | (inline, pending) | **当前 48k/50k, ~1h 到 50k** |
| 49999 | — | — | — | — | (final, pending) | — |

**Run-B best (截至目前)**: step 40000, MAE@1 = **0.0086** (resumed inline; offline 30k=0.0089)。当前 **48k/50k, 即将完成 (~1h)**。

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

> A = Run-A pi05_base (全 inline, 200f 子采样); B = Run-B mixed1 (10/20/30k offline full, 40k inline 200f)。

⚠️ **方法学差异**: Run-A 当前仅 inline-eval (200-frame 子采样), Run-B §2.2 是 offline full eval — step 也未对齐。**严格对比需两 Run 都用 offline full eval 在对齐 step 上重测** (见 §5)。

### 观察 (interim, 待严格对比确认)
- **mixed1 (Run-B) 早期领先**: 10k @1=0.0106, 优于 Run-A 16k 的 0.0141 — warmed init 起跑优势明显。
- **pi05_base (Run-A) 冷启后追平并反超**: 8k=0.0272 → 32k=**0.0082** (@1 持续单调降); Run-A 32k (@1=0.0082) 已 **≤ Run-B 40k (@1=0.0086, 更高 step)**, 长 horizon 更明显 (Run-A 32k @50=0.0421 vs Run-B 40k @50=0.0702)。
- **后期 inline 可比**: Run-A 全程 inline, Run-B 40k 也是 inline (resumed) → 32k(A) vs 40k(B) 同方法 → pi05_base 在更低 step 已不输 mixed1。早期 B 领先靠 warmed init 起跑, 后期 pi05_base 收敛更充分。
- 符合假说 (init 影响早期收敛、后期趋同); 但 step 未对齐 + 早期 B 含 offline, **final 结论待两 Run 到 50k 统一 offline-eval** (§5)。

---

## 4. Ckpt 位置

| Run | step | 路径 |
|---|---|---|
| Run-A (cnbj) | 49999 | `/vePFS-North-E/vis_robot/.../checkpoints/pi05_flatten_fold_A_0423_0527/A_0423_0527_pi05_JAX/49999/` |
| Run-B (cnsh) | 10k/20k/30k(+40k/48k/49999) | `/vePFS/.../checkpoints/pi05_flatten_fold_A_0423_0527/A_0423_0527_mixed1_JAX/{step}/` |

---

## 5. 待办

- [ ] Run-A (cnbj) 跑到 50k, Run-B (cnsh) 续跑到 50k
- [ ] 两 Run 都到 50k 后, 统一 offline full eval 全部 kept ckpt (10k/20k/30k/40k/49999), 做严格 step-aligned 对比 (消除 inline vs offline 方法学差异)
- [ ] 双 init final 对照 — 验证"50k 后趋同"假说
- [ ] 真机测试 (验证排除 5-16~5-21 校准漂移段效果, 见 plan §6 决策树); Run-A 优先, Run-B 视 A 结果
- [ ] 更新 master `00_training_history.md`
