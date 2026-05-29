# A_0423_0527 (除 5-16~5-21) — 双 init JAX 训练 (验证校准漂移段排除效果)

> **实验目的**: 在**排除 5-16~5-21 校准漂移段**后的全部 dates (4-23~5-27 共 13 dates) 上, 分别用 `pi05_base` 和 `mixed_1` init 训练 JAX 模型, 验证:
> 1. 排除 5-16 (stay-still D1) + 5-18/19/20/21 (gripper firmware 校准漂移期 v7) 后, 真机表现是否回到 smooth_800 水平 (闭合稳定 / 不 oscillate / 不松开)
> 2. 同时双 init 对照, 量化在大数据集 (~1000+ ep) 上 pi05_base vs mixed_1 init 的 final 差异 (历史经验: 50k step 后近似消除)
>
> **状态**: ⏳ pending
> **日期**: 2026-05-27
> **关联**:
> - v7 决定性发现: [`../../analysis/vis_v2_full_data_audit.md`](../../analysis/vis_v2_full_data_audit.md) §0.NEXT-v7 (Gripper 校准漂移)
> - v8 校准后扫描: §0.NEXT-v8 (R_grip 过度校准, 不应直接用校准后数据)
> - smooth_800 anchor (4-23~5-09 work): [`../../history/experiments/task_a_new_smooth_800_new_norm_results.md`](../../history/experiments/task_a_new_smooth_800_new_norm_results.md)
> - 5day_recent anchor (5-18~22 真机 fail): [`../../history/experiments/task_a_vis_curated_subset_experiments.md`](../../history/experiments/task_a_vis_curated_subset_experiments.md)
> - 数据集 README: [`/transfer-shanghai/KAI0/Task_A/base/README.md`](/transfer-shanghai/KAI0/Task_A/base/README.md)

---

## 1. 实验设计 — 两个 Run

| Run | Exp name | Init | 数据 | 目的 |
|---|---|---|---|---|
| **Run-A** | `A_0423_0527_pi05_JAX` | `pi05_base` | 13 dates (排 5-16~5-21) | 与 vis_v2_full pi05_base init (1406 ep, broken) 对照, 验证排除 5-16~5-21 后能 work |
| **Run-B** | `A_0423_0527_mixed1_JAX` | `mixed_1` | 同 Run-A | 与 smooth_800 mixed_1_clean init (811 ep, work) 对照, 双 init 配比 |

> ⚠️ **命名说明**: 用户在原始 message 两次写了 `A_0423_0527_pi05_JAX`, 推测一个应是 `A_0423_0527_mixed1_JAX` (mixed_1 init 版本). 如需要不同命名, 在跑训前 update.

---

## 2. 数据集 `A_0423_0527`

### 2.1 Date 组成 (排除 5-16~5-21)

| date | 场景 (README §1) | ep (raw) | Class C ep | 净 ep |
|---|---|---:|---:|---:|
| 2026-04-23-v2 | 早期未分类 | 21 | 1 | 20 |
| 2026-04-24-v2 | 早期未分类 | 187 | 22 | 165 |
| 2026-04-25-v2 | 简单叠衣 ⭐ | 96 | 3 | 93 |
| 2026-04-28-v2 | 简单叠衣 ⭐ | 152 | 8 | 144 |
| 2026-04-29-v2 | 简单叠衣 ⭐ | 100 | 3 | 97 |
| 2026-04-30-v2 | 杂乱→整齐 ⭐⭐ | 83 | 20 | 63 |
| 2026-05-06-v2 | 杂乱→整齐 ⭐⭐ | 100 | 10 | 90 |
| 2026-05-07-v2 | 杂乱→整齐 ⭐⭐ | 20 | 2 | 18 |
| 2026-05-08-v2 | 杂乱→整齐 ⭐⭐ | 101 | 10 | 91 |
| 2026-05-09-v2 | 杂乱→整齐 ⭐⭐ | 30 | 3 | 27 |
| ~~2026-05-16-v2~~ | ❌ 排除 (stay-still ideal, D1) | (16) | (0) | — |
| ~~2026-05-18-v2~~ | ❌ 排除 (gripper 校准漂移期 v7) | (100) | (3) | — |
| ~~2026-05-19-v2~~ | ❌ 排除 (同) | (100) | (5) | — |
| ~~2026-05-20-v2~~ | ❌ 排除 (同) | (100) | (8) | — |
| ~~2026-05-21-v2~~ | ❌ 排除 (同) | (100) | (6) | — |
| 2026-05-22-v2 | 杂乱投放 ⭐⭐⭐ | 100 | 10 | 90 |
| 2026-05-26-v2 | 杂乱投放 ⭐⭐⭐ | 100 | 6 | 94 |
| 2026-05-27-v2 | 杂乱投放 ⭐⭐⭐ | 100 | 9 | 91 |
| **TOTAL (含 Class C)** | — | **1,190** | **107** | — |
| **TOTAL 排 Class C (实际 build 结果, 2026-05-27 verify)** | — | — | — | **1,083 ep** |
| End-snap 截尾 (含在 1083 中) | — | — | — | 5 ep 末段 1-58 帧裁掉 |

> ⚠️ 实测 2026-05-27: 5-27 数据有 100 ep (不是 README §2 标的 76, README 可能未及时更新), 故总 raw=1190 而不是 1166. 排 Class C 后 1083 ep, End-snap 5 ep 在 4-24/28/29 早期已含截尾.

**对比** (从 README §2):
- smooth_800 范围: 4-23~5-09 共 10 dates / 811 ep (X1 cleaned)
- vis_v2_full 范围: 4-23~5-22 共 16 dates / 1406 ep (未排 Class C / 含 5-16)
- **本 A_0423_0527: 4-23~5-27 排 5-16~5-21, 共 13 dates / 1,083 ep (排 Class C + 5 End-snap 截尾)** ⭐

### 2.2 数据集 build 脚本要点 (新建)

```python
# 类似 build_vis_v2_full.py, 但 dates 不同 + 含 Class C filter
SRC_ROOT = Path("/transfer-shanghai/KAI0/Task_A/base")
DST = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/A_0423_0527")

# 排除的 dates (校准漂移段 + ideal)
EXCLUDE_DATES = {"2026-05-16-v2", "2026-05-18-v2", "2026-05-19-v2",
                 "2026-05-20-v2", "2026-05-21-v2"}

# 排除 Class C (README §3.2 / analysis/07_classC_blacklist.csv)
import pandas as pd
bl = pd.read_csv("/transfer-shanghai/KAI0/Task_A/base/analysis/07_classC_blacklist.csv")
EXCLUDE_CLASSC = set(zip(bl.date, bl.ep))   # 129 个 (date, ep_filename)

# 截尾 End-snap 5 ep (06_end_snap_trim.csv)
trim = pd.read_csv("/transfer-shanghai/KAI0/Task_A/base/analysis/06_end_snap_trim.csv")
TRIM_MAP = dict(zip(zip(trim.date, trim.ep), trim.T_new))   # 5 个 (date, ep) → keep_frames

# 在 build 循环里 filter:
for src_date in sorted(SRC_ROOT.iterdir()):
    if src_date.name in EXCLUDE_DATES: continue
    for parquet in src_date.glob('data/chunk-*/episode_*.parquet'):
        key = (src_date.name, parquet.name)
        if key in EXCLUDE_CLASSC: continue
        T_keep = TRIM_MAP.get(key, None)  # None = 全保留
        # rebuild ep with [:T_keep] if specified
```

### 2.3 数据路径 (build 后)

| 路径 | 用途 |
|---|---|
| `/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/A_0423_0527/` | cnsh 训练数据 |
| `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/A_0423_0527/` | cnbj 训练数据 (用前需 sync) |
| `/vePFS/.../A_0423_0527/meta/norm_stats.json` | 重新算的 norm_stats |

### 2.4 Val 数据

| Val | 路径 | 说明 |
|---|---|---|
| Cross val (与 v8 一致) | `vis_v2_merged_val` (30 ep, 04-23/24 dates) | 与 vis_v2_full / 5day_recent 同 val, 可直接对照 |
| Native val (可选) | `A_0423_0527_val` (从 13 dates 各抽 hold-out) | 训练时 inline-eval 用 |

---

## 3. 训练配置 (Run-A + Run-B 共享)

| 项 | 值 |
|---|---|
| **Model** | `pi0_config.Pi0Config(pi05=True)` |
| **训练框架** | **JAX/Flax NNX** (`scripts/train.py`) |
| **Dataset** | `A_0423_0527` (~1059 ep) |
| Prompt | `"Flatten and fold the cloth."` |
| `use_delta_joint_actions` | False (absolute, 与 smooth_800 / vis_v2_full 一致) |
| **LR schedule** | Cosine, warmup_steps=1_000, peak_lr=**1.5e-5**, decay_steps=50_000, decay_lr=**1.5e-6** |
| **EMA decay** | 0.9999 |
| **Steps** | **50,000** |
| **Batch size** | **128** |
| **Cluster** | 8× NVIDIA GPU (单节点 FSDP) |
| `fsdp_devices` | 8 |
| `keep_period` | 10_000 |
| `save_interval` | 2_000 |
| `num_workers` | 16 (8 GPU × 默认), uc 集群 64 |
| `inline_eval_val_root` | `vis_v2_merged_val` (cross val 30 ep) |
| `inline_eval_n_frames` | 200 |
| `inline_eval_every` | 每 4k step |

### 3.1 Run-A: `A_0423_0527_pi05_JAX`

```python
TrainConfig(
    name="pi05_flatten_fold_A_0423_0527",  # base config, 两 run 共用 + override init
    model=pi0_config.Pi0Config(pi05=True),
    data=LerobotAgilexDataConfig(
        repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/A_0423_0527",
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,
    ),
    # === Run-A init ===
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
    ),
    ema_decay=0.9999,
    num_train_steps=50_000,
    keep_period=10_000,
    save_interval=2_000,
    num_workers=16,
    batch_size=128,
    fsdp_devices=8,
    inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_merged_val",
    inline_eval_n_frames=200,
    inline_eval_every=4,
),
```

启动:
```bash
python -u scripts/train.py pi05_flatten_fold_A_0423_0527 \
  --exp-name A_0423_0527_pi05_JAX \
  --batch-size 128 --fsdp-devices 8 --num-workers 16 \
  --no-wandb-enabled
```

### 3.2 Run-B: `A_0423_0527_mixed1_JAX`

同 Run-A, 但 `weight_loader.params-path` 改为 `mixed_1` (或 `mixed_1_clean`):

```python
# 同 base config, 仅 weight_loader 改:
weight_loader=weight_loaders.CheckpointWeightLoader(
    "/home/tim/local_ckpts/Task_A_init/mixed_1_clean/params"  # 或 cnbj 对应路径
),
```

启动:
```bash
python -u scripts/train.py pi05_flatten_fold_A_0423_0527 \
  --exp-name A_0423_0527_mixed1_JAX \
  --weight-loader.params-path /path/to/mixed_1_clean/params \
  --batch-size 128 --fsdp-devices 8 --num-workers 16 \
  --no-wandb-enabled
```

---

## 4. 期望对照表

### 4.1 与 anchor 模型对比 (cross val MAE@1 在 vis_v2_merged_val 30 ep)

| 模型 | 数据 | ep | Init | Cross val MAE@1 | 真机 |
|---|---|---:|---|---:|---|
| smooth_800 | 4-23~5-09 X1 cleaned | 811 | mixed_1_clean | (待 cross val 补测, native 0.0089) | ✅ work |
| vis_5day_recent | 5-18~22 (校准漂移期) | 498 | pi05_base | 0.0086 | ❌ broken |
| vis_v2_full | 4-23~5-22 (含 D1+D7) | 1406 | pi05_base | 0.0131 | ❌ broken |
| **Run-A pi05** ⭐ | **4-23~5-27 排 5-16~5-21** | **1083** | pi05_base | **期望 ≤ 0.0100** | **期望 work (smooth-class)** |
| **Run-B mixed1** ⭐ | 同 Run-A | 1083 | mixed_1 | 期望 ≤ 0.0095 | 同 Run-A |

### 4.2 数据组成对比 (smooth_800 + 后期非校准漂移段)

```
Run-A/B = smooth_800 范围 (4-23~5-09) +  ⭐⭐⭐ 后期非校准段 (5-22, 5-26, 5-27)
        - 5-16 (D1 stay-still)
        - 5-18~21 (D7 gripper 校准漂移)
        - Class C 黑名单
        - End-snap 5 ep 截尾
```

**Run-A/B 与 smooth_800 + 5day_recent 的关键差异**:
- **Run 含 5-22/5-26/5-27 共 3 dates (~251 ep)** — README §1 标"杂乱投放 ⭐⭐⭐ 最难场景"
- **不含 5-18~5-21** — 这正是 v7 v8 发现的 gripper 校准漂移段
- **5-22 是临界 date** (mtime 2026-05-27 也 update 了, 见 v8) — 也校准过, 需 verify 是否仍含 R_grip 过度问题

### 4.3 与 5day_recent 真机 fail 对照

| 差异 | 5day_recent (fail) | Run-A (期望 work) |
|---|---|---|
| 5-16 ideal | 不含 (天然) | 不含 ✓ |
| 5-18~21 校准漂移 | **全含 (399 ep)** | **不含 ✓** |
| 5-22+ 杂乱投放 | 含 (100 ep) | 含 (~251 ep, 3 dates) |
| smooth 4-23~5-09 | **不含** | **含 (~808 ep)** ✓ |

→ Run-A/B = "smooth_800 优势 (含早中期 + X1 cleaned) + 5day_recent 优势 (含后期 ⭐⭐⭐ 场景) - 校准漂移段". 理论上是**最优组合**.

### 4.4 双 init 对照 (Run-A vs Run-B)

| 维度 | Run-A pi05_base | Run-B mixed_1 |
|---|---|---|
| 起点 MAE@1 (step 4k) | ~0.05 (raw 起点) | ~0.012 (Task_A warmed) |
| Final MAE@1 (50k) | 期望 ≈ Run-B (历史经验) | 期望 ≈ Run-A |
| 真机表现 | 可能差异不大 | 同 |
| 用途 | 与 vis_v2_full pi05_base 直接对照 | 与 smooth_800 mixed_1_clean 直接对照 |

---

## 5. 实施步骤

| Step | 内容 | ETA |
|---|---|---:|
| T1 | 写 `build_A_0423_0527.py` (参考 build_vis_v2_full.py + Class C/End-snap filter) | 2h |
| T2 | Run build script: TOS → `/vePFS/.../A_0423_0527/` (cnsh + cnbj 各 build 一份) | 4h (含 video symlink) |
| T3 | 计算 norm_stats | 0.5h |
| T4 | 加 base config `pi05_flatten_fold_A_0423_0527` 到 `config.py` | 0.5h |
| T5 | Smoke test 1 GPU (Run-A pi05_base init) ~1k step | 2h |
| T6 | **Run-A full** 8 GPU × 50k step | ~30-40h |
| T7 | **Run-B full** 8 GPU × 50k step (并行 if 资源允许, 否则 sequential) | ~30-40h |
| T8 | Offline eval: 两 ckpt × vis_v2_merged_val cross val | 2h |
| T9 | 真机测试 (Run-A 优先, Run-B 视 A 结果) | 1 day |
| T10 | 写 results 文档 + 更新 00_training_history | 1h |

总 ETA: **3-5 day** (并行 if 16 GPU 可用 2 节点)

---

## 6. 决策树

```
Run-A/B 真机测试结果?
├── ✅ 三症状全消 (闭合稳定, 不 oscillate, 不松开)
│   → v7 校准漂移假说 100% 证实
│   → 直接生产部署 (Run-A 或 Run-B, 看 MAE 谁更好)
│   → 后续 5-18~5-21 数据先放着, 等 R_grip 校准脚本修复 (F_v8_A)
│
├── ⚠️ 部分症状消失 (比 vis_v2_full 好, 但比 smooth_800 差)
│   → 5-22+ 后期数据可能仍有 R_grip 校准残留 (5-22 mtime 也被 update 11:27)
│   → 进一步排除 5-22, 仅用 4-23~5-09 + 5-26/27, 重训
│
├── ❌ 三症状仍在
│   → v7 假说被反驳, 真因不是校准
│   → 重审 wrist 漂移 + Class C 边际 + inference G0 生效 status
│
└── 🆕 Run-A vs Run-B 真机不同 → init 在大数据集仍有影响 (与历史经验反向, 需调查)
```

---

## 7. 风险 + 应对

| # | 风险 | 应对 |
|---|---|---|
| 1 | 5-22 数据也含 R_grip 过度校准 (v8 §0.NEXT 显示 5-22 mtime 11:27 也 update) | 在 T1 build 后, 重扫 5-22 R_grip 分布与 smooth 时期对比, 若过度归零则也排除 |
| 2 | 1083 ep 在 50k step 下 epoch 数 = 50k × 128 / ~1.4M ≈ 4.6 (略低于 smooth 5.4 / 5day 7.7) | 接受 (与 vis_v2_full 3.3 接近), 若结果不好考虑加 step 到 75k |
| 3 | Build script 时间 (TOS 同步 video symlinks 慢) | 在 gf0/gf3 (mount TOS) 上跑, 不要 scp |
| 4 | Run-A + Run-B 并行需 16 GPU (2 节点) | 优先 Run-A 单跑, 出 eval 后再决定 Run-B |
| 5 | mixed_1 init 在 cnbj 路径是否就位? | T1 前 verify `/path/to/mixed_1_clean/params` 存在或 sync |

---

## 8. 与 v8 audit 文档协同

本实验**直接验证** v8 audit §0.NEXT-v8 推断:
- v7 推断: 5-18~5-21 是 gripper 校准漂移期 (与 README §4 校准说明完美对应)
- v8 推断: 校准后 R_grip 过度归零, 不能直接用校准后数据训练
- **Run-A/B = "排除校准漂移期" 的对照实验**
- 如果 Run-A 真机 work → v7 假说证实, v8 R_grip 过度校准只需后续修脚本不影响当前训练

---

## 9. 文件 / Ckpt 位置 (训练后)

| Run | Ckpt 路径 |
|---|---|
| Run-A | `/vePFS/.../checkpoints/pi05_flatten_fold_A_0423_0527/A_0423_0527_pi05_JAX/49999/` (cnsh) |
| Run-B | `.../A_0423_0527_mixed1_JAX/49999/` |
| Build dataset | `/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/A_0423_0527/` |
| Build script | `train_scripts/data/build_A_0423_0527.py` (待写) |
