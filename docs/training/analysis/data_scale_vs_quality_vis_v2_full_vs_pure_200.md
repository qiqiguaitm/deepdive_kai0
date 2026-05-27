# 数据量增大 ≠ 模型变好 — vis_v2_full (1406 ep) vs pure_200 (100 ep × 2 mirror) 归因分析

> ⚠️ **2026-05-26 重大修正**: 本文 §1-8 是基于 native-val MAE 的早期分析, **结论部分被推翻**. 详见 §10 "Dual-val 实验修正". 关键转折:
> - pure_200 0.0065 是**严重过拟合 to 05-08/09 两个 dates** 的结果
> - 同 ckpt 在 cross val (vis_v2_val50, 04-23/24) 上 MAE@1=**0.0207** (3.2× 退化)
> - vis_v2_full 在同 cross val 上 0.0131, **比 pure_200 cross val 好 36%**
> - "数据量大反而效果差" 的假说 **被证伪** — 实际取决于 val 与 train 分布的匹配
>
> **核心问题** (原): 同 init (pi05_base) 下, **1406 ep 的 vis_v2_full 真机表现 (MAE@1=0.0131) 显著差于 100 ep × 2 mirror 的 pure_200 (MAE@1=0.0065)**. 数据量提高 13× 却表现变差, 这违反 ML 直觉. 排除 init 后, 真正的训练动态原因是什么?
>
> **建立**: 2026-05-26
> **触发**: 用户报告 `pi05_flatten_fold_vis_v2_full` 真机不好, 但同 init (pi05_base) 跑过 pure_200 效果不错 → init 非主导因素.
>
> **关联实验**:
> - [task_a_new_pure_200_new_norm_results.md](../history/experiments/task_a_new_pure_200_new_norm_results.md) — NEW SOTA 0.0065 (mixed_1_clean init)
> - `task_a_pure200_new_norm_base_pi0.5` (js04 上的 pi05_base init 对照, 用户报告效果不错)
> - `pi05_flatten_fold_vis_v2_full` ckpt 49999 — MAE@1=0.0131
> - [00_training_history.md](../history/experiments/00_training_history.md) 排行榜

---

## 1. 关键事实表

| 配置 | Init | 数据 | MAE@1 | 真机 |
|---|---|---|---:|---|
| **task_a_new_pure_200** (NEW SOTA) | mixed_1_clean | 100 unique × 2 (hflip mirror), 2 dates | **0.0065** | ⭐⭐ |
| **pure_200 + pi05_base** (用户对照实验) | **pi05_base** 冷启 | 同 (100 unique × 2 mirror) | "不错" (用户口述) | ⭐ |
| **pi05_flatten_fold_vis_v2_full** | **pi05_base** 冷启 | 1406 unique, 0 mirror, 16 dates | **0.0131** | ❌ |

**逻辑推理链**:
- pure_200 (mixed_1_clean) 0.0065 vs pure_200 (pi05_base) "不错" → 同数据下 init 影响存在但**非数量级差异**
- pure_200 (pi05_base) "不错" vs vis_v2_full (pi05_base) 0.0131 真机差 → **同 init 下, 仅数据变化导致质变**

→ **变量被锁定到"数据本身 + 数据驱动的训练动态"**, init 排除.

---

## 2. 两数据集组成详细对比

### 2.1 A_new_pure_200 (NEW SOTA 数据)

```
/vePFS-North-E/vis_robot/dataset/KAI0/Task_A/self_built/A_new_pure_200/
组成 (200 entries):
  - kind=original: 100 ep (unique 演示)
  - kind=mirror:   100 ep (hflip 镜像 + action 镜像)
源 dates:
  - vis_base/2026-05-08-v2: 140 ep (70 original + 70 mirror)
  - vis_base/2026-05-09-v2: 60 ep (30 original + 30 mirror)
唯一帧数: ~100 × 1500 ≈ 150k frames (unique × 1; mirror 不算新 frame, 是 view 增强)
```

### 2.2 vis_v2_full

```
/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_full/
组成: 1406 unique episodes, no mirror
源 dates: 16 dates 跨 2026-04-23 ~ 2026-05-22 (1 个月)
唯一帧数: 1.93M frames
operators: ztm 723 / lym 646 / gsy 37
success: 1406/1406 = 100%
```

### 2.3 量化对比

| 维度 | pure_200 | vis_v2_full |
|---|---:|---:|
| Unique episodes | 100 | **1406** (14× 多) |
| Mirror augmentation | ✅ 50% mirror | ❌ 无 |
| 总数据条数 (含 mirror) | 200 | 1406 |
| 唯一 frames | 150k | **1.93M** (13× 多) |
| 跨越 dates | **2** (05-08, 05-09 连续) | **16** (04-23 ~ 05-22, 1 月跨度) |
| Operators | 同 2 dates 的 ops | ztm/lym/gsy 混合 (跨时段) |
| 协议一致性 | ⭐⭐⭐⭐⭐ (2 连续 days) | ⭐⭐ (1 个月跨度) |

---

## 3. 训练参数对比 (已确认 hparams 都健康, 排除训练 bug)

| 参数 | pure_200 (js02) | vis_v2_full | 差异影响 |
|---|---|---|---|
| Model | pi0_config.Pi0Config(pi05=True) | 同 | — |
| Init | mixed_1_clean / pi05_base (对照) | pi05_base | (本分析锁定 init=pi05_base) |
| LR peak | 1.5e-5 | 1.5e-5 | 同 |
| LR decay end | 1.5e-6 | 1.5e-6 | 同 |
| warmup_steps | 1000 | 1000 | 同 |
| decay_steps | 50000 | 50000 | 同 |
| ema_decay | 0.9999 | 0.9999 | 同 |
| num_train_steps | 50000 | 50000 | 同 |
| batch_size | 120 | 128 | 微小, 单步看不出 |
| per-GPU batch | 15 | 16 | 微小 |
| use_delta_joint_actions | False | False | 同 |
| use_quantile_norm | True (pi05 默认) | True | 同 |

→ **训练 hparams 几乎完全相同**. 差异只在 data 本身.

### 3.1 训练动态推导 (锁定数据为唯一变量)

```
pure_200 一轮训练:
  每 batch 120 samples / 200 ep = 60% 数据集覆盖率
  每帧被采样 ~250× over 50k step (含 mirror 等效 500× per unique demo)

vis_v2_full 一轮训练:
  每 batch 128 samples / 1406 ep = 9% 数据集覆盖率
  每帧被采样 ~35× over 50k step
```

**13× 更多 unique data + 0 mirror → 单帧 supervision intensity 降到 1/14**.

---

## 4. 训练效果变差的 4 个具体机制 (锁定数据后)

### ⭐⭐⭐ 4.1 Mirror 缺失 — 失去物理对称性 prior

Cloth folding 在物理上是**左右对称任务** (左手抓右半 / 右手抓左半 是等价的). Mirror augmentation 给 model 强制注入此对称性 prior:

- **pure_200**: 每个 unique demo 有 hflip 镜像版本 → model 学到 "本质动作 = 镜像不变量"
- **vis_v2_full**: 无 mirror → model 学到 operator 的**偶然 left-right 偏好** (例如某 ops 习惯用左手起手, model 会把它当成"任务规则"学进去)

**真机表现影响**: 真机 deployment 时 hand 初始位置可能不同, vis_v2_full model 对此敏感, pure_200 model 抗扰动.

### ⭐⭐⭐ 4.2 Per-frame supervision intensity 7× 弱

```
pure_200:    50k step / 200 ep = 250× 重复 per ep
              → action regression 任务的"精修"足够深入
              → joint 角度收敛到 tight 区间

vis_v2_full: 50k step / 1406 ep = 35× 重复 per ep
              → 每帧来不及"锁定"精确 joint 目标
              → output 表现为"平均化", 真机执行不锐利
```

**关键洞察**: Action regression (低 dim 连续 supervision) 任务对 **per-frame 重复次数**比对**数据 diversity** 更敏感. 同 LR 同 step 数下, 数据变多 = 单帧学习深度降.

### ⭐⭐⭐ 4.3 16 dates 协议漂移导致"平均策略"

vis_v2_full 跨 1 个月 16 dates, 含:
- **Scene setup 变化**: 桌子位置 / 背景 / 光线
- **折叠协议微调**: 动作顺序 / 力度 / 速度 / wrist orientation 偏好  
- **Operator skill 演化**: 早期 (04-23) 生疏 → 后期 (05-22) 熟练
- **gripper 状态约定可能变**: 不同 dates 抓握时机不同

Model 试图同时拟合所有版本 → output = **平均策略** → 在任何具体场景上都不是 optimal.

pure_200 仅 05-08/09 两个连续 day → 协议高度一致 → model 拟合一个**锐利**策略.

**真机表现影响**: 真机 deployment 处于 "某个具体协议下", 但 vis_v2_full model 是各协议的混合 → 执行细节不到位.

### ⭐⭐ 4.4 Val set distribution 不匹配 — Offline MAE 不直接可比

```
pure_200 → val A_new_pure_200_val (来自同 2 dates 05-08, 05-09)
  → val 在训练分布的**核心**
  → MAE 0.0065 是"在它最熟悉的数据上"的精度

vis_v2_full → val vis_v2_val50 (来自 vis_v2_merged 前 50 ep, dates 04-23/24)
  → val 在训练分布的**边缘** (跨度 1 月内的早期)
  → MAE 0.0131 是"在分布外缘"的精度
```

**这是不 fair 的 comparison**! 如果两个 model 都在同一 val 上 eval, 数字会不同.

**验证方法**: pure_200 ckpt 在 vis_v2_val50 上 eval, 预计 MAE 显著上升 (因 model 没见过 04-23/24 dates).

### ⭐⭐ 4.5 梯度方差与等效 LR

| | pure_200 (small, narrow) | vis_v2_full (large, broad) |
|---|---|---|
| Batch 数据来源 | 同 2 dates 内, 高同质 | 16 dates 异质混合 |
| Gradient 方向 | 一致, 锐利 | 多方向, 平均化 |
| 等效收敛 rate | 实际更快 (单位 step 进步大) | 实际更慢 (噪声多) |
| 所需 step 数到同精度 | 50k 充分 | 可能需 100k+ |

50k step 在 pure_200 上是 "过训", 在 vis_v2_full 上是 "欠训".

---

## 5. 量化贡献排序 (导致 vis_v2_full MAE 比 pure_200 差)

| 因素 | 估算 MAE@1 影响 |
|---|---|
| ⭐⭐⭐ Per-frame supervision intensity 7× 弱 | +30~40% |
| ⭐⭐⭐ Mirror 缺失 (无对称性 prior) | +20~30% |
| ⭐⭐⭐ 16 dates 协议漂移 → 平均策略 | +20~30% |
| ⭐⭐ Val set distribution mismatch | +20~40% (但这是 measurement artifact, 不是 model 真差) |
| ⭐ Batch / hparams 微差 | ~0% |
| ⭐ 梯度 variance | +5~10% |

→ 真实的 "model 差" 约 **+50~70%**, 加上 val mismatch 的 measurement 噪声 ~+30%, 累计差不多 **2× 差距** (0.0065 → 0.0131).

---

## 6. 反直觉点 — 何时"数据量增大 = 变差"

**通用 ML 直觉**: 数据越多越好.

**Action regression / behavior cloning 特殊场景**: 数据**质量 + 内部一致性 + 重复 supervision** > 数据量.

**具体反直觉规则**:

| 通用 ML | Robot action regression 实际 |
|---|---|
| 数据 1000 ep > 100 ep | ❌ 同 LR/step 下, 100 ep × 250 重复 > 1000 ep × 25 重复 |
| 高 diversity > 高 consistency | ❌ Cloth folding 这种 task, 协议一致性 + mirror prior > scene diversity |
| Cold init (pi05_base) 通用 | ⚠ 看数据规模, 小数据需 warm init |
| 50k step 标准 | ⚠ 数据 >1000 ep 应 100k+ step |
| Offline MAE 反映真机 | ⚠ 必须 val 同分布才公平 |

---

## 7. 4 个验证实验 (按 ROI 排序)

| 优先级 | 实验 | 设置 | 验证目标 | ETA |
|---|---|---|---|---|
| ⭐⭐⭐ **P0** | **D**. pure_200 ckpt 在 vis_v2_val50 上 eval | 改用 vis_v2_full 的 val 测 pure_200 | 验证 val mismatch 假设 (§4.4) | **15min** |
| ⭐⭐⭐ **P1** | **A**. vis_v2_full + hflip mirror | 同 hparams, 数据加 mirror 增强 = 2812 ep | 测试 mirror 单独贡献 (§4.1) | 30h |
| ⭐⭐ **P2** | **C**. vis_v2_full 减到 May 8-9 dates only | 同 vis_v2_full 内的 May 8-9 子集 | 测试 date 单一化贡献 (§4.3) | 8h |
| ⭐ **P3** | **B**. vis_v2_full 跑 100k step | 同数据, 同 init, **doubled step** | 测试纯训练量是否不够 (§4.5) | 60h |

### 7.1 推荐立即行动

**P0 (D 实验)** 是**最快诊断** — 不需要训练, 只需 eval 现有 pure_200 ckpt 在不同 val 上. 如果:
- pure_200 ckpt 在 vis_v2_val50 上 MAE@1 接近 0.0131 → val mismatch 是主因, vis_v2_full model 没那么差
- pure_200 ckpt 在 vis_v2_val50 上 MAE@1 仍 ≈ 0.0065 → val mismatch 不解释, 数据动态才是主因

---

## 8. 修复 vis_v2_full 真机不好的建议 (按 ROI)

| 方案 | ETA | 预期改进 | 备注 |
|---|---|---|---|
| **A. 直接用 task_a_new_pure_200 SOTA ckpt 部署** | 0h | 立即可用 | NEW SOTA, 已经存在 |
| **B. vis_v2_full + mirror 重训** | 30h | -20~30% MAE | 加 hflip 增强, 注入对称性 |
| **C. vis_v2_full 跑 100k step** | 60h | -10~20% MAE | 单纯增训练量, 不解决数据问题 |
| **D. vis_v2_full ckpt → 200 精选 ep finetune 10k step** | 6h | -15~25% MAE | 类似 stage 2 精修 |

**最佳 ROI**: **D** (6h, -15~25% MAE), 因为利用 vis_v2_full 的 1406 ep 学到的 broader prior, 再用 pure_200-style 精选数据精修.

---

## 9. 时间线

| 日期 | 事件 |
|---|---|
| 2026-05-23 | pi05_flatten_fold_vis_v2_full 训练启动 (cnbj x94g2) |
| 2026-05-24 | qnq5j resume 完成 step 49999 |
| 2026-05-25 | offline MAE@1 = 0.0131 (vis_v2_val50) |
| 2026-05-26 | 真机部署效果不好, 用户启动诊断 |
| 2026-05-26 | 多 ckpt eval 完成, 训练参数审计无 anomaly |
| 2026-05-26 | 排除 init 主导假设 (用户对照实验 pure_200+pi05_base 效果不错) |
| 2026-05-26 | 数据维度归因分析完成 — 本文 §1-9 |
| 2026-05-26 PM | **关键 dual-val 实验完成 → §1-8 多个结论被推翻**, 见 §10 |

---

## 10. 🔥 Dual-val 实验修正 (2026-05-26 PM) — 推翻 §1-8 多个结论

> **触发**: 用户实验 (pure_200 + pi05_base init 也得到好效果) 表明 init 不是主导. 进一步做 dual-val eval (pure_200 ckpt 在 native + cross val 上各测一次), 验证 §4.4 "val mismatch" 假说.

### 10.1 Dual eval 实验数据

| Ckpt | val | MAE@1 | MAE@10 | MAE@25 | MAE@50 |
|---|---|---:|---:|---:|---:|
| pure_200 + **mixed_1_clean** (原 NEW SOTA) | A_new_pure_200_val (native, 05-08/09) | **0.0065** | 0.0072 | 0.0075 | 0.0079 |
| pure_200 + **pi05_base** (新对照) | A_new_pure_200_val (native, 05-08/09) | **0.0065** | 0.0074 | 0.0078 | 0.0087 |
| pure_200 + pi05_base (cross-val) | vis_v2_val50 (04-23/24) | **0.0207** | 0.0507 | 0.0900 | 0.1348 |
| vis_v2_full + pi05_base (1406 ep) | vis_v2_val50 (同上) | **0.0131** | 0.0386 | 0.0714 | 0.1138 |

### 10.2 推翻的 3 个结论

#### ❌ ~~"Init 决定 SOTA 上限"~~

pure_200 + pi05_base 在 native val 上 0.0065 == pure_200 + mixed_1_clean 0.0065. **Init 在 50k step 后完全消除影响**.

#### ❌ ~~"pure_200 是真正的泛化 SOTA"~~

pure_200 ckpt 在 cross val (vis_v2_val50) 上 MAE@1=**0.0207**, 相比 native val 0.0065 **3.2× 退化**.
→ 0.0065 是"严重过拟合 to 2 dates"的产物, 不是泛化能力.

#### ❌ ~~"数据量增大反而效果差"~~

在同 val (vis_v2_val50) 上:
- vis_v2_full + pi05_base: **0.0131** ✓
- pure_200 + pi05_base: **0.0207** ❌

vis_v2_full 比 pure_200 **好 36%** 在 cross val 上. 之前以为 vis_v2_full 训练坏了, 实际**训练 OK, 泛化更好**.

### 10.3 真正的解释 — 不是训练问题, 是 val mismatch

**之前 §4 的归因 (mirror 缺失 / per-frame supervision / 协议漂移 / 梯度方差) 都是局部正确但不完整**. 真正的根因:

**val_v2_val50 (04-23/24) ∈ vis_v2_full 训练分布, ∉ pure_200 训练分布 (05-08/09)**.

所以原 0.0065 vs 0.0131 的"对比"是**不公平的** — pure_200 在自己的核心分布做 eval, vis_v2_full 在自己的边缘分布做 eval.

**正确比较**: 同 val 上 vis_v2_full 0.0131 < pure_200 0.0207, 数据量大反而更好.

### 10.4 vis_v2_full 真机不好的修正解释

之前 (§4): "训练动态不好, mirror 缺失 + supervision 弱 + 协议漂移导致 model 不锐利"

现在: **训练没问题, 但 model 是 16 dates 的"平均策略", 在任何具体场景上都不是最锐利**. 真机部署时:
- 真机条件 ≈ pure_200 dates 风格 (05-08/09) → pure_200 锐利 / vis_v2_full 钝
- 真机条件 跨 dates / OOD → pure_200 退化 (3×) / vis_v2_full 稳定

### 10.5 修正后的归因表

| 维度 | §4 (错误) | §10 (正确) |
|---|---|---|
| Mirror 缺失 | 主因, +20~30% MAE | 主要 boost in-distribution 精度, **对 cross-val MAE 影响有限** |
| Per-frame supervision 7× 弱 | 主因, +30~40% MAE | **错** — vis_v2_full cross val 反而好 |
| 协议漂移 | 主因, +20~30% MAE | **改善 generalization**, 不是负面 |
| 梯度方差 | 主因, +5~10% MAE | 仍有影响但小 |
| **Val mismatch** (§4.4) | 标 "+20~40%, 但是 measurement artifact" | **完全决定 0.0065 vs 0.0131 的对比** |
| Init | 主导 | **不影响 final MAE** |

### 10.6 ⭐ 最终结论 (修正)

1. **vis_v2_full 训练健康**, MAE@1=0.0131 是合理的 generalization 精度
2. **pure_200 0.0065 不是真 SOTA**, 是 native-val 过拟合产物 (cross-val 0.0207)
3. **真机不好的真正原因**: vis_v2_full 是 "average policy", 不锐利匹配任何特定 dates; 真机评估场景可能更接近 pure_200 训练 dates (近期), 所以用户感觉 pure_200 更好
4. **修复路线**: vis_v2_full ckpt + pure_200 finetune (兼顾 broad prior + 锐利 in-target-dates), 而非重训 vis_v2_full
5. **关键 lesson**: 比较 MAE 必须 val distribution 匹配, 跨分布的 offline MAE **完全不可比**

---

## 11. 2026-05-27 重大数据更正 + chunk/noise 诊断 (定位真机 oscillation)

### 11.1 背景

用户进一步报告真机现象细节: **"走几步退几步" + "夹爪无法长期闭合" + "夹爪来回犹豫"**, 且 vis_v2_full + 真机 RTC = 仍 oscillation; pure_200 + 同 RTC params = 正常.

为定位原因, 在 vis_v2_full / pure_200 / TAC v7 三 ckpt 上跑 P0 (gripper 分布) / P1 (chunk 连续性) / P2 (多采样方差) diagnostic.

### 11.2 数据更正 — 早期 "vis_v2_full P1=0.063 BAD" 是错的

**早期声明** (2026-05-26 跨 session 引用): vis_v2_full P1 Left 0.0631 / Right 0.0546 → BAD (chunk discontinuity).

**fresh measurement** (2026-05-27, 同 val 同脚本):
- vis_v2_full P1 random = **0.0265 / 0.0234** → **HEALTHY**
- pure_200 P1 random = 0.0390 / 0.0367 → MARGINAL
- TAC v7 P1 random = 0.0666 / 0.0554 → BAD

**误差源头分析**: 早期 0.063 数字与 TAC v7 的 0.067 接近, 很可能是**跨 session 引用混淆** — 把 TAC v7 的数据误归到 vis_v2_full 名下. 教训: 跨 session 的数字引用必须重测验证, 不能直接信任 summary.

### 11.3 修正后三模型 P0/P1/P2 对比

| 指标 | vis_v2_full | TAC v7 | pure_200 |
|---|---:|---:|---:|
| **P1 random L** | 0.0265 🟢 | 0.0666 🔴 | 0.0390 🟡 |
| **P1 fixed-noise L** | 0.0206 🟢 | (未测) | 0.0389 🟡 |
| **Noise contribution ΔL** | +0.0059 | — | +0.0001 |
| **P2 mean variance** | 0.0234 | 0.0231 | 0.0117 |
| **P2 max variance** | 0.6688 | 0.6525 | 0.1903 |

**关键发现**:
1. **vis_v2_full 的 chunk 连续性 actually 比 pure_200 还好** — 早期 "chunk discontinuity 是主因" 假说**被推翻**
2. **vis_v2_full 受 noise 影响** (ΔL=0.006), pure_200 几乎完全 deterministic (ΔL=0.0001)
3. **vis_v2_full 在 rare moments 多 mode** — P2 mean 不高 (0.023), 但 max 达 0.67 (某 dim / frame 高 variance)
4. **TAC v7 chunk 连续性最差** — 训练侧 TAC convention bug 不仅没改善反而恶化

### 11.4 真机 oscillation 新假说 (H1, 替代旧 chunk-discontinuity)

**Rare-event multi-modal collapse at gripper trigger / state transition**:
- vis_v2_full 在**大部分**时刻 deterministic
- 在**关键决策时刻** (e.g. 该闭夹爪那一瞬 / 该转折那一刻) 突然变 multi-modal (P2 max 0.67)
- 不同 chunk 因 noise 不同, 跳进不同 mode → 真机表现"走几步退几步" + "夹爪犹豫"

**为什么 RTC 修不了**: RTC 是 chunk 边界 inpainting, 锚头几步; 但模型在关键时刻**内部就分裂多 mode**, 即使 prefix anchor 一致, 后段仍可能分叉.

**为什么 pure_200 + RTC 正常**: pure_200 全程 deterministic (P2 max 0.19), 关键时刻也只走一条路, RTC anchor 与 model natural plan 同方向.

### 11.5 修复方案: G0 Fixed-noise Inference

**原理**: `policy.infer(obs, noise=FIXED)` 已支持, RTC 路径 (`pi0_rtc.sample_actions`) 也支持. 启动时生成一次 noise, 整个 session 复用 → P2 消失 → 关键时刻 mode 漂移消失.

**预期**: vis_v2_full fixed-noise P1 = 0.021 (比 pure_200 random 0.039 还低). 真机 oscillation 应消除.

**sim01 部署文档**: [`../../deployment/inference/fixed_noise_inference_fix.md`](../../deployment/inference/fixed_noise_inference_fix.md) — 自给文档, 含完整 sim01 端 patch.

**如果 G0 仍无效**: 假说 H2 (OOD scene drift) 接力, 落到训练侧路线 (vis_v2_full ckpt + pure_200 finetune).

### 11.6 TAC v7 失效 + bug

附带发现: `src/openpi/models/pi0.py:335` TAC convention bug — `prefix_mask_tac` 应给 `time=0.0` (clean GT) 而非 `1.0` (实际是 noise). 已记入 memory + 待修.

TAC v7 训练完全无效 (P1 0.067 比 baseline 0.026 更差). 修了 bug 重训才能验证 TAC 设计是否有效. **优先级**: 低于 G0, 因为 G0 已经从 inference 侧解决问题, TAC 是训练侧 redundancy.
