# Task A new_pure_200 训练结果 ⭐⭐ 全任务 NEW SOTA

> **结论先行**: js02 (单机 8 GPU) 上 `task_a_new_pure_200_new_norm` 从 step 22000 resume 到 49999, final **MAE@1 = 0.0065** ⭐, 全任务新 SOTA, 比之前 best (`task_a_new_pure2_1800_6000_new_norm` 0.0085, uc 集群 7900 ep) 低 **24%**。**200 ep 精选 + mixed_1_clean init** 推翻"数据规模 > 数据质量"假设, 数据质量 (`-new` 限定 + mirror 增强) 占主导。

## 1. 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_pure_200_js` |
| Model | pi05 (`Pi0Config(pi05=True)`) |
| **Init** | `/mnt/data/tim/kai0_ckpts/Task_A/mixed_1_clean/params` (MA-merged Task_A warmed-up) |
| **Dataset** | `/mnt/data/tim/data/Task_A/self_built/A_new_pure_200` (200 ep 精选, JuiceFS:visincept) |
| - 来源 | `-new` 日期限定 + hflip mirror 增强, 仅高质量子集 |
| Val | `/mnt/data/tim/data/Task_A/self_built/A_new_pure_200_val` |
| Prompt | "Flatten and fold the cloth." |
| `use_delta_joint_actions` | False |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50k, decay_lr=1.5e-6 |
| EMA decay | 0.9999 |
| Steps | 50,000 |
| Batch | 120, fsdp_devices=8 (单机全 FSDP `[1,8]`) |
| Save | every 2,000 step, keep_period=2,000, max_to_keep=1 |
| inline_eval | every 2 saves (= 每 4k 步), 200 frames |
| Seed | 42 |
| Server | **js02** (A800-SXM4-80GB × 8) |
| 启动 | **从 step 22000 resume** (原 ckpt 在 `/mnt/data/tim/checkpoints/.../22000`), 续训到 49999 |
| WandB | offline (`--no-wandb-enabled`) |
| 训练时长 | resume window 16h11m (2026-05-14 08:08 → 2026-05-15 00:18 CST) |

### 1.1 关键差异 (vs 老 SOTA `pure2_1800_6000`)

| 维度 | pure_200 (本实验) | pure2_1800_6000 (老 SOTA) |
|---|---|---|
| **数据规模** | **200 ep** (1/40 of SOTA) | 7900 ep |
| **Init** | mixed_1_clean (Task_A 微调过) | pi05_base (原始) |
| 集群规模 | 单机 8 GPU | 24 GPU (uc01+02+03) |
| Batch | 120 | 120 |
| Final MAE@1 | **0.0065** ⭐ | 0.0085 |
| Gap | — | **+30.8% (worse)** |

## 2. 完整 inline-eval MAE@{1,10,25,50} 曲线

> 因 resume from step 22000, 仅记录 step 24000 起的 eval。

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 24000 | 0.0071 | 0.0086 | 0.0095 | 0.0115 | (resume baseline) |
| 28000 | 0.0069 | 0.0082 | 0.0088 | 0.0102 | -2.8% |
| 32000 | 0.0067 | 0.0078 | 0.0084 | 0.0094 | -2.9% |
| 36000 | 0.0066 | 0.0076 | 0.0080 | 0.0088 | -1.5% |
| 40000 | 0.0066 | 0.0074 | 0.0078 | 0.0085 | 0.0% |
| 44000 | 0.0065 | 0.0073 | 0.0076 | 0.0081 | -1.5% |
| 48000 | 0.0065 | 0.0072 | 0.0075 | 0.0080 | 0.0% |
| **49999** | **0.0065** | **0.0072** | **0.0075** | **0.0079** | 0.0% |

**Best**: step 49999, MAE@1 = **0.0065** ⭐ (44k/48k 也是 0.0065, @50 在 49999 最低 0.0079)

## 3. 训练动力学

- resume @ 22k 的起点未直接 eval, 但 step 24k 已达 0.0071 — mixed_1_clean init 下经过 24k step (相当于从 mixed_1 起步训了 24k) 已显著优于 pi05_base init 同步数
- 收敛极快: step 24k → 36k 在 12k 步内从 0.0071 跌到 0.0066 (-7%), 但 36k 后进入 long plateau
- @1 在 44k-49999 持续 0.0065 (4 个连续 eval 持平), 训练已饱和
- @50 (long-horizon) 持续小幅下降到 step 49999 (0.0115 → 0.0079, **-31%**) — 长 horizon planner 后期仍在精修, 与 pure2_1800_6000 趋势一致
- 全程 0 NaN, 无 div, eval 时长 754-962s (均值 768s)

## 4. SOTA 关键洞察

1. **数据质量 > 数据规模**: 200 ep (`-new` 限定 + mirror) 比 7900 ep 大杂烩 MAE 低 24%。**关键变量**: pure_200 是高度精选的 `-new` 日期数据 + hflip mirror 双对增强, 比 mix b6000+p1200 大量 `kai0_base / kai0_advantage` 旧数据干净得多。
2. **mixed_1_clean init 上限被低估**: 之前 pure2_1800_6000 用 pi05_base init 得到 0.0085 (老 SOTA), 推断 "干净起点天花板更高"。但 pure_200 用 mixed_1_clean (Task_A warmed-up) 反而打到 0.0065 — 说明在**高质量小数据**场景下, Task_A 适配过的 init 反而是更好的起点 (节省 24k step 的"重新学 Task_A"开销)。
3. **小数据 + 长训** 配合 EMA decay 0.9999 不发生 overfit: 200 ep × 50k step = ~250 epoch, 仍单调下降, 没看到 val MAE 反弹。
4. **resume 不影响收敛趋势**: 22k resume 起点没破坏 cosine LR schedule (continued from same step counter), 后续 28k step 平滑收敛。
5. **@1 vs @50 差异启示**: @1=0.0065 (单步精度), @50=0.0079 (50 步长 horizon) — gap 仅 22%, 明显小于其他实验 (pure2_1800_6000 @1=0.0085 @50=0.0337, gap 296%)。说明该模型不仅单步准, 长 horizon planner 也极稳。

## 5. 最佳 ckpt 位置

```
js02:/mnt/data/tim/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_new_pure_200_new_norm/49999/
```

⚠️ JuiceFS:visincept 共享路径 — js01-js04 任一节点都能直接读, 不需 SSH 跨节点拷贝。

**完整 ckpt 列表** (每 2k 步): 22000, 24000, 26000, ..., 48000, 49999  
**附属文件**: 同目录 `norm_stats.json`, `wandb_id.txt`

## 6. 后续计划

- **对照实验**: 改用 pi05_base init (而非 mixed_1_clean) 跑同样的 pure_200 数据, 验证"高质量小数据下哪个 init 更优"。新 exp `task_a_pure200_new_norm_base_pi0.5` 已在 js04 单卡启动。✅ **2026-05-26 完成, 见 §7**
- **deploy 到 sim01**: ckpt 49999 应优先 pack + 上 sim01 测试 (老 SOTA pure2_1800_6000 已上, 这个有望大幅优化 chunk planner)。
- **解析为啥 -24%**: 是数据质量, 还是数据量本身就够? 跑 pure_200 / pure_400 / pure_800 sweep 即可 disentangle。

---

## 7. 2026-05-26 更新 — pi05_base init 对照实验完成 + 关键修正

> 🔥 **本节推翻 §1-5 中的 3 个核心假说**. 新 dual eval 数据揭示: init **不是主导因素**, pure_200 的 0.0065 是**严重过拟合 to 2 dates** 的产物, vis_v2_full 在 generalization 上反而**比 pure_200 好**.

### 7.1 pi05_base init 实验完成

| 项 | 值 |
|---|---|
| Ckpt | `/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/task_a_pure200_base_pi05_step49999` |
| 训练 | gf3 (1 H20 smoke), batch=16, fsdp_devices=1 |
| 数据 | 同 A_new_pure_200 (100 unique × 2 mirror, 2 dates 05-08/09) |
| Init | **pi05_base 冷启** (vs NEW SOTA 的 mixed_1_clean warm init) |
| Steps | 50000 (full schedule) |

### 7.2 Dual eval 结果

| Ckpt | val 数据集 | MAE@1 | MAE@10 | MAE@25 | MAE@50 |
|---|---|---:|---:|---:|---:|
| **pure_200 + mixed_1_clean** (NEW SOTA) | native A_new_pure_200_val | **0.0065** ⭐ | 0.0072 | 0.0075 | 0.0079 |
| **pure_200 + pi05_base** (新对照) | native A_new_pure_200_val | **0.0065** ⭐ | **0.0074** | **0.0078** | **0.0087** |
| **pure_200 + pi05_base** (cross-val) | vis_v2_val50 (来自 04-23/24) | **0.0207** ❌ | 0.0507 | 0.0900 | 0.1348 |
| **vis_v2_full + pi05_base** (1406 ep, 16 dates) | vis_v2_val50 (同上) | **0.0131** ✓ | 0.0386 | 0.0714 | 0.1138 |

### 7.3 🔥 推翻的 3 个假说

#### 推翻 1: ❌ ~~"mixed_1_clean init 在小数据下更优"~~ (§1, §5)

**新数据**: pure_200 + pi05_base 在 native val 上 **0.0065 == pure_200 + mixed_1_clean 0.0065**.

→ **Init 不影响 final native MAE**. 之前 §5 关于 init 的所有讨论都需修正. 关键在数据 (pure_200 精选 + mirror) 而非 init.

#### 推翻 2: ❌ ~~"pure_200 是真正的 SOTA"~~

**新数据**: 同 ckpt 在 cross-val (vis_v2_val50) 上 MAE@1 从 0.0065 → **0.0207 (3.2× 退化)**.

→ **0.0065 是"过拟合 to 2 dates" 的产物**, 不代表真泛化能力. 在 vis_v2_full 训练集分布的早期 dates (04-23/24) 上, pure_200 model 性能差.

#### 推翻 3: ❌ ~~"vis_v2_full 数据量增加反而效果差"~~

**新数据**: 在同 val (vis_v2_val50) 上对比:
- vis_v2_full + pi05_base: **0.0131**
- pure_200 + pi05_base: **0.0207** (差 36%)

→ vis_v2_full 在跨日期 val 上**比 pure_200 好 36%**. 1406 ep 跨 16 dates 训练让 model 学到 broader generalization prior. 训练没坏, 反而是 pure_200 在 cross 表现不好.

### 7.4 真正的解释 — Val distribution 决定 MAE 数量级

| Train | Native val (in-distribution) | Cross val (vis_v2_val50, 04-23/24) |
|---|---|---|
| pure_200 (05-08/09, 2 dates) | 0.0065 ✓ (val ∈ train dates) | 0.0207 ❌ (val ∉ train dates, OOD) |
| vis_v2_full (04-23~05-22, 16 dates) | n/a (no test) | **0.0131 ✓ (val ∈ train dates)** |

**vis_v2_val50 (04-23/24) ∈ vis_v2_full 训练分布, ∉ pure_200 训练分布**.
- vis_v2_full 在 vis_v2_val50 上的 0.0131 是 in-distribution 精度
- pure_200 在 vis_v2_val50 上的 0.0207 是 OOD 精度

之前的 NEW SOTA 0.0065 vs vis_v2_full 0.0131 是**不公平比较** — 两 model 各自在自己分布的核心做 eval.

### 7.5 真机表现 vs Offline MAE 的真正关系

```
真机部署 in conditions ≈ 05-08/09 dates 风格 (近期采集):
  → pure_200 表现 ≈ 0.0065 (优秀)
  → vis_v2_full 表现 ≈ 0.0131 (一般, 因为是 16 dates 的"平均策略")
  → pure_200 在此特定场景胜出

真机部署 in conditions varied / OOD:
  → pure_200 表现 ≈ 0.0207+ (3× 退化)
  → vis_v2_full 表现 ≈ 0.0131 (稳定)
  → vis_v2_full 在 generalization 上胜出
```

### 7.6 vis_v2_full 真机不好的修正解释

之前归因 (错): "训练没问题, init 弱 / mirror 缺失 / 协议漂移"

现在归因 (对): **vis_v2_full 训练没问题, generalization 实际比 pure_200 好**. 真机不好可能因为:

1. **真机评估场景与训练 dates 不匹配**: vis_v2_full 训练分布跨 04-23~05-22, 真机可能在更新的 dates 风格下评估 (e.g., 05-26) → OOD
2. **vis_v2_full 学到"平均策略", 不锐利**: 跨 16 dates 的"折中" → 在任何具体场景上都不是最锐利, 但比 pure_200 在 OOD 时稳定
3. **Real-world vs offline gap**: offline val 是录制好的数据, 真机有 lighting/setup variations 等 noise

### 7.7 修正后的归因表

| 维度 | 之前 (错误) | 现在 (正确) |
|---|---|---|
| Init 影响 | 主导因素, mixed_1_clean 优 | **无影响**, 同 0.0065 |
| Mirror 缺失 | 主因之一 | 主要 boost in-distribution 精度, 对 generalization 影响小 |
| 数据量更大反而差 | 是 (基于 native val) | **错** — vis_v2_full 在 vis_v2_val50 上 0.0131 < pure_200 cross val 0.0207 |
| 16 dates 协议漂移 | 主因之一 (训练不锐利) | 部分对 — 但同时给了 generalization prior |
| Val 选择 | (未考虑) | **决定性因素** |
| 真机表现 | offline MAE 越低越好 | 取决于真机条件与 train distribution 距离 |

### 7.8 下一步可立即跑的实验

| 优先级 | 行动 | 预期发现 |
|---|---|---|
| ⭐⭐⭐ P0 | **真机评估 pure_200 vs vis_v2_full 在近期/旧 dates 多场景** | 验证 §7.5 假设 |
| ⭐⭐ P1 | 构建 universal val (从 vis_v2_full 16 dates 各抽 ~2 ep) eval 所有 ckpt | 公平 generalization 比较 |
| ⭐⭐ P2 | **vis_v2_full ckpt 49999 + 100 unique pure_200 ep finetune 10k step** | 兼顾 broad prior + 锐利 |
| ⭐ P3 | pure_200 用更多 dates (e.g., 05-08~05-22 = 5 dates curated × mirror) 重训 | 测 broader train + curated data 是否最优 |

### 7.9 关键 Takeaway (修正版)

1. ✅ **小 curated 数据 + 重复 = 在 in-distribution val 上极佳**, 但是**严重过拟合, 在 cross-val 3.2× 退化**
2. ✅ **大 diverse 数据 + 跨日期 = 在 broader val 上更稳** (vis_v2_full 0.0131 < pure_200 cross 0.0207)
3. ❌ ~~Init 决定上限~~ — 数据决定上限, init 影响收敛速度但不影响 final 精度
4. ❌ ~~vis_v2_full 训练失败~~ — 训练 OK, 在跨日期 val 上反而比 pure_200 好 36%
5. ⚠️ **必须用 train/val distribution 匹配的 val 比较 MAE**, 跨分布的 MAE 不可比
6. ⚠️ **真机 deployment 应根据真机场景 vs train distribution 距离选择 model**: 近期场景 → pure_200, 跨日期/OOD → vis_v2_full

---

## 8. 2026-05-27 Chunk/Noise diagnostic 更新

为定位 vis_v2_full 真机 "走几步退几步 + 夹爪犹豫" 问题, 在 3 个 ckpt 上跑了 P0/P1/P2 + fixed-noise 诊断. pure_200 这边的 fresh measurement:

| 指标 | pure_200 | vis_v2_full | TAC v7 |
|---|---:|---:|---:|
| P1 random L | 0.0390 🟡 | 0.0265 🟢 | 0.0666 🔴 |
| P1 fixed-noise L | 0.0389 🟡 | 0.0206 🟢 | (未测) |
| **Noise contribution ΔL** | **+0.0001** | +0.0059 | — |
| P2 mean variance | **0.0117** | 0.0234 | 0.0231 |
| P2 max variance | **0.1903** | 0.6688 | 0.6525 |

**pure_200 的特征**:
- ✅ **几乎完全 deterministic** — noise 贡献 0.0001 (vis_v2_full 是 0.006, 大 60×)
- ✅ **P2 max 0.19** (低 — 所有时刻所有 dim 都 deterministic)
- 🟡 **P1 = 0.039 MARGINAL** — chunk-to-chunk diff 主要来自**输入变化** (state@k → state@k+5), 不是 noise

**为什么 pure_200 真机 work 而 vis_v2_full 不 work**: pure_200 的 deterministic 行为意味着同一 obs 始终输出相同 action chunk, **关键时刻不会跳 mode**. vis_v2_full 大多数时刻 deterministic 但在 rare 关键时刻 (P2 max 0.67) 多 mode → 真机看到的 oscillation.

**修复路线**: vis_v2_full 走 [G0 fixed-noise inference fix](../../../deployment/inference/fixed_noise_inference_fix.md), 不需要重训. 详见 [analysis §11](../../analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md#11-2026-05-27-重大数据更正--chunknoise-诊断-定位真机-oscillation).
