# vis_v2_full 数据侧 Audit — 真机 oscillation 根因 (cumulative 2026-05-27 v8)

> **文档定位**: 数据侧 audit, 定位真机 oscillation / 夹爪犹豫 / 抓取后松开 的**数据侧根因**。版本演进 v1→v8. **当前结论见 §0.NEXT-v8 校准后 mixed 结果**.
>
> **🔑 当前结论 (v8)**: 用户已校准 TOS 5-18~22 数据 (2026-05-27 11:27 完成). 重扫: ✅ L_grip 完全修复, ❗❗ **R_grip 过度校准 (over-correction)** 比校准前更糟. ❌ Wrist 漂移 / Class C / σ 收窄 没修. **仍不可直接用于真机部署, 需先解决 R_grip 过度校准**.
>
> **🚀 下一步**: 见 §0.NEXT-v8 — 优先 fix R_grip 校准脚本 (推测过严 threshold), 再重训.
>
> **权威依据**:
> - [`/transfer-shanghai/KAI0/Task_A/base/README.md`](/transfer-shanghai/KAI0/Task_A/base/README.md) — 数据集 maintainer 官方描述
> - [`/transfer-shanghai/KAI0/Task_A/base/analysis/`](/transfer-shanghai/KAI0/Task_A/base/analysis/) — 8 CSV/JSON 分析包 (含 07_classC_blacklist.csv 129ep / 06_end_snap_trim.csv 5ep)
> - [`task_a_vis_curated_subset_experiments.md`](../history/experiments/task_a_vis_curated_subset_experiments.md) — smooth_800 + 5day_recent 联合实验报告 (anchor 数据来源)
>
> **关联文档**:
> - [`data_scale_vs_quality_vis_v2_full_vs_pure_200.md`](data_scale_vs_quality_vis_v2_full_vs_pure_200.md) — 上游诊断 (P0/P1/P2 + H1 假说)
> - [`task_a_new_pure_200_new_norm_results.md`](../history/experiments/task_a_new_pure_200_new_norm_results.md) — pure_200 对照
> - [G0 sim01 fix](../../deployment/inference/fixed_noise_inference_fix.md) — fixed-noise inference 修复

---

## 0.NEXT-v8 🟡 **校准后 mixed verdict** (2026-05-27 晚 v8) ⭐ 最新

> **背景**: 用户根据 v7 发现, 对 TOS 上 5-18~5-22 数据做了校准 (2026-05-27 11:27 完成, mtime confirmed). 立即重扫.

### v8 数据修改时间 (verify 校准已应用)

```
TOS /transfer-shanghai/KAI0/Task_A/base/
  2026-05-18-v2  ep_000000 mtime: 2026-05-27 11:27:50  ← 今天刚校准
  2026-05-19-v2  ep_000000 mtime: 2026-05-27 11:27:53  ← 同
  2026-05-20-v2  ep_000000 mtime: 2026-05-27 11:27:48  ← 同
  2026-05-21-v2  ep_000000 mtime: 2026-05-27 11:27:49  ← 同
  2026-05-22-v2  ep_000000 mtime: 2026-05-27 11:27:50  ← 同 (5-22 也被 update)
  2026-04-25-v2  ep_000000 mtime: 2026-05-06 04:01    ← 旧, 校准前原始
  2026-05-08-v2  ep_000000 mtime: 2026-05-11 10:13    ← 旧, 校准前原始
```

✅ 5-18~5-22 全部已校准. 4-23~5-09 仍是校准前.

### v8 校准后 gripper 精细分布 (smooth 4-23~5-09 排 Class C vs 5day-NEW 5-18~22 已校准)

#### L_grip — ✅ **基本修好**

| bin | smooth % | 5day-NEW % | Δ pp | v7 校准前 Δ | verdict |
|---|---:|---:|---:|---:|---|
| **[0, 0.001) "完全归零"** | **15.00** | **15.75** | **+0.75** | ❌ -5.78 | ✅ **修好** |
| **[0.001, 0.005) "微开"** | 31.62 | **37.27** | **+5.65** | ❌ +11.33 | ⚠️ 半修, 仍 5.65pp 偏差 |
| [0.050, 0.060) "中开" | 15.98 | 13.92 | -2.06 | -2.04 | 弱漂 |
| [0.070, 0.080) "完全张开" | 9.72 | 6.71 | -3.01 | -2.91 | 弱漂 |

#### R_grip — ❗❗ **过度校准, 反向更糟!**

| bin | smooth % | 5day-NEW % | Δ pp | v7 校准前 Δ | verdict |
|---|---:|---:|---:|---:|---|
| **[0, 0.001) "完全归零"** | **21.71** | **53.60** | **+31.89** ❗❗ | ❌ -11.67 | **过度归零!** |
| **[0.001, 0.005) "微开"** | 41.96 | **10.70** | **-31.26** ❗❗ | ❌ +7.16 | **微开被 erase** |
| [0.060, 0.070) "中-大开" | 4.85 | 7.42 | +2.57 | +2.91 | 弱漂 |

**R_grip 校准过度 — 把所有 [0.001, 0.005) 微开 action 都 clip 到 0**:
- 校准前: R 闭合状态 21.5% 完全归零 + 41.96% 微开 0.001-0.005 (自然机械动态)
- 校准后: R 闭合状态 **53.6% 完全归零 + 10.7% 微开** (微动态被 erase)
- 推测校准脚本规则: `if r_grip < 0.005: r_grip = 0`, 过激

#### Mean 统计 (R_grip 因为 [0.06, 0.07) 升了 +2.57pp 补偿了归零, 整体 mean 没变)

| | smooth | 5day-NEW | Δ |
|---|---:|---:|---|
| L_grip mean | 0.0298 | 0.0258 | **-13.3%** ❗ (仍偏低) |
| R_grip mean | 0.0212 | 0.0212 | +0.3% ✅ |
| L_grip 闭合比 (<0.02) | 51.06% | 56.67% | +5.61pp ⚠️ |
| R_grip 闭合比 (<0.02) | 65.66% | 65.34% | ✅ 一致 |

### v8 Wrist joints — ❌ **没修 (校准不在范围内)**

| joint | smooth μ | 5day-NEW μ | Δμ(°) | σ ratio | v7 校准前 |
|---|---:|---:|---:|---:|---|
| **L_腕yaw** | -0.155 | **+0.018** | **+9.9°** ❗ | 0.79× | +10.0° (没变) |
| L_腕pit | 0.719 | 0.761 | +2.4° | 0.81× | +2.5° (没变) |
| L_腕rol | 0.157 | 0.045 | -6.4° | 0.77× | -6.4° (没变) |
| R_腕yaw | 0.338 | 0.241 | -5.5° | 1.00× | -6.0° (微改善) |
| R_腕rol | -0.235 | -0.152 | +4.8° | 0.86× | +5.0° (微改善) |

**L_腕yaw 漂移 +9.9° 仍存在** — 这是操作员姿态习惯差异, 不是 firmware 校准能修的.

### v8 Action |Δa| max — ❌ **Class C spike 没修**

| 维度 | smooth max | 5day-NEW max | v7 校准前 | verdict |
|---|---:|---:|---:|---|
| L_肩pit | 1.94 | 1.18 | (1.18) | ⚠️ 仍 spike |
| L_肘 | 1.61 | 1.27 | (1.27) | ⚠️ 仍 spike |
| **R_肩pit** | **1.02** | **1.65** ❗❗ | (1.02) | **校准后反升!** |
| R_肘 | 1.29 | 1.24 | (1.24) | 相近 |
| L_grip | 0.080 | 0.080 | ✅ | (gripper 物理上限) |
| R_grip | 0.079 | 0.079 | ✅ | (gripper 物理上限) |

⚠️ **R_肩pit max 反而从 1.02 升到 1.65 rad** — 不在 Class C 黑名单的某 ep 引入了新 spike? 需要 verify.

### v8 综合 verdict 表

| # | 问题 | v7 校准前 | v8 校准后 | 修了? |
|---|---|---|---|:-:|
| 1 | L_grip 完全归零 | ❌ -5.78pp | **+0.75pp** | ✅ |
| 2 | L_grip 微开 [0.001, 0.005) | ❌ +11.33pp | ⚠️ +5.65pp | 半修 |
| 3 | **R_grip 完全归零** | ❌ -11.67pp | ❗❗ **+31.89pp** | **过度校准!** |
| 4 | **R_grip 微开 [0.001, 0.005)** | ❌ +7.16pp | ❗❗ **-31.26pp** | **过度校准!** |
| 5 | L_腕yaw +10° 漂移 | ❌ +10.0° | ❌ +9.9° | 没改 |
| 6 | Wrist σ -20% 收窄 | ❌ | ❌ | 没改 |
| 7 | L_肘 max |Δa| (Class C) | ❌ 1.61 | ❌ 1.27 | 略减 |
| 8 | **R_肩pit max |Δa|** | ❌ 1.02 | ❗ **1.65** | **反升** |

### v8 真机预测

**如果用校准后 5day 数据重训** ckpt + 真机部署:

- L gripper 行为应**改善** (修复了归零)
- **R gripper 行为可能恶化** (binary 化, 缺中间力度调节, 抓物体过急/不稳)
- Wrist L_腕yaw +10° 漂移仍导致 wrist 视野 OOD
- Class C spike 仍可能引起偶发大跳
- 整体: **不可直接部署, 需 first fix R_grip 过度校准**

### v8 修复路径

#### F_v8_A ⭐⭐⭐⭐⭐ (最优先): 修 R_grip 校准脚本
**问题**: R_grip 校准把 [0.001, 0.005) 全 clip 到 0.
**修复**:
- 检查校准脚本的 R_grip threshold 逻辑
- 期望行为: 保留 [0.001, 0.005) 自然微动态 (类似 smooth 时期的 41.96%)
- 重新跑校准, 5day-NEW R_grip [0, 0.001) 应回到 21% 左右

#### F_v8_B ⭐⭐⭐⭐ (并行): 检查 R_肩pit 1.65 rad spike 来源
- 不在 Class C 黑名单? 检查具体 ep
- 可能是新校准引入的 mid-spike, 或之前 missed 的 ep

#### F_v8_C ⭐⭐⭐ Wrist 漂移问题 (操作员习惯)
**无法用 firmware 校准修** — 只能:
- 联合 smooth_800 数据训练 (平均化 wrist mean) → 实验 E v5/v6
- 或 后续录数据时刻意让操作员对齐姿态

#### F_v8_D ⭐⭐ 推理 hack (备用)
如果 F_v8_A 修不了, fallback:
- Inference 时把 model 输出的 R_grip > 0 但 < 0.01 的 value 加噪声扰动到 [0.001, 0.005) 自然分布
- 不优雅但可救急

### v8 决策点

**当前状态**: 数据校准修了 L_grip, 但 R_grip 过度校准, wrist + spike 仍未修. **不应直接重训**.

**推荐顺序**:
1. ⭐ F_v8_A: 修 R_grip 校准脚本 (找用户确认校准脚本逻辑)
2. ⭐ F_v8_B: 检查 R_肩pit 1.65 rad 新 spike 来源 (是否需要扩 Class C 黑名单)
3. 重扫 verify R_grip + spike 都符合 smooth 时期范围
4. 然后走 实验 E v6 (smooth_800 + 5day-NEW 联合 1277 ep + Class C filter, 30h)

---

## 0.NEXT-v7 🔴🔴🔴 **Gripper 校准漂移 — 真机失败真正根因** (2026-05-27 晚 v7, archive)

> ⚠️ v6 仅扫到 EE μ/σ 差异 (L_腕yaw +10°, wrist σ -20%), 推断"prior 过窄". v7 进一步精细扫 gripper 分布 → **发现 gripper 校准漂移**, 这才是真机失败真因.

### v7 决定性证据 — Gripper "完全闭合" 行为根本不同

精细 bin 扫描 (smooth 4-23~5-09 排 Class C vs 5day 5-18~22):

| Gripper | bin (m) | smooth % | 5day % | Δ pp |
|---|---|---:|---:|---:|
| **L_grip** | **[0, 0.001) "完全归零"** | **14.78** | **9.00** | **-5.78** ⭐ |
| **L_grip** | **[0.001, 0.005) "微开 1-5mm"** | 31.71 | **43.04** | **+11.33** ⭐⭐⭐ |
| L_grip | [0.050, 0.060) "中开" | 16.03 | 13.99 | -2.04 |
| L_grip | [0.070, 0.080) "完全张开" | 9.96 | 7.04 | -2.91 |
| **R_grip** | **[0, 0.001) "完全归零"** | **21.51** | **9.85** | **-11.67** ⭐⭐⭐ |
| **R_grip** | **[0.001, 0.005) "微开 1-5mm"** | 42.24 | **49.40** | **+7.16** ⭐⭐ |
| R_grip | [0.005, 0.010) "更微开" | 1.25 | **4.58** | **+3.33** ⭐ |

**核心 pattern**:
- smooth 时期闭合 → action **真的归零** (R_grip 21.5% in [0, 0.001))
- 5day 时期闭合 → action **永远带 1~5mm 微开** (R_grip [0, 0.001) 仅 9.85%, 集中在 [0.001, 0.010))

### 🔑 README §4 完美对应

> README §4 (引用):
> > 5-18 ~ 5-21 夹爪 dim 已于 2026-05-27 完成**线性拉伸校准** (左 → [0, 0.0797], 右 → [0, 0.0795]), 原文件备份在 `/data2/visrobot_backup/datasets/KAI0/Task_A_backup/base/`

**5-18~21 段 gripper firmware 经历了线性拉伸校准**, 这正是 5day_recent 训练数据来源时段! 校准前后 action value 与物理 grip 距离的 mapping **变了**.

### v7 真机失败机理 (替代之前 H_v5/H_v6 假说)

```
训练数据 gripper 表征:
  smooth_800 (4-23~5-09):  firmware 旧 → action=0 = 物理完全闭合
  5day_recent (5-18~22):   firmware 已校准 → action=0.003 = 校准后等效闭合

真机当前 firmware: 大概率仍旧版 (因为 smooth_800 ckpt 用户实测 work)

真机执行 5day_recent ckpt:
  model 输出 action=0.003 (它学到的"闭合"值)
  → 真机旧 firmware 解读: "gripper 应有 3mm 缝隙"
  → 物理上 gripper 没夹紧 (实际开度 ~3mm)
  → 衣服自然滑出 ❌

这与用户真机症状完全吻合:
  "夹取没多久就松开 → 衣服脱落 → 回到初始阶段"
```

### v7 与其他真机现象的对应

| 真机症状 | v5/v6 旧解释 | v7 新解释 |
|---|---|---|
| 完全闭合夹取**变多** (G0 之后) | G0 让 noise deterministic | G0 后 model 总是输出 0.003 (5day 学到的闭合值) — 在 *gripper 控制 level* 上比 random noise 一致 |
| **没多久就松开** ⭐ | model 学到短抓握 mode | **训练 action=0.003 vs 真机 firmware 解读=3mm 缝隙 → 物理上没夹紧 → 衣服滑出** |
| 回到初始阶段 | model 跳回"去抓" mode | 衣服在物理上滑落, 后续 obs 重置 → model 重新进入抓取流程 |

### v7 双手联动模式 + Hold duration (作为辅助证据)

**双手联动**:
| pattern | smooth | 5day | Δ |
|---|---:|---:|---:|
| L closed / R closed | 42.45% | **45.89%** | +3.44pp |
| L closed / R open | 8.45% | 10.87% | +2.42pp |

5day 双手同闭合更频繁 — 操作员更"双手并用", 但差异不致命.

**Hold duration**:
| 维度 | smooth median | 5day median | 短抓握(<60f) % |
|---|---:|---:|---|
| L_grip | 88 帧 | **127 帧** | smooth 29.7% → 5day 23.2% (-6.4pp) |
| R_grip | 143 帧 | 153 帧 | smooth 16.4% → 5day 8.2% (-8.2pp) |
| any | 124 帧 | 158 帧 | smooth 27.2% → 5day 17.2% (-10pp) |

✅ **5day 抓得更稳更久** (median 124→158f). 与"短抓握污染"假说 (v2 D6) 完全反向 — 真机松开**不是 model 学到的行为**, 是 **firmware/数据 mismatch**.

**Gripper |Δa| 单帧大 Δ**:
- 5day 所有 |Δa| 阈值 < smooth (ratio 0.37~0.98×)
- 5day 操作员**不会 snap 释放**, model 不应输出大 Δ
- 进一步证伪"model 主动松开" 假说

### v7 修复路径修正

#### F_v7_A ⭐⭐⭐⭐⭐ (最便宜验证): Gripper 缩放推理 hack
推理时把 model 输出的 grip action 做**反校准映射**:
```python
# 在 ShmServer/WS infer wrapper 中:
def correct_grip(action):
    # 假设 5day 数据闭合 ≈ 0.003, 真机 firmware 闭合 = 0
    # 把所有 action[6] / action[13] < threshold 的值 clip 到 0
    a = action.copy()
    a[..., 6]  = np.where(a[..., 6]  < 0.010, 0.0, a[..., 6])
    a[..., 13] = np.where(a[..., 13] < 0.010, 0.0, a[..., 13])
    return a
```
- ETA: **< 1 hour 代码改 + 真机测试**
- 如果 work → 5day_recent ckpt 可直接用 (zero retrain)
- 如果不 work → 假说错, 走 F_v7_B

#### F_v7_B ⭐⭐⭐⭐ Gripper 数据预处理重训
重 build 5day_recent 数据集 + 把 gripper [0.001, 0.005) 区间 clip 到 0:
```python
# 在 dataloader 或 build_vis_v2_full 里:
for parquet in date_5_18_to_5_22:
    action[..., 6]  = np.where(action[..., 6]  < 0.010, 0.0, action[..., 6])
    action[..., 13] = np.where(action[..., 13] < 0.010, 0.0, action[..., 13])
```
- ETA: 30h 重训
- 如果 F_v7_A work, F_v7_B 不必做 (inference hack 已够)

#### F_v7_C ⭐⭐⭐ 用校准前数据重 build
用 README §4 提到的 `/data2/visrobot_backup/datasets/KAI0/Task_A_backup/base/` (校准前备份) 重 build 5day_recent
- ETA: 30h 重训 + 数据 sync
- 与 F_v7_B 等效但更"干净"

#### F_v7_D ⭐⭐ smooth_800 + 5day_recent 联合 (实验 E v5/v6, 1277 ep)
- 加入 smooth_800 后 mean grip 闭合值会落在 0 vs 0.003 之间, 仍可能 mismatch
- 除非同时 F_v7_B 校准处理
- ETA: 30h

### 🚀 推荐立即行动

**Priority 1** (1h): **F_v7_A inference hack + 真机测试**
- 验证 gripper 校准假说
- 用 5day_recent 现成 ckpt + 推理时 grip clip → 真机
- 如 work → v7 主因证实, **不需要重训**

**Priority 2** (并行): 直接问数据集 maintainer:
- 5day_recent 训练数据是 校准前 还是 校准后?
- 真机当前 firmware 是哪版?
- (这 2 个 yes/no 决定 F_v7_A 是否能 work)

**Priority 3** (如 P1 P2 都不通): F_v7_B/C 重训 30h

---

## 0.NEXT-v6 🔴 5day_recent 真机失败 — EE 分布漂移分析 (2026-05-27 晚 v6, archive)

**❗ Update**: 用户真机测试 5day_recent **不 work**, v5 假说 (D1 是真主因) 被反驳. 现重点扫**末端执行器 (EE) 数据差异**.

### v6 关键发现 — smooth_800 (4-23~5-09 排 Class C) vs 5day_recent (5-18~22) EE 对比

数据样本: smooth ~197 ep / 259k frames; 5day ~150 ep / 251k frames (随机抽样).

#### EE-1 ⭐⭐⭐ **L_腕yaw mean +10° 反向漂移** (最大单维差)

| Wrist joint | smooth μ | 5day μ | Δ角度 | σ ratio (5day/smooth) |
|---|---:|---:|---:|---:|
| **L_腕yaw (dim 3)** | -0.154 | **+0.020** | **+10.0°** ❗ | **0.79×** |
| L_腕pit (dim 4) | 0.720 | 0.764 | +2.5° | 0.81× |
| L_腕rol (dim 5) | 0.156 | 0.044 | -6.4° | 0.77× |
| R_腕yaw (dim 10) | 0.341 | 0.235 | -6.0° | 1.02× |
| R_腕pit (dim 11) | 0.886 | 0.827 | -3.4° | 0.96× |
| R_腕rol (dim 12) | -0.238 | -0.151 | +5.0° | 0.86× |

- **L_腕yaw 漂移 +10°** (从 -8.8° 翻到 +1.1°) — 这是 SE3 表示下显著的姿态翻转
- 与 cross_embodiment_strategy.md §2.1 实测的 **"R 腕 yaw -16.8°" KAI0↔vis 跨 robot 漂移同量级**

#### EE-2 ⭐⭐⭐ **Wrist σ 全面收窄 ~20%** (5day prior 更窄)

5day_recent 全部 wrist joints σ ratio 在 **0.77~1.02×** (主导 < 1.0×):
- 后期 lym 单 operator + 5 连续日 → 操作姿态固定, **prior 收窄 20%**
- smooth_800 多 op (ztm/lym/gsy) + 11 天跨度 → prior 更宽

**含义**: 5day_recent model **prior 过窄**, 真机部署若稍偏离 lym 训练姿态 → OOD → 失控.

#### EE-3 ⭐⭐⭐ **L_grip 操作风格显著差异**

| Gripper | smooth | 5day_recent | Δ |
|---|---:|---:|---:|
| **L_grip mean** | 0.02989 | 0.02614 | **-12.6%** ❗ |
| L_grip 闭合比 (<0.02) | 50.9% | **56.8%** | **+5.9pp** |
| L_grip 完全开比 (>=0.05) | 37.0% | 30.9% | -6.2pp |
| R_grip mean | 0.02116 | 0.02271 | +7.3% (相近) |
| R_grip 闭合比 | 65.7% | 65.0% | -0.7pp (相近) |

- **L_grip 在 5day 期间更多保持闭合** (后期 lym 习惯)
- R_grip 两期接近 (R 操作习惯一致)
- → model 在 L gripper 上学到的释放阈值不同

#### EE-4 Arm workspace σ 略大 (5day +6~13%)

| Arm joint | smooth σ | 5day σ | σ ratio |
|---|---:|---:|---:|
| L_肩pit | 0.553 | 0.605 | 1.09× |
| L_肘 | 0.436 | 0.466 | 1.07× |
| R_肘 | 0.506 | 0.570 | **1.13×** |
| R_肩pit | 0.568 | 0.603 | 1.06× |

- 5day_recent arm workspace 略广 (适应"杂乱投放 ⭐⭐⭐" 场景)
- 但 wrist 收窄 vs arm 扩张 — 表示"在更大 workspace 内用单一 wrist 姿态完成任务"

#### EE-5 Action 速度 — mean/p95 接近, 但 5day 含 max>1 rad spike

| dim | smooth mean |Δa| | 5day mean | 5day p95 | 5day max |
|---|---:|---:|---:|---:|
| L_肘 | 0.00904 | 0.00832 | 0.04232 | **1.27** ❗ Class C |
| R_肘 | 0.00842 | 0.00910 | 0.04645 | **1.24** ❗ Class C |
| L_腕rol | 0.00564 | 0.00397 | 0.02297 | 0.78 |

5day_recent 仍含 32 Class C ep, max=1.27 rad 单帧跳变可见 (但 mean/p95 接近 smooth)

### v6 真机失败新假说

**单一假说不足以解释 — 多因素叠加**:

| ID | 假说 | 证据 |
|---|---|---|
| **H_v6_A** ⭐⭐⭐ | **Operator prior 过窄** (wrist σ -20%) → 真机 OOD | EE-2 全 wrist σ <1.0× |
| **H_v6_B** ⭐⭐⭐ | **L_腕yaw +10° 漂移** → 真机 wrist 视野/抓取角度 OOD | EE-1 唯一大维度差异 |
| **H_v6_C** ⭐⭐ | **L_grip 操作风格漂移** (闭合比 +5.9pp) → 释放时机不一致 | EE-3 |
| **H_v6_D** ⭐⭐ | **Class C 32 ep 仍贡献 spike** | EE-5 max 1.27 rad |
| **H_v6_E** ⭐ | **真机当前部署场景与 lym 后期录制场景距离远** | 推测 (lighting/setup 漂移) |
| **H_v6_F** ⭐ | **G0 fixed-noise 没真生效** (inference 配置 bug) | 需 verify log |

### v6 修复方向修正

**不再依赖单一 ckpt** — smooth_800 + 5day_recent 各有缺陷, 单跑不行:

| 方案 | 优势 | 劣势 |
|---|---|---|
| smooth_800 单独 (用户已测 work) | prior 广 | 缺后期"杂乱投放" supervision |
| 5day_recent 单独 (真机 fail) | cross val 0.0086 ⭐ | prior 过窄 + wrist 漂移 |
| **smooth_800 + 5day_recent 联合 (实验 E v5, 1309 ep)** | 兼顾广 prior + 后期场景 | 需 30h 重训 |
| smooth_800 + 5day_recent + X1 cleanup (实验 E v6, 1277 ep) | 上述 + 排 Class C | 30h |

### v6 推荐: 走实验 E v6 (1277 ep 联合 + X1)

**理由**:
1. smooth_800 真机 work → 早中期 supervision 可靠 (含 simple-fold + 杂乱→整齐 ⭐⭐)
2. 5day_recent 后期 lym 含 ⭐⭐⭐ 最难场景 — net positive, 但需要 smooth 早中期"拉宽 prior"
3. 联合后 wrist σ 应回到广值 (smooth 主导), L_grip 风格平均化
4. 同时排 Class C → 净干净

**Dataset 组成**:
```
= smooth_800 (811 ep, X1 cleaned 早中期 4-23~5-09)
  + vis_v2_full 5-18~22 (498 ep) - Class C 32 ep = 466 ep
  - End-snap 5 ep 截尾
= ~1272-1277 ep / ~1.65M frames
```

**Hparams**:
- Init: **pi05_base** (与 5day_recent 一致, smooth_800 用 mixed_1_clean 仅是节省 warmup, 50k step 后无差异)
- 同 vis_v2_full (peak_lr 1.5e-5, 50k step, batch 128, EMA 0.9999, fsdp 8/16)
- 有效 epoch = 50k × 128 / 1.65M = **3.88 epoch**

**期望**:
- ✅ 三症状全消 (smooth_800 已验证, 加 5day 不应破坏)
- ✅ Wrist prior 不过窄 (smooth_800 拉宽)
- ✅ Long-horizon 优于 smooth_800 (加后期 lym 补 ⭐⭐⭐ supervision)
- ✅ Class C 全排, 无 spike

---

## 0.NEXT 🚀 真机测试 5day_recent — 执行 Checklist (2026-05-27 晚)

⚠️ **结果已知 (2026-05-27 晚 v6)**: **真机 fail**. 详见 §0.NEXT-v6 EE 分布漂移分析.

**目标**: 验证 v5 主结论 (D1 是真主因, 5day_recent 应在真机 work)

**前置条件** (已就绪):
- ✅ 5day_recent ckpt 49999 在 cnbj: `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_vis_5day_recent/pi05_flatten_fold_vis_5day_recent/49999/`
- ✅ Cross val MAE@1 = 0.0086 ⭐ (offline 已最优)
- ✅ G0 fixed-noise inference 修复已部署 sim01

### Step 1 — Ckpt 传输 cnbj → sim01 (~20-30 min)

```bash
# 在 gf3 (cnbj) 上
cd /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_vis_5day_recent/pi05_flatten_fold_vis_5day_recent
# 只传 params + assets + _CHECKPOINT_METADATA (12.5G, 不含 train_state 25G)
tosutil cp -r 49999/params tos://transfer-shanghai/ckpts/vis_5day_recent_49999/params -j 32
tosutil cp -r 49999/assets tos://transfer-shanghai/ckpts/vis_5day_recent_49999/assets -j 16
tosutil cp 49999/_CHECKPOINT_METADATA tos://transfer-shanghai/ckpts/vis_5day_recent_49999/

# 在 sim01 上拉
mkdir -p ~/local_ckpts/vis_5day_recent_49999
cd ~/local_ckpts/vis_5day_recent_49999
tosutil cp -r tos://transfer-shanghai/ckpts/vis_5day_recent_49999/ . -j 32
```

### Step 2 — 启动推理 server (with G0)

```bash
# 在 sim01 上
cd ~/workspace/deepdive_kai0/kai0
export OPENPI_FIXED_NOISE_SEED=0   # G0 fixed-noise 启用
.venv/bin/python -u scripts/serve_policy_v1.py \
  --ckpt-path ~/local_ckpts/vis_5day_recent_49999 \
  ... (其他启动 flags 复用 vis_v2_full 启动脚本) \
  2>&1 | tee /tmp/serve_5day_recent.log

# 验证 log 出现:
#   "G0 fixed-noise inference enabled (seed=0, shape=(50, 32))"
#   "Loaded checkpoint from .../vis_5day_recent_49999"
```

### Step 3 — 真机测试 (重点观察 3 个症状)

| 症状 | vis_v2_full | smooth_800 | **5day_recent 预期** |
|---|---|---|---|
| 走几步退几步 / oscillation | ❌ 有 | ✅ 无 | ✅ **无** |
| 夹爪犹豫 / 反复开合 | ❌ 有 | ✅ 无 | ✅ **无** |
| 夹取后短时间松开 / 衣服脱落 | ❌ 有 | ✅ 无 | ✅ **无** (推测) |
| Long-horizon (折叠完成度) | 差 | 中等 (@50=0.0636) | **应好于 smooth_800** (@50=0.0630 + 后期 supervision) |

### Step 4 — 结果决策树

```
真机测试结果?
├── ✅ 三症状全消 + 折叠流畅
│   → 直接生产部署, 不重训
│   → 实验 F v5 (§4) 成功, 30min total cost
│
├── ⚠️ 三症状消失但 long-horizon 差 (折叠中途卡)
│   → 加 smooth_800 早中期数据补 simple-fold supervision
│   → 走实验 E v5 (1309 ep, 30h)
│
├── ⚠️ 仅部分症状消失 (e.g., 不 oscillate 但仍偶尔松开)
│   → D7 (Class C 32 ep) 在 6.4% 仍有边际影响
│   → 走 "5day_recent + Class C filter (466 ep)" 30h
│
└── ❌ 三症状全在 (类似 vis_v2_full)
    → D1 假说被反驳, 重新审视
    → 检查: G0 是否真生效? ckpt 加载是否对? noise shape (50, 32) 是否对?
    → 可能是 inference 配置问题, 不是数据问题
```

### Step 5 — 反馈到文档

测试后, 在 §8 时间线追加一条:
```
| 2026-05-XX | 真机测试 5day_recent: <结果>. <下一步> |
```

---

## 0.0 ⭐⭐⭐⭐ **决定性证据 v5 — 三角对照 (smooth_800 + 5day_recent + vis_v2_full, 2026-05-27 晚)**

**三个 anchor 模型实测**:

| 模型 | 数据组成 | ep | dates / op | Class C | 5-16 stay-still | Cross val MAE@1 | 真机 |
|---|---|---:|---|---:|:-:|---:|---|
| **smooth_800** | 早中期 + X1 cleaned | 811 | 4-23~5-09 (10 dates, mix op) | ✅ 排 81 | ✅ 排 (天然) | (native 0.0089) | ✅ **闭合稳定, 不 oscillate, 不松开** |
| **vis_5day_recent** ⭐ | 后期 lym | **498** | 5-18~5-22 (5 dates, 全 lym) | ❌ 含 **32** (6.4%) | ✅ 排 (天然) | **0.0086** ⭐ 最优 | ⚠️ (推荐 sim01 但用户未明确真机已测) |
| **vis_v2_full** | 早中期 + 5-16 + 后期 | 1406 | 4-23~5-22 (16 dates) | ❌ 含 **113** (8.0%) | ❌ 含 16 ep | 0.0131 | ❌ **oscillate + 松开** |

**关键对照**:

```
对照 A: smooth_800 (work) vs vis_v2_full (broken)
  差异: vis_v2_full 多 81 Class C + 16 stay-still + 466 后期 + 5 End-snap
  → 多个污染源同时引入, 无法单独定位

对照 B (v5 新增): 5day_recent (含 32 Class C 但 work) vs vis_v2_full (broken)
  差异: vis_v2_full = (smooth_800 + 5day_recent + 5-16 + 5 End-snap), 5day_recent 部分相同
  → 5day_recent 单独含 32 Class C 仍 cross val 最优
  → Class C 32 ep / 6.4% 在小数据集不致命!
  → 推断: vis_v2_full broken 主因 = D1 (5-16 stay-still), 而非单纯 D7 (Class C)

对照 C (v5 新增): smooth_800 (work, 排 Class C) vs 5day_recent (work, 含 Class C)
  → 两者都 work, 共同点: 都不含 5-16 stay-still
  → Class C 阈值 ≤ 6.4% 在 ~500-800 ep 量级可被高一致性数据稀释
```

**v5 修正主因排序**:

| Rank | 根因 | v4 评级 | **v5 评级** | 修正理由 |
|---|---|:-:|:-:|---|
| 1 | **D1** stay-still ideal | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ **真主因** | smooth_800 + 5day_recent 都不含 5-16, 都 work; vis_v2_full 含 16 ep, broken |
| 2 | D7 Class C 跳变 | ⭐⭐⭐ | ⭐⭐ **次要** | 5day_recent 含 32 Class C (6.4%) 仍 cross val 最优, 单独 D7 在 ≤6.4% 不致命 |
| 3 | D8 End-snap | ⭐⭐ | ⭐ | 5 ep / 0.36%, smooth_800 + 5day_recent 都没修 |
| 4 | D9 早期 spike density | ⭐ | ⭐ | smooth_800 含早期但 X1 cleaned, 可能掩盖 |

**新洞察 — 时间连续性 > 全期清洗**:
- smooth_800 (跨 4-23 ~ 5-09 11 dates, X1 cleaned) → 0.0089 native (cross val 未测)
- 5day_recent (5 连续日, **无清洗**) → **0.0086 cross val**, **更优**
- 解读: **5 日连续 + 单一 op (lym) 提供的协议一致性**, **>** 全期数据 + 清洗的多样性 + 漂移
- 与 "杂乱投放 ⭐⭐⭐ 最难场景" 也匹配真机部署分布 (5-18~22 是 lym 后期, README §1 标注最难场景)

**v5 修复路径 (锐化)**:

| Rank | 方案 | 数据 | ETA | 期望 |
|---|---|---|---:|---|
| ⭐⭐⭐⭐⭐ | **最便宜: 直接部署 5day_recent** | (现成 ckpt, cnbj) | **~30 min** ckpt → TOS → sim01 | 如真机 work, 直接生产 |
| ⭐⭐⭐⭐ | 5day_recent + Class C filter (466 ep) 重训 | 466 ep / ~775K frames | 30h | marginal 改进 5day_recent, 验证 D7 边际 |
| ⭐⭐⭐ | **5day_recent + smooth_800 联合 (1309 ep)** | 811 + 498 = 1309 ep / ~1.65M frames | 30h | 加早中期简单场景 supervision, 改 long-horizon |
| ⭐⭐ | 5day_recent + 加 hflip mirror | 498 ep ×1.5 (25% mirror) | 35h | 如果 5day cross val 0.0086 还有空间 |

⚠️ **关键 missing info**: **5day_recent 真机是否已测?** 决定走哪条路径:
- 已测且 work → 直接走 ⭐⭐⭐⭐⭐ (部署), 不重训
- 未测 → 先测 5day_recent 真机 (0h), 再决定

---

## 0.0-archive ⭐⭐⭐ 决定性证据 (v4 — smooth_800 单对照, 2026-05-27 晚)

> ⚠️ **v4 已被 v5 supersede**: v4 推断 D7 主因, 但 v5 引入 5day_recent (含 32 Class C 仍 work) 后, 真主因修正为 D1. v4 内容保留作为推理证据链 reference, 不再是 active 结论.

**smooth_800 真机表现 ≫ vis_v2_full** (用户实测: 闭合稳定 / 不 oscillate / 不松开). 数据组成对比揭露**数学级 root cause**:

```
vis_v2_full 1406 ep = smooth_800 811 ep
                     + 81 ep Class C 跳变 (D7 早中期段, README/X1 应排但 vis 没排)
                     + 16 ep 5-16 stay-still (D1, 应单独用但混入)
                     + 466 ep 后期 lym 干净段 (5-18~22)
                     + 32 ep Class C 跳变 (D7 后期段)
                     + ~5 End-snap (D8)
                     + (可能其他小项)
```

**Smoking gun 计算 (verify 2026-05-27)**:
- vis_v2_full 4-23~5-09 早中期段: **892 ep** (从 vis_v2_full/meta/episodes.jsonl 实测)
- smooth_800 (X1 cleaned): **811 ep** (从 task_a_new_smooth_800_new_norm_results.md)
- **差异: 81 ep**
- Class C 黑名单同期段 (07_classC_blacklist.csv 聚合): **82 ep**
- **Gap: 仅 1 ep** ← 数学等价!

**结论**:
1. **X1 cleanup ≈ README Class C 标准** (|Δ|>0.5 rad 排除整 ep)
2. **vis_v2_full 是从 raw `vis_base_real` 重 build 的, 失去了 smooth_800 已做的 X1 cleanup** — 这是 dataset build pipeline 的关键缺陷
3. **smooth_800 真机 work 完全归因 X1 cleanup (= D7 修复)** — 真因 D7 100% 证实
4. 后期 lym 数据 (5-18~22) 466 ep 是 **net positive** (提供"杂乱投放" supervision), 但被 D1+D7 污染掉

**修复路径锐化**:
- 最优 = `smooth_800 + 后期 lym (5-18~22) X1 cleaned`
- ≈ 811 + 466 = **1277 ep / ~1.7M frames** (减去 End-snap 5 ep 截尾 → 1272)
- 这正是 README §3.3 "纯净 4-29+ ~1322 eps" 方案的核心

**为什么 G0 fixed-noise 部分有效**:
- smooth_800 用 mixed_1_clean init + 50k step, **action distribution 干净** → 闭合 deterministic, 不 oscillate, 不松开
- vis_v2_full 加入 81+32=113 个 Class C 跳变, **学到 spike action mode** → fixed-noise 让 spike deterministic 出现, gripper 被推开 → "夹取后松开"
- vis_v2_full 加入 16 个 stay-still + 同 prompt → "obs(已叠)→stay" mode → "走几步退几步" (G0 部分救)

---

## 0. TL;DR (v3 — README 校准后)

**真机问题三根因合力, 全部能从数据集 README 直接读出**:

| Rank | 根因 | 严重 | 真机症状 | README 直接证据 |
|---|---|:-:|---|---|
| **D7** ⭐⭐⭐ | **Class C 大跳变 129 ep (~9%) 未排除** | 🔴 | G0 后"夹取后短时间松开" / 衣服脱落 | §3.2 "建议训练时 exclude" + `task_a_base_classC_blacklist.csv` |
| **D1** ⭐⭐⭐ | **5-16 ideal 16 ep 混入主 BC 训练 + 同 prompt** | 🔴 | G0 前"走几步退几步" / 夹爪犹豫 | §3.3 "**单独作为状态识别样本, 不混入主 BC 训练**" |
| **D8** ⭐⭐ | **End-snap 5 ep 末段 1-58 帧归零未截** | 🟠 | 可能贡献"突然停止" pattern | §3.2 "建议截掉末段 1~58 帧" |
| D9 ⭐ | 早期 4-23~28 spike 密度 10× 高 (sign-flip 4% vs 后期 2%) | 🟠 | 抖动 prior | §3.1 平滑度对比表 |
| D5 | 有效 epoch 3.3 (vs pure_200 ~40) | ⭐⭐ | 放大其他问题 | (推导) |
| D2 | ep 长度跨 date 2-3× 漂移 | ⭐ | 弱 | README §1 设计意图: 不同场景难度自然导致 |
| D4 | 左右臂不对称 | ⭐ | 弱 | (mirror 因 top head >3cm 偏移不可用) |

**G0 fixed-noise 真机实测 (2026-05-27 PM)**:
- ✅ 闭合次数变多, oscillation 减轻 → noise sampling 是 oscillation 部分驱动 (H1 部分成立)
- ❌ **夹取后短时间松开, 衣服脱落, 回到初始** → 数据里 model 学到了 spike action (D7 Class C 跳变), fixed-noise 救不了"学到的"行为

**核心 insight (README 校准后)**:
- pure_200 真机 work 是因为它来自 **稳定期 05-08/09** (杂乱→整齐场景, ⭐⭐ 难度), spike 密度低 + 无 stay-still + 无 Class C 黑名单 ep
- vis_v2_full 把 stay-still ideal (1.84%) + Class C 跳变 (~9%) + End-snap (5 ep) 全部混入主 BC, **三个数据质量问题都没按 README 建议处理**

**推荐修复 (v3, 按 README 训练建议)**:
1. ⭐ 最便宜验证 (3h): `vis_v2_full ckpt + pure_200 5k step finetune` — pure_200 自然避开三个问题
2. ⭐⭐⭐ **按 README 建议彻底修 (30h)**: 排除 Class C 129 ep + 5-16 stay-still 16 ep + End-snap 5 ep 截尾 → ~1260 ep 重训
3. ⭐⭐ 纯净 4-29+ 版本 (30h): 排除早期 spike-prone dates + 上述三类

**⚠️ 关键校正 (v3 删除 v2 D6)**:
- v2 曾错把"早期 04-23~28 短抓握 41%" 当独立 root cause (D6)
- README §1 揭示: 早期是 **简单叠衣场景** (ep 仅 22-27s), hold 短是任务自然规模, **不是 grip artifact**
- D6 删除, 早期问题真正来源是 D9 (spike 密度) 而非短抓握

---

## 1. 数据集基本信息

| 维度 | 值 | 备注 |
|---|---:|---|
| Total episodes | 1406 | |
| Total frames | 1,927,064 | |
| Dates 数 | 16 (04-23 ~ 05-22) | 1 个月跨度 |
| Operators | 3 | ztm (51%) / lym (46%) / gsy (3%) |
| fps | 30 | |
| 唯一 prompt | 1 | `"Flatten and fold the cloth."` (所有 ep) |
| Mirror augmentation | ❌ 无 | (与 pure_200 200ep 用 50% hflip 形成对比) |
| Success flag | 100% True | 操作员自标, 不代表 trajectory 质量 |
| Note | 99.9% `pedal` | pedal-triggered 录制 |
| **设计意图** | smooth_800 + 5-16 之后数据 | 用户描述: 在 smooth_800 基础上扩展. 但 build 时从 raw 重 build, 失去 X1 cleanup ⭐ |
| **实际 vs 设计** | ❗ 多 81+32=113 Class C + 16 stay-still + 5 End-snap | dataset build pipeline 缺陷, 见 §0.0 |

### 1.1 跨 date / operator 分布

| date | operator | ep | frames | avg_len | median | min/max |
|---|---|---:|---:|---:|---:|---:|
| 04-23 | gsy | 20 | 15,522 | 776 | 711 | 174/1935 |
| 04-24 | lym (147) + ztm (38) + gsy (1) | 186 | 151,280 | 813 | — | — |
| 04-25 | ztm | 100 | 67,539 | 675 | 662 | 355/2559 |
| 04-28 | ztm (151) + lym (1) | 152 | 104,120 | 685 | — | — |
| 04-29 | ztm | 100 | 118,429 | 1184 | 1157 | 769/1868 |
| 04-30 | ztm | 83 | 167,577 | 2019 | 1936 | 1036/**5346** |
| 05-06 | ztm | 100 | 185,446 | 1854 | 1823 | 852/3243 |
| 05-07 | ztm | 20 | 37,430 | 1871 | 1845 | 1050/3073 |
| 05-08 | ztm | 101 | 161,766 | 1602 | 1519 | 978/3332 |
| 05-09 | ztm | 30 | 51,866 | 1729 | 1669 | 1043/2688 |
| **05-16** | **gsy** | **16** | **38,904** | **2432** | **2154** | **1757/4129** |
| 05-18 | lym | 100 | 168,141 | 1681 | 1725 | 756/3178 |
| 05-19 | lym | 99 | 171,687 | 1734 | 1564 | 1078/**5279** |
| 05-20 | lym | 99 | 163,039 | 1647 | 1625 | 1092/2641 |
| 05-21 | lym | 100 | 163,274 | 1633 | 1616 | 1072/2700 |
| 05-22 | lym | 100 | 161,044 | 1610 | 1568 | 990/2337 |

---

## 2. 5 个发现 (按证据强度 + 与真机症状相关性排序)

### D1 ⭐⭐⭐ Stay-still ideal 数据与 fold 数据 prompt 不区分 (强证据, 主因候选)

#### D1.1 事实

**Stay-still 定义**: 整 ep 的 `action_range_max < 1e-3` (机械臂从头到尾完全没动)

Sweep 全 1406 ep 找 stay-still:

| ep_idx | date | operator | length | a_range_max | 备注 |
|---:|---|---|---:|---:|---|
| 892 | 05-16 | gsy | 2205 | 0.0000 | |
| 894-907 (14 ep) | 05-16 | gsy | 1757~3685 | 0.0000 | |
| 297 | 04-25 | ztm | 713 | 0.0000 | **非 05-16, 待确认是否同性质** |

**Near-static (1e-3 ≤ a_range < 0.05)**:
| 274 | 04-25 | ztm | 788 | 0.0020 | |
| 275 | 04-25 | ztm | 2559 | 0.0064 | |
| 893 | 05-16 | gsy | 4129 | 0.0019 | |

**Stay-still + near-static 合计**: 18 ep / 41,243 frames / **2.14% 数据集**

#### D1.2 关键性质

| 性质 | 值 |
|---|---|
| Prompt | `"Flatten and fold the cloth."` (与正常 ep **完全一致**) |
| template_id | `task_a_base` (与正常 ep **完全一致**) |
| Pose | 机械臂停在**折叠完成位置** (L_肘 ≈ -0.46, L_腕pit ≈ 0.75) |
| 正常 ep 末帧 pose 对比 | 05-08 ep 761 末帧 L_肘=-0.47, L_腕pit=0.79 — **几乎重合** |
| 视觉 (推测, 未直接验证) | "已折叠衣服 + 机械臂在保持位" |

#### D1.3 训练中的实际效果 (理论估算)

| 项 | 值 |
|---|---|
| Stay-still frames 数 | 41,243 |
| 占数据集比例 | 2.14% |
| 50k step × 128 batch 总样本 | 6.4M |
| stay-still 被采样次数 (random sampling) | **~137,000 次** |
| 平均每 batch 含 stay-still frame 数 | **~2.74 个** |
| 训练中等价 "教 model 输出 zero action" 的梯度更新次数 | ~50,000 (每次 update 都被 stay-still 影响) |

#### D1.4 与真机症状的因果对应

```
模型学到 mixture mode:
  Mode A (fold):  obs(衣服未叠) → action(继续折)
  Mode B (stay):  obs(已叠完) → action=0

真机执行:
  T=0..N:   obs 接近 "正在折叠中" → Mode A 触发 → 正常动
  T=N+1:    obs transition 到 "接近折叠完成" → Mode A/B mixture
  T=N+2:    state 微小变化 + visual 进一步接近 ideal → Mode B 触发 → action=0 (停)
  T=N+3:    state 自然漂移 + visual 又稍偏离 ideal → Mode A 又触发 → 又动
  ↺ 循环: oscillation / 走几步退几步
```

**对症状的直接解释**:
- "走几步退几步" = mode A ↔ B 之间在 transition 区漂移
- "夹爪无法长期闭合" = gripper trigger 后 visual 接近 ideal → Mode B 触发 → gripper 释放
- "夹爪犹豫" = 同一 visual 帧, mode mixture 让 gripper 在开/合间 漂移

#### D1.5 为什么 G0 fixed-noise 救不了

`policy.infer(obs, noise=FIXED)` 让推理变 deterministic across noise samples, 但 **model output 仍是 (obs, noise) 的函数**。当 obs 进入 mode A/B 边界时:
- Fixed noise + 不同 obs → 不同 mode 选择 (不是 noise 驱动, 是 obs 驱动)
- 真机每 chunk obs 都在变化 (机器在动 + 相机有 jitter)
- → 跨 chunk model 仍在 mode A/B 间跳

#### D1.6 与诊断数据 P2 max=0.67 的对应

诊断显示 vis_v2_full P2 max=0.67 (rare frames, 大部分 frame 是 0.023)。
- **这些 rare high-variance frames 大概率就是 mode A/B 边界帧** — 这是可以验证的, 让 user 跑 diagnostic 时挑 P2 max 的具体 (frame, dim), 对照图像看 visual 是否接近 "折叠完成态"。

#### D1.7 待用户确认

- [ ] 05-16 这 16 ep 的视觉 state — 是 "已折叠平整衣服" 还是 "机器臂悬停在空衣服上方"?
- [ ] 04-25 ep 297 是否也是 ideal? 还是录制故障 / 操作员暂停?
- [ ] 04-25 ep 274 / 275 (a_range 0.002 / 0.006) 是 ideal 还是 setup 阶段?
- [ ] 设计这 16 ep 时, 是否考虑过给它们用 distinct prompt 来区分?

---

### D2 ⭐⭐ Episode 长度跨 date 漂移 2-3 倍 (中证据)

#### D2.1 事实

| 阶段 | 代表 dates | avg_len | 极端 |
|---|---|---:|---|
| 早期 | 04-23 ~ 04-28 | 700-800 frames (~25s) | min 174 (~6s) |
| 中期 | 04-29 ~ 04-30 | 1184-2019 frames | max 5346 (~178s) |
| 后期 | 05-06 ~ 05-22 | 1610-2432 frames | max 5279 (~176s) |

**Top-10 最长 ep** (>3000 frames, 约 ~100s):
- 04-30 ep 569: 5346 frames (~178s)
- 05-19 ep 1015/1037/1025: 5279/4912/4115
- 05-16 ep 893: 4129 (这是 stay-still ep!)
- 04-30 ep 563/558: 3743/3476
- 05-08 ep 851: 3332
- 05-06 ep 684: 3243

#### D2.2 推断

操作员"完整折叠流程" 1 个月内扩展 2-3 倍:
- 早期: 可能只是单次折叠
- 后期: 多次重叠 / 整理 / 修整
- 极端长 ep: 可能含多次失误重做

#### D2.3 与真机症状关系

- model 学到 "任务时长" prior 不锐利
- transition 到 "应该结束" 边界模糊
- 与 D1 stay-still 叠加: 在不确定 "还要继续叠多久" 的时刻, mode B (stay) 容易插入

#### D2.4 待用户确认

- [ ] 长 ep (>3000 frames) 是不是真的"完整任务", 还是含 setup / 等待 / 操作员调整时段?
- [ ] 是否要 filter 异常长 ep?

---

### D3 ⭐ 跨 date norm shift 5-8° (弱-中证据)

#### D3.1 事实

(排除 18 stay-still ep 后) 每 date 抽 5 ep 算 action mean 跨 date std:

| dim | label | 跨 date std (rad) | 跨 date std (°) |
|---:|---|---:|---:|
| 0 | L 肩yaw | 0.061 | 3.5° |
| 1 | L 肩pit | 0.080 | 4.6° |
| 2 | L 肘 | 0.087 | 5.0° |
| **3** | **L 腕yaw** | **0.146** | **8.4°** |
| 4 | L 腕pit | 0.054 | 3.1° |
| 5 | L 腕rol | 0.113 | 6.5° |
| 6 | L grip | 0.004 | — |
| 7 | R 肩yaw | 0.093 | 5.3° |
| **8** | **R 肩pit** | **0.113** | **6.5°** |
| **9** | **R 肘** | **0.133** | **7.6°** |
| 10 | R 腕yaw | 0.093 | 5.3° |
| 11 | R 腕pit | 0.097 | 5.6° |
| 12 | R 腕rol | 0.074 | 4.2° |
| 13 | R grip | 0.004 | — |

#### D3.2 推断

- joint mean 跨 date 漂移 5-8°, 不算极大
- gripper σ=0.004 几乎一致 (operator behavior 在 gripper 上很统一)
- **per-dataset norm 可吸收大部分** (vis_v2_full 用全数据集合并 norm), **不是主因**

---

### D4 ⭐ 左右臂强不对称 + hand-dominance bias (弱证据, 但 mirror 路线已排除)

#### D4.1 事实

(抽 20 ep / 28874 帧 concat)

| dim | L μ | R μ | \|L+R\| (镜像下应≈0) | L σ | R σ |
|---|---:|---:|---:|---:|---:|
| 0 肩yaw | -0.076 | 0.041 | 0.035 | 0.213 | 0.236 |
| 1 肩pit | 1.329 | 1.456 | **2.786** | 0.582 | 0.585 |
| 2 肘 | -1.175 | -1.324 | **2.499** | 0.432 | 0.600 |
| 3 腕yaw | -0.075 | 0.331 | 0.256 | 0.424 | 0.302 |
| 4 腕pit | 0.727 | 0.872 | **1.599** | 0.258 | 0.226 |
| 5 腕rol | 0.101 | -0.215 | 0.113 | 0.406 | 0.263 |
| 6 grip | 0.029 | 0.021 | 0.008 | 0.029 | 0.029 |

#### D4.2 解读

- 镜像对称下 |L+R| 应近 0, 实测 1.6-2.8 — 数据强不对称
- 大概率原因: 操作员惯用手 (右手主导 fold 动作)
- **与 oscillation 关系不直接**

#### D4.3 ❌ Mirror augmentation 在 vis_v2_full 上**不可用** (user 2026-05-27 决定)

**硬件约束**: top head 相机相对于双臂中线偏移 **> 3cm / > 3°** (用户实测).

**Mirror 后 visual artifact 强度评估**:

| top head 偏移 | mirror visual artifact | 是否可用 |
|---|---|---|
| < 1 cm / < 1° | 噪声级, model 可吸收 | ✅ 50% mirror 可加 |
| 1-3 cm / 1-3° | 轻微 systematic bias | ⚠️ 推荐 20-25% 低比例 |
| **> 3 cm / > 3°** ⭐ vis_v2_full 实际 | **显著 systematic bias, mirror image 在真实分布外** | ❌ **跳过 image mirror** |

**结论**:
- vis_v2_full 不加 mirror augmentation
- pure_200 100 ep 当时用 mirror 是因为 **数据少不得不扩增**, 即使 visual artifact 也接受 — 但 1400 ep 数据量下, 数据扩增收益 < visual artifact 代价
- D4 hand-dominance bias 仍存在, 但**不能用 mirror 解决**
- 替代路径: (a) 数据采集时刻意平衡左右手主导任务 (long-term); (b) 用其他 augmentation (ColorJitter / RandomResizedCrop, 不依赖左右对称); (c) 接受 hand-dominance bias 作为 model 固有特性 (反正真机部署也是同一台机器)

#### D4.4 与 oscillation 的关系 (彻底解耦)

D4 在文档里保留作为数据画像 reference, **但与真机 oscillation 修复路径完全无关** — mirror 路线已排除, hand-dominance bias 不修也不影响 stay-still ideal mode mixture (D1) 的修复。

修复路线中 (实验 A/B/C) **完全不加 mirror**。

---

### D7 ⭐⭐⭐ **(v3 主因)** Class C 大跳变 129 ep 未排除 → "夹取后短时间松开"

#### D7.1 README + Analysis CSV 直接证据

`README §3.2`:
> **Class C 中段大跳变** (`\|Δ\|>0.5 rad`): **129 eps (~9%)**, 多数为 CAN 包丢失/状态突变, 建议训练时 exclude. 详见 [`analysis/07_classC_blacklist.csv`](analysis/07_classC_blacklist.csv)

CSV 结构: `date, ep, T, n_mid_spike, worst_val_mid, worst_joint` (每行一个 ep)

**Per-date 分布** (从 `07_classC_blacklist.csv` 聚合):

| date | n | date | n | date | n |
|---|---:|---|---:|---|---:|
| 04-23 | 1 | 04-30 | 20 | 05-19 | 5 |
| **04-24** | **22** ❗ | 05-06 | 10 | 05-20 | 8 |
| 04-25 | 3 | 05-07 | 2 | 05-21 | 6 |
| 04-28 | 8 | 05-08 | 10 | 05-22 | 10 |
| 04-29 | 3 | 05-09 | 3 | 05-26 | 6 |
| | | 05-18 | 3 | 05-27 | 9 |

**Total: 129 ep 跨 17 dates 全分布** — 不是早期独有。早期 4-24 占最多 (22 ep), 整体~9% 数据。

`README §3.3`:
> 完整训练: 全集 1682 eps **减去 Class C 黑名单 129 ep = 1553 eps**

**vis_v2_full 训练现实**: 未排除, 全量混训 → 9% 数据带 CAN 跳变进入主 BC supervision。

#### D7.1b 跳变样例 (worst_val_mid)

| ep | spike 帧 | worst joint | 跳变值 (rad) | 相当于 |
|---|---|---|---:|---|
| 05-20 ep067 | 1 | R_j2 (R 肘) | **1.84** | **105°** 单帧跳! |
| 05-08 ep042 | 1 | L_j2 (L 肘) | 1.82 | 105° |
| 04-29 ep023 | 1 | L_j2 | 1.73 | 99° |
| 04-24 ep116 | 1 | L_j4 (L 腕pit) | 1.71 | 98° |

**物理常识**: 30Hz 单帧 33ms, Piper 关节最快 ~3 rad/s, 单帧 max ~0.1 rad (5.7°). **CSV 里这些 100°+ 跳变绝对是 CAN 丢包**。

#### D7.2 训练效果机理

```
单帧 |Δaction| > 0.5 rad = 单帧 ~29° 跳变 (joint), 物理上不可能 (30Hz 单帧 33ms)
→ 这些是 CAN 包丢失 / state spike artifact
→ model 学到 "obs[t] → action 可以瞬间跳 29°"
→ flow matching 在 denoising 时, 这些 spike supervision 会污染 action distribution
→ 推理时 model 输出有概率出现 spike action
```

#### D7.3 与 G0 后真机症状的精确对应

**症状**: G0 后 "完全闭合夹取的情况变多了, 但是依然夹取没多久夹爪就松开导致衣服脱落"

**机理**:
- G0 让 base noise variance 消失 → grip 输出 deterministic 闭合
- 但 Class C 跳变 supervision 让 model 在某些 chunk **预测一个 spike action** (其中可能包含 grip 释放)
- gripper 在闭合状态下被 spike 推到 0.06 (打开) → 衣服脱落
- 物体脱落后 obs 重置 → 下一 chunk model 又进入"去抓"模式 → 循环

**为什么 G0 fixed-noise 救不了**: spike 是**训练 supervision 直接教的 action**, 不是 noise sampling 引起的随机性。fixed-noise 让 model deterministic 表达学到的 mode, 包括"可以输出 spike"这个 mode。

#### D7.4 Pure_200 为什么没有这个问题

pure_200 来自 05-08/09 ("杂乱→整齐"场景, ⭐⭐ 难度, **4-29 后采集流程改进期**):
- README §3.1: 4-29 后 spike/kFrame 从 45-75 降到 3-6 (**10× 降低**)
- pure_200 source 大概率不在 Class C 黑名单
- → model 学到的 action distribution **没有 spike supervision**

#### D7.5 修复优先级

⭐ **最高优先级** — README 已明确建议排除, 几乎零工程成本 (用现成 CSV).

---

### D8 ⭐⭐ End-snap 5 ep 末段 1-58 帧归零 artifact

#### D8.1 README + CSV 直接证据

`README §3.2`:
> **End-snap** (末尾归零 artifact): **5 eps** — 录制延续到 teleop 释放后, 建议截掉末段 1~58 帧. 详见 [`analysis/06_end_snap_trim.csv`](analysis/06_end_snap_trim.csv)

CSV 全清单 (5 行):

| date | ep | T_orig | snap_t | trim_n | T_new | pre_arm_abs | post_arm_mean | jump_val |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 04-24 | ep116 | 856 | 799 | **57** | 799 | 1.251 | 0.383 | 1.706 |
| 04-28 | ep102 | 665 | 622 | 43 | 622 | 1.032 | 0.615 | 0.36 |
| 04-28 | ep104 | 519 | 487 | 32 | 487 | 1.027 | 0.512 | 0.404 |
| 04-29 | ep023 | 1517 | 1459 | **58** | 1459 | 1.03 | 0.291 | 1.73 |
| 04-29 | ep098 | 791 | 790 | **1** | 790 | 1.109 | 0.267 | 2.171 |

**判定标准** (INDEX.md §阈值): 末段最后一次大跳变后 ≤90 帧均 `arm_joints |mean| < 1.0`

#### D8.2 机理

teleop 释放后录制延续 1-58 帧, 这段 action 归零 / 跳到 reset pose.
- model 学到 "在某些 obs 后 action 应该突然归零"
- 真机执行可能触发 "瞬间停止 / 突然 reset"

#### D8.3 与症状相关性

- 中等严重性 (5 ep 很少, 占比 0.36%)
- 但 末段是**任务完成附近**, model 在 "接近完成" 的 obs 上看到"action 突然归零" — 与 D1 stay-still 协同放大 "完成附近行为异常"
- 推荐顺手修, 与 D7/D1 同时排除

#### D8.4 修复

按 README 截掉末段 1-58 帧 (用现成 CSV), 或直接删除这 5 ep。

---

### D9 ⭐ 早期 4-23~28 spike 密度 10× 高 + sign-flip 4% (README §3.1)

#### D9.1 README + Analysis CSV 直接证据

`README §3.1 平滑度`:
> **4-29 是关键分水岭** — 4-29 之后采集流程改进, spike 密度降低约 10×, 高频抖动 (sign-flip 比例) 从 4.0% 降至 2.0%.

实际 per-date 数据 (从 [`analysis/03_instability_summary.csv`](analysis/03_instability_summary.csv)):

| date | spike_arm /kFrame | flip_ratio (sign-flip 比例) | static_runs (内部静止段) |
|---|---:|---:|---:|
| 04-23 | 45.8 | 3.97% | 1 |
| **04-24** | **44.0** ❗ | 3.70% | 28 |
| **04-25** | **70.7** ❗ | 4.12% | 5 |
| **04-28** | **75.2** ❗ 最高 | 4.10% | 12 |
| 04-29 ⭐ 分水岭 | 3.29 | 2.28% | 89 |
| 04-30 | 5.96 | 2.08% | 22 |
| 05-06 | 3.43 | 2.01% | 26 |
| 05-08 (pure_200 源) | 6.09 | 2.20% | 79 |
| 05-09 (pure_200 源) | 2.74 | 2.01% | 10 |
| **05-16 ideal** | 0.00 ✅ | 0% ✅ | 17 (全 ep static) |
| 05-18 | 5.31 | 2.37% | 29 |
| 05-19 | 3.33 | 2.27% | 103 |
| 05-20 | 4.36 | 2.28% | 109 |
| 05-21 | 3.69 | 2.32% | 168 |
| **05-22** | 2.75 | 2.50% | **192** ❗ 高 static_runs |
| 05-26 | 0.72 | 2.37% | 170 |
| 05-27 | 0.50 | 2.29% | 117 |

**关键观察 (CSV 实证)**:
- **04-25/28 spike_arm/kF = 70-75 是早期最严重段** (4-29 后降到 3-6, 22× 改善)
- 04-23~28 sign-flip ratio 3.7-4.1% 与 4-29+ 的 2.0-2.5% 相比 **几乎 2×** 高频抖动
- **05-21+ static_runs 跃升到 109-192** — 后期数据有大量"内部停顿段", 可能对应"操作员暂停思考"或"等待 cloth settle"
  - 与 5-16 ideal (static_runs=17, 全 ep static) 不同 — 后期 ep 是部分 static, 不是全静止

#### D9.2 与 v2 D6 的关系 (v3 校正)

> ⚠️ **v2 文档曾把"04-23~28 短抓握 41%" 当 root cause (D6 ⭐⭐⭐), 这是误判**
>
> 实际:
> - 早期 ep 仅 22-27s (README §1 "简单叠衣场景", ep 短是任务本身规模小)
> - hold duration 短是 ep 短的副产品, 不是 grip artifact
> - 真正问题是 **spike 密度** (D9), 不是 "短抓握"

#### D9.3 修复

- 与 D7 协同: Class C 黑名单已 cover 部分早期数据
- 更彻底: 按 README §3.3 "纯净版" 建议, 只用 4-29+ 数据 (~1322 ep, ~17h) → 实验 C v2

---

### 🔍 pure_200 vs vis_v2_full 对比 (v3 校准)

| 项 | pure_200 (work) | vis_v2_full (broken) |
|---|---|---|
| 训练数据时段 | 仅 2 dates (05-08/09, **4-29+ 流程改进期**) | 16 dates (含 04-23~28 高 spike 期) |
| Stay-still ideal (D1) | 0 ✅ | 18 ep / 1.84% ❌ |
| **Class C 大跳变 (D7)** | **0% (推断, 4-29+ 数据)** | **~9% (129 ep) ❌** |
| **End-snap (D8)** | 0 ✅ | 5 ep ❌ |
| **Spike density (D9)** | 3-6 / kFrame ✅ | 含 4-23~28 区段 45-75 /kF ❌ |
| 短抓握 (<2s) 比例 | 23.5% (任务 rhythm) | 26.1% (含早期短 ep 比例, **不是 artifact**) |
| 长抓握 (>5s) 比例 | 49.1% | (推断) ~45% |
| Hold 中位数 | 147 帧 | 139 帧 |

→ pure_200 之所以 work, **是因为它的 source dates (05-08/09) 自然规避了 D1/D7/D8/D9 四大数据质量问题**, 不是因为 "200 ep 数据少所以学锐利"

> ⚠️ **v2 → v3 校正**: D6 "短抓握污染" 误判已删除 — 早期 hold 短是 ep 长度短的副产品 (README §1 简单场景 22-27s), **不是 grip artifact**. 真正问题是 D9 (spike 密度) + D7 (Class C 跳变).

---

### D5 ⭐⭐ 有效 epoch 仅 3.3 — rare-event under-supervision (推导证据)

#### D5.1 数据 vs supervision 强度对比

| 模型 | 唯一 frames | step × batch | 有效 epoch | rare frame (~5%) 出现次数 |
|---|---:|---:|---:|---:|
| pure_200 | ~150k | 50k × 120 = 6M | ~40 epoch | 6M × 5% = **~300,000** |
| **vis_v2_full** | **1.93M** | 50k × 128 = 6.4M | **3.3 epoch** | 6.4M × 5% = ~320,000 (但分散到 12.9× 多 frame) |

#### D5.2 关键 insight

绝对数 (rare frame 总出现次数) 相近, 但 vis_v2_full 把这 320k supervision **分散到 12.9× 多 frame**, 每 frame **平均梯度更新次数 ≈ pure_200 的 1/13**。

→ 单 rare frame 的 supervision 不够 → multi-modal collapse 严重 → 与诊断 P2 max=0.67 完全一致

#### D5.3 修复

- 加训 step (100k vs 50k): 6.6 epoch, 仍少
- + mirror aug (×2 dataset): 等效 6.6 epoch on 3.86M frames
- + filter 长 ep: 减少 dataset 但提高质量比

---

## 3. 根因排序 (v5, 2026-05-27 晚, 5day_recent 三角校准)

| Rank | 编码 | 根因 | 严重 | G0 后症状阶段 | 修复方案 | 来源 | ETA |
|---|---|---|:-:|---|---|---|---:|
| 1 | **D1** ⭐⭐⭐⭐⭐ | 5-16 stay-still ideal 混入主 BC + 同 prompt | 🔴 **真主因** (v5 升级) | G0 前 oscillation + G0 后残留 | F1: 排除 / F1b: distinct prompt | 三角对照: smooth_800 + 5day_recent 都不含, 都 work | 同 D7 |
| 2 | **D7** ⭐⭐ | Class C 大跳变 113 ep (8%) 未排除 → spike action | 🟠 次要 (v5 降级) | 可能贡献 "夹取后松开" 但非必要 | F7: 用 `07_classC_blacklist.csv` 排除 | 5day_recent 含 32 (6.4%) 仍 work → 单独 D7 在 ≤6.4% 不致命 | 30h |
| 3 | **D8** ⭐ | End-snap 5 ep 末段 1-58 帧归零 | 🟠 微 | 顺手修 | F8: 用 `06_end_snap_trim.csv` 截尾 | smooth_800 + 5day_recent 都没修也 work | 同 F7 |
| 4 | **D9** ⭐ | 早期 4-23~28 spike 密度 10× | 🟠 微 | 与 D7 协同 | F9: 纯净 4-29+ 版本 | smooth_800 X1 cleaned 后无显著 | 30h |
| 5 | **D5** | 有效 epoch 3.3 | ⭐⭐ | 放大其他 | F4: 100k step | 5day_recent 50k × 128 / 827k = 7.7 epoch, 更高 → cross val 更优 | 60h |
| 6 | **D2** | Ep 长度跨 date 2-3× | ⭐ | 弱 | n/a | README §1 设计差异 | n/a |
| 7 | **D4** | 左右臂不对称 | ⭐ | 弱 | ❌ Mirror 不可用 (top head >3cm) | (内部测量) | n/a |
| 8 | **D3** | 跨 date norm shift 5-8° | ⭐ | per-dataset norm 吸收 | n/a | (内部测量) | n/a |
| ~~D6~~ | ❌ 误判 (v2) | "短抓握污染" | — | — | 删除 | v3 校正 | — |

**v5 关键变化**:
- **D1 升级为唯一真主因** ⭐⭐⭐⭐⭐ — 三角对照证实 (5day_recent 不含 5-16 也 work)
- **D7 从主因降为次要** — 5day_recent 含 32 Class C (6.4%) 仍最优, 单独 D7 不致命
- 修复路径: 优先排 5-16 (D1), Class C 顺手修 (锦上添花)

**新发现 (v5)**:
- **D10 (新)**: 时间连续性 + 单 operator (5day_recent 5 连续日 全 lym) **>** 全期清洗 + 多 op 多月 (smooth_800)
- 5day_recent 有效 epoch 7.7 (498 ep 数据 50k step) vs vis_v2_full 3.3 → 数据量精选 + step 充足 = best 配方

---

## 4. 修复路线

### 实验 A — 最便宜验证 D1 (~3h, 推荐先做) ⭐

**方案**: vis_v2_full ckpt + pure_200 100ep 5k step finetune

**机理**:
- pure_200 训练数据**完全没有 stay-still 样本**
- 5k step finetune 把 vis_v2_full 学到的 stay-still mode "用 pure_200 的连续 fold supervision 压制下去"
- 模型仍保留 vis_v2_full 的 broad visual prior, 但 action mode 锐化为 "always fold"

**期望**:
- 真机 oscillation **完全消失** → D1 确认为主因
- 真机仍 oscillate → D1 不是主因 (或 obs jitter H9 / OOD H2 才是)

**Config**:
```python
TrainConfig(
    name="pi05_vis_v2_full_pure200_finetune_5k",
    model=Pi0Config(pi05=True),
    weight_loader=CheckpointWeightLoader(
        "cnbj:/vePFS-North-E/.../pi05_flatten_fold_vis_v2_full/49999/params"
    ),
    data=LerobotAgilexDataConfig(
        repo_id=".../A_new_pure_200",   # pure_200 data
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,
    ),
    num_train_steps=5_000,
    batch_size=128,
    peak_lr=5e-6,           # 低 LR finetune (避免破坏 broad prior)
    lr_schedule=cosine_warmup_1k_decay_5k,
    ema_decay=0.9999,
)
```

**Resource**: 8 H20 单节点, ~3h

### 实验 B v3 — 按 README §3.3 训练建议彻底修 D7+D1+D8 (~30h) ⭐⭐⭐ **推荐主线**

**方案**: 严格按 README + analysis CSV 排除三类异常 ep:

| # | 处理 | 数据源 | ep 数 |
|---|---|---|---:|
| 1 | 排除 **Class C 黑名单** (D7) | `analysis/07_classC_blacklist.csv` | 129 |
| 2 | 排除 **5-16 stay-still ideal** (D1) | (16 ep date=2026-05-16-v2) | 16 |
| 3 | 截尾 **End-snap** trim_n 帧 (D8) | `analysis/06_end_snap_trim.csv` | 5 |

**Dataloader 集成代码** (INDEX.md 提供):
```python
import pandas as pd
ANALYSIS_ROOT = "/transfer-shanghai/KAI0/Task_A/base/analysis"

# Stage 1: 排除 Class C 129 ep + 5-16 stay-still 16 ep
bl = pd.read_csv(f"{ANALYSIS_ROOT}/07_classC_blacklist.csv")
exclude_classC = set(zip(bl.date, bl.ep))
exclude_stay = {("2026-05-16-v2", f"episode_{i:06d}.parquet") for i in range(892, 908)}  # ep 892~907 全 16 ep

# Stage 2: End-snap 5 ep 应用 trim
trim = pd.read_csv(f"{ANALYSIS_ROOT}/06_end_snap_trim.csv")
trim_map = dict(zip(zip(trim.date, trim.ep), trim.T_new))

# 在 build_vis_v2_full.py 里 filter:
for src_date, src_ep_parquet in ...:
    key = (src_date, src_ep_parquet)
    if key in exclude_classC: continue
    if key in exclude_stay: continue
    T_keep = trim_map.get(key, None)  # None = 不截尾
    # ... rebuild parquet with [:T_keep] if specified
```

**Dataset 变化**:
- 原 base 1682 ep (vis_v2_full 是 base 子集, 1406 ep ≈ base 含 5-26/5-27 共 1682 减去 5-26/5-27 的 176 ep + 04-23 加权差异)
- 注: **vis_v2_full 1406 ep ≠ README base 1682 ep**, 需先做 mapping (build_vis_v2_full.py 已存 `_src_dir` + `_src_idx` 字段)
- 估算 vis_v2_full 内排除: ~108 Class C (假设按比例) + 16 stay-still + 5 截尾 ≈ -10% 数据
- 剩余 **~1280 ep / ~1.74M frames** (估算)

**Hparams**: 与 vis_v2_full 完全一致 (peak_lr 1.5e-5, 50k step, batch 128, EMA 0.9999, pi05_base init)
- 有效 epoch = 50k × 128 / 1.74M = **3.68 epoch** (与原 3.3 接近)

**期望**:
- D7 主修: spike action 消失 → 真机"夹取后松开"消失 (G0 后剩余主症状)
- D1 主修: stay-still mode 消失 → 真机"走几步退几步" 消失 (G0 前主症状)
- D8 顺手修: 末段归零 artifact 消失
- val MAE 可能略升 (数据少 10%) 但**真机三大症状应全消**
- 与 G0 fixed-noise 同时启用最佳

### 实验 C v3 — 纯净 4-29+ 版本 (~30h) ⭐⭐

**方案**: 按 README §3.3 "纯净版" 建议, 只用 4-29 之后数据 (D9 + 上述三类全排除)

**Dataset 变化**:
- 排除 04-23 ~ 04-28 早期 dates (~459 ep)
- 排除 Class C 中后期残余 (~50-80 ep)
- 排除 5-16 stay-still (16 ep)
- 排除 End-snap (5 ep)
- 剩余 **~850 ep / 1.43M frames**

**有效 epoch**: 50k × 128 / 1.43M = **4.48 epoch**

**期望**: 数据最纯净, 真机表现期望最好. trade-off 是数据量 -40%, 可能 generalization 略弱 (未必 — README 推荐这个配置).

⚠️ **风险**: D7/D8/D9 修了, 但如果 4-23~28 数据中包含某些**长尾场景** (简单叠衣 ⭐ 难度), model 可能在真机简单场景下泛化弱化.

### 实验 F v5 — 直接部署 5day_recent (现成 ckpt, ~30 min) ⭐⭐⭐⭐⭐⭐ **v5 第一推荐**

**前提**: 5day_recent ckpt 已存在 cnbj (`/vePFS-North-E/.../vis_5day_recent/49999/`), cross val 已最优 (0.0086).

**步骤**:
1. cnbj ckpt → TOS (~10 min): `tosutil cp -r /vePFS-North-E/.../vis_5day_recent/49999 tos://transfer-shanghai/checkpoints/vis_5day_recent_49999/`
2. sim01 拉取 (~10 min)
3. sim01 部署 + G0 fixed-noise 启用 (`OPENPI_FIXED_NOISE_SEED=0`)
4. 真机测试

**期望**:
- Cross val 已是最优 (0.0086 vs vis_v2_full 0.0131 vs pure_200 0.0207)
- D1 (5-16) 天然不含, oscillation 应消失
- Class C 32 ep 在 6.4% 比例下可控 (smooth_800 排了仍 work, 但 5day_recent 含也 work → 阈值)
- 真机应 ≥ smooth_800 (闭合稳定, 不松开), 且 long-horizon 比 smooth_800 好 (cross val @50 0.0630 vs smooth native @50 0.0636)

⚠️ **如果真机 work** → 不需要重训, 节省 30h
⚠️ **如果真机 still 有问题** → 走实验 E v5 (1309 ep 联合)

---

### 实验 E v5 — 5day_recent + smooth_800 联合 (1309 ep, ~30h) ⭐⭐⭐⭐

**适用**: 实验 F (直接部署 5day_recent) 真机不充分时, 加 smooth_800 早中期补 long-horizon supervision.

**方案**: smooth_800 数据组成 + 后期 lym (5-18~22) — 见 §0.0 三角对照

**Dataset 组成**:
```
= smooth_800 数据 (811 ep, X1 cleaned 早中期 4-23~5-09)
  + vis_v2_full 5-18~22 段 (498 ep) - Class C 32 ep = 466 ep
  - End-snap 5 ep 截尾 (用 06_end_snap_trim.csv)
  - 5-16 stay-still 16 ep (D1, 已天然不含)
= ~1272 ep / ~1.65M frames
```

**Init**: mixed_1_clean (与 smooth_800 一致, 节省 24k step Task_A warmup)
- 或 pi05_base (与 vis_v2_full 一致, 直接对照)

**Hparams**: 同 vis_v2_full (peak_lr 1.5e-5, 50k step, batch 128, EMA 0.9999, fsdp 16)
- 有效 epoch = 50k × 128 / 1.65M = **3.88 epoch** (与原 3.3 接近)

**期望**:
- ✅ 真机三大症状全消 (smooth_800 已验证: 闭合稳定 / 不 oscillate / 不松开)
- ✅ Long-horizon (@50) **比 smooth_800 改善** (因为加了后期 lym 466 ep, 多 57% 数据 + 杂乱投放场景 supervision)
- ✅ Offline MAE@1 应 ≤ smooth_800 0.0089 (更可能 0.0070~0.0080)
- 部署直接替换 vis_v2_full ckpt

**Resource**: 8 H20 单节点 30h, 或 16 H20 ~15h

**关键 build script 改动** (相对于 build_vis_v2_full.py):
```python
import pandas as pd
ANALYSIS = "/transfer-shanghai/KAI0/Task_A/base/analysis"
bl = pd.read_csv(f"{ANALYSIS}/07_classC_blacklist.csv")
exclude_classC = set(zip(bl.date, bl.ep))   # 129 ep, 但 vis_v2_full 范围内有效 114
exclude_dates = {"2026-05-16-v2"}             # D1 ideal
trim = dict(zip(zip(pd.read_csv(f"{ANALYSIS}/06_end_snap_trim.csv").date,
                    pd.read_csv(f"{ANALYSIS}/06_end_snap_trim.csv").ep),
              pd.read_csv(f"{ANALYSIS}/06_end_snap_trim.csv").T_new))

for src_date in DATES:
    if src_date in exclude_dates: continue
    for src_ep_parquet in glob...:
        key = (src_date, src_ep_parquet)
        if key in exclude_classC: continue
        T_keep = trim.get(key, None)  # None = 全保留
        # write [:T_keep] if specified else full
```

---

### 实验 D — 终极版 (修全部 + 100k step + distinct prompt, ~60h) ⭐⭐⭐⭐ (v3)

**方案**: 在 B v3 基础上 + D9 (纯净 4-29+) + D1 改 distinct prompt + 100k step

**Dataset 重 build**:
- 5-16 stay-still 16 ep prompt 改为 `"Hold still at folded pose."` (而不是删除, 保留作辅助任务)
- 排除 Class C 黑名单 (D7)
- 排除 / 截尾 End-snap (D8)
- 排除 4-23 ~ 4-28 (D9, 早期 spike)
- 剩 ~850 ep / 1.43M frames

**训练**:
- 100k step × 128 / 1.43M = **8.95 effective epoch** (3× D5 改善)

**期望**:
- 训练 val MAE 接近或好于原 vis_v2_full 0.0131
- 真机三症状全消, 同时 model 学到 "prompt distinct ideal" 作 boundary detector
- D4 hand-dominance bias 仍存在但不修 (接受作为 model 固有特性)

**Resource**: Robot-North-H20 16 H20, ~60h

---

## 5. 推理侧 G0 失败的对应解释

### 为什么 G0 fixed-noise 无效

G0 修复 假设是 **chunk 之间 noise sample 不同导致 mode 跳变** (H1)。但实际机理:

| 假说 | G0 fix 有效吗? |
|---|---|
| H1: noise sample 驱动 mode 跳变 | ✅ 有效 (P2 → 0) |
| **D1: obs 驱动 mode 跳变 (stay vs fold mixture)** | ❌ **无效** (obs 仍变化) |

实测真机失败 → **D1 才是真因, H1 只是表面现象**。

### 与 P2 max=0.67 的对应

P2 max=0.67 出现在 rare frames, 大概率就是 mode A/B 边界帧 (visual 接近 "折叠完成" 但还没真到 ideal)。让 user 验证: 跑 P2 诊断时挑 max variance 的 (frame, dim), 打印对应 frame_index 与 ep, 然后看 video 该时刻视觉是否接近"折叠完成态"。

**如果验证成立** → D1 主因确认, 走实验 A / B / C
**如果验证不成立** (P2 max 来自其他 frame) → D1 不是主因, 重新分析

---

## 6. 关键洞察 + Lessons (v5 更新)

1. **(v5) 三角对照打破单变量假说** ⭐⭐⭐⭐ — v4 以为 "vis_v2_full - smooth_800 = 81 Class C → D7 是主因". 但 v5 加入 5day_recent (含 32 Class C 仍 work) 后, 这个推论被打破. **多 anchor 三角对照** 是辨别 confounder 的唯一方法. 单对照 (smooth vs vis) 把 D1+D7 confound 在一起, 加 5day_recent 才能分离.

2. **(v5) Stay-still ideal 是 silent killer** ⭐⭐⭐⭐ — D1 看起来只 16 ep / 1.84%, 但因为它 prompt 与正常 fold 完全重叠, model 学到 mode mixture. 这种"小占比但语义冲突"的污染比"大占比但语义同向"的 Class C 更致命. **prompt distinction 是数据集 maintainer 必须显式标注的, 不能依赖训练者自己注意**.

3. **(v5) 时间连续性 > 全期质量清洗** ⭐⭐⭐ (D10 新) — 5day_recent 5 连续日 + 单 op (lym) + **无清洗** > smooth_800 全期 + X1 清洗. 启示: 数据集设计上, **缩窄时间窗 + 锁定 op** 比扩大时间窗 + 清洗更省力且更有效 (对单一部署场景而言).

4. **Dataset build pipeline 必须 inherit cleanup, 不能从 raw 重 build** ⭐⭐⭐ (v4) — vis_v2_full 是从 raw vis_base_real 重 build 的, **失去了 smooth_800 已做的 X1 cleanup**. 下次 build derived dataset 应 inherit upstream cleanup metadata.
2. **数据集 README 是 ground truth** — 自己扫数据猜测的"短抓握污染" (v2 D6) 被 README §1 推翻 (是简单场景 ep 短). 以后做数据 audit **先读 maintainer README**, 再扫数据 verify, 不要反过来.
3. **对照实验有数学级证据** ⭐⭐ (v4) — smooth_800 (811 ep, X1 cleaned, 真机 work) 与 vis_v2_full 早中期段 (892 ep, raw) 的差 81 ep ≈ Class C 黑名单 82 ep, **gap 仅 1 ep**. 这种数学级 anchor 比任何理论推断更有力, 优先找 ground truth 对照点.
4. **Offline MAE ≠ 真机表现**: vis_v2_full 0.0131 比 pure_200 cross val 0.0207 好 36%, 但真机更差. MAE 不捕捉 spike action / mode mixture.
5. **数据规模大了反而出问题**: 不是数据本身的问题, 而是数据**质量过滤**没跟上. README 明确建议排除 9% Class C 黑名单, 但 vis_v2_full build script 没读这个建议.
6. **prompt 是 model 的唯一显式区分通道**: 5-16 ideal 与 fold 用同 prompt → mode mixture. README 推荐"不混 BC", 实际可以 distinct prompt 保留数据.
7. **pure_200 work 的真正原因 (v3)**: 不是"数据少所以锐利", 是它的 source dates (05-08/09) **自然落在 4-29+ 流程改进期**, 自动规避 D7/D8/D9 三个数据问题.
8. **G0 fixed-noise 是必要不充分**: 它修了 noise variance (oscillation 部分), 但**数据里学到的 spike action** (D7) 它管不了.
9. **真机症状 → 根因映射的因果链需要分层**:
   - 表层 (G0 前): noise variance 主导 oscillation
   - 中层 (G0 后): spike supervision 导致主动释放
   - 深层 (无论 G0): mode mixture (stay-still ideal) 导致 transition oscillate
   - **三个根因独立但叠加**, 修复需多管齐下

---

## 7. 决策状态 (v5 精简)

### 🚀 待执行 (Next Action)
- **真机测试 5day_recent** ckpt (见 §0.NEXT checklist) — 决定走 F (直接部署) vs E (1309 ep 重训) vs 其他

### ✅ 已决策 / 已 README+Analysis 解答
1. **5-16 ideal 数据设计意图** — README §3.3: 状态识别辅助任务样本, **不应混入主 BC**
2. **早期 hold 短不是 anomaly** — README §1: 早期是简单场景 22-27s (v2 D6 误判已删除)
3. **Mirror augmentation** — ❌ 不加 (top head 偏移 >3cm/>3°, visual artifact)
4. **Class C / End-snap CSV** — ✅ 已在 [`/transfer-shanghai/KAI0/Task_A/base/analysis/`](/transfer-shanghai/KAI0/Task_A/base/analysis/)
5. **真主因** — D1 (5-16 stay-still). D7 (Class C) 在 ≤6.4% 不致命 (5day_recent 证)

### 🔄 后续讨论 (真机测试结果出来后)
- 真机测试结果决定: 直接部署 / 1309 ep 重训 / 还需要调试
- vis_v2_full → KAI0/base 编号 mapping (如需重训才做)
- prompt distinct 是否值得探索 (优先级降, README 推荐直接排除)

---

## 8. 时间线

| 日期 | 事件 |
|---|---|
| 2026-05-27 | 真机测试 G0 fixed-noise 失效 |
| 2026-05-27 | 启动数据侧 audit, 发现 18 stay-still ep / 2.14% 数据 / prompt 无区分 |
| 2026-05-27 | 完成 D1-D5 五维度分析, 形成本文档第一版 |
| 2026-05-27 | 用户确认 top head 偏移 >3cm/>3° → **D4 mirror 路线排除** (visual artifact 不可接受). 实验 C 移除 mirror. |
| 2026-05-27 PM | 用户真机测 G0 fixed-noise: ✅ 闭合次数变多, ❌ 但夹取没多久就松开. 初判 D6 短抓握污染 (v2). |
| 2026-05-27 PM-v3 | 用户指出 `KAI0/Task_A/base/README.md` 是数据集 official 描述. README §3 揭示**真实质量问题**: Class C 大跳变 129 ep (~9%) + End-snap 5 ep + 5-16 应单独用. **v2 D6 误判 (短抓握是 ep 长度副产品), 删除**. 引入 v3 D7/D8/D9 主因. 实验 B 重写为按 README 训练建议 (排除 Class C + stay-still + End-snap). |
| 2026-05-27 PM-v3.1 | 用户告知 README 已更新指向 `analysis/` 子目录, 含 8 个 CSV/JSON official 分析包 (INDEX.md + 07_classC_blacklist.csv 129 行 + 06_end_snap_trim.csv 5 行 + 03_instability_summary.csv per-date + ...). 文档全部 CSV path 替换为正式路径, Class C per-date 分布 + spike density per-date 等数据全部 from CSV. |
| 2026-05-27 晚 v4 | **决定性证据 v4**: 用户告知 smooth_800 真机表现 ≫ vis_v2_full (闭合稳定, 不 oscillate, 不松开). 数学比对: vis_v2_full 早中期 892 ep - smooth_800 811 ep = **81 ep ≈ Class C 黑名单 82 ep** (gap 1). 证实 **X1 cleanup ≈ Class C filter** + vis_v2_full 是从 raw 重 build 失去了 cleanup. D7 主因 100% 证实. **实验 E v4 (smooth_800 思路 + 后期 lym, 1272 ep)** 为新推荐主线. |
| 2026-05-27 晚 v5 | **三角对照修正**: 用户告知新实验 `vis_5day_recent` (498 ep 5-18~22 lym, 无 X1, **含 32 Class C / 6.4%**), cross val MAE@1 = **0.0086 ⭐ 最优**. 三角对照 (smooth_800/5day_recent/vis_v2_full) 揭示: **D1 (5-16) 是真主因 (升级为 ⭐⭐⭐⭐⭐), D7 降级为次要 (5day 含 32 Class C 仍 work)**. 新洞察: 时间连续性 + 单 op > 全期清洗. 最便宜路径: **直接部署 5day_recent ckpt** (现成, ~30 min ckpt→sim01). |
| 2026-05-27 晚 v5.1 | 文档清理: §0.NEXT 加真机测试可执行 checklist, §7 精简至已决策/Next Action 两栏, v4 archive 标注已 supersede. **当前状态: pending user 真机测试 5day_recent**. |
| 2026-05-27 晚 v6 | **真机测试结果: 5day_recent 不 work**. v5 (D1 单一主因) 被反驳. 立即扫 smooth vs 5day EE 数据差异: L_腕yaw +10° 漂移, wrist σ -20% 收窄, L_grip 闭合比 +5.9pp. 假说: prior 过窄. |
| 2026-05-27 晚 v7 | **决定性发现**: 用户要求精细扫 gripper 开合 → **5day 与 smooth 的"完全闭合"行为根本不同** (R_grip [0, 0.001) 占比 21.5% → 9.85%, -11.67pp, 集中漂到 [0.001, 0.005) 微开). **完美对应 README §4 "5-18~21 夹爪 2026-05-27 完成线性拉伸校准"**. 新主因: **Gripper firmware 校准漂移**. |
| 2026-05-27 晚 v8 | **用户对 TOS 5-18~5-22 数据做了校准 (mtime 11:27)**. 重扫: ✅ **L_grip 修好** (完全归零 -5.78pp → +0.75pp), ❗❗ **R_grip 过度校准** (完全归零 -11.67pp → **+31.89pp**, 微开 [0.001,0.005) 比例从 +7.16pp 跌到 -31.26pp, 推测校准脚本对 R_grip 用了过严 threshold). ❌ **L_腕yaw +9.9° 漂移 / Wrist σ 收窄 / R_肩pit max |Δa| 反升到 1.65 rad** 均没修. **当前结论**: 不应直接重训, 需先修 R_grip 校准脚本 (F_v8_A). |
| (待续) | F_v8_A: 修 R_grip 校准脚本 (检查 threshold 逻辑) → F_v8_B: 检查 R_肩pit 1.65 rad 新 spike 来源 → 重扫 verify → 实验 E v6 (1277 ep 联合 30h) |

---

## 附录 A. 验证脚本

诊断扫描:
```python
import json, pyarrow.parquet as pq
from pathlib import Path
import numpy as np

ROOT = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_full"
eps = [json.loads(l) for l in open(f'{ROOT}/meta/episodes.jsonl')]

# 找所有 stay-still ep
for e in eps:
    ep_idx = e['episode_index']
    chunk = ep_idx // 1000
    p = Path(ROOT) / 'data' / f'chunk-{chunk:03d}' / f'episode_{ep_idx:06d}.parquet'
    t = pq.read_table(str(p), columns=['action'])
    a = np.array(t.column('action').to_pylist())
    ar = float((a.max(axis=0) - a.min(axis=0)).max())
    if ar < 1e-3:
        print(f'STAY-STILL: ep {ep_idx} date={e["_src_dir"]} op={e.get("operator")} len={a.shape[0]}')
```

跨 date norm shift 算法:
- 每 date 抽 5 ep, concat action, 算 mean per dim
- 16 dates × 14 dim 矩阵, 算 column-wise std → 跨 date 漂移

镜像对称性:
- L vs R (dim 0-6 vs 7-13): `abs(L_mu + R_mu)` 衡量非对称

---

## 附录 B. ckpt + dataset 路径

- **数据集 README (权威)**: `/transfer-shanghai/KAI0/Task_A/base/README.md`
- **Analysis 索引**: `/transfer-shanghai/KAI0/Task_A/base/analysis/INDEX.md`
- **Class C 黑名单 CSV**: `/transfer-shanghai/KAI0/Task_A/base/analysis/07_classC_blacklist.csv` (129 ep)
- **End-snap 截尾 CSV**: `/transfer-shanghai/KAI0/Task_A/base/analysis/06_end_snap_trim.csv` (5 ep)
- Per-date 平滑度统计: `analysis/03_instability_summary.csv`
- Per-date 异常分类 (A/B/C): `analysis/05_instability_classes.csv`
- Per-date spike 位置 (head/mid/tail): `analysis/04_spike_position.csv`
- All spike ep raw JSON: `analysis/08_all_spike_eps_raw.json` (1559 行)
- vis_v2_full ckpt: `cnbj:/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_vis_v2_full/pi05_flatten_fold_vis_v2_full/49999/`
- vis_v2_full data: `/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_full/` (cnsh)
- vis_v2_full data: `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_full/` (cnbj)
- pure_200 ckpt: `js02:/mnt/data/tim/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_new_pure_200_new_norm/49999/`
- pure_200 data: `/vePFS/.../A_new_pure_200/`
- 诊断 raw: 见 `data_scale_vs_quality_vis_v2_full_vs_pure_200.md` §B

## 附录 C. README 关键引用 (随时回查)

来源: `/transfer-shanghai/KAI0/Task_A/base/README.md` (2026-05-27)

**§1 场景分类** (与 D9 关联):
- 04-25 ~ 4-29: 简单叠衣 ⭐ (衣服提前铺开)
- 4-30 ~ 5-18 (排 5-16): 杂乱→整齐 ⭐⭐
- 5-16: **静止参考 / ideal** (非常规)
- 5-19 ~ 5-27: 杂乱投放 ⭐⭐⭐ (最难)

**§3.1 平滑度分水岭** (与 D9 关联):
- 4-23 ~ 4-28: spike 45-75 /kF, sign-flip 4%
- 4-29 ~ 5-09: spike 3-6 /kF
- 5-18 ~ 5-27: spike 0.5-5 /kF
- "4-29 之后采集流程改进, spike 密度降低约 10×"

**§3.2 异常清单**:
- End-snap 5 ep: `4-29 ep023/098`, `4-24 ep116`, `4-28 ep102/104`
- **Class C 中段大跳变 129 ep (~9%), 建议训练时 exclude**
- 5-16 ideal 16 ep: 设计如此, 非异常

**§3.3 训练建议**:
- 完整训练: 1682 - 129 = **1553 eps**
- 5-16: **单独用作辅助任务, 不混入主 BC 训练**
- 渐进训练: Stage 1 简单 → Stage 2 +杂乱→整齐 → Stage 3 +杂乱投放
- 纯净版: 仅 4-29+ (~1322 eps)

**INDEX.md §阈值约定**:
- 大跳变: `|Δaction|/frame > 0.30 rad` (5× Piper 物理限速)
- 黑名单 (Class C): `|Δ|/frame > 0.50 rad`, 排除头尾各 30 帧
- 末尾归零 (end-snap): 末段最后一次大跳变 + 跳变后 ≤90 帧均 `arm_joints |mean| < 1.0`

**INDEX.md §训练应用方式** (官方推荐代码):
```python
import pandas as pd
ANALYSIS = "/transfer-shanghai/KAI0/Task_A/base/analysis"

# Stage 1: 排除 Class C 黑名单
bl = pd.read_csv(f"{ANALYSIS}/07_classC_blacklist.csv")
exclude_keys = set(zip(bl.date, bl.ep))   # 129 个 (date, ep)
# 在 dataloader 里过滤

# Stage 2: 对 5 个 end-snap ep 应用 trim_n 截断
trim = pd.read_csv(f"{ANALYSIS}/06_end_snap_trim.csv")
trim_map = dict(zip(zip(trim.date, trim.ep), trim.T_new))   # 5 个 (date, ep) → keep_frames
```
