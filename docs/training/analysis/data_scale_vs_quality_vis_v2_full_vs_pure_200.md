# vis_v2_full vs pure_200 — 真机 oscillation 归因 (cumulative 2026-05-26 ~ 2026-05-27)

> **文档定位**: 跨实验对比 + 反直觉归因. 主线问题: 为什么 vis_v2_full (1406 ep, MAE@1=0.0131) 真机表现差, 而 pure_200 (100 ep × 2 mirror, MAE@1=0.0065) 表现好?
>
> **当前认知 (截至 2026-05-27)**:
> 1. **不是训练失败**, vis_v2_full 训练健康
> 2. **不是 chunk discontinuity**, vis_v2_full 在 chunk 连续性上 **比 pure_200 还好**
> 3. **是 rare-event multi-modal collapse** — vis_v2_full 在**关键决策时刻**对 noise 高度敏感, pure_200 全程 deterministic
> 4. **可 inference 侧修** — G0 fixed-noise 不需要重训
>
> **本文阅读顺序**: §0 TL;DR → §1 当前事实表 → §2 现在的根因假说 → §3 修复路线 → §A 附录走过的弯路.
>
> **关联**: [task_a_new_pure_200_new_norm_results.md](../history/experiments/task_a_new_pure_200_new_norm_results.md) · [00_training_history.md](../history/experiments/00_training_history.md) · [G0 sim01 fix](../../deployment/inference/fixed_noise_inference_fix.md)

---

## 0. TL;DR

**症状** (用户报告):
- vis_v2_full 真机: 走几步退几步 (oscillation), 夹爪无法长期闭合, 夹爪来回犹豫
- vis_v2_full + 真机 RTC: 同样 oscillation
- pure_200 + 同 RTC params: 正常

**根因** (定位结果):
- 真机现象不是 chunk 连续性问题 (P1 vis_v2_full 0.026 < pure_200 0.039, vis_v2_full 实际更好)
- 真机现象是 **flow matching noise sensitivity** 在 rare moments 失控 (P2 max vis_v2_full 0.67 vs pure_200 0.19)
- 当 noise 不同时, vis_v2_full 在关键时刻 (gripper trigger / 转折) 跳进**不同 attractor** → 真机看到 oscillation

**修复**: G0 **fixed-noise inference** — `policy.infer(obs, noise=FIXED)` 启动时生成一次 noise 复用. RTC 兼容. 0 训练成本. ([sim01 部署文档](../../deployment/inference/fixed_noise_inference_fix.md))

**MAE 数字解读** (避免误导):
- pure_200 native-val 0.0065 ≠ 真泛化能力, 是**严重过拟合 to 05-08/09 两个 dates**
- vis_v2_full 同 val (vis_v2_val50, 04-23/24) 0.0131 比 pure_200 cross-val 0.0207 **好 36%**
- offline MAE 高低不直接对应真机表现 — 真机现象由 noise sensitivity 主导, 不是 MAE

---

## 1. 当前事实表 (fresh measurement, 2026-05-27)

### 1.1 数据集对比

| 维度 | pure_200 | vis_v2_full |
|---|---:|---:|
| Unique episodes | 100 | **1406** (14× 多) |
| Mirror augmentation | ✅ 50% hflip mirror | ❌ 无 |
| 总条数 (含 mirror) | 200 | 1406 |
| 唯一 frames | ~150k | **1.93M** (13× 多) |
| 跨越 dates | 2 连续 (05-08, 05-09) | **16** (04-23 ~ 05-22, 1 月跨度) |
| 协议一致性 | ⭐⭐⭐⭐⭐ | ⭐⭐ |

### 1.2 训练 hparams 对比 (排除 hparams 差异)

两侧 hparams **几乎完全相同** (model = `Pi0Config(pi05=True)`, peak_lr = 1.5e-5, decay_steps=50k, ema=0.9999, batch=120/128, fsdp=8). Init 锁定为 pi05_base (用户对照实验 pure_200+pi05_base 也好, 排除 init 主导).

→ **唯一变量是数据**.

### 1.3 Offline MAE (同/异 val 双视角)

| Ckpt + Init | val | MAE@1 | MAE@10 | MAE@25 | MAE@50 |
|---|---|---:|---:|---:|---:|
| pure_200 + mixed_1_clean | A_new_pure_200_val (native, 05-08/09) | **0.0065** | 0.0072 | 0.0075 | 0.0079 |
| pure_200 + pi05_base | A_new_pure_200_val (native, 05-08/09) | **0.0065** | 0.0074 | 0.0078 | 0.0087 |
| pure_200 + pi05_base | vis_v2_val50 (cross, 04-23/24) | **0.0207** | 0.0507 | 0.0900 | 0.1348 |
| **vis_v2_full + pi05_base** | vis_v2_val50 (cross, 04-23/24) | **0.0131** | 0.0386 | 0.0714 | 0.1138 |

读法:
- pure_200 在 native val 上 0.0065 是过拟合到 2 个 dates 的产物
- 在 cross val 上 pure_200 退化 3.2× (0.0065 → 0.0207)
- 同 cross val 比较: vis_v2_full 0.0131 < pure_200 0.0207, vis_v2_full 泛化更好 36%

### 1.4 Chunk/Noise diagnostic (定位真机 oscillation 真因)

诊断脚本: P0 gripper 分布 / P1 chunk-to-chunk |diff| / P2 multi-sample variance.

| 指标 | vis_v2_full | TAC v7 (buggy) | pure_200 |
|---|---:|---:|---:|
| **P1 random L** | 0.0265 🟢 HEALTHY | 0.0666 🔴 BAD | 0.0390 🟡 MARGINAL |
| **P1 fixed-noise L** | 0.0206 🟢 | — | 0.0389 🟡 |
| **Noise 贡献 ΔL** | **+0.0059** | — | **+0.0001** |
| **P2 mean variance** | 0.0234 | 0.0231 | **0.0117** |
| **P2 max variance** | **0.6688** | 0.6525 | **0.1903** |

读法:
- **vis_v2_full chunk 连续性比 pure_200 好** (0.026 < 0.039) — chunk discontinuity 不是问题
- **pure_200 几乎完全 deterministic** (noise 贡献 0.0001), vis_v2_full 受 noise 影响 (0.0059, ×59)
- **vis_v2_full 大部分时刻稳定** (P2 mean 0.023 不算高), **某 rare 帧某 dim std 高达 0.67** (multi-modal collapse)
- **TAC v7 不仅没改善反而恶化** — 见 §A.3 TAC bug 附录

---

## 2. 根因假说 (当前 live)

### H1 ⭐ 主假说: Rare-event multi-modal collapse at critical decision moments

**机制**:
1. vis_v2_full 在 1.93M frames 上学到一个 flow matching policy
2. 大部分时刻 (持续运动) 这个 policy 是 deterministic 的
3. 在**关键决策点** (gripper 开/合切换 / 抓取/折叠之间的 state transition / wrist 翻转), policy 的 mode 分布存在 multi-modal landscape
4. flow matching ODE 从 noise 出发去噪, 不同 noise 起点会去到不同 attractor
5. 不同 chunk 用不同 noise → 跳进不同 attractor → 真机表现 oscillation / 犹豫

**支持证据**:
- P2 mean 0.023 (大部分 deterministic) + max 0.67 (rare 关键时刻多 mode) — 直接对应这个 pattern
- 用户实测: vis_v2_full + RTC 仍 oscillate (RTC 只锚 chunk 边界, 不修内部 mode 分裂)
- pure_200 P2 max 仅 0.19 (rare moments 也是 deterministic) + pure_200 + RTC 真机 work

**与症状对应**:
- "走几步退几步" → 走到 transition 时 model 跳 mode
- "夹爪犹豫" → gripper trigger 时 model 在"开/合"两个 mode 间漂移
- "夹爪无法长期闭合" → 闭合后下一 chunk noise 不同 → 跳回"半开" mode

### H2 (备选): OOD scene drift

真机场景与训练分布距离远 → 即使 fixed noise 也 oscillate.

验证: G0 真机测试. 如果 fixed noise 仍 oscillate, H1 不充分, H2 接力.

### H3 (已证伪): Chunk discontinuity

早期假说 — 已被 P1=0.026 (HEALTHY) 推翻. 详见 §A.4.

---

## 3. 修复路线

### G0 ⭐ Fixed-noise inference (推荐, 0 训练)

**原理**: `policy.infer(obs, noise=FIXED)` 已支持. 启动时生成一次 noise, 整个 session 复用. P2 → 0 (完全 deterministic across chunks).

**预期效果**:
- vis_v2_full + fixed noise: P1 降到 0.021 (实测), 关键时刻不再跳 mode
- 真机 oscillation 应消除

**部署**: [`docs/deployment/inference/fixed_noise_inference_fix.md`](../../deployment/inference/fixed_noise_inference_fix.md) — sim01 端 self-contained patch, env-var-gated, RTC 兼容.

**验证步骤**: see deploy doc §5 (seed=0/1/2/3 各试, 找 best seed).

### G1 Fallback (如果 G0 不充分)

| 优先级 | 方案 | 训练 | 预期 |
|---|---|---:|---|
| ⭐⭐ | vis_v2_full ckpt + pure_200 100ep finetune 10k step | 6h | 兼顾 broad prior + 锐利 (推荐) |
| ⭐ | vis_v2_full + hflip mirror 重训 | 30h | 注入对称性 prior |
| ⭐ | vis_v2_full 跑 100k step | 60h | 加深 supervision (不解决 multi-mode) |

⚠️ TAC 路线 (G2) 已确认无效 — 见 §A.3 TAC bug. 修了 bug 重训才能验证 TAC 设计的真实效果, 但优先级低于 G0 (inference 侧已能修).

---

## 4. 关键 Lesson (跨实验沉淀)

1. **Offline MAE ≠ 真机表现**: vis_v2_full 0.0131 < pure_200 cross val 0.0207 (MAE 更好), 但真机更差 — 真机现象由 flow matching noise sensitivity 主导, 不直接由 MAE 决定
2. **Val distribution 必须匹配才可比**: pure_200 native-val 0.0065 比 vis_v2_full 0.0131 看起来好 50%, 但同 val 下反过来 — native val 测的是过拟合, 不是泛化
3. **跨 session 引用数字必须 fresh 重测**: 早期 vis_v2_full P1=0.063 BAD 是跨 session 数字混淆 (实际是 TAC v7 数据). 详见 [feedback memory](../../../../home/tim/.claude/projects/-home-tim-workspace/memory/feedback_verify_cross_session_numbers.md)
4. **Init 不主导**: 50k step 后 pi05_base vs mixed_1_clean 几乎完全消除影响. 之前 "init 决定上限" 假说被推翻
5. **"多数据 = 多坏" 不成立 (有条件)**: 在公平 val 下数据多反而好. "数据量增大反而 MAE 变差" 是 val mismatch artifact

---

## 5. 时间线 + 验证记录

| 日期 | 事件 |
|---|---|
| 2026-05-23 | vis_v2_full 训练启动 (cnbj) |
| 2026-05-24 | step 49999 完成 |
| 2026-05-25 | offline MAE@1=0.0131 (vis_v2_val50) |
| 2026-05-26 | 真机不好, 启动诊断. 第一版分析 (§A.1) — **多数结论后续被推翻** |
| 2026-05-26 PM | dual-val 实验完成 — 推翻"训练失败 / init 主导 / 数据量大反而差"三个假说 (§A.2) |
| 2026-05-27 | 用户报告真机现象细节 (oscillation / gripper hesitate) — 启动 chunk/noise diagnostic |
| 2026-05-27 | 跑 P0/P1/P2 三 ckpt, fresh 数字推翻"chunk discontinuity 是主因"假说 (§A.4) |
| 2026-05-27 | TAC v7 验证: 完全无效, 发现 pi0.py:335 convention bug (§A.3) |
| 2026-05-27 | Fixed-noise diagnostic: vis_v2_full P1 fixed=0.021 < pure_200 random 0.039 — H1 假说成立, G0 路线确认 |
| 2026-05-27 | sim01 部署文档 commit + push (8d3802f) |

---

## 附录 A. 走过的弯路 (避免后续重复)

### A.1 ❌ 假说: "vis_v2_full 训练失败" (2026-05-26 AM, 后被推翻)

当时认为是训练动态问题, 提出 4 个具体机制:
1. Mirror 缺失 → 失去对称性 prior (+20~30% MAE)
2. Per-frame supervision 7× 弱 → 不锐利 (+30~40% MAE)
3. 16 dates 协议漂移 → 平均策略 (+20~30% MAE)
4. 梯度方差 → 等效 LR 不够 (+5~10% MAE)

**为什么错**: 没控制 val distribution. 同 val 下 vis_v2_full 反而好.

**部分仍 valid**: Mirror 缺失对 in-distribution 精度有影响 (但小); 协议漂移 actually 是 generalization advantage.

### A.2 ❌ 假说: "init (pi05_base vs mixed_1_clean) 决定上限" (2026-05-26 PM, 推翻)

用户对照实验 (pure_200 + pi05_base 也 0.0065) → 50k step 后 init 完全消除影响.

### A.3 ❌ 假说: "TAC 训练能修 chunk 不连续" (2026-05-27 AM, 推翻 + 发现 bug)

TAC v7 (vis_v2_full_tac) 训完 50k step × 8 H20:
- P1 = 0.0666 (BAD, 比 baseline vis_v2_full 0.026 更差)
- P2 与 baseline 基本一致

**发现 bug**: `src/openpi/models/pi0.py:335`
```python
# 当前 (buggy):
time_per_token = jnp.where(prefix_mask_tac, 1.0, time[..., None])
# openpi convention: t=1=noise, t=0=clean GT
# 所以 prefix tokens 拿到的是 pure noise, 不是 paper 设计的 "clean GT prefix"
# 应该是:
time_per_token = jnp.where(prefix_mask_tac, 0.0, time[..., None])
```

**TAC pure_200 (`t-20260526154023-7fg82`)**: 用户决定不 kill, 跑完作为 buggy baseline (见 [memory](../../../../home/tim/.claude/projects/-home-tim-workspace/memory/project_tac_implementation_bug.md)). 等修了 bug 重训为 `tac_v2` 才能验证 TAC 设计真实效果.

### A.4 ❌ 假说: "vis_v2_full chunk discontinuity 导致 oscillation" (2026-05-27 PM, 推翻)

早期诊断声明 vis_v2_full P1 Left=0.0631 / Right=0.0546 BAD. 据此构建 F0/F1 (RTC inpainting + TAC retrain) 路线.

**Fresh 重测推翻**: vis_v2_full P1 实际 = **0.0265 / 0.0234 HEALTHY**, 比 pure_200 还好.

**误差源**: 跨 session 数字混淆, 把 TAC v7 (0.067) 误归到 vis_v2_full 名下. 详见 [feedback memory](../../../../home/tim/.claude/projects/-home-tim-workspace/memory/feedback_verify_cross_session_numbers.md).

**推翻后**: F0 inference RTC 用户已经实测过没用, 因为 RTC 只锚 chunk 边界. 真因是 §2 H1 multi-modal collapse, 修复改走 G0 fixed-noise inference.

---

## 附录 B. 验证脚本与 ckpt 路径

| 脚本/产物 | 路径 |
|---|---|
| 诊断 P0/P1/P2 脚本 | `/tmp/diagnose_vis_v2_full.py` |
| 诊断 fixed-noise 变体 | `/tmp/diagnose_fixed_noise.py` |
| vis_v2_full ckpt | `cnbj:/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_vis_v2_full/pi05_flatten_fold_vis_v2_full/49999` |
| pure_200 ckpt | `/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/task_a_pure200_base_pi05_step49999` |
| TAC v7 ckpt | `cnbj:/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_vis_v2_full_tac/pi05_flatten_fold_vis_v2_full_tac/49999` |
| 诊断 raw JSON: pure_200 | `/tmp/diagnose_pure_200_on_vis_v2_val50.json` |
| 诊断 raw JSON: vis_v2_full fixed vs random | `cnbj:/tmp/diag_vis_v2_full_fixed_vs_random.json` |
| 诊断 raw JSON: pure_200 fixed vs random | `/tmp/diag_pure_200_fixed_vs_random.json` |
