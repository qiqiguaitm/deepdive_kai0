# AWBC × 零训练 milestone-value:A/B 对照实验 plan

> **目的**: 验证零训练 V2.4 milestone-value(见 [cross_episode_recurrence_value_METHOD.md](cross_episode_recurrence_value_METHOD.md))在 AWBC 里的两种用法,真机对照已跑的 pi0-AE 基线(C)。
> **核心问题**: V2.4 该 **(A) 直接当 value 源打 advantage**(取代 Stage 0+1,零训练),还是 **(B) 当自动标注器训一个 AE**(取代人工 Stage 0、保留 Stage 1 训练)?哪个真机更好。
> **状态(2026-06-13)**: 📝 待执行。C 已跑(离线 MAE 0.0079,真机 Tier3 未跑)。
> **上游**: [awbc_implementation_plan.md](../../../../deployment/strategy/awbc_implementation_plan.md)(4-step 总纲)· 复用 [awbc_viva_value_comparison_plan.md](../awbc_viva_value_comparison_plan.md) 的"换 value 源"接口与单变量对照结构。

---

## 1. 实验设计:单变量 = "如何获得 value 函数"

**锁定**(三臂逐字段一致,隔离唯一变量):数据集 `A_smooth800_dagger_all`(1124ep,与 C 同)· warm-start `task_a_new_smooth_800_step49999`(SFT MAE@1=0.0089)· AWBC config `pi05_flatten_fold_awbc`(bs128/fsdp8/EMA0.9999/~15-20k step)· **quantile-matched discretize**(见 §5,保证三臂 pos/neg 比例一致)· 同 seed/同 val 集/同 sim01 评估口径。

**唯一变量** = value 的来源:

| 臂 | value 获取 | 取代 AWBC 的 | 训练成本 | 在线能力 |
|---|---|---|---|---|
| **A** V2.4 直接 | 零训练挖掘的 per-frame value 直接当 advantage 源 | Stage 0+1 全跳过 | **0**(几分钟挖掘) | ❌ 离线 DP |
| **B** V2.4→AE 蒸馏 | V2.4 value 当伪 `stage_progress_gt` 训 AE,再用 AE 打 advantage | Stage 0(人工标注) | 1-2 天 GPU + 7GB | ✅ 参数化 |
| **C** pi0-AE(已跑) | 人工标 `stage_progress_gt` 训 AE | — (完整 RECAP) | 1 周标注 + 1-2 天 GPU | ✅ 参数化 |

**A/B 各自验证的假设**:
- **A**: V2.4 value 信息无损直接可用 → 零成本替代 estimator。
- **B**: 用 pi0 强 backbone 拟合 V2.4 弱标签,可发生 **weak-to-strong**(pi0 特征纠正 V2.4 frozen-DINOv2 看不清的随机错误),value 质量**反超 V2.4 自身**;且获得在线推理能力。
- **A vs C / B vs C**: V2.4 标签能否替代 1 周人工 `stage_progress_gt` 标注(自动化 Stage 0 的价值)。
- **A vs B**: 蒸馏那一步(训 AE)值不值——weak-to-strong 是否真发生,还是纯离线打标多此一举。

---

## 2. 共同前置:V2.4 给全 1124ep 打 value(Phase 0)

A/B 都需要 V2.4 在 `A_smooth800_dagger_all` 全集上的 per-frame value。

1. **三路特征提取**(GPU):`A_smooth800_dagger_all` 的 1124ep 各提 raw-DINOv2 ⊕ armmask-DINOv2 ⊕ proprio。现有 `tcc_smooth800_{armmask 806, raw 565+}`,**dagger 313ep 的两路特征需补提**(复用 `train_scripts/kai/data/{extract_masked_features.py, generic raw 提取}`)。
2. **挖掘 + 打 value**:用 V2.4(`smooth800_v24_ep0.py` 的挖掘逻辑,k=96/三路/增分子 cov/进度分桶/端点锚/DP)在全集挖 milestone,对每条 ep 产 per-frame value `v[t]∈[0,1]`。
3. **质量过滤**(借 viva 的 `corr_filter`):per-ep 算 `corr(v, 归一化时间)`(demo/dagger 域 progress≈时间),**剔除 |corr|<0.5 的 ep**(V2.4 挖掘失败的 ep,约占比需统计);剔除清单记入 plan。
4. 输出:每 ep parquet 加 `mv_value` 列(V2.4 per-frame value)。

> ⚠️ **neg 信号检查**: V2.4 在 demo/dagger 域主产**正 progress**;负 advantage 来自进度停滞/退步段的差分。先统计 `mv_value` 差分的 neg 比例,对照 C 的 25.2%neg——若显著偏低,说明 V2.4 在 dagger 纠错段的"低 advantage"判别弱(它的强项是显式退步,见 rollout 验证),这是关键观察点,用 §5 quantile-match 兜底但要在结论里讨论。

---

## 3. A 臂执行(V2.4 直接当 value 源,零训练)

V2.4 value 与 pi0-AE 的 `absolute_value` **语义一致**(都是 0→1 递增的 from-start progress),**符号不用翻**(不像 ViVa 是递减剩余比例)。

1. **薄 adapter** `milestone_value_to_advantage.py`(类比 `viva_value_to_advantage.py`):
   - `absolute_value[n] = mv_value[n]`
   - `absolute_advantage[n] = clip(mv_value[n+50] − mv_value[n], −1, 1)`(Δ=50,与 pi0-AE Stage 2 同窗口;末 50 帧用 `mv_value[end]−mv_value[n]`)
   - 写回 `absolute_advantage` 列(**与 pi0-AE 输出同列名 → 下游零改动**)
2. **Stage 3 离散化**(quantile-matched,见 §5):
   ```
   python kai0/stage_advantage/annotation/discretize_advantage.py <ds_A> \
       --discretion-type binary --advantage-source absolute_advantage --threshold <q_match>
   ```
3. **Stage 4 AWBC**:
   ```
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc \
       --exp_name=awbc_mv_A --repo_id <ds_A> --batch-size 128 --fsdp-devices 8
   ```
   warm-start `task_a_new_smooth_800_step49999`,~15-20k step。

**A 臂全程零模型训练**(除最后 AWBC policy 本身,这是三臂共有的)。

---

## 4. B 臂执行(V2.4 标签蒸馏一个 AE)

1. **写伪 GT**:`mv_value` → 写入 `stage_progress_gt` 列(替代人工 Stage 0)。
   - ⚠️ **语义对齐**: 人工 `stage_progress_gt` 是 per-subtask 段内单调;V2.4 是全局 0→1。两种处理:(a) 直接用全局 progress 当 GT(简单,estimator 学全局进度);(b) 按 V2.4 milestone 边界切 subtask 重标段内 progress(更贴原格式)。**先试 (a)**,若 AE 拟合差再上 (b)。
2. **Stage 1 训 AE**(= 蒸馏):
   ```
   uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
       --exp_name=ae_mv --repo_id <ds_B_with_pseudo_gt> --save_interval 10000
   ```
   pi0 backbone,2-step relative advantage regression(n vs n+50),50k step → `adv_est_mv`。
3. **Stage 2 用 AE_mv 打 advantage**(标准 eval.py,产 `absolute_advantage`):
   ```
   python kai0/stage_advantage/annotation/eval.py Flatten-Fold KAI0 <ds_B> \
       --ckpt adv_est_mv/49999
   ```
4. **Stage 3/4**:与 A 臂、C 完全相同(quantile-matched discretize → `pi05_flatten_fold_awbc` warm-start)。`--exp_name=awbc_mv_B`。

**B 臂关键 = 第 2 步训 AE**:这是 A 没有、C 有(但 C 用人工 GT)的环节。B vs A 直接量化"训这个 AE 值不值";B vs C 量化"V2.4 伪 GT vs 人工 GT"。

---

## 5. 关键技术对齐(否则对照不干净)

| 点 | 处理 | 为什么 |
|---|---|---|
| **value 符号** | V2.4=递增 progress,**不翻号**(adv=v(t+Δ)−v(t)) | 与 pi0-AE `absolute_value` 同语义;ViVa 才需翻 |
| **discretize 阈值** | **quantile-matched**: A/B 各自取使 pos/neg 比例 = C 的 25.2%neg 的分位点,而非同一绝对阈值 | V2.4 与 pi0-AE 的 advantage **标度不同**;锁比例才隔离"value 源"而非"阈值"的影响 |
| **Δ 窗口** | 50 帧(与 pi0-AE Stage 2 一致) | 同窗口才可比 |
| **质量过滤** | `corr_filter` |corr|≥0.5 剔坏 ep(A/B 用同一份过滤后集) | V2.4 部分 ep 挖掘失败,不过滤会污染 |
| **neg 比例** | 若 V2.4 自然 neg ≪ 25% → quantile-match 强制对齐,但在结论讨论"V2.4 在纠错段判别弱" | V2.4 强项是显式退步非细微纠错 |
| **prompt 字节精确** | 推理喂 `"Flatten and fold the cloth. Advantage: positive"`(句号/大小写精确) | 错一字节掉点 |

---

## 6. 评估与判据

**Tier 1(离线 MAE,sanity-only)**: val 集 MAE@1/10/25/50,对照 SFT 0.0089 / C 的 0.0079。⚠️ **MAE 对 AWBC 不敏感**(positive-prompt 离线 MAE 看不出 advantage 加权收益),仅作 sanity。

**Tier 3(sim01 rollout,决定性)**: positive-prompt 部署,measure **成功率 / 完成帧数(throughput)/ 关键子阶段(抓·对折)通过率 / 夹爪稳定性**。N trials(建议 ≥20/臂,与 C 同条件补跑——C 真机也还没跑,本实验**同时补上 C 的 Tier3**)。

**判据矩阵**(真机成功率):

| 结果 | 结论 |
|---|---|
| **A ≥ C** | 零训练直接打标 ≥ 人工 GT 训 AE → **V2.4 完全替代 Stage 0+1**,省 1 周标注+1-2 天训练 ✅✅ 最优 |
| A < C 且 **B ≥ C** | 直接用不够、蒸馏行 → **weak-to-strong 发生**,B 最优(V2.4 自动 GT 替代人工 GT,自动化 Stage 0) |
| **B > A**(且都≥C) | 蒸馏那步值得(pi0 强特征提纯 + 在线能力) |
| **B ≈ A** | 纯离线打标蒸馏多余,直接用 A |
| A<C 且 B<C | V2.4 标签质量不足以替代人工 GT;V2.4 退守"数据质量筛选"等用途(见 METHOD §应用) |

> **决定性对照 = A vs B vs C 真机三角**。A vs C 答"零训练能否替代";B vs C 答"自动 GT 能否替代人工 GT";A vs B 答"蒸馏值不值/weak-to-strong"。

---

## 7. Phase 拆分与工作量

| Phase | 任务 | 工作量 | 关键文件/命令 |
|---|---|---|---|
| **P0a** | dagger 313ep 补提三路特征 | 0.5-1d(GPU) | `extract_masked_features.py` + generic raw |
| **P0b** | V2.4 全 1124ep 挖掘+打 `mv_value` + corr_filter | 0.5d | `smooth800_v24_ep0.py` 挖掘逻辑扩全集 |
| **A1** | `milestone_value_to_advantage.py` adapter + discretize | 0.5d | 新薄脚本 + `discretize_advantage.py` |
| **A2** | AWBC 训练 A(uc/volc 8 卡 ~15-20k) | 1-2d | `train.py pi05_flatten_fold_awbc` |
| **B1** | 写伪 GT + Stage 1 训 AE_mv(50k) | 1-2d(GPU) | `train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` |
| **B2** | AE_mv 打 advantage(Stage 2)+ discretize | 0.5d | `eval.py` + `discretize_advantage.py` |
| **B3** | AWBC 训练 B | 1-2d | 同 A2 |
| **EVAL** | Tier1 MAE(A/B)+ **Tier3 sim01(A/B/C 三臂)** | 1-2d | 补 C 的真机 |

**总计 ~6-9 天**(A 臂 ~3-4 天最快;B 臂多 Stage1 训练 ~2 天)。建议**先跑 A**(零成本、最快出真机数),A≥C 则 B 的优先级取决于是否需要在线 value。

---

## 8. 风险与兜底

| 风险 | 概率 | 兜底 |
|---|---|---|
| V2.4 在 dagger 纠错段 neg 信号弱(neg≪25%) | 中 | quantile-match 强制比例;或往数据集掺真退步 rollout 段补 neg |
| B 臂全局 progress 当 GT 拟合差(语义不匹配 per-subtask) | 中 | 切 milestone 边界重标段内 progress(§4 方案 b) |
| V2.4 系统性误差被 AE 继承(非随机) | 中 | weak-to-strong 不发生 → B≈A,结论仍有效(说明该蒸馏) |
| C 真机也没跑→无对照基线 | 高(已知) | 本实验 Tier3 **同时补 C**,三臂同口径 |
| 特征提取/挖掘在 1124ep 上挖掘退化 | 低 | 已验证 500ep 挖掘稳;全集只增样本 |

---

## 9. 链接

- 方法 → [cross_episode_recurrence_value_METHOD.md](cross_episode_recurrence_value_METHOD.md) · 探索 → [..._plan.md](cross_episode_recurrence_value_plan.md)
- AWBC 总纲 → [awbc_implementation_plan.md](../../../../deployment/strategy/awbc_implementation_plan.md)
- 复用接口/结构 → [awbc_viva_value_comparison_plan.md](../awbc_viva_value_comparison_plan.md)
- Stage 2-4 机理 → [advantage_pipeline_and_visual_subgoal.md](../advantage_pipeline_and_visual_subgoal.md)
- C 结果 → awbc_implementation_plan.md §结果回填(2026-06-09)
