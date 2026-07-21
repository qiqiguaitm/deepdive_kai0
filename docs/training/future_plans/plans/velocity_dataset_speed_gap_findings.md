# 模型运动速度 vs 数据集速度的差异度 —— 调研 + 离线实证 findings

> **建立**: 2026-07-18 · **性质**: 研究调研 + pilot 实证(非训练归档)
> **一句话**: 模型比数据集慢约 **5–7%**(温和 mode-averaging 税, 非塌缩);**冻结不是全局速度属性**, 两条离线分布内探针都抓不到它 —— 冻结被钉死为**闭环 + 状态特异的自锁**。
> **数据**: `docs/training/future_plans/plans/data/velocity_fidelity/{velfid,cshift}_*.{json,npz}`
> **脚本**: `train_scripts/kai/eval/velocity_fidelity.py`(全局保真度) · `velocity_condition_shift.py`(pos/neg 条件位移)

---

## 1. 问题与动机

"用数据集训出来的模型, 在运动速度上和数据集差多少?" 这是我们两周 freeze 工作的**定量母题**: 冻结 = 速度→0 的极端点, 而"速度分布保真度"是完整谱系(平均化变慢 / 平滑 / idle 塌缩)。主流 VLA 评测只看成功率 + 动作 MAE, **两者都看不见速度分布塌缩**(MAE 对"整体慢 7%"无感, 成功率要真机才暴露)。

**数据集自身就是双峰**(先扣掉再谈模型差异): base(expert) 均速 0.037 / 静止 27%; dagger(拼接) 均速 0.027 / 静止 47%。

## 2. 方法(离线开环, 同一 val=vis_v2_merged_val 30ep, 恒喂部署 prompt)

- **单变量 ckpt 对**(freeze 诊断原生对照, 同机同 init, 唯一差 = dagger 边界段):
  - **frozen** = `pi05_v4_awbc_plus_freshdagger/49999`(任务②, 真机确认冻结)
  - **good** = `pi05_v4_awbc/49999`(不冻基线)
- **探针①ap 全局保真度**: 每 val 帧预测 50 步 chunk → chunk 内臂速分布(排除夹爪) vs GT。指标 W1/JS/中位比/静止%。
- **探针①(conditioning)**: 同一状态分别喂 `Advantage: positive` / `negative`, 测速度位移(冻结机制应是 positive→决策态静止)。

## 3. 结果

**探针①ap — 全局速度保真度(frozen ≈ good, 都轻微慢于数据):**

| ckpt | 中位比(模型/GT) | p90 比 | 模型静止% | GT静止% | W1 | JS |
|---|---|---|---|---|---|---|
| good | 0.943 | 0.935 | 40.3 | 39.2 | 0.0041 | 0.0511 |
| frozen | 0.931 | 0.932 | 40.9 | 39.2 | 0.0040 | 0.0475 |

**探针① — pos/neg 条件位移(几乎为零, frozen ≈ good):**

| ckpt | med_pos | med_neg | pos/neg | pos/gt | 静止%_pos | pos<neg 状态% | pos→static% |
|---|---|---|---|---|---|---|---|
| good | 0.0268 | 0.0265 | 1.012 | 0.951 | 40.2 | 45.5 | 0.8 |
| frozen | 0.0265 | 0.0260 | 1.019 | 0.939 | 40.7 | 47.2 | 1.7 |

## 4. 结论(两条独立离线探针一致收敛)

1. **速度差异度(直接答案)**: 模型比数据集**慢约 5–7%**(中位比 0.93–0.95), p90 低 ~6.5%, 静止占比 ~40% 与数据持平, W1≈0.004 / JS≈0.05 = 分布贴合良好。**是温和 mode-averaging/chunk 低通税, 非塌缩**。**positive/negative 对速度幅度几乎无影响**(pos/neg≈1.0)—— 条件化改的是动作方向/模式, 不是速度。
2. **离线分布内探针抓不到冻结**: 全局保真度 + 条件位移两条都**无法分开 frozen/good**(W1 0.0040 vs 0.0041; pos/neg 1.019 vs 1.012)。原因: (a) teacher-forced 每步从 GT 重置 → 打断自锁; (b) base val 不含"回折决策/接管卡住"OOD 态。**冻结 = 闭环 + 状态特异自锁**(covariate shift + copycat latching, 文献 theme 3&4), 分布内单发预测不可见。**这解释了离线 MAE/速度都正常、真机却冻。**
3. frozen 仅存极微弱方向性(pos→static 1.7% vs 0.8%; pos/gt 0.939 vs 0.951), 量级远不足当检测器。

## 5. 文献 + novelty

机制齐全(Diffusion Policy 2303.04137 均值化 / ACT 2304.13705 · RTC 2506.07339 chunk 低通 / Error-Aware IL 2112.05251 idle 卡死 / DAgger · Three Regimes 2102.02872 协变量漂移); "策略比示范慢"有人做(**DemoSpeedup 2506.05064** · **SAIL 2506.11948**, 但用完成时间比而非分布散度); 标量运动指标有(**Embodied Efficiency 2603.19131**: 完成时间/路径长/jerk, 但对自己基线)。

**空白/novelty**(文献扫确认): (a) "**策略 rollout vs 数据集的速度边缘分布保真度**"(W1/KL/JS on speed histogram)作为命名诊断基本没人做; (b) "**advantage 条件化如何移动执行速度**"没人测 —— 正是我们 CRAVE/AWBC 这条线。本 pilot 的负结果本身即贡献: **该保真度在分布内不能检测 freeze, 把 freeze 钉死在闭环**。

## 6. 下一步(离线路已探明为负 → 需换探测面)

- **探针② 决策态定向**: 不用 base val 全局, 专测 dagger **回折/接管态**帧的速度(freeze 高发区)—— 若冻结是状态特异, 这里才可能分开。**便宜, 仅需挑帧。**
- **探针③ 闭环**(sim01/真机): 唯一能拿真实 rollout 速度 + 自锁的办法。
- **可发表的实测**: "VLA 继承 ~7% 的 mode-averaging 速度税 + 40% idle, 而 advantage 条件化对速度近零影响" 是干净的分布级测量, 补 2603.19131 缺的分布层。

## 关联
- 冻结根因/修复: [`dagger_launchpoint_trim_freeze_fix_plan.md`](dagger_launchpoint_trim_freeze_fix_plan.md)(§9 Δprogress+门控) · [`pi05_v4_awbc_modeB_freeze_diagnosis_plan.md`](pi05_v4_awbc_modeB_freeze_diagnosis_plan.md)
