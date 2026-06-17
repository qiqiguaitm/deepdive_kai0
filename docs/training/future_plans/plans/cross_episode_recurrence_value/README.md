# 跨 Episode 重复度挖掘 → 自动 Milestone / Value(零训练 milestone-value 工作集)

> **Method name: CRAVE** — **C**ross-episode **R**ecurrence **a**s **V**alue **E**stimation
> *(Training-free dense value from what demonstrations repeat. 实现代号 = milestone-value / V2.4。)*

> 一条独立工作线的归档目录。核心命题:**同任务多条 demo 中反复出现的状态 = 任务必经 milestone**;
> 把跨 episode 的统计重复性 → 自动浮现的 milestone → 经 Viterbi-DP 读出为稠密单调 progress value,
> **全程零训练**(frozen DINOv2-small + KMeans + Viterbi-DP,零梯度更新),替代 AWBC 逐帧监督回归(pi0-AE)。
>
> **三支柱**:① 跨 episode 统计重复性揭示任务结构 → ② 重复态 = 自动 milestone → ③ 零训练产稠密 value。
> 已验证:demo 域干净 0→1、撞色衣物兜底、跨天 16/16、rollout 退步+恢复、kai0 GT MAE 0.105;
> **跨数据集强泛化**(新本体 XVLA corr 0.956 / 真实 ALOHA coffee corr 0.988);
> 单 episode 对比监督 pi0-AE(METHOD §3.3b):in-distribution 与 AE 平手且更平滑(corr 0.82,单调 100% vs 52%);out-of-distribution 明显更稳。

## 文档结构(2026-06-17 整理后)

**核心方法(干净版,先读这些):**

| 文档 | 作用 |
|---|---|
| [METHOD](cross_episode_recurrence_value_METHOD.md) | **离散主线 V2.4**:9 步配方 + 四场景验证 + 否决死路 + 结论。**实现请读这个。** |
| [CONTINUOUS](cross_episode_recurrence_value_CONTINUOUS.md) | **连续 value 形态**:端到端 TCC + 细 bin DP 时序证据读出。advantage 密集(81-96%)、平滑无崩塌;含完整推导。 |
| [GENERALIZATION](cross_episode_recurrence_value_GENERALIZATION.md) | **跨数据集泛化实证**:XVLA soft_fold corr 0.956、真实 ALOHA coffee corr 0.988,配方逐字不改。 |

**定位与落地:**

| 文档 | 作用 |
|---|---|
| [CRAVE_positioning_and_roadmap](CRAVE_positioning_and_roadmap.md) | **定位 / 场景 / roadmap(合一)**:前沿地图 + vs GVL + 工作项 A/B/C/D(含 B1✅/B2❌ 验证)+ 6 场景×SOTA + 分阶段安排。 |
| [value_advantage_methods_comparison](value_advantage_methods_comparison.md) | **机理对比**:kai0-AE(监督进度差)vs π\*0.6-RECAP(RL 回报优势)vs CRAVE(零训练离散),逐维表 + 分场景结论。 |
| [awbc_milestone_value_AB_plan](awbc_milestone_value_AB_plan.md) | **下游 A/B 对照执行 plan**:A=直接当 value 源 / B=蒸馏训 AE,对照已跑的 C=pi0-AE(当前 A 臂三档训练中)。 |
| [frequency_window_params](frequency_window_params.md) | **频率窗参数**:lam ∝ fps、时间窗按秒;3Hz/30Hz 落表 + 标定规则 + 实测(窗标定让抖动降 8×)。 |

**溯源:**

| 文档 | 作用 |
|---|---|
| [cross_episode_recurrence_value_plan](cross_episode_recurrence_value_plan.md) | **探索记录索引存根**:结论速览表 + 迭代索引(保 §-锚点)+ 文献 + 工件清单。详细叙述已收口进上方干净文档。 |
| [crave_interpretability.md](../../../../visualization/cross_episode_recurrence_value/crave_interpretability.md) | **关键上升/下降点严格可解释分析**:每个 milestone 跨越的相机帧 + 可分离三路归因 + grounded 判据。脚本 `crave_interpretability.py`。 |
| [crave_grounded_advantage.md](../../../../visualization/cross_episode_recurrence_value/crave_grounded_advantage.md) | **可解释分析 II**:真机退步归因 + grounded 过滤接进 AWBC advantage(neg 假标 5.2%→3.9%)。脚本 `crave_grounded_advantage.py`。 |

## 两条 value 形态
- **离散 CRAVE V2.4**(主交付,[METHOD](cross_episode_recurrence_value_METHOD.md)):milestone 阶梯,零训练,跨数据集强泛化。AWBC 打标用 `smooth_monotone(w∝fps)` 连续读出即可。
- **连续 TCC+DP**([CONTINUOUS](cross_episode_recurrence_value_CONTINUOUS.md)):端到端 TCC 进度感知特征 + 相似度场 Viterbi-DP + 子 bin 软期望 → 逐帧连续 value;消 fold 凹口、保留真回退;跨数据集泛化 corr 0.94-1.00。

## 关联资源(目录外)
- **统一计算库**:`train_scripts/kai/data/crave_value.py`(`FeatureSpace` + `DiscreteValue` + `ContinuousValue`)
- 可视化图/预览帧:[`docs/visualization/cross_episode_recurrence_value/`](../../../../visualization/cross_episode_recurrence_value/)(含 `generalization/` 子目录)
- 评估脚本(任意 HDF5 / LeRobot-v3 数据集即插即用):`train_scripts/kai/data/{hdf5,lerobot_v3}_extract_features.py` + `{hdf5,lerobot}_v24_eval.py`
- 上游 AWBC 总纲:[`awbc_implementation_plan.md`](../../../../deployment/strategy/awbc_implementation_plan.md)
- Web 技术报告:`web/showcase/reports/crave_interp/`(showcase tab-bar「CRAVE 技术报告」)

## 状态
✅ 离散 V2.4 收口/四场景/跨数据集 · ✅ 连续 TCC+DP 收口(跨数据集泛化) · ✅ value 计算重构为统一库(crave_value.py) · 🔄 AWBC A/B 真机对照(A 臂三档集群训练中,Tier3 sim 待跑)
