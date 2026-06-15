# 跨 Episode 重复度挖掘 → 自动 Milestone / Value(零训练 milestone-value 工作集)

> 一条独立工作线的归档目录。核心:**同任务多条 demo 中反复出现的状态 = 任务必经 milestone**,
> 从跨 episode 结构挖 value,**零训练**(frozen DINOv2-small + KMeans + Viterbi-DP)替代 AWBC 的逐帧监督回归。

## 文档导航

| 文档 | 作用 |
|---|---|
| [cross_episode_recurrence_value_METHOD.md](cross_episode_recurrence_value_METHOD.md) | **最终方法 V2.4**(9步配方 + 四场景验证 + 否决死路 + 结论)。干净版,先读这个。 |
| [cross_episode_recurrence_value_GENERALIZATION.md](cross_episode_recurrence_value_GENERALIZATION.md) | **跨数据集泛化实证**:XVLA soft_fold(新本体)corr 0.956/100%≥0.7;真实 ALOHA coffee(新任务)corr 0.988/单调100%。配方逐字不改。 |
| [cross_episode_recurrence_value_plan.md](cross_episode_recurrence_value_plan.md) | **完整探索记录**:18 次迭代 + 56 图 + 文献调研 + 所有否决死路的诊断过程。 |
| [awbc_milestone_value_AB_plan.md](awbc_milestone_value_AB_plan.md) | **下游落地 A/B 对照 plan**:A=V2.4 直接当 value 源 / B=蒸馏训 AE,对照已跑的 C=pi0-AE。 |

## 关联资源(目录外)
- 可视化图/预览帧:[`docs/visualization/cross_episode_recurrence_value/`](../../../../visualization/cross_episode_recurrence_value/)(含 `generalization/` 子目录)
- 评估脚本(任意 HDF5 / LeRobot-v3 数据集即插即用):`train_scripts/kai/data/{hdf5,lerobot_v3}_extract_features.py` + `{hdf5,lerobot}_v24_eval.py`
- 上游 AWBC 总纲:[`awbc_implementation_plan.md`](../../../../deployment/strategy/awbc_implementation_plan.md)

## 状态
✅ 方法收口(V2.4)· ✅ 四场景验证 · ✅ 跨数据集强泛化 · ⏳ AWBC A/B 真机对照(plan 已就绪,待执行)
