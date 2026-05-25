# strategy/ — 战略与上游对照

> **场景**: 高层方向决策 / 跨本体战略 / Task 路线图 / 与官方 fork 差异分析。

## 文件清单

| 文件 | 用途 |
|---|---|
| [`cross_embodiment_strategy.md`](cross_embodiment_strategy.md) | ⭐ 跨本体战略主文档 — 3 异构机器人 (KAI0/vis/XVLA) / 4-层 ROI / norm-stats 实证 / 假说 H1-H4 / Conditioning / RTC-TAC / Tri-track / 决策点 |
| [`task_a_master_plan.md`](task_a_master_plan.md) | Task A (cloth fold) 主规划 — 训练 + 部署 Roadmap |
| [`official_diff_and_risk_analysis.md`](official_diff_and_risk_analysis.md) | deepdive_kai0 vs 官方 kai0 fork 差异 + 风险分析 + 修复记录 |

## 按需求找文件

| 你想做什么 | 文件 |
|---|---|
| 看 3 个机器人 (KAI0/vis/XVLA) 区别 / 跨本体方案 | cross_embodiment_strategy.md |
| 看 Conditioning (Hard / Soft / Action Head) 设计选择 | cross_embodiment_strategy.md §5 |
| 看 RTC / TAC 集成 | cross_embodiment_strategy.md §6 |
| 看 Tri-track (A SSL / C Action Cond / X X-VLA 官方) 架构 | cross_embodiment_strategy.md §7 |
| 看 Task A 整体 roadmap | task_a_master_plan.md |
| 看本 fork 与官方差异 / 改了什么 / 为什么 | official_diff_and_risk_analysis.md |

## 跨场景跳转

- 战略对应的具体 plan 与 history → `../../training/future_plans/` 和 `../../training/history/`
- ckpt 部署到真机 → `../inference/`
- 训练集群运维 → `../training_ops/`
