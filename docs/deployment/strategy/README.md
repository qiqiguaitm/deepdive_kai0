# strategy/ — 战略与上游对照

> **场景**: 高层方向决策 / 跨本体战略 / Task 路线图 / 与官方 fork 差异分析。

## 文件清单

| 文件 | 用途 |
|---|---|
| [`cross_embodiment_strategy.md`](cross_embodiment_strategy.md) | ⭐ 跨本体战略主文档 — 3 异构机器人 (KAI0/vis/XVLA) / 4-层 ROI / norm-stats 实证 / 假说 H1-H4 / Conditioning / RTC-TAC / Tri-track / 决策点 |
| [`task_a_master_plan.md`](task_a_master_plan.md) | Task A (cloth fold) 主规划 — 训练 + 部署 Roadmap |
| [`dagger_implementation_plan.md`](dagger_implementation_plan.md) | SFT-DAgger 主轨实施方案 — 任务路由 / 5 阶段闭环 / Form C 录制设计 / 5 Phase / 失败模式分类 |
| [`awbc_implementation_plan.md`](awbc_implementation_plan.md) | AWBC / RECAP advantage 升级 — 4-Step pipeline (Stage 0 标 progress → Stage 1 估 advantage → Stage 2 预测 → Stage 3 discretize → Stage 4 AWBC 训练) |
| [`rlt_implementation_plan.md`](rlt_implementation_plan.md) | RLT 实施方案 — 6 集成点 (含 Hybrid Fork + Adapter) / 4-Phase 实施 / 代码改动清单 / vis_v2_full gripper POC |
| [`official_diff_and_risk_analysis.md`](official_diff_and_risk_analysis.md) | deepdive_kai0 vs 官方 kai0 fork 差异 + 风险分析 + 修复记录 |

## 按需求找文件

| 你想做什么 | 文件 |
|---|---|
| 看 3 个机器人 (KAI0/vis/XVLA) 区别 / 跨本体方案 | cross_embodiment_strategy.md |
| 看 Conditioning (Hard / Soft / Action Head) 设计选择 | cross_embodiment_strategy.md §5 |
| 看 RTC / TAC 集成 | cross_embodiment_strategy.md §6 |
| 看 Tri-track (A SSL / C Action Cond / X X-VLA 官方) 架构 | cross_embodiment_strategy.md §7 |
| 看 Task A 整体 roadmap | task_a_master_plan.md |
| 看 SFT-DAgger 主轨闭环流程 / Form C 决策 / 失败模式分类 | dagger_implementation_plan.md |
| 看 AWBC / RECAP advantage 4-Step pipeline / 升级路径 | awbc_implementation_plan.md |
| 看 RLT 实施方案 / 4-Phase 路线 / Hybrid Fork + Adapter | rlt_implementation_plan.md |
| 看 DAgger vs AWBC vs RLT 任务路由 (我的失败该走哪条?) | dagger_implementation_plan.md §2 + awbc_implementation_plan.md §2 + rlt_implementation_plan.md §2 |
| 看本 fork 与官方差异 / 改了什么 / 为什么 | official_diff_and_risk_analysis.md |

## 跨场景跳转

- 战略对应的具体 plan 与 history → `../../training/future_plans/` 和 `../../training/history/`
- ckpt 部署到真机 → `../inference/`
- 训练集群运维 → `../training_ops/`
