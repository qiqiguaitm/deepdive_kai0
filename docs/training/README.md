# docs/training — 训练文档总索引

> **目的**: 提供 deepdive_kai0 训练相关全部文档的导航入口, 帮助 AI agent 与人按需精准定位。
> **更新**: 2026-05-25 重构成 "顶层索引 + future_plans/ + history/" 三层结构。

---

## 目录结构

```
docs/training/
├── README.md                                # ← 你在这里 (顶层索引)
├── future_plans/                            # 未来训练计划
│   ├── README.md                            # 全部未来计划简表 + 状态总览
│   └── plans/                               # 单个未来实验计划 (一文件一任务)
│       ├── pytorch_native_vis_v2_full.md
│       ├── ssl_phase_pretrain_pipeline.md
│       ├── xvla_track_x_curriculum.md
│       └── ...                              # 旧 *_plan.md 系列
├── history/                                 # 训练历史记录
│   ├── README.md                            # 全部历史实验简表 + best 排行榜
│   └── experiments/                         # 单个实验记录 (一文件一实验或一主题)
│       ├── 00_training_history.md           # ⭐ master 历史汇总 (TL;DR 排行榜)
│       ├── training_reproduction_log.md     # kai0 复现日志
│       ├── task_a_*_results.md              # Task A 各组结果
│       ├── task_p_*_results.md              # Task P 全解冻对照
│       ├── xvla_conditioning_methods_results.md
│       ├── conditioning_vs_action_representation_ablation.md
│       ├── training_cli_notes.md            # CLI 用法 (reference)
│       ├── wandb_monitoring.md              # W&B 监控 (reference)
│       └── dynamic_dataset_workflow.md      # 数据集动态 workflow (reference)
└── analysis/                                # 跨实验对比 + 反直觉归因 (新增 2026-05-26)
    ├── README.md                            # 索引 + 与其他目录的边界
    └── data_scale_vs_quality_vis_v2_full_vs_pure_200.md   # 数据量增大反而 MAE 变差归因
```

---

## AI 逐层查找指南

当一个 agent 接手训练相关任务时, **按以下顺序查找**:

### Step 1: 判断意图 — 这是要做的事 (plan) 还是已经做过的事 (history)?

| 你想做什么 | 去哪里 |
|---|---|
| **启动新实验** / 看下次该跑什么 | `future_plans/README.md` |
| **看某个实验跑得怎么样** / 复盘 | `history/README.md` |
| **看历史最好 MAE / 排行榜** | `history/experiments/00_training_history.md` |
| **跨实验对比 / 反直觉归因 / "为什么 X 比 Y 差"** | `analysis/README.md` |
| **vis 数据精选子集对比 (smooth_800 + 5day_recent 时间精选)** | `history/experiments/task_a_vis_curated_subset_experiments.md` |
| **CLI 怎么写 / wandb 怎么看 / 数据集 workflow** | `history/experiments/training_cli_notes.md` / `wandb_monitoring.md` / `dynamic_dataset_workflow.md` |

### Step 2: 进入对应子目录, 读它的 README

- `future_plans/README.md` 列出所有未来计划的简表 + 状态, 并标注每个 plan 文件名
- `history/README.md` 列出所有已完成实验的简表 + best 结果, 并标注每个 record 文件名

### Step 3: 打开具体单文件

按 README 指引精读对应文件。

---

## 上游文档 (跨目录)

未来计划的**战略上游** (跨数据集/跨模型架构的整体规划) 在 `docs/deployment/` 下:

| 文档 | 用途 |
|---|---|
| `docs/deployment/strategy/cross_embodiment_strategy.md` | ⭐ 跨本体战略 (3 robots / 4-层 ROI / norm-stats 实证 / Conditioning / RTC-TAC / Tri-track / 决策点) |
| `docs/deployment/inference/realtime_vla/` (4 文件: strategy / roadmap / v1_triton_log / layer_b_plan) | 实时推理优化路线 (PyTorch+Triton 选项 X) |
| `docs/deployment/training_ops/` (overview + storage_and_env + ssh_and_credentials + data_sync_tos + submission/) | 训练集群 (uc01-03 / Robot-North-H20 / robot-task / gf0-3) |
| `docs/deployment/README.md` | ⭐ deployment 总入口 (6 场景多叉树, 跨场景任务流程) |

**何时去看 deployment**: plan 涉及 Track A/C/X 的战略边界决策 (3 robot embodiment gap / Conditioning 注入点选择 / TAC 集成方案), 或需要查 norm-stats / 假说矩阵的实证依据时。

---

## 写新文档的约定

- 新计划: 加文件到 `future_plans/plans/<name>.md`, 同时在 `future_plans/README.md` 表里加一行。
- 实验完成: 移动该 plan 到 `history/experiments/<name>_results.md` (重命名加 `_results` 后缀), 同时在 `history/README.md` 表里加一行 + 更新 `00_training_history.md` 排行榜。
- 单文件结构尽量简洁: 顶部 1 段背景, 表格化的状态/超参/资源, 末尾决策点。
- 跨 plan 引用用相对路径 (`../plans/foo.md` / `../../history/experiments/bar.md`)。
