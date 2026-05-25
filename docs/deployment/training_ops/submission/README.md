# submission/ — 训练任务提交 (3 路径)

> **场景**: 在 deepdive_kai0 项目里有 3 条互补的"提任务"路径, 本目录每条一份文档。

## 3 路径对比

| 路径 | 适用场景 | 状态 |
|---|---|---|
| **`volc_ml_platform.md`** | 提 Volc ML Platform 集群任务 (cn-beijing Robot-North-H20 / cn-shanghai robot-task), 16 卡 + 集群 RDMA | 主要生产路径 |
| **`gf0_control_plane.md`** ⭐ | 在 gf0 一台机器上统一管理 Volc 任务 + uc01/02/03 任务 (2026-05-21 起推荐) | 日常运维推荐 |
| **`uc_cluster_jobs.md`** | uc01-03 直连启动 + 24 GPU HSDP/FSDP 3-host 集群 (RDMA) | 大集群训练 |

## 文件清单

| 文件 | 行数 | 用途 |
|---|---|---|
| [`volc_ml_platform.md`](volc_ml_platform.md) | ~172 | Volc YAML/SDK 模式 + 16 卡 H20 YAML 配置要点 + region/queue mapping + image_cr |
| [`gf0_control_plane.md`](gf0_control_plane.md) | ~361 | gf0 安装 volcengine SDK / mlp CLI 速查 / queue mapping / 镜像选择 / vsubmit 工具 |
| [`uc_cluster_jobs.md`](uc_cluster_jobs.md) | ~423 | gf0 → SSH 管理 uc01-03 + uc 单机 8 GPU 启动 + uc01+uc02+uc03 24 GPU RDMA HSDP/FSDP 集群训练 |

## 按需求找文件

| 你想做什么 | 去 |
|---|---|
| 提 Volc 任务但还没在 gf0 上设置 | volc_ml_platform.md (基础 SDK + YAML) |
| 用 mlp CLI 列/停/详情查任务 | gf0_control_plane.md (CLI 速查) |
| 批量提交多个 YAML 任务 | gf0_control_plane.md (vsubmit + SDK auto-submit) |
| 知道 cn-beijing / cn-shanghai 哪个 queue 跑哪种任务 | volc_ml_platform.md 或 gf0_control_plane.md (queue mapping 表) |
| 经 gf0 ssh 到 uc 跑训练 / 收 log | uc_cluster_jobs.md (gf0 → uc SSH 管理段) |
| 配 uc 三机 RDMA HSDP 24 GPU 训练 | uc_cluster_jobs.md (§12 3-host HSDP/FSDP 段) |

## 跨场景跳转

- 提任务前需要确认数据/ckpt 在位 → `../storage_and_env.md` + `../data_sync_tos.md`
- 服务器全景 / 单机 quick start → `../overview.md`
- SSH 设置前置 → `../ssh_and_credentials.md`
