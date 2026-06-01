# submission/ — 训练任务提交 (3 路径)

> **场景**: 在 deepdive_kai0 项目里有 3 条互补的"提任务"路径, 本目录每条一份文档。

## ⭐ 提交前检查清单 (Pre-Submit Checklist)

> 提**任何**新训练任务前逐项确认。前 3 项是迁移 / git-pull 后新增的硬约束, 漏了会读错数据 / 跑旧 config / norm 报错。

1. **数据在 `self_built` 规范位置** — 三端 (gf0/gf3/uc) 已统一为 `kai0/data/Task_A/`:
   - 构建数据集 → `self_built/<name>/`;原始采集 base → `vis_base/`;HF 官方 → `kai0_base/kai0_dagger/kai0_advantage/`。
   - config.py 的 `repo_id` / `repo_ids` / `inline_eval_val_root` 指向对应机器路径(gf0 `/vePFS/...`、gf3 `/vePFS-North-E/...`、uc `/data/shared/.../kai0/data/...`)。
   - 规范详见 `../storage_and_env.md §2.3` + `train_scripts/kai/data/README.md`。

2. **norm_stats 已算** ⚠️ — `train.py` **不自动算** norm_stats(只 `shutil.copy`)。提交前必须在数据所在机器跑:
   ```bash
   python scripts/compute_norm_states_fast.py --config-name <config>
   ```
   否则 Normalize transform 会用错/缺失统计(详见 `uc_cluster_jobs.md §12.8 陷阱 D`)。

3. **config 已 commit + push** ⚠️ (gf3 / uc 关键) — gf3 + uc01/02/03 由 **1-min git pull cron 镜像 GitHub main (`reset --hard`)**。
   - 改完 `config.py` 等代码 **必须在 gf0 `git commit && git push origin main`**, 等 ~1 分钟让目标机 pull 到, 再提交训练。否则目标机跑的是**旧 config**(路径/超参不一致 → 崩或读错数据)。
   - **不要直接在 gf3/uc 改代码**(会被下次 reset 覆盖)。gf0 本地即 main 源, 改完即时生效。

4. **init ckpt 在位** — `weight_loader` 指向的 base ckpt(如 `base_init_ckpts/pi05_base/params`、`checkpoints/Task_A/mixed_1/params`)在目标机存在。

5. **queue 有余量 + 镜像/挂载正确** — `mlp job list` 查目标 queue 空闲 GPU(见 `gf0_control_plane.md §5.6.c.2`);`ImageUrl` 拼写正确(`cn-beijing` 别拼成 `bejing`);cn-beijing 队列 vePFS 必须配 `SubPath: /vis_robot`。

6. **ckpt/log 落地路径** — 单机训练走 symlink trick 落本地盘(**别直接写 NFS/vePFS 的 `checkpoints/` 真实路径**);volc 任务写 vePFS `checkpoints/<config>/<exp>/`, 日志重定向到 vePFS `logs/`。

## ⚠️ 踩坑经验 (提交/排障必读)

| 文档 | 范围 |
|---|---|
| [`training_pitfalls_common.md`](training_pitfalls_common.md) ⭐ | **跨集群共性坑** — norm_stats 不自动算 / 绝对 repo_id 被新 hub 拒 / 数据集视频目录命名 / init 按 size 校验 / TOS 嵌套 / eval prompt 默认错 / inline-eval 静默失败 / config 先 push。文末附"一个新数据集→提交训练完整前置链"7 步速查 |
| [`volc_ml_platform.md`](volc_ml_platform.md) §"Volc 特有踩坑" | Volc cnbj/cnsh — 卡 Deploying=资源被占(gang-sched)/镜像缓存 vs 多机 tradeoff / VOLC_REGION 必设 / SubPath 否则 403 / Status.State 字段 / 多机 orbax race |
| [`uc_cluster_jobs.md`](uc_cluster_jobs.md) §12.10-12.11 | uc01/02/03 — ssh alias↔eth1 错位 / JAX coordinator 启动时序(proc0 先听 :15830)/ ICMP 滤但 TCP 通 / SSH 过载走 NFS 读 log / num_workers=16×节点 |

## 3 路径对比

| 路径 | 适用场景 | 状态 |
|---|---|---|
| **`volc_ml_platform.md`** | 提 Volc ML Platform 集群任务 (cn-beijing Robot-North-H20 / cn-shanghai robot-task), 16 卡 + 集群 RDMA | 主要生产路径 |
| **`gf0_control_plane.md`** ⭐ | 在 gf0 一台机器上统一管理 Volc 任务 + uc01/02/03 任务 (2026-05-21 起推荐) | 日常运维推荐 |
| **`uc_cluster_jobs.md`** | uc01-03 直连启动 + 24 GPU HSDP/FSDP 3-host 集群 (RDMA) | 大集群训练 |

## 文件清单

| 文件 | 行数 | 用途 |
|---|---|---|
| [`training_pitfalls_common.md`](training_pitfalls_common.md) ⭐ | ~76 | 跨集群共性踩坑 (数据/init/eval/config) + 新数据集→提交 7 步前置链 |
| [`volc_ml_platform.md`](volc_ml_platform.md) | ~230 | Volc YAML/SDK 模式 + 16 卡 H20 YAML 配置要点 + region/queue mapping + image_cr + "Volc 特有踩坑" |
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
