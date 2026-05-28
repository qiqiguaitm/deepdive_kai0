# DAgger / SFT 迭代实施方案

> **目的**: 把 deepdive_kai0 现有的 "录数据 → 训练 → 部署 → 找失败 → 加新示范" 闭环固化为可执行的 master plan, 与 RLT 轨 ([`rlt_implementation_plan.md`](rlt_implementation_plan.md)) 互为分工.
> **现状**: **主轨, 持续运行中, 完成度 ≈ 80-85%**. Task_A flatten-fold (mix_apr28_450 / vis_v2_full / Task_A_base 系列) 已经在这条闭环上迭代了若干轮. ROS2 采集 + Web Data Manager + 训练管线均 production-ready; 仅 3 个自动化 Gap (失败检测 / best ckpt 自动选择 / failure auto-trigger) 待补.
> **本文角色**: 不是教程, 是 master plan — 已实现代码盘点 (§3) + Gap 优先级 (§4) + 流程拓扑 (§5) + 决策判断 (§6).

---

## 1. 算法定位 (一句话)

监督学习 + DAgger (Dataset Aggregation): VLA 部署后跑真机, 操作员看到失败时 *完整 teleop* 把这条轨迹做对, 把 `(obs, a_human)` 加入训练集, 重新 SFT 整个 VLA (~5B 参数). 学习信号是 100% 的 imitation loss (flow matching MSE), 不依赖 reward.

---

## 2. DAgger 适配 / 不适配场景 (任务路由)

| 任务类型 | 是否走 DAgger | 理由 |
|---|---|---|
| **Task_A flatten-fold 主轨** | ✅ 主线 | 长程任务, 多 sub-phase, reward 难定义, SFT 通用性强 |
| **多任务通用 ckpt (mix 系列)** | ✅ 必选 | RLT task-specific, 不适合 mix; mix_apr28_450/600 路线由 DAgger 持续滚动 |
| **新本体能力 bootstrapping** | ✅ 必选 | 任何 0 → 1 的能力建立都需要 demo 提供初始分布 |
| **任务一 hand-cube / 任务二 hand-stack** | ✅ 主轨 | 已部署, 失败时优先走 DAgger 加示范 (除非确认是 action edit 类问题) |
| **vis_v2_full 夹爪 mid-stuck (0.027 卡)** | ❌ 不推荐 | reward 极简 + action edit 单 channel, 走 RLT 4–8h 搞定 vs SFT 数天 |
| **未来 Piper 精细插拔 / 螺丝** | ❌ 不主推 | 论文证据: RLT 速度上限可超 teleop, DAgger 上限 ≤ teleop |
| **EE jiggle / 走 3 退 1** | ❌ 不相关 | deploy 层问题, 不是模型能力问题 |

**DAgger 适配的失败模式特征**: (a) 失败是 "模型不知道这种情况怎么办" (需要新示范扩分布), (b) 失败横跨多个 sub-phase (单点 reward 难拆分), (c) 任务长程 (≥ 1 min/episode).

---

## 3. 已实现代码盘点 (2026-05-26 现状)

> **当前完成度 ≈ 80-85%** — 主轨已 production-ready, 缺口集中在 *自动化决策* (失败检测 / best ckpt 自动选择). 不影响手工迭代.

### 3.1 ROS2 采集端

| 文件 | 行数 | 角色 | 状态 |
|---|---|---|---|
| `ros2_ws/src/piper/scripts/dagger_recorder_node.py` | 601 | **DAgger 失败接管状态机 + 录制** — 四态机 POLICY_RUN → ALIGNING → HUMAN_RECORD → RETURNING; 物理按键触发 intervention; 调用 EpisodeWriter 落 LeRobot | 4/5 |
| `ros2_ws/src/piper/scripts/autonomy_recorder_node.py` | 368 | **自动推理落数据** — 首帧触发, 一启动一 episode, 落 parquet + mp4 + zarr | 4/5 |
| `ros2_ws/src/piper/scripts/arm_master_servo_node.py` | — | **Master arm 伺服 + 按键发布** — `/master/button_pressed` (Bool) 聚合两臂按键给 dagger_recorder | 3/5 |
| `ros2_ws/src/piper/scripts/arm_teleop_node.py` | — | **Master arm 遥操作配置** — passive_init + `/teach/master_config_*` 切换 0xFA/0xFC 模式 | 4/5 |
| `ros2_ws/src/piper/launch/dagger_launch.py` | — | **DAgger ROS2 launch** — IncludeLaunchDescription autonomy_launch.py + master_servo + dagger_recorder; `record_enable:=false` 禁 autonomy_recorder 避免重复 | 5/5 |
| `start_scripts/start_dagger_collect.sh` | — | **一键启动** — wrapper 复用 start_autonomy.sh CAN/Camera 基础设施, 支持 `--task / --prompt / --subset` 参数 | 4/5 |

### 3.2 Web Data Manager (sim01)

| 文件 | 行数 | 角色 | 状态 |
|---|---|---|---|
| `web/data_manager/backend/app/dataset_writer.py` | 424 | **LeRobot 格式化写盘** (EpisodeWriter) — 30Hz 同步采集 14 维 state/action; 三路视频并行 AV1; parquet 流式; TOS sync 钩子 | **5/5 生产级** |
| `web/data_manager/backend/app/recorder.py` | — | **UI 状态机** — StartRecordingReq → 校验 → _EpisodeWriter → 后台 worker 拉帧 → Finalize / Discard; 支持踏板 auto-trigger | 4/5 |
| `web/data_manager/backend/app/ros_bridge.py` | — | **ROS2 → WebSocket 桥** — 订阅相机/关节 topic, 推流前端; 与 dagger_recorder 共享 14d state/action layout | 4/5 |
| `web/data_manager/backend/app/stats_service.py` | — | **数据统计独立服务** — 磁盘扫描 parquet/mp4 真实计数; watchdog 增量; SQLite 索引; 不信任录制 worker | 4/5 |
| `web/data_manager/backend/app/templates.py` | — | **任务 / Prompt 模板存储** — Task/Subset/Prompt 元组; 采集员下拉只读, 管理员可编辑; YAML 持久化 | 3/5 |
| `web/data_manager/backend/app/main.py` | 392 | **FastAPI 主服务** — REST + WebSocket 双通道; `/recorder/start\|save\|discard` `/api/episodes` `/api/stats` `/ws/status`; 双模式权限 | 4/5 |
| `web/data_manager/frontend/src/components/*.tsx` (9 个 React 组件) | — | **前端 UI** — StatusBar / CameraGrid / ArmsPanel / Controls / ReplayPanel / StatsCard / TemplateManager 等; MJPEG 三路相机; 回放 Range mp4 + 关节曲线; 按 Task/Subset/操作员分组柱状图 | 4/5 |

### 3.3 训练侧数据管线

| 文件 | 角色 | 状态 |
|---|---|---|
| `train_scripts/data/build_task_a_mixed.py` | **多源数据 mix** — 聚合 base + dagger + advantage 三类; 支持 dated/flat layout; norm_stats 统一; parquet 重索引 | 4/5 |
| `train_scripts/data/label_dagger_positive.py` | **DAgger 标签化** — 复制 Task_A/dagger 为 dagger_labeled, 所有帧标 `Advantage: positive`, 复用 AWBC 训练格式 | 4/5 |
| `train_scripts/launch/run_awbc_daggeronly_gf0.sh` | **纯 DAgger 数据训练入口** — config `pi05_flatten_fold_awbc`, batch=256, fsdp=1; inline-eval MAE@1 监控 | 3/5 |
| `train_scripts/launch/run_dagger_infer_gf0.sh` / `gf1.sh` | **集群推理 (stage 分类)** — 对 dagger 数据跑 advantage 估计器, 输出 metrics JSON | 3/5 |
| `train_scripts/stage_classifier/viz_dagger.py` | **DAgger episode 可视化** — 读 dagger + stage 预测 JSON, 生成 10 帧 contact sheet PNG + timeline | 3/5 |
| `train_scripts/stage_classifier/infer_dagger.py` | **Stage classifier 推断** — 对 DAgger episode 跑 stage-progress estimator, 标注 pseudo-GT 用于失败分类 | 2/5 |

### 3.4 已 production-ready 文档

| 文档 | 角色 |
|---|---|
| [`../data_collection/data_manager_plan.md`](../data_collection/data_manager_plan.md) | Data manager UI 完整规范 (采集员/管理员双模式, 权限矩阵, 10+ 功能模块) — 5/5 完整 |
| [`../data_collection/teleoperation_guide.md`](../data_collection/teleoperation_guide.md) | 遥操作 SOP |
| [`../training_ops/data_sync_tos.md`](../training_ops/data_sync_tos.md) | TOS 中心枢纽 + 跨服务器 sync |
| [`../training_ops/submission/`](../training_ops/submission/README.md) | 3 种提任务路径 (Volc / gf control plane / uc) |

---

## 4. 未实现的 Gap (按优先级)

| # | Gap | 优先级 | 预估工作量 | 当前手工兜底 | 建议解决方案 |
|---|---|---|---|---|---|
| G1 | **Stage 5 自动失败检测** | 中 | 1-2 周 | 100% 人工按键触发 | 已被 §4.5 决策部分解决 — 物理柔性开关本身就是 trigger, 自动检测可后置 |
| G2 | **Stage 4 best ckpt 自动选择** | 高 | 3-5 天 | 文档启发式 (MAE@1 最低点 ±2k step EMA), 手工 pack | 把启发式脚本化, 嵌进 `run_awbc_*` 训练后步骤, 自动 pack tar → TOS → sim01 |
| G3 | **Stage 1 失败 auto-trigger** | 低 | 已并入 §4.5 | start_dagger_collect.sh 支持参数但流程引导不足 | 物理柔性开关 = 自然 trigger, 见 §4.5 |
| G4 | **录制设计决策 (Form A/B/C)** | **高** | 4-5 天 | 当前 Form A (无 inference 段), 与官方 KAI0 不兼容, 也不支持 RECAP | **§4.5 已决定 — Form C + 物理柔性开关** |

**ROI 顺序建议**: G4 (Form C + 物理开关, 4-5 天, 同时解决 G1/G3 + piper_review O3 剩余 40%) → G2 (3-5 天, 立即省人时, 风险最低) → G1 自动失败检测 (1-2 周, 可作为 G4 之后的增量) → 未来 RECAP advantage estimator 升级 (见 §8.2).

---

## 4.5 录制设计决策 — Form C + 物理柔性开关 (2026-05-26 拍板)

### 背景: 三种形态的对比

| Form | 录什么 | Episode 边界 | schema | 训练用法 |
|---|---|---|---|---|
| **A (当前 deepdive_kai0)** | 仅 (1,1) human 段 | 每次接管 = 1 ep | 无 mask | 全标 positive, AWBC |
| **B (用户初提案)** | (0,0) + (1,1) 都录, 一个 mixed ep | (0,0)→(1,1)→(0,0) cycle = 1 ep | 加 intervention_mask 列 | HG-DAgger / IL+RL |
| **C (官方 KAI0 + 推荐)** | (0,0) 录到 `inference/`, (1,1) 录到 `dagger/`, 两 dataset 分离 | inference 段 d-trigger 截断 / dagger 段 Space-trigger 开始 | 无 mask (路径区分) | RECAP advantage 升级 / AWBC / 自由 mix |

### 决定: Form C + 物理柔性开关

**双 dataset 分离 + 双柔性开关二段 trigger** = 官方设计哲学的物理化实现.

**理由**:
1. **与官方 KAI0 上游 100% 兼容** — `_inference_hdf5/` + `_dagger_hdf5/` 路径分离, schema 不动
2. **与现有 `train_scripts/data/label_dagger_positive.py` 自然衔接** — 现在的 "全 dagger 标 positive" 是 RECAP Stage 3 的简化, 未来可无缝走完整 RECAP (见 §8.2)
3. **物理开关 = 官方键盘 `d`+`Space` 二步 trigger 的物理化** — 操作员本来就要打开两个柔性开关, 不引入额外动作
4. **完美 fit piper_review §B (软件级主从)** — 全程 firmware role 不变, MotionCtrl_1 软件触发拖动 (合并 O3 剩余 40%)
5. **解决静止数据问题** — 第二个开关打开瞬间 = 操作员手已到位 + 已对齐, 不需要运动检测 gate

### 状态机 (基于 `(L_open, R_open)` 二位组)

```
(0,0) POLICY_RUN     — 策略跑, slave 跟策略; *持续录 inference episode* 到 <task>/inference/<date>/
       │ L=1                                            │ R=1
       ▼                                                ▼
(1,0) ALIGN_LEFT     — 策略立即停; *finalize 当前 inference ep*;     (0,1) ALIGN_RIGHT (对称)
      slave hold; 左 master 已进物理拖动 (操作员握住);
      右臂仍在 hold (不动); 不录任何数据
       │ R=1
       ▼
(1,1) HUMAN_RECORD   — *启 dagger EpisodeWriter* 写 <task>/dagger/<date>/;
      双 master 拖动 → 双 slave 跟随, 正式录制
       │ L=0                                            │ R=0
       ▼                                                ▼
(0,1) RETURN_RIGHT   — *finalize dagger ep*; 等另一个开关也关         (1,0) RETURN_LEFT (对称)
       │ R=0
       ▼
(0,0) POLICY_RUN     — master grag_teach=0x02 软件复位; 策略 resume;
                      *启动新 inference EpisodeWriter*
```

### 数据落盘路径

```
/data1/DATA_IMP/KAI0/<task>/
├── inference/<date>-v2/
│   ├── data/chunk-000/episode_*.parquet      ← 策略 rollout 段
│   ├── videos/chunk-000/{top_head,hand_left,hand_right}/episode_*.mp4
│   └── meta/{episodes.jsonl, tasks.jsonl}
└── dagger/<date>-v2/
    ├── data/chunk-000/episode_*.parquet      ← 人类接管段
    ├── videos/chunk-000/{top_head,hand_left,hand_right}/episode_*.mp4
    └── meta/{episodes.jsonl, tasks.jsonl}
```

**Episode 索引独立** (与官方 KAI0 一致). 一次完整 cycle 产 1 个 inference ep + 1 个 dagger ep.

### 4 个边界 case 兜底

| Case | 处理 |
|---|---|
| 开关 toggle vs deadman | Piper 当前柔性开关是 deadman 类型 (松手回弹), 完美 fit. 若未来换硬件, 加 30s 无运动 timeout 兜底 |
| (1,0) 状态操作员放弃 (L 又关) → (0,0) | 不启 dagger writer, 也不重启 inference writer (继续用刚 finalize 的 inference ep 的延续); 不报警 |
| 录制中途任一开关关 ((1,1) → (1,0)) | 立即 `dagger_writer.finalize()` 保住已录数据, 进 RETURN 子状态等另一开关也关 |
| 误触回弹 (< 300ms) | ROS subscriber 层 debounce 300ms, 状态稳定持续才转移 |

### 实现工作量与 Phase 拆分 (见 task list)

| Phase | 任务 | 工作量 | 关键文件 |
|---|---|---|---|
| **D1** | arm_master_servo 发布 `/dagger/switch_state_left,right` + 300ms debounce | 0.5 天 | `ros2_ws/src/piper/scripts/arm_master_servo_node.py` |
| **D2** | dagger_recorder 改 6 态状态机, inference + dagger 双路径 | 1.5 天 | `ros2_ws/src/piper/scripts/dagger_recorder_node.py` |
| **D3** | arm_teleop_node 4 处 MasterSlaveConfig 替换为 MotionCtrl_1 (合 O3 剩余 40%) | 1 天 | `ros2_ws/src/piper/scripts/arm_teleop_node.py` |
| **D4** | 状态机转移加 GetArmStatus 健康检查 (piper_review O2) | 1 天 | `dagger_recorder_node.py` + `_utils.py` |
| **D5** | 真机验证 5 cycle + linkage_config 全程 0xFC 验证 | 1 天 | — |

**总计 5 天** — 一次性解决: 录制设计 + 静止数据 + 物理开关 trigger + O3 firmware 软件化 + 健康检查.

---

## 5. 标准 DAgger 闭环 (5 阶段, 含实现文件)

```
        ┌──────────────────────────────────────────────────────────────┐
        │                  当前 ckpt 真机部署                           │
        │       sim01 V1 Triton (P50 32ms) → ROS2 → Piper             │
        └────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Stage 5: 失败模式分析 / 决定下一轮要加什么 demo   [50% — Gap G1]│
        │     • 录失败 episode (dagger_recorder POLICY_RUN 阶段)        │
        │     • viz_dagger.py 生成 contact sheet + timeline             │
        │     • 分类: (a) 分布外? (b) action edit? (c) prompt 错?       │
        │     • (b) → 转 RLT; (a)/(c) → 继续 DAgger                     │
        │     ⚠️ 目前 100% 人工触发, 缺自动失败检测                       │
        └────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Stage 1: 数据采集 (master arm teleop)   [85% — Gap G3]       │
        │     • start_dagger_collect.sh → dagger_launch.py              │
        │     • dagger_recorder_node.py 状态机                          │
        │       POLICY_RUN → ALIGNING → HUMAN_RECORD → RETURNING        │
        │     • 物理按键触发 intervention (arm_master_servo_node)        │
        │     • 操作员针对失败做 demo (10–100 ep)                       │
        └────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Stage 2: 数据预处理 + Web UI 元数据   [100%]                  │
        │     • dataset_writer.py::EpisodeWriter (30Hz, AV1, parquet)   │
        │     • web/data_manager UI: Task/Subset/Prompt 模板下拉        │
        │     • stats_service.py: SQLite 索引 + watchdog 增量            │
        │     • compute_norm_stats_fast.py: norm stats                  │
        └────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Stage 3: Sync 到训练集群   [100%]                            │
        │     • dataset_writer 内置 TOS sync 钩子                       │
        │     • gf1/gf0 ← TOS (data_sync_tos.md)                       │
        │     • 老数据保留 (不 --delete hand-cube/hand-stack)            │
        └────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Stage 4: 集群 SFT 训练   [95% — Gap G2]                      │
        │     • build_task_a_mixed.py: base + dagger + advantage 聚合   │
        │     • label_dagger_positive.py: AWBC 标签化                   │
        │     • run_awbc_daggeronly_gf0.sh / run_dagger_infer_*.sh      │
        │     • config: pi05_flatten_fold_awbc, batch=256, fsdp=1       │
        │     • inline-eval MAE@1/@10/@25/@50 实时监控                  │
        │     ⚠️ best ckpt 自动 pack 缺, 仍需手工选 EMA 平均点            │
        └────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  新 ckpt 真机部署 → 循环回到 Stage 5                          │
        └──────────────────────────────────────────────────────────────┘
```

各 Stage 详细文件清单见 §3.1-3.4; 本节是流程拓扑 + Gap 标记.

---

## 6. 各 Stage 关键决策点 (DAgger 特有的判断)

### Stage 5 — 失败模式分类 (决定走 DAgger 还是 RLT)

| 观察到的失败 | 分类 | 路径 |
|---|---|---|
| 同样的物体姿态多次, 模型动作完全不一致 | 分布外 / 数据稀缺 | DAgger (加 demo) |
| 物体 / 任务认错 (拿成另一个 / 步骤跳跃) | prompt 理解错 | DAgger (加 demo) 或 prompt 工程 |
| 动作连贯但精度差几 mm (插不进 / 抓不稳) | action edit 类 | **RLT** (走 [`rlt_implementation_plan.md`](rlt_implementation_plan.md)) |
| 夹爪 mid-stuck / 速度永远偏慢 | action 单 channel 偏移 | **RLT** |
| 长程组合错误 (前 80% 对, 最后 critical phase 错) | 看 critical phase 类型 — 精度差 → RLT; 步骤错 → DAgger | 二者择一 |

### Stage 1 — Demo 数量决策

| 失败严重度 | 建议 demo 数 | 示例 |
|---|---|---|
| 单一新状态没覆盖 | 5–20 ep | 物体新颜色 / 新摆放角度 |
| 整个 sub-phase 不会 | 30–80 ep | 第一次教 "对折" |
| 任务从零起 | 200–500 ep | 新任务 bootstrap |

### Stage 4 — config 选择 (摘 [`../training_ops/submission/`](../training_ops/submission/README.md))

| 实验类型 | config 名 | 训练步数 | 集群 |
|---|---|---|---|
| 单任务 pi0.5 from base | `pi05_<task>_finetune` | 30k–60k | gf1 |
| 单任务 pi0 from base | `pi0_<task>_finetune` | 30k–60k | gf1 |
| Mix 多任务 | `pi05_mix_<dataset>` | 60k–100k | gf0 + gf1 并行 |
| ablation (norm_stats / data subset) | 同上 + suffix | 通常 30k 够 | 任一 |

监控指标: `val MAE@1` (论文级 metric, 单步精度) 是首要看的, 不是 `train loss`. 内联 eval 每 2000 step 一次.

### Stage 4 → Stage 5 衔接: best ckpt 选择

不是直接选 final step. 标准做法 (从历史 `00_action_only_finetune_history.md`):
1. 看 `inline-eval` 曲线, 找 MAE@1 最低点
2. 该 step 在 +/-2000 step 内挑 EMA 平均最稳的一个
3. auto-pack 自动 tar 这个 ckpt 到 `/vePFS/tim/workspace/deepdive_kai0_tmp/data/<run>_best.tar`
4. tosutil 拉到 sim01, 解压到 `/data1/DATA_IMP/checkpoints/ckpt_v1/<run>/`

---

## 7. 当前活跃 DAgger 实验追踪 (动态)

| 实验线 | 当前 ckpt | 状态 | 触发原因 |
|---|---|---|---|
| **Task_A flatten-fold mix 系列** | `mix_apr28_450` | 已部署, 持续加 demo | 持续优化 long-horizon 表现 |
| **Task_A 单任务 base** | `Task_A_base_delta_step49999` | 已部署 | 含 idle demo 的纯单任务对比 |
| **vis_v2_full 单任务** | `vis_v2_full_step49999` | 已部署, gripper 卡 | 准备切给 RLT POC ([`rlt_implementation_plan.md`](rlt_implementation_plan.md) Phase 2) |
| **任务一 hand-cube (PP)** | (待补) | 已部署 | 观察 1–2 周 |
| **任务二 hand-stack (PS)** | (待补) | 已部署 | 观察 1–2 周 |

**每次新一轮 DAgger 启动时, 更新这张表**, 历史 ckpt 状态归档到 [`../../training/history/00_action_only_finetune_history.md`](../../training/history/00_action_only_finetune_history.md).

---

## 8. DAgger 升级路径

DAgger 主轨完成后, 有 2 条独立升级路径. 路径选择由失败模式驱动.

### 8.1 转 RLT (Physical Intelligence RL Token)

DAgger 不能解决的 4 种失败 → 转 RLT (详见 [`rlt_implementation_plan.md`](rlt_implementation_plan.md) §2):

1. **action edit 类失败** — 模型动作大方向对, 但精度差几 mm
2. **单 channel offset** — 夹爪 / 单关节系统性偏向某值
3. **速度上限受 teleop 限制** — DAgger 上限 ≤ 操作员; RLT 可突破 (论文 Ethernet 一半 ep 比所有 teleop 快)
4. **critical phase 隔离明确** — 任务可定义 ≤ 1 min 的 critical phase, 该 phase reward 信号简单

**hand-off 物理动作**:
- DAgger 训出的最新 ckpt 当作 RLT Stage 0 的 frozen VLA
- 共享同一份 LeRobot 数据 (RLT Stage 1 也用)
- RLT 产物 (actor) 可以单独 deploy, 也可以反过来 — 把 RLT 学到的 residual *蒸馏* 进 DAgger ckpt (论文 §V.B 提到的 "conclude with a short fine-tuning phase"), 但本方案 v1 不做这一步

### 8.2 走 RECAP advantage 升级 (KAI0 上游已开源)

§4.5 的 Form C 决策为这条路径打开了大门 — inference 段的存在是 advantage estimator 训练的必需输入. 详细 4-step pipeline (Stage 0 标 progress → Stage 1 训 estimator → Stage 2 预测 → Stage 3 discretize → Stage 4 AWBC 训练) 见独立方案文档:

→ [`awbc_implementation_plan.md`](awbc_implementation_plan.md)

**简述触发条件**: 简化版 `train_scripts/data/label_dagger_positive.py` 训出的 ckpt 已无法继续提升 (val MAE@1 平台) + 失败仍是 "DAgger 适配" 类 (非 "action edit") → 升级 AWBC.

### 8.3 升级路径对照表

| 当前失败模式 | 推荐路径 | 工作量 |
|---|---|---|
| 分布外 / 模型不会某 sub-phase | DAgger 加 demo 继续 | 1-2 周 / 实验 |
| DAgger 加 demo 仍训不出 / val MAE 卡 | **RECAP / AWBC 升级** → [`awbc_implementation_plan.md`](awbc_implementation_plan.md) | 2-3 周 |
| 单 channel offset / action edit / 速度上限 | **RLT** → [`rlt_implementation_plan.md`](rlt_implementation_plan.md) | 6 周 |

**两条升级路径不冲突, 可串行**: AWBC 训出更强 VLA → RLT 再加 actor 修 critical phase.

---

## 9. 风险与兜底

| 风险 | 概率 | 影响 | 兜底 |
|---|---|---|---|
| 加新 demo 导致旧任务 catastrophic forgetting (mix 实验时常见) | 中 | 中 | 留 base ckpt 不动, 新数据走 mix config 重训; 用 norm_stats ablation 对照 |
| 长时间没复现失败模式, demo 收集成本高 | 中 | 中 | 优先复现率高的 deterministic failure; 难复现的留给视觉化 episode review |
| inline-eval MAE@1 看不出真机表现 (val/train mismatch) | 中 | 大 | 必须真机 eval, 不能只看 MAE; 见 `00_action_only_finetune_history.md` 历史教训 |
| 数据集 schema mismatch (新批次字段不一致) | 低 | 中 | data_manager UI 强制 schema 校验 |
| 集群训练中断 / TOS 凭据过期 | 低 | 中 | training_ops/ssh_and_credentials.md 凭据轮换 SOP |

---

## 10. 与上游 / 相关文档跳转

- 双轨另一支 (RLT 实施方案) → [`rlt_implementation_plan.md`](rlt_implementation_plan.md)
- 跨本体战略主文档 → [`cross_embodiment_strategy.md`](cross_embodiment_strategy.md)
- Task A 项目级主规划 (算力 / 集群分配) → [`task_a_master_plan.md`](task_a_master_plan.md)
- 数据采集 SOP → [`../data_collection/teleoperation_guide.md`](../data_collection/teleoperation_guide.md) / [`data_manager_plan.md`](../data_collection/data_manager_plan.md)
- 跨服务器数据 sync → [`../training_ops/data_sync_tos.md`](../training_ops/data_sync_tos.md)
- 集群任务提交 → [`../training_ops/submission/README.md`](../training_ops/submission/README.md)
- ckpt 命名 / 目录规范 → [`../training_ops/checkpoints_layout.md`](../training_ops/checkpoints_layout.md)
- 训练实验历史 (mix_apr28_450 / vis_v2_full / Task_A_base 等) → [`../../training/history/00_action_only_finetune_history.md`](../../training/history/00_action_only_finetune_history.md)
- sim01 部署 (DAgger 评估端) → [`../inference/sim01_deployment.md`](../inference/sim01_deployment.md)
