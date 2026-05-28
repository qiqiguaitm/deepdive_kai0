# RLT 实施方案 (Physical Intelligence RL Token)

> **目的**: 在已部署的 VLA ckpt 上挂一个轻量 actor + critic, 通过 sparse human reward 在真机 online 微调 critical phase, **不动 VLA 主体**.
> **现状**: 仅方案设计阶段, 未启动. 第 0 个 milestone 是 vis_v2_full gripper mid-stuck POC.
> **互补关系**: 本方案与 [`dagger_implementation_plan.md`](dagger_implementation_plan.md) 是分工而非替代关系 — DAgger 做"任务级新能力", RLT 做"critical phase 精修". 任务路由见本文 §2.
> **上游参考**:
> - 论文: <https://pi.website/research/rlt> (Xu et al. 2025)
> - 非官方实现: <https://github.com/yknxh/rlt-openpi> (本仓库 `/data1/tim/workspace/rlt-openpi`)

---

## 1. 算法定位 (一句话)

VLA 冻结, 在它输出的 action chunk 上加一个 *residual* 修正项. residual 由小 MLP actor (~600K 参数) 输出, 用 TD3-style RL + BC 正则训练. RL state 是 VLA prefix embedding 经过 encoder-decoder 信息瓶颈压成的 `z_rl ∈ ℝ^{2048}`. 训练发生在真机 in-the-loop, 数据 (15 min – 5 h) 来自人按键 reward (s/f/p) + 偶尔 VR/Master arm 接管.

详细算法/pipeline 参考 `/data1/tim/workspace/rlt-openpi/init.md`.

---

## 2. RLT 适配 / 不适配场景 (任务路由)

| 任务类型 | 是否走 RLT | 理由 |
|---|---|---|
| **vis_v2_full 夹爪 mid-stuck (0.027 卡)** | ✅ 首选 POC | reward 极简 (gripper > 0.06 阈值), action edit 单 channel, 4–8h 真机搞定 |
| **未来 Piper 高精度插拔 / 螺丝** | ✅ 首选 | 论文 4 个任务直接对标 (Ethernet / charger / screwdriver / ziptie) |
| **Task_A critical sub-phase 后期卡死** | ⏳ 触发后再定 | action edit 能修 → RLT; 模型表达不够 → DAgger |
| **Task_A flatten-fold 主轨** | ❌ 不适合 | 长程任务, reward 难设计, 走 DAgger ([`dagger_implementation_plan.md`](dagger_implementation_plan.md)) |
| **多任务通用 ckpt (mix 系列)** | ❌ 不适合 | RLT 是 task-specific (一个任务一个 actor head) |
| **EE jiggle / 走 3 退 1** | ❌ 不相关 | 不是 task 层问题, 保持现有 Layer 1 deploy 层方案 |

**RLT 适配的失败模式特征**: (a) 失败可被局部 action edit 修复 (不是 prompt 理解错), (b) success/failure 可用单一信号判定, (c) critical phase 时长可控 (< 1 min/episode).

---

## 3. RLT 接入 deepdive_kai0 的 6 个集成点

RLT 不需要新搭一套, 而是把 rlt-openpi 的抽象接口替换成 kai0 已有的等价物, 并通过 git submodule 隔离 openpi 版本冲突.

### 3.1 Env factory — ROS2 包装代替 DROID

`rlt-openpi/src/rlt_openpi/rollout/robot_env.py` 的三回调接口 (step_fn / reset_fn / get_obs_fn) 与现有 ROS2 stack 一一对应:

| rlt-openpi 接口 | kai0 现有等价物 |
|---|---|
| `step_fn(action)` 单步发送 | `policy_inference_node._publish_action(act)` 内部方法或直接 publish `/joint_command` |
| `reset_fn()` home pose | 现有 reset 服务 (master arm 回 home) |
| `get_obs_fn()` 返回 dict | 现有 `obs_assembler` 已组装好, key 改成 OpenPI schema 即可 |

**新增**: `kai0/src/kai0/rlt_adapters/env_factory.py::make_piper_env`, ~150 行.

### 3.2 Intervention manager — Master arm 代替 Oculus VR

逻辑等价, 替换 controller 接口:

```
论文 (VR):                              kai0 (Master arm):
  按住 grip 键 → 接管                    按下 master arm 启用按钮 → 接管
  VRPolicy.forward(robot_state) → cmd    master arm 关节读数 → cmd
  A 键 → success                         脚踏 / 键盘 s → success
  B 键 → failure                         脚踏 / 键盘 f → failure
```

**新增**: `kai0/src/kai0/rlt_adapters/intervention.py::MasterArmInterventionManager`, ~200 行. 关键点: master → slave 的 action 表示必须与 actor 输出空间对齐 (joint_velocity / ee_pose 残差选定其一).

### 3.3 Reward listener — ROS2 topic 代替 termios 键盘

`rlt-openpi/src/rlt_openpi/rollout/reward.py::HumanReward` (termios cbreak) 在真机现场不适用. 改为 ROS2 subscriber:

```
/rlt/reward_signal  (std_msgs/String, "s" | "f" | "p")
```

操作员通过 VR / 脚踏 / 蓝牙键盘 publish 信号. **修改** `RobotEnv.step` 用 rclpy subscriber 替换 `self._feedback.check()`, ~30 行.

### 3.4 推理路径 — actor 接进 `policy_inference_node`

现有路径:
```
camera → preprocess → VLA (V1 Triton) → ā ∈ ℝ^{H×d} → RTC smooth → publish
```

加入 RLT 后:
```
camera → preprocess → VLA (V1 Triton) ─┬─→ ā ∈ ℝ^{H×d} ──┐
                                       │                  │
                                       └─→ z_{1:M} ─→ ③Encoder → z_rl ─┐
                                                                       │
                                                                       ▼
                                          cat(z_rl, proprio) → x      │
                                                            │          │
                                                            ▼          │
                                                         ④Actor(x, ā[:C])
                                                            │          │
                                                            ▼          │
                                          a = ā[:C] + residual ←───────┘
                                                            │
                                                            ▼
                                                     RTC smooth → publish
```

**RTC × RLT 决策**: 训练时 RTC 关 (RLT chunk 边界 hard switch), 部署时 RTC 工作在 actor 输出 a 上 (不工作在 ā 上), `latency_k` 6 → 12 让 residual 不被立刻覆盖. 若后续发现 jiggle 加剧, 备选: 训练时学 "RTC 平滑后的 ā" 的残差 (改 ReplayBuffer 的 a_tilde 字段).

**Inference latency 增量**: encoder ~5ms (2 层 Transformer on ~500 tokens) + actor ~1ms < 10ms, 远小于 V1 P50 32ms 预算. 直接走 PyTorch, 不需要 Triton 化.

**修改** `ros2_ws/src/piper/scripts/policy_inference_node.py`: 加 `--rlt-actor-checkpoint` 选项, 加载 actor + encoder, 在 ā 出来之后插 actor.forward, ~80 行. 无此 flag 时跟现在一模一样.

### 3.5 训练机 — sim01 复用 V1 Triton

sim01 现在跑 V1 inference server. RLT Stage 2 训练需要同时:
- VLA forward (复用现有 V1 Triton, 走 websocket)
- actor / critic forward + backward (PyTorch, 小, 可走 CPU)
- 与真机 ROS2 通讯 (rclpy in conda env)

不需要新机器. 启动方式:
```
Terminal A: 现有 V1 server (不变)
Terminal B: 新增 RLT trainer, --vla-server ws://localhost:8000 复用 V1 加速
```

**新增** `kai0/src/kai0/rlt_adapters/vla_websocket_client.py`: 改 VLAWrapper 走 websocket 而不是本地 load model, ~150 行. 这是与 rlt-openpi 上游最大的偏离, 但能省真机训练时的 VLA forward 时间. 此修改也可考虑 PR 回 fork (`submodules/rlt-kai0`) 作为 vla_wrapper 的 websocket 模式选项, 见 §3.6.

### 3.6 仓库与依赖隔离 — Hybrid Fork + Adapter

deepdive_kai0 内的 `kai0/` 已经是 openpi 的一个 fork (5 处 model code fix + V1 Triton 后端), 与 rlt-openpi 上游 `pyproject.toml` 钉死的 `openpi @ fdc03f5` 不兼容. 不能在同一 venv 内同时满足两者. 解决方案: **Fork rlt-openpi 作为 git submodule, kai0-specific 接线放在 `kai0/src/kai0/rlt_adapters/`**.

#### 仓库结构

```
deepdive_kai0/
├── kai0/                              ← 现有 kai0 fork (含 5 处 model fix + V1 Triton)
│   └── src/kai0/rlt_adapters/         ← kai0-specific 接线 (新)
│       ├── env_factory.py             make_piper_env (ROS2 桥接)
│       ├── intervention.py            MasterArmInterventionManager
│       ├── reward_listener.py         ROS2 topic subscriber
│       ├── data_transforms.py         Kai0Inputs/Outputs
│       └── vla_websocket_client.py    替换 vla_wrapper 走 V1 server
├── submodules/
│   └── rlt-kai0/                      ← Fork from yknxh/rlt-openpi (新, git submodule)
│       ├── pyproject.toml             改一行: openpi → ../../kai0 (editable)
│       └── src/rlt_openpi/            算法核心, 几乎不改 (~5000 行不计入仓库膨胀)
```

#### 关键修改清单 (在 submodules/rlt-kai0 fork 内)

| 改什么 | 内容 | 行数 |
|---|---|---|
| `pyproject.toml` | `[tool.uv.sources] openpi = { path = "../../kai0", editable = true }` | 1 行 |
| `src/rlt_openpi/vla/vla_wrapper.py` | 加 websocket 模式 (与 V1 Triton server 通讯, 跳过 in-process load 5B 模型) | ~80 行 |
| (可选) `src/rlt_openpi/training/online_rl_trainer.py` | 加 ROS2 reset 同步 hook | ~20 行 |

合计 fork 内改动 **< 110 行**, 上游升级时 `git pull && resolve` 冲突面极小.

#### Venv (新建第 4 个)

`deepdive_kai0/.venv_rlt`:
- Python 3.11
- `pip install -e kai0/` (kai0 fork, 解决 model fix 问题)
- `pip install -e submodules/rlt-kai0/` (RLT 算法)
- `pip install rclpy` (ROS2 jazzy 兼容)
- `pip install msgpack websockets` (走 V1 server)

不需要 `kai0/.venv_5090_trt` (那个是给 V1 server 用的, RLT trainer 是 client).

#### 命令入口

```
# Stage 1 (离线, 在 sim01 GPU 上跑)
conda activate rlt_kai0
python -m rlt_openpi.scripts.train_rl_token \
    --train.vla-config-name kai0_piper_finetune \
    --train.vla-checkpoint-dir ckpt_v1/vis_v2_full/model.safetensors \
    --repo-id task_a_v2 \
    --data-transforms-fn kai0.rlt_adapters.data_transforms.piper_three_camera
#                        ↑ 动态 import deepdive_kai0 内的 adapter

# Stage 2 (真机 in-the-loop)
python -m rlt_openpi.scripts.train_online_rl \
    --env-factory kai0.rlt_adapters.env_factory.make_piper_env \
    --intervention-factory kai0.rlt_adapters.intervention.make_master_arm_intervention \
    --vla-server ws://sim01:8000 \                    # 复用现有 V1 Triton server
    --rl-token-checkpoint .../rl_token.pt \
    --task-prompt "..."
```

`importlib` 在 `rlt-kai0/src/rlt_openpi/rollout/factory.py::_ensure_project_root_on_path()` 中已动态加载用户提供的 import path, 这是 rlt-openpi 现成的解耦机制.

#### 推理时接入 `policy_inference_node`

- 训练完产物 `online_rl_ep<N>.pt` 拷到 sim01 / 控制机
- `policy_inference_node.py` 加 `--rlt-actor-checkpoint`, 用 `from rlt_openpi.models.actor import Actor` 加载
- 部署侧不需要 kai0 fork 与 rlt-kai0 同时存在 — actor 是纯 PyTorch 模块, state_dict 加载即可, 不依赖 openpi 本身

#### 为什么不选其他方案

| 方案 | 不选的原因 |
|---|---|
| (a) 完全 vendor rlt-openpi 进 deepdive_kai0/kai0/src/kai0/rlt/ | 5000 行维护成本; 上游升级需手动 diff merge ~50 文件; 仓库膨胀 ~2.4 MB |
| (b) 完全独立 (用 rlt-openpi 自带的 conda env 'rlt') | openpi @ fdc03f5 与 kai0 V1 ckpt 不兼容 (5 处 model fix 不在上游); 该 venv 无 rclpy 无法接 ROS2 stack, Phase 1-3 全部做不了 |
| **(c) Hybrid: Fork submodule + adapter (本节方案)** | ✅ 改动总量最小 (~110 行 fork + ~650 行 adapter); 上游升级简单; 边界清晰, 后续可单独 publish fork |

---

## 4. 4-Phase 实施路线

每个 phase 有明确 entry / exit criteria, 上一个 phase pass 才进下一个.

### Phase 0 — Stage 1 RL Token 训练 (离线, 不动真机)

**周期**: 1 周
**目标**: 在 vis_v2_full / mix_apr28_450 ckpt 上跑出 `rl_token_step5000.pt`, 验证 encoder 工作.

| 步骤 | 内容 |
|---|---|
| 1 | `git submodule add` fork `rlt-kai0` 到 `submodules/`; 改 `pyproject.toml` 1 行指向 `../../kai0`; `setup_rlt_env.sh` 创建 `.venv_rlt` |
| 2 | 选目标 ckpt (建议 vis_v2_full_step49999); 注册 OpenPI sidecar config (复用 sim01 部署模板, 命名 `kai0_piper_finetune`) |
| 3 | 写 `kai0/src/kai0/rlt_adapters/data_transforms.py::piper_three_camera` (Kai0Inputs/Outputs, ~100 行) |
| 4 | 现有 Task_A LeRobot 数据 (~890 ep), Stage 1 frozen mode (α=0) 5000 step |
| 5 | 验收: 重建 loss < 0.01; load encoder 后 inference 与原 VLA bit-exact |

**产出**: `checkpoints/rl_token/vis_v2_full_rl_token_step5000.pt` (~200MB)
**风险**: 数据 schema mismatch (Task_A image key vs DroidInputs 默认 schema) — 由步骤 3 的 adapter 解决.

### Phase 1 — 真机基础接线 (无 RL 训练, 仅验证 pipeline)

**周期**: 2 周
**目标**: Piper 上跑一次 VLA-only warmup (无 actor, 等价 SFT 部署), 验证 env_factory + intervention + reward 三接口工作.

| 步骤 | 内容 |
|---|---|
| 1 | 写 `make_piper_env` (§3.1) |
| 2 | 写 `MasterArmInterventionManager` (§3.2) |
| 3 | 写 ROS2 reward listener (§3.3) |
| 4 | 跑 `train_rlt.py --warmup-steps 250 --max-env-steps 0` (训练步数=0, 仅 warmup) |
| 5 | 验收: 250 chunks 入 buffer; s/f/p 正确 latch / consume; master arm 接管能正确记录 chunk; 机器人无硬碰撞 |

**产出**: `warmup_buffer.pt` (复用给 Phase 2 跳过 warmup)
**风险**: action space mismatch (V1 ee_pose 16d vs OpenPI normalize 链可能 joint 8d). Phase 1 必须先验证 ckpt action_dim 与 actor action_dim 一致, 不一致需选定统一空间 (建议 ee_pose 与 deploy 链对齐).

### Phase 2 — vis_v2_full Gripper POC (RLT 完整闭环)

**周期**: 1–2 周 (其中真机 ~8h)
**目标**: 把 vis_v2_full 的 gripper mid-stuck 问题用 RLT 修掉. **整个方案的 go / no-go 节点**.

**Reward 设计** (这个 phase 最关键, POC 阶段从最简单开始):
```
首版: 操作员按 s 表示"成功抓取" → reward = 1.0, done = True
     timeout 60 step 未按 s → reward = 0, done = True
     按 p (progress, 例如初步触碰物体) → reward += 0.5, episode 继续
后续: 接 force sensor 阈值 / 视觉 grasp detection 实现半自动 reward
```

**Stage 2 训练参数**:
| 参数 | 值 | 来源 |
|---|---|---|
| `--rl-token-checkpoint` | Phase 0 输出 | — |
| `--chunk-length` | 5 | gripper 任务短, 与论文 exp/stage2.sh 同 |
| `--max-episode-chunks` | 30 | 60 step 抓不到就 timeout |
| `--warmup-buffer` | Phase 1 输出 | 跳过 warmup |
| `--max-env-steps` | 6000 | ~200 episode, 论文体量 |

**预估真机时间**: 200 ep × 30 step × 50Hz = ~2 min/ep × 200 = **6.7h**, 加上重置 + 操作员歇息 → 总 ~8h.

**验收**:
- 训练曲线: actor_loss 下降, critic Q 上升, episode reward 从 ~0.2 → ~0.8
- 真机 eval (50 ep): success rate base ~30% → RLT ~80%+
- 不出现 ee jiggle 加剧 (用 Layer 1 工具量); jiggle 加剧则调大 RTC blend window

**产出**: `online_rl_vis_v2_gripper_ep200.pt` + `docs/rlt/01_vis_v2_gripper_poc.md` 报告.

### Phase 3 — Production 集成 (POC 成功后)

**周期**: 1 周
**目标**: 把 RLT 训出来的 actor 接进 `policy_inference_node` 作为常规推理路径.

| 步骤 | 内容 |
|---|---|
| 1 | 改 `policy_inference_node.py` 加 `--rlt-actor-checkpoint` 选项 (§3.4) |
| 2 | 改 `start_autonomy_v1.sh` 加 `RLT_ACTOR` env var |
| 3 | RTC blend window 试 12 / 16, 选 jiggle 不恶化的那个 |
| 4 | 写 `docs/deployment/inference/realtime_vla/rlt_integration.md` |
| 5 | 验收: 真机长跑 1h, success rate 维持 Phase 2 水平, jiggle metrics 不超过 baseline 1.2× |

**产出**: 主分支 commit, RLT 成为 V1 pipeline 的可选模块.

### Phase 4 (Future) — 复制到下一个任务

- 任务一 hand-cube / 任务二 hand-stack 若出现 critical phase 失败 → 直接复制 Phase 2 流程
- Piper 高精度插拔 / 螺丝任务 → 完整走 Phase 0–3

---

## 5. 代码改动清单

### 新增 (Adapter, 在 deepdive_kai0 内)

| 路径 | 内容 | 行数估计 |
|---|---|---|
| `kai0/src/kai0/rlt_adapters/env_factory.py` | `make_piper_env` + ROS2 bridge | ~150 |
| `kai0/src/kai0/rlt_adapters/intervention.py` | `MasterArmInterventionManager` | ~200 |
| `kai0/src/kai0/rlt_adapters/reward_listener.py` | ROS2 reward topic subscriber | ~50 |
| `kai0/src/kai0/rlt_adapters/data_transforms.py` | Kai0Inputs / Outputs (替 DroidInputs) | ~100 |
| `kai0/src/kai0/rlt_adapters/vla_websocket_client.py` | 替 VLAWrapper 走 V1 server | ~150 |
| `setup_rlt_env.sh` | 一键创建 `.venv_rlt` (含 kai0 fork + rlt-kai0 fork + rclpy) | ~50 |
| `docs/deployment/inference/realtime_vla/rlt_integration.md` | 集成文档 (Phase 3 产出) | ~300 |
| `docs/training/rlt_experiments/*.md` | 每个 task 的 results.md | 持续 |

合计 (代码): ~650 行 + docs.

### 新增 (Fork, git submodule)

| 路径 | 内容 | 行数估计 |
|---|---|---|
| `submodules/rlt-kai0/` | Fork from yknxh/rlt-openpi, 算法核心代码 5000 行不计入 deepdive_kai0 仓库膨胀 | 仅 fork 内改动 ~110 |
| `submodules/rlt-kai0/pyproject.toml` (改) | 1 行: openpi 依赖指向 `../../kai0` editable | 1 |
| `submodules/rlt-kai0/src/rlt_openpi/vla/vla_wrapper.py` (改) | 加 websocket 模式 (PR 候选) | ~80 |
| `submodules/rlt-kai0/src/rlt_openpi/training/online_rl_trainer.py` (可选改) | ROS2 reset 同步 hook | ~20 |

合计 (fork 内): ~110 行, 上游升级时冲突面极小.

### 修改

| 路径 | 改什么 | 行数 |
|---|---|---|
| `ros2_ws/src/piper/scripts/policy_inference_node.py` | 加 `--rlt-actor-checkpoint`, 加载 + actor.forward 插入 | ~80 |
| `ros2_ws/src/piper/launch/autonomy_launch.py` | 加 `rlt_actor_checkpoint` launch arg | ~10 |
| `start_scripts/start_autonomy_v1.sh` | 加 `RLT_ACTOR` env var | ~5 |

合计: ~95 行修改.

### 不动

- 现有 SFT 训练 pipeline (gf1 mlp / TOS sync / auto-pack)
- V1 Triton inference (RLT 走 websocket 复用)
- 现有 Layer 1 EE 平滑 (与 RLT 解耦)
- ckpt_v0 / v1 / others 目录结构

---

## 6. 资源与时间预算

| 资源 | 估计 |
|---|---|
| **工程时间** | Phase 0 一周 + Phase 1 两周 + Phase 2 两周 + Phase 3 一周 = **6 周** 到完整 POC 上线 |
| **一次性 setup** | git submodule + fork pyproject 改一行 + `.venv_rlt` 创建 ≈ **半天** |
| **真机时间** | Phase 1 ~1h + Phase 2 ~8h + Phase 3 ~2h = **~11h 总真机** |
| **GPU 时间** | Stage 1 sim01 单卡 ~2h; Stage 2 算力极小 (actor 4 MB) |
| **存储** | 每个 RLT ckpt ~250MB (encoder + actor + critic + buffer 100k entry) |
| **仓库膨胀** | deepdive_kai0 增 ~650 行 (adapter) + 1 submodule reference; fork 内 ~110 行改动, 上游 5000 行不计 |
| **操作员时间** | Phase 2 期间需要操作员**全程在场** ~8h (按键 + 接管). RLT 最大的人力成本 |

---

## 7. 风险登记 + 兜底

| 风险 | 概率 | 影响 | 兜底 |
|---|---|---|---|
| Phase 2 POC 失败 (RLT 在 kai0 上 reward 信号不够强) | 中 | 大 | 退回纯 DAgger; 保留 Phase 1 接线代码作为以后再试基础 |
| Action space 不一致 (ee_pose vs joint_velocity) 导致 BC 拉错方向 | 中 | 大 | Phase 1 完整 dry-run, 用 `--max-env-steps 0` 验证 buffer 数据 |
| RTC + actor 冲突, jiggle 反而恶化 | 中 | 中 | Phase 3 备选: 训练时学 *RTC 平滑后的 ā* 的残差 (改 ReplayBuffer.a_tilde) |
| 真机训练时机器人乱动伤人/伤物 | 低 | 高 | 软件 fence (joint limit + collision check), 操作员手不离 e-stop |
| Master arm action 转换 joint_velocity 时漂移 | 中 | 中 | Phase 1 测试 100 接管 chunk, 看 buffer 里 a_human 与 a_executed 是否一致 |
| reward labeling 主观偏差导致 critic 学错 | 低 | 中 | 多操作员交叉打 reward, 或尽快接半自动 reward (force sensor 阈值) |

---

## 8. 首要决定点

| # | 决策 | 候选 | 结论 |
|---|---|---|---|
| Q1 | Phase 2 首个 RLT POC 任务 | (a) vis_v2_full gripper / (b) 其他 | ⏳ **待定** — 看 Phase 0/1 结果后再决定 |
| Q2 | RLT 训练时 action space | (a) ee_pose 16d / (b) joint_velocity 8d × 2 | ✅ **(a) ee_pose 16d** — 与 V1 deploy 链对齐 |
| Q3 | Reward 首版来源 | (a) 操作员键盘 / (b) 半自动 (sensor / 视觉) | ✅ **(a) 操作员键盘** — POC 阶段 reward source 越简单越好 |
| Q4 | 代码组织 | (a) 完全 vendor 进 `kai0/src/kai0/rlt/` / (b) 完全独立 rlt-openpi venv / (c) Hybrid Fork submodule + adapter | ✅ **(c) Hybrid** — Fork rlt-openpi 作 git submodule (`submodules/rlt-kai0/`), 改 1 行 pyproject 指向 kai0 fork; kai0-specific 接线放 `kai0/src/kai0/rlt_adapters/`. 详见 §3.6 |

---

## 9. 与上游 / 相关文档跳转

- 双轨另一支 (SFT-DAgger 实施方案) → [`dagger_implementation_plan.md`](dagger_implementation_plan.md)
- RLT 论文与实现细节速查 → `/data1/tim/workspace/rlt-openpi/init.md` (700 行, 涵盖算法 / pipeline / hyperparameters / 与论文 deviation)
- RLT 算法 fork (Hybrid §3.6) → `submodules/rlt-kai0/` (待 add)
- RLT 上游原始仓库 → <https://github.com/yknxh/rlt-openpi> (本地工作副本 `/data1/tim/workspace/rlt-openpi/`)
- 跨本体战略主文档 → [`cross_embodiment_strategy.md`](cross_embodiment_strategy.md) (3 异构机器人, 4-层 ROI, Tri-track)
- Task A 主规划 → [`task_a_master_plan.md`](task_a_master_plan.md)
- 实时推理优化历史 (EE jiggle Layer 1) → [`../inference/realtime_vla/ee_stability_layer1.md`](../inference/realtime_vla/ee_stability_layer1.md)
- V1 Triton 推理日志 (RLT 复用此 server) → [`../inference/realtime_vla/v1_triton_log.md`](../inference/realtime_vla/v1_triton_log.md)
- sim01 部署 (RLT Stage 2 训练机) → [`../inference/sim01_deployment.md`](../inference/sim01_deployment.md)
- 训练集群运维 (Stage 1 在 sim01 / gf 跑) → [`../training_ops/`](../training_ops/README.md)
