# Realtime-VLA 推理优化 — 战略 (Strategy)

> 本文档是 `realtime_vla_optimization_analysis.md` (1687 行, 已拆) 拆分后的"战略层"。包含决策摘要、上下文基线、优化项目排序与复杂度评估、Fallback 方案 (选项 Y)。
>
> **同 series 文档**: `roadmap.md` (5 阶段实施路线) / `v1_triton_log.md` (V1 已实施日志) / `layer_b_plan.md` (Layer B 未来 plan)

---

## 0. 决策摘要

### 已选方案: 选项 X — 双推理架构并存

| 维度 | 决策内容 |
|---|---|
| 训练侧 | JAX (现状) + PyTorch (新增, 用 `train_pytorch.py`) 并存, 新 fine-tune 按需选 |
| 推理侧 | JAX backend (`serve_policy.py:8000`) + PyTorch backend (新增 `serve_policy_pytorch.py:8001`), sidecar `framework` 字段分发 |
| 旧 JAX ckpt | 不迁, 走 JAX 推理 + #6 浅层优化 (1.5-2× baseline) |
| 新 ckpt | 走 V1 Triton 路径 (复用 + 5090 重 autotune), **实测 P50=32 ms / 8.0× vs eager**, 见 §6 |
| 痛点状态 | 双实现痛点 (数值对齐 / 双实现同步 / bug 翻倍 定位) **不存在**; 剩余痛点是"两个独立项目"成本 |

### 5 阶段实施一览

| 阶段 | 目标 | 主要项目 | 期望收益 | 触发条件 |
|:---:|---|---|---|---|
| **1** | 短期热身 | #8 延迟标定 + #6 JAX 浅层 + 真机测试 1/2 | 抖动 -30-50%, 推理 1.5-2× | 立即可启动 |
| **2** | 任务速度 | #5 timeaxis_smooth QP | 任务耗时 1.5-2× | 阶段 1 #8 完成 |
| **3** | 选项 X 落地 | PyTorch 训等效 POC + V1 Triton 推理 (✅ 完成) + sidecar | **P50=32 ms (8.0×)** ✅ | 阶段 1 真机测试 1 完成 |
| **4** | 任务质量 | #4 速度自适应学习 (油门数采 + 回归 head) | 精细阶段成功率 ↑, 耗时 -50% | 阶段 2 QP 落地 |
| **5** | 推理极致 (可选) | #3 Flash 推测推理 + #7 客户端 MPC | 复合 10-20× / 跟踪误差 -50% | 阶段 3 完成; #7 需 t_motion > 50ms |

### 文档使用方法

- 按 §3 各阶段文档逐步实施, 每阶段都给出具体改动文件 + 步骤 + 验证标准
- §2 排序表是排序与复杂度参考, 阶段顺序并非简单按"#编号"
- §4 是真机测试脚本说明, 阶段 1 用
- §5 是 fallback (Y 选项) 触发条件 + 切换方法
- 任何决策点变更, 在 §6 修订历史里追加版本

---

## 1. 上下文与基线

### 1.1 deepdive_kai0 现状

#### 推理栈
- **框架**: JAX + Flax + Orbax, sim01 上 `serve_policy.py` WebSocket 服务 (`:8000`)
- **模型**: π₀.₅, action_horizon=50, joint_dim=14 (双臂)
- **ROS2 节点**: `policy_inference_node.py` 拉取相机 + 关节, 走 WebSocket 推理, 返回 chunk
- **双时钟**:
  - publish_rate = 30 Hz
  - inference_rate = 3.0 Hz (≈ 333 ms/cycle)
  - chunk_size = 50, latency_k = 8 (头部裁剪步)
- **RTC 已实装**:
  - 模型层: `Pi0RTC.sample_actions` 在去噪迭代内对 `[d, exec_h)` 区间引导 (`rtc_execute_horizon=16`, `max_guidance_weight=0.5`)
  - 运行时层: `StreamActionBuffer.integrate_new_chunk` latency 裁剪 + 8 步线性 overlap 平滑
- **XLA**: 已有 `start_server_xla_cache.sh` 预编译缓存

#### 硬件
- **部署机 sim01**: Ubuntu 24.04, **2× RTX 5090 32GB** (Blackwell sm_120)
- **训练机 gf0**: 8× A100 80GB; uc01-03 / js01-04 GPU 集群
- **机械臂**: 双 Piper, CAN slave 控制
- **相机**: 1× D435 (top) + 2× D405 (左右腕), 三路 RGB @ 30fps (depth 关闭)

#### 现状弱点
- RTC `latency_k=8` 是手调常量, 没数据校准
- 系统延迟未量化 (相机曝光 / 读出 / proprio / motor 滞后均无测量)
- 速度全局一致 (30Hz 一档), 折叠/对位等精细阶段无降速逻辑
- chunk 边界靠 8 步线性平滑兜底, 物理量 (速度/加速度) 无约束

#### 已存在的 PyTorch 训练侧 (影响 §1.4 决策)
deepdive_kai0 **已在维护双栈**:
- `kai0/scripts/train_pytorch.py` (646 行, DDP/torchrun 支持)
- `kai0/src/openpi/models_pytorch/`:
  - `pi0_pytorch.py` (646 行): `PI0Pytorch` 类支持 pi05 (`PI0Pytorch.pi05` 字段, train_pytorch.py:410 读 `config.model.pi05`); 含 `AdvantageEstimator`
  - `gemma_pytorch.py` (281 行)
  - `preprocessing_pytorch.py` (358 行)
- 实际使用: `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` config 走 PyTorch 训练 advantage/stage 模型
- **主流 pi05 fine-tune 仍走 JAX `scripts/train.py`** — 选项 X 阶段 3.1 需 POC 验证等效性

### 1.2 三论文核心思想速览

| 论文 | 核心技术 | 论文报告收益 | 与 deepdive_kai0 关系 |
|---|---|---|---|
| **V1** *Running VLAs at Real-time Speed* | PyTorch + Triton 内核全栈优化 (CUDA Graph + GEMM 分块 + QKV/RMSNorm/RoPE fusion + Stream 并发) | 105ms → 27.3ms 双视角 @ 4090 | 阶段 3.2 推理 serve 直接套用 |
| **Flash** *Realtime-VLA FLASH* | 110M draft + flow-matching 端点重建并行验证 + 阶段感知 fallback | 3.04× 加速, 成功率 -0.3pp | 阶段 5 #3, 与阶段 3 复合 |
| **V2** *Learning to Run VLAs Fast, Smooth, and Accurate* | 4 类延迟标定补偿 + 服务端时间轴 QP + 客户端 MPC + 速度自适应学习 | 任务耗时 2-4×, 成功率保持 | 阶段 1 #8, 阶段 2 #5, 阶段 4 #4, 阶段 5 #7 |

### 1.3 关键洞察

#### A. 333ms 是 timer 节流, 不是模型上限
`inference_rate=3.0` 是 ROS2 timer 参数, 推理线程每 333ms tick 一次。模型实际推理时间估计 100-200ms (5090 上, 50 步去噪 × 2-4ms/step + VLM prefill ~30ms)。

**影响**:
- 加速模型本身不能让推理"看起来更快" — timer 还是 333ms 一发
- 但加速模型可以**让你提高 inference_rate** (例如 6-10 Hz), 让 RTC 收敛更稳
- 真机测试 1 (§4.1) 是后续推理优化路线决策的关键依据

#### B. deepdive_kai0 双栈已存在
PyTorch 训练侧 (`train_pytorch.py` + `models_pytorch/`) 已在维护, advantage / stage classifier 已在用。"双栈维护"对 deepdive_kai0 不是 0→1 的新增, 而是 0.5→1 的扩张。

#### C. 双推理架构下"双实现痛点"消失
关键认识: 7 个双栈痛点中, 第 1-3 项 (数值对齐 / 双实现同步 / bug 定位翻倍) 都是"**同一模型两份实现要同步**"的成本。

如果每个 ckpt 只活在一个框架里 (JAX ckpt → JAX 推理, PyTorch ckpt → PyTorch 推理), 这部分成本完全没有。剩余痛点 4-7 (sidecar 扩展 / 工具链分裂 / AI 上下文翻倍 / upstream 同步) 是"维护两个独立项目"成本, 数量级低于"双实现"。

| # | 痛点 | 双推理架构下 |
|:---:|---|---|
| 1 | 数值对齐 | **消失** |
| 2 | 架构改动双实现 | **大部分消失** |
| 3 | Bug 定位翻倍 | **消失** |
| 4 | sidecar 扩展 | 保留 (加 framework 字段) |
| 5 | 工具链分裂 | 弱化 (仅交叉对比时显现) |
| 6 | AI/人上下文翻倍 | 弱化 (per-ckpt 上下文固定) |
| 7 | upstream openpi 同步 | 可选 (哪边活跃跟哪边) |

#### D. agent 时代成本评估标准变化
不再用"人·周"度量, 改用 4 维任务本征复杂度。代码量大但纯逻辑的项 (V1 全栈端口) AI 可加速; 需要真机/数据迭代验证的项 (Flash 阈值调参、速度自适应数采、MPC 硬件辨识) 卡物理时间。详见 §2.1。

### 1.4 选项 X 决策依据

#### 用户优先级 (Q1+Q5)
- Q1: 探索性优化, 不指向单一痛点
- Q5: 推理速度 + 任务速度都要, **任务速度优先**

#### 同架构选项对比 (双推理架构假设 + 忽略 100+ JAX ckpt 迁移)

| 选项 | 训练 | 推理 | 旧 ckpt | 新 ckpt | 选定状态 |
|---|---|---|---|---|:---:|
| **X** | JAX (现状) + PyTorch (新增) | JAX backend + PyTorch+Triton backend | 1.5-2× (#6) | **5-10×** | ✅ |
| Y | JAX only | JAX + JAX→ONNX→TRT (按 ckpt 选) | 1.5-2× / 3-5× | 3-5× | Fallback |
| Z | JAX + PyTorch 都活跃 | 三 backend | 自选 | 自选 | 过度设计, 不选 |

#### 为什么选 X
1. **旧 ckpt 不迁** — 走 JAX backend 拿 1.5-2× 足够日常
2. **新 ckpt 拿满 5-10×** — 走 PyTorch+Triton 路径 A
3. **`train_pytorch.py` + `PI0Pytorch` 代码层已支持 pi05**, 非从零搭
4. **避开 ONNX 导 flow loop 的技术风险点** (Y 的主要技术风险)
5. **双实现痛点 1-3 不存在** (双推理架构使然)

#### Y 作为 fallback (§5)
若阶段 3.1 PyTorch 训练等效性 POC 失败, 退回 Y。

---

## 2. 优化项目排序与复杂度评估

### 2.1 成本维度 (agent 时代 4 维)

| 维度 | 含义 | AI 可压缩? |
|---|---|:---:|
| **逻辑复杂度** | 算法/数学/工程难度 (代码量 + 框架端口规模) | ✅ 大幅 |
| **决策密度** | 需要人工拍板的关键架构 + 阈值选择数量 | △ 部分 |
| **实验迭代周期** | 真机/数据迭代验证的轮次 × 单轮耗时 | ❌ 卡物理时间 |
| **架构持续耦合** | 长期维护成本 + 是否锁死未来选项 | ❌ 常量 |

**关键认识**: agent 时代真正的瓶颈是"需要真机/数据迭代验证的轮次", 不是代码量。

### 2.2 8 项优化排名表

按期望收益 (gain × 兑现概率) 排序:

| 排名 | 优化项 | 来源 | 期望收益 | 逻辑复杂度 | 决策密度 | 实验迭代 | 架构耦合 | 阶段 |
|:---:|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **#1** | V1 PyTorch+Triton 全栈端口 | V1 | 推理 333ms → 30-60ms (**5-10×**) | 中 | 中 | 高 | 中 (X 下) | 3.2 |
| **#2** | V1 JAX→ONNX→TensorRT 半端口 | V1 思想 | 推理 60-100ms (**3-5×**) | 中 | 低 | 中 | 低-中 | Fallback Y |
| **#3** | Flash 草稿 + 推测推理 | Flash | 推理 **2-3×** (与 #1/#2 复合) | 高 | 高 | 高 | 中-高 | 5 |
| **#4** | V2 速度自适应学习 | V2 §4.4 | 关键阶段任务耗时 **-50%, 成功率 +10-20%** | 中 | 中 | **极高** (数采) | 低 | 4 |
| **#5** | V2 时间轴 QP 重参数化 | V2 §4.3.1 | 任务耗时 **1.5-2×** | 低 | 低 | 低 | 低 | 2 |
| **#6** | JAX/XLA 推理微优化 | V1 思想 | 推理 **1.5-2×** (333→150-200ms) | 低 | 低 | 低 | 极低 | 1 |
| **#7** | V2 客户端 MPC + 1 阶滞后辨识 | V2 §4.3.2 | 跟踪误差 -50% (若 t_motion > 50ms) | 中 | 中 | 高 | 中 | 5 (条件) |
| **#8** | V2 延迟标定 + 感知对齐 | V2 §4.2 | 抖动 -30-50%, 为 #5/#7 提供准数 | 极低 | 极低 | 低 | 极低 | 1 |

#### Pareto 分布

| 象限 | 优化项 |
|---|---|
| 高收益 × 低复杂度 (最佳 ROI) | #5 (QP), #2 (Y fallback) |
| 高收益 × 高复杂度 (值得投入) | #1 (X 阶段 3.2), #3 (Flash), #4 (速度自适应) |
| 中收益 × 极低复杂度 (首发热身) | #6 (浅层 JAX), #8 (延迟标定) |
| 中收益 × 高复杂度 (条件投入) | #7 (MPC, 需 #8 数据验证) |

### 2.3 兼容性矩阵

| 优化项 | RTC | sidecar | 训练管线 | ROS2 节点 |
|---|:---:|:---:|:---:|:---:|
| #1 V1 全栈端口 | 不冲突 (算法层) | 推理 sidecar 加 framework 字段 | 训练侧选 X 时需切 PyTorch | 服务端新增 `serve_policy_pytorch.py` |
| #2 V1 TRT 半端口 (Y) | 不冲突 | sidecar 加 ONNX/engine 路径 | 训练 JAX 不变 | 服务端逻辑微调 |
| #3 Flash | 与 RTC 重叠 (都在去噪过程) | 需 draft sidecar | 加 draft 训练 job | 改 inference 调用图 |
| #4 速度自适应 | 通过 dt 联动 | 需新 head 权重 | 加 head 训练 | 改 teleop + 推理后处理 |
| #5 时间轴 QP | 正交 | 不影响 | 不影响 | WebSocket 后处理 + 客户端定时器改造 |
| #6 JAX 浅层 | 不影响 | 不影响 | 不影响 | 不影响 |
| #7 MPC | 不影响 | 不影响 | 不影响 | 新 `mpc_tracker_node` |
| #8 延迟标定 | 增强 | 不影响 | 不影响 | 改 `policy_inference_node._get_synced_frame` |

**冲突点提醒**:
- **#5 QP 与 RTC `latency_k` 单位变化**: 从"步数"改"时间", 阶段 2 主要工程量
- **#3 Flash 的 K 验证步与 RTC prefix weights**: 两者都在去噪迭代内部, 实施时建议 RTC 先生成完整 chunk 候选 → Flash 验证哪些步可执行
- **#1 / #2 与 #3 复合**: V1 的 Triton kernel 是为 pi05 完整去噪写的, Flash 的并行验证调用模式不同, 内核可能要分叉两套

---

---

## 5. Fallback 方案 (选项 Y)

### 触发条件
阶段 3.4.1 PyTorch 训练等效性 POC 失败, 例如:
- 同 config 下 PyTorch 训出 ckpt 的 inline-eval MAE > JAX 对照 10%+
- DDP 不稳定 (NaN / 梯度爆炸 / hang)
- 收敛速度显著慢于 JAX (单 step 时间慢 50%+)

### 选项 Y 内容
| 维度 | Y 决策 |
|---|---|
| 训练侧 | JAX only, 完全不变 |
| 推理侧 | JAX 默认 + JAX→ONNX→TRT (高性能版, 按 ckpt 配置) |
| 旧 ckpt | 走 JAX 推理 + #6 浅层 (1.5-2×) |
| 重要 ckpt | 跑 `pack_inference_trt.py` 转 TRT engine, 走 TRT serve (3-5×) |

### 切换步骤
1. **新增** `kai0/scripts/pack_inference_trt.py`:
   - 读 JAX ckpt → JAX→ONNX 导出 (用 `jax2tf` 或 `flax2onnx`)
   - ONNX → TRT engine (用 `trtexec` 或 `tensorrt` Python API)
   - 输出附在 ckpt 目录 `<ckpt>/inference_engine.trt`
2. **新增** `kai0/scripts/serve_policy_trt.py`:
   - 类似 PyTorch serve 但加载 TRT engine
   - WebSocket 协议与 JAX serve 一致
3. **修改** `start_autonomy_from_ckpt.sh`:
   - sidecar `backend: "jax" | "trt"` 字段分发 (与 X 的 `framework` 字段功能类似)
4. **POC 验证点**: pi05 flow matching 10 步去噪循环的 ONNX 导出是否可拆 (主要技术风险)

### Y 期望收益
- 旧 ckpt: 1.5-2× (走 JAX + #6) / 3-5× (走 TRT) 自选
- 新 ckpt: 3-5× (走 TRT)
- 训练侧零变更, 无 X 阶段 3.1 的等效性验证风险

---


---

## 8. 修订历史

| 版本 | 时间 | 内容 |
|:---:|---|---|
| v0.1 | 2026-05-19 | 初版, P1/P2/P3 排序 + 三论文逐项判断 |
| v0.2 | 2026-05-19 | 成本框架重构: agent 时代 4 维任务复杂度评估, 不再因工时拒绝方案。V1 全栈端口从"不做"调整为 #1。新增 V1 三路径 A/B/C 对比 |
| v0.3 | 2026-05-19 | 整合 Q1-Q5 答案; 发现 PyTorch 训练侧已实装 (advantage 管线), 双栈是既成事实; 新增同架构 4 选项 a/b/c/d, 选项 d 是"无双栈痛点"折中; 关键洞察: 333ms 是 timer 节流; 加真机测试方案 |
| v0.4 | 2026-05-19 | Q4 round 2: 假设双推理架构并存 + 忽略 ckpt 迁移风险。痛点 1-3 消失, 4-7 弱化。新增选项 X/Y/Z, 推荐 X |
| v0.5 | 2026-05-19 | 用户决策选定选项 X。重写 §7 为 5 阶段实施路线图, 每步给出具体改动文件 + 风险点 |
| **v0.6** | **2026-05-19** | **文档全面整理: 加目录; 删冗余 (原 §3 推荐落地路径 ↔ §7 阶段细节重复, 原 §5 Q&A round 1 a/b/c/d 详细对比); §0 收敛为决策摘要 + 5 阶段一览; §1 整合现状/三论文/关键洞察/决策依据; §2 排序与复杂度; §3 实施路线图为主干; §4 真机测试; §5 Fallback Y 详细方案。从 756 行精简到 ~570 行, 主线清晰** |
| v0.7 | 2026-05-19 | **关键前置障碍**: 实测发现 sim01 5090 sm_120 与 `kai0/.venv` PyTorch (2.7.1+cu126) 不兼容; §3.4.2 阶段 3 落地前必须升级 PyTorch nightly / cu128。建议独立 venv `kai0/.venv_5090` 隔离 |
| v0.8 | 2026-05-19 | **pi05 推理 5-backend 实测完成** (`optimize/benchmark_pi05_inference.py`): E max-autotune P50=43.5ms (5.52×), B compile-default 2.18× (纯 fusion), D reduce-overhead 4.98× (+CUDA Graph), V1 论文 CUDA Graph 单独 2× 被印证 (D vs B = 2.13×)。**结论: 策略 B 饱和, 不需要 V1 手写 Triton 路径**。已知 4 个 PI0Pytorch model code 内部 dtype 问题需阶段 3.2 修 (sample_noise/dt/time/RMSNorm output, embed_prefix att_masks list→tensor) |
| v0.9 | 2026-05-19 | §3.4.2 扩展实测结果展示: 加分位数 (P50/P95/P99) 含义说明 + 5-backend 详细描述表 (Python/Inductor/CUDA Graph/autotune 4 层) + 实测数据表 + 分量贡献分解 + 抖动分析 (Std/P99-P50) + cold-start 开销 |
| v0.10 | 2026-05-19 | **根据实测结果更新实验计划**: §3.1 总体阶段图加 "阶段 0 真机推理基线" + 阶段 3 工程量减 50% (3-5 周→1.5-2.5 周, 不需 Triton); 新增 §3.1.1 实测对计划影响对比表; §3.1.2 PI0Pytorch 6 处 model code fix 清单 (P0-P6); §3.1.3 子任务清单 (已完成/待做); §3.6 #3 Flash 降级为研究项 (baseline 43.5ms × 3× 边际效用低) |
| **v0.11** | **2026-05-20** | **V1 Triton 推理优化全程实施 (合并自 `optimize/v1_triton/PROGRESS.md`)**: 新增 §6 完整记录 Step 0-9 的 9 个优化步骤. 路径决策从"自写 PyTorch+Triton"改为"复用 V1 `pi05_infer.py` + 5090 重 autotune", 工程量 1-2 周 → 3 天. **最终 P50 = 32.05 ms (8.00× vs eager, 比 §3.4.2 max-autotune 43.5ms 再快 26%)**. 关键发现: 5090 sm_120 "小 BLOCK_N 大 BLOCK_K" 反直觉最优 (Step 6 单步 -8.8%); decoder GEMM memory-bound, pipelining/encoder sweep 噪声内; 继续突破 30ms 需结构性 kernel fusion (Step 11, 3-5 天) 或 wgmma 重写 (Step 13, 5-10 天). 独立 PROGRESS.md 已删除 |
| **v0.12** | **2026-05-20** | **针对性整理**: §0/§3.1 总体阶段图按 v0.11 实施现状重写 (阶段 0 完成 / 阶段 3 推理 serve 已 ✅); §3.1.1 (原计划影响对比表) 删除, 改为"PI0Pytorch fix 备选路径清单"; §3.1.2 子任务清单状态更新 (V1 路径 4 项完成); §3.4 顶部加 v0.11 现状引导; §3.4.2 末尾自写 Triton 计划折叠 (8 步细节 → 3 行实施路径摘要); §3.6 Flash 降级理由瘦身 (用 32ms 而非 43.5ms 数据). TOC 补 §3.1.1/3.1.2. 总行数 950 → ~900 |
| **v0.13** | **2026-05-20** | **新增 §7 Layer B 系统级优化 plan**: 单 5090 真机约束确认 (排除多 GPU 选项). 子项 B4 (V1 serve 包装, 主线) → B1 (全链路 11 段 latency profile) → B2 (preprocess GPU 化, 数据驱动). 1.5-2 周, 关键里程碑 B4 = 真机 V1 推理首跑通. Layer A (kernel fusion/wgmma) 暂缓 (推理 32ms 已远 < timer 周期, ROI 低). §7 修订历史 → §8. TOC 更新 |
| **v0.14** | **2026-05-20** | **Q2 sim01 JAX 推理延迟实测完成 + B4 Phase 1 serve_policy_v1.py 落地**: §4.1 加 1299-sample 实测表 (P50=196 ms / P95=221 / P99=232 / Std=13.2 / jitter=25 ms), 落在 "100-200ms 标准 5090 baseline" 档, 确认 V1 路径 6.1× 加速空间. JAX 抖动 P95-P50=25ms 不需 AOT compile. 推理 196ms vs timer 333ms = 59% utilization, V1 落地后可拉 inference_rate 到 20-30 Hz. 新增 `kai0/scripts/serve_policy_v1.py` (B4 Phase 1, 343 行) + `start_scripts/kai/diag/measure_jax_infer_latency.sh` (Q2 helper) |
| **v0.15** | **2026-05-20** | **§4.1 扩为正式实验报告 (8 子节)**: 4.1.1 测量方法 (含 policy_inference_node.py:2085-2148 timer 源码片段); 4.1.2 实验配置 (config / ckpt / asset_id / 硬件 / timer 等 9 项 metadata); 4.1.3 原始 log 样本 (前 5 条 + JIT outlier 标注); 4.1.4 分位数表; 4.1.5 **ASCII 分布直方图 9 bucket** (180-220 ms 集中 91%, 单峰窄分布 CV=6.6%, 无长尾); 4.1.6 V1 对比加 jitter / Std/Mean 行 + WebSocket overhead 补偿估算; 4.1.7 决策映射 4 行结论 (V1 路径✅ / AOT❌ / inference_rate 提升潜力⏳ / #6 价值低); 4.1.8 复跑命令. TOC 加 4.1 子节链接 |
| **v0.16** | **2026-05-20** | **新增 §3.4.5 TensorRT 路径回顾**: 沉淀 TRT 攻关失败记录, 防止重复趟坑. 6 子节 A-F: A 已就绪资产 (`.venv_5090_trt` Python 3.10 + PyTorch 2.7.1+cu128 + TRT 10.14, `pi05_trt_pipeline.py` 367 行 5-stage 流水线, AOTI 6.3GB 产物); B **5 个阻塞点** (sm_120 / torch_tensorrt CUDA 13 / pypi hang / Python ABI / **未解 ONNX flow loop**); C AOTInductor Backend H 同期阻塞 (compile OK 但 load fail); D V1 §4.2.2 **8/8 优化已被 Inductor 自动捕获** (PyTorch 工具链 41ms 极限); E **4 条重启路径** (选 1 等 PyTorch 2.13 stable; 选 4 当前接受 V1 32ms); F 相关链接 6 个文件. TOC 加 3.4.5 子节链接 |
| **v0.17** | **2026-05-20** | **B4 Phase 2 + B1 server-side profile 完成**: 实现 `SentencepieceStateEncoder` (kai0 同款 prefix `"Task: {p}, State: {s};\n"` + 256-bin 离散化 + PaliGemma embed lookup + scale√2048), 绕开 V1 prebaked language_embeds via `v1_forward_with_state()` 直写 encoder_x. 新增 `expand_v1_pkl_for_phase2.py` (扩 pkl `language_embeds` 7→200 行, 为 prompt+state 留位). V1Policy.infer() 加 5 段 timing (preproc / state_encode / infer / postproc / total). 本机 smoke test (5 iter): **total ~40.5 ms** (preproc 6 + state 0.3 + infer 34 + post 0.2), state 切换→action max diff 0.286 (验证 state 流入). vs Q2 JAX 196ms = **4.9× server-side speedup**. §7.2 加实测表 |
| **v0.18** | **2026-05-23** | **§7.6/7.7/7.8 新增 — 20Hz cycle 攻关启动**: §7.6 Step 0 baseline 全链路 11 段实测 (cycle P50=80.5ms / P95=89.1, cam→emit P50=134.8ms, server_infer 34ms = V1 floor 已触顶); §7.7 P1 image_age root cause 定位 = `SingleThreadedExecutor` 在 80ms cycle 期间阻塞所有 image callbacks → deque[-1] 不更新 → 测得 age 含 20-30ms executor lag; §7.8 排除 forward 优化后按 ROI 重排执行优先级 (P1.a executor 改 / P1.b 相机配置 / P2 GPU preprocess / P3 SHM transport) + 跟踪表 (baseline + 每步 cycle/image_age/cam→emit Δ). 本次执行 P1.b (C2 fps 30→60 + C7 关 D435 depth) |
| **v0.21** | **2026-05-23** | **C.4 SHM v2 + C.3 buffer_integrate vectorize + 20Hz 启用**: (1) **C.4 SHM transport** 替 WS+msgpack TCP loopback — 新 `kai0/scripts/shm_transport.py` 含 `ShmServer/ShmClient`, 协议: 4MB POSIX shm region `/dev/shm/kai0_v1_obs` (header 64B + image 451KB zero-copy memcpy + metadata msgpack) + 64KB resp region, 同步 hybrid busy-poll (200µs 硬 spin + `time.sleep(0)` yield + soft sleep). 关键 v2 优化: `np.frombuffer(shm_buf).reshape(3,3,224,224)` view 单 memcpy/cam (跳 `np.stack`+`tobytes` 中间 alloc). 实测 cycle 43.87→**40.05** P50 / 49.64→**44.25** P95 (-3.8 / -5.4ms), ws_overhead 5.30→2.18 / 8.31→5.45 (-3 / -2.9ms). 默认 `transport=ws` (JAX legacy), V1 路径 `transport:=shm` opt-in. (2) **C.3 buffer_integrate vectorize**: smooth 段 Python list comprehension 改成 numpy broadcast 矢量化, 数值等价. 调查发现 P95 spike 25% cycle 在 smooth 段, 但 vectorize 后 spike 不变 (-0.37ms 边际收益) — **真凶是 Linux CFS scheduler 抢占 inference thread** (~5ms 时间片), 修需 SCHED_FIFO (CAP_SYS_NICE / root) 留 C.7. (3) **20Hz timer + RTC 适配**: start_autonomy_v1.sh `inference_rate:=20.0` (period 100→50ms), RTC 比例缩放 `latency_k:=2 min_smooth_steps:=3 rtc_execute_horizon:=4`. cycle P95 44.25 < 50ms period, headroom 5.75ms. 跟踪表新增 C.3+C.4+20Hz 行; 进展总览 ASCII 加 C.2+C.4 列. 真机 1414 cycle execute 验证 OK, infer 37-42ms 稳定, RTC ratio 0.5-0.85 健康. JAX 路径 bit-identical (transport 默认 ws). 待续: 20Hz execute 真机验证 + 若 jitter overrun 降到 inference_rate=18 fallback |
| **v0.20** | **2026-05-23** | **C.2 image_age 测量修复 — 揭示真实 cam→emit ~126ms (旧 metric 42-58ms 是测错对象的偏倚样本)**: 新 `_get_observation_with_stamp()` 方法返 `(obs, head_stamp_ns)` tuple, stamp 走**tuple 侧通道**而非 obs dict, 兼容 JAX/V1 transform 链 (obs dict bit-identical 旧版). `ObsPrefetchWorker.queue` 改存 tuple. `_record_latency_sample` 加 `obs_stamp_ns` 参数, 优先用它. 旧 `_get_observation` 保留 wrapper 兼容. JAX 路径不传 stamp → fallback deque (legacy 行为). 实测: image_age 0% NaN (vs A.2 70% NaN); P50=125.7ms P95=148.5ms = **本次推理实际用的 head 帧→action 发出的真实总 age**. 拆解 = 物理 sensor 50ms + worker prefetch lag 22ms + cycle work 44ms + variance 10ms. **不是性能回归** — 旧 metric 读 `deque[-1]` 是"record 时的 latest 帧", 不是"本次实际用的帧" — 流水线下两者完全不同. RTC chunk merge 让等效控制延迟 ≈ cycle_period (50-100ms), 不由 image_age 直接决定. cycle work 仍 44ms (22.6Hz, P95 49.7) — **当前 timer 仍是 inference_rate=10Hz 节流**, 拉到 18-20 可解锁真 20Hz 推理频率 |
| **v0.19** | **2026-05-23** | **P1.b/P1.a/P2/A.2 全部实测落地 — 20Hz cycle 目标达成 (22.6Hz)**: 跟踪表重构为 A 已落地 / B 失败归档 / C 待实施 三段. 实施进展总览 ASCII 图清晰展示 4 代 baseline→P1.b→P2→A.2 各指标 Δ. **A 段全部完成**: (1) P1.b-partial (head depth off, 隔离测试无效, image_age 噪声内 +0.3, 验证 root cause 不在 depth pipeline); (2) P1.a 两轮 MultiThreaded attempt 都 hang (B.2 GIL 抖动 / B.3 num_threads=2 sensor cb starve), 回退 SingleThreaded; (3) **P2 Step 1+2** ✅ fast_obs_pipeline=true 跳 JPEG+CvBridge+cvtColor (训练 collect 路径用 AV1/h264 编 MP4, JPEG roundtrip 本就不对齐), **cycle 80→62ms (-18.3) = 16Hz, obs_construct 35.7→14.9 (-20.7)**, 真机 execute 验证通过; (4) **A.2 异步流水线** ⭐ 新增 `ObsPrefetchWorker` 在背景线程持续 pop `_get_observation()` 放 maxsize=1 queue, main 推理直接取 prefetched obs, 等于 obs_construct 15ms 藏到 forward 35ms 背后, **cycle 62→44ms (-17.9) = 22.6Hz 超 20Hz 目标 ✓**, obs_construct 14.9→0.03ms, 真机 execute 验证通过. **B 段**: P1.b-C2 60fps 被 D405 hand_left USB 2.0 限制 (rs-enumerate 确认硬件 mode list 仅 30/15/5fps); P1.a 两轮 multi-thread hang. **总体 Δ baseline→A.2**: cycle 80.5→44.3ms (-45% time, +88% throughput), cam→emit 134.8→~86ms (-36%). 后续 image_age 还可砍 (C.5/C.6) 把 cam→emit 进一步压到 ~60-70ms |
