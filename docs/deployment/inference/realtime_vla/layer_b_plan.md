# Realtime-VLA — Layer B 系统级优化 Plan (Next Phase)

> 本文档是 `realtime_vla_optimization_analysis.md` (1687 行, 已拆) 拆分后的"未来 plan 层"。V1 Triton 完成后的下一阶段 — Layer B 系统级优化路线 (异步流水线 / SHM / multi-rate)。
>
> **同 series 文档**: `strategy.md` (战略) / `roadmap.md` (5 阶段路线) / `v1_triton_log.md` (V1 已实施)

---

## 7. Layer B 系统级优化 plan (next phase)

> **路径选择**: kernel 内配置 tune 已尽 (§6.6 Step 11+ 结构性优化是单点突破, ROI 中低). 下一阶段转向**端到端系统级优化** — 量化全链路, 把视野从"推理 32ms"扩大到"相机曝光 → motor 响应"全链.

### 7.1 范围 + 约束

#### 硬件约束 (单 5090)
sim01 真机部署**只用 1 个 5090** (尽管机器物理装 2 张). 这排除了"多 GPU async pipeline"等架构选项:

| 已排除 | 原因 |
|---|---|
| 双 5090 load balance (vision 1 卡, decoder 另 1 卡) | 单 GPU 约束 |
| Multi-GPU batch | 同上 |
| Cross-GPU stream overlap | 同上 |

#### 已选 3 子项 (按依赖顺序)

```
B4 (V1 serve 包装) ──→ 真机可跑 ──→ B1 (profile) ──→ B2 (定向 preprocess)
```

#### 暂不做 (Layer A + Layer C 其他项)

| 项 | 决策 | 原因 |
|---|---|---|
| Layer A1 kernel fusion (-1-3%) | 暂缓 | 收益小, 推理 32ms 已远 < timer 周期 |
| Layer A4 wgmma 重写 (-5-10%) | 暂缓 | 5-10 天高风险, B 层未做完前不投入 |
| Layer C2 inference_rate 调参 | 阶段 1 (并行) | 真机测试 1 完成后即可做 |
| Layer C3 与 QP 联动 | 阶段 2 (依赖 §3.3 #5) | QP 落地后 |
| Layer C4 客户端 MPC | 阶段 5 (条件) | 真机测试 2 测出 t_motion > 50ms 才上 |

### 7.2 B4 V1 serve 包装 ✅ Phase 2 完成 (2026-05-20)

**目标**: 把 §6 的 `Pi05InferenceTuned` 包装成 WebSocket 服务, ROS2 client 无感切换.

#### 实测结果 (kai0/.venv_5090_trt 本机 smoke test, 5 iter Phase 2)

| 段 | iter 2-5 平均 (ms) | 说明 |
|---|---:|---|
| **total** (server side) | **~40.5** | 含全部 5 段 |
| preproc | 5.7-6.7 | PIL resize 224 + bf16 cast (B2 优化候选) |
| state_encode | 0.3-0.4 | sentencepiece + PaliGemma embed lookup |
| infer (V1 forward) | 34.0 | offline P50=32 + 2ms inference_mode/sync overhead |
| postproc | 0.1-0.2 | action denorm + .cpu() |

vs Q2 JAX 196ms: **4.9× 加速**已实现 (server side). 加 WebSocket + ROS2 transit (B1 待量化) 后客户端 RTT 估 ~50-55 ms (vs JAX 196 ms 仍 3.5-4× 加速).

**State conditioning sanity check**: 切换 state ([0.5,-0.3,...] → [0,0,...]), action[0] max diff = 0.286 → state 编码确实流入 action.

#### 实施步骤

#### 步骤

1. **新增** `kai0/scripts/serve_policy_v1.py` (复制 `serve_policy.py` 骨架, WebSocket payload schema 不变)
   - 启动: load V1 pickle (`task_a_mix_b6000_p1200_v1.pkl`, 6.7 GB) → `Pi05InferenceTuned`
   - `predict()`: 调用 `infer.forward(image, noise)`, 输出 action chunk (50, 14)
   - 端口 `:8002` (JAX :8000, PyTorch 备用 :8001 之外)

2. **数据格式 adapter**:
   - V1 期望: `image (num_views, 224, 224, 3) bf16 CUDA` + `noise (50, 32) bf16 CUDA`
   - JAX serve 接收: `(num_views, H_orig, W_orig, 3) uint8 numpy`
   - 写 adapter: decode → resize 224×224 → bf16 → CUDA (B2 优化点)
   - `norm_stats` 复用: 加载 deepdive_kai0 同一份 `assets/<asset_id>/norm_stats.json` 做 denorm

3. **sidecar 协议扩展**:
   ```jsonc
   {"base_config_name": "...", "framework": "jax"}        // 默认, 走 :8000
   {"base_config_name": "...", "framework": "v1_triton"}  // 新增, 走 :8002
   ```

4. **启动脚本分发** (`start_scripts/start_autonomy_from_ckpt.sh`):
   ```bash
   FRAMEWORK=$(python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json')).get('framework', 'jax'))")
   case "$FRAMEWORK" in
     v1_triton) ENTRY=serve_policy_v1.py;  PORT=8002 ;;
     pytorch)   ENTRY=serve_policy_pytorch.py; PORT=8001 ;;
     *)         ENTRY=serve_policy.py;     PORT=8000 ;;
   esac
   ```

5. **WebSocket 协议验证**: ROS2 client 零改动, sim01 跑 autonomy 看推理回包形状一致.

**输出**: `serve_policy_v1.py` + sidecar 协议 + 启动脚本分发 + 1 个真实 ckpt 跑通端到端.

**验收**: ROS2 `policy_inference_node` 调 `:8002` 完成一次 autonomy 周期, action chunk 形状 (50, 14), 推理 RTT 与 §6 benchmark 一致 (~35-40 ms 含 WebSocket).

### 7.3 B1 全链路 latency profiling (2-3 天)

**目标**: 量化 11 段延迟, 找 P50/P95 拐点, 决定 B2 优化方向.

#### 11 段切片

```
t0 相机曝光             ─┐
t1 USB readout          │  → t_camera (~50ms, V2 估)
t2 ROS2 transport      ─┘
t3 client → server WebSocket send
t4 server preprocess     → t_preproc (B2 候选)
t5 server inference      → t_infer (32 ms, 已知)
t6 server postproc / norm denorm
t7 server → client WebSocket recv
t8 ROS2 publish (action chunk)
t9 Piper CAN write
t10 motor 响应           → t_motion (~50-150ms 估, §4.2 测试)
```

#### 步骤

1. **改 `policy_inference_node.py`** (client 侧): `_inference_loop` 每段加 `time.perf_counter()` 串, 落 CSV (列 = t0..t10, 100 cycle 取 P50/P95/P99)
2. **改 `serve_policy_v1.py`** (server 侧): `predict()` 内部 t4/t5/t6 用 GPU event timer (`torch.cuda.Event(enable_timing=True)`)
3. **跑 1-2 个真任务 autonomy session** (sim01, ≥200 cycle), 收集 latency CSV
4. **出表 + 推荐**: 11 段 P50/P95/P99, 标最大头 + 抖动最大段, 落 `docs/deployment/latency_profile_v1.md`
5. **副产物**: 同时拿到 §3.2 #8 `latency_k` 的数据驱动值 (一举两得)

**输出**: `docs/deployment/latency_profile_v1.md` (11 段表 + 优化方向推荐).

**验收**: 找到 ≥1 个 P50 > 5ms 的非推理段, 或确认链路已无 > 5ms 段.

### 7.4 B2 Preprocess 全 GPU 化 (1-3 天, 数据驱动)

**目标**: B1 profile 指出 t4 > 5ms 时才做; 否则确认链路已优化.

#### 候选优化 (按可能瓶颈)

| 候选 | 现状假设 | 改成 | 预期收益 |
|---|---|---|---|
| Image resize | CPU OpenCV / PIL | GPU `torchvision.transforms.v2` 或 V1 vision_encoder 内嵌 | -2-5 ms |
| Normalize (mean/std) | CPU numpy | GPU torch op | -1-2 ms |
| Uint8 → bf16 cast | CPU | GPU `.to(torch.bfloat16, non_blocking=True)` | -0.5-1 ms |
| host→device copy | 阻塞 | Pinned memory + `non_blocking=True` 异步 | -1-2 ms |
| JPEG decode (若 client encode) | CPU PIL | nvJPEG | 视 client 编码方式而定 |

#### 步骤

1. 看 B1 数据定优先级 (只改 > 1ms 段)
2. 改 `serve_policy_v1.py` 内 preprocess pipeline
3. 重测 B1 验证

**验收**: t4 降到 < 3ms, 或确认无优化空间.

### 7.5 时序 + 风险

#### 时序总览

| 周 | 主任务 | 验收 |
|:---:|---|---|
| 1 (Mon-Fri) | B4 V1 serve 包装 + sidecar | sim01 跑通真实 ckpt, ROS2 节点无感 |
| 1 weekend / 2 (Mon-Wed) | B1 profile + autonomy session | latency_profile_v1.md, 11 段延迟表 |
| 2 (Thu-Fri) | B2 定向 preprocess GPU 化 | t4 < 3ms 或确认无空间 |

**总工程**: 1.5-2 周, 主线 B4. 关键里程碑: **B4 完成 = 真机 V1 推理首跑通**.

#### 风险点

| 风险 | 缓解 |
|---|---|
| V1 推理需要的 image preprocessing 与 JAX serve 不一致 (resize 策略 / normalize 参数) | 取 JAX `_preprocess_observation` 作对照, V1 同输入对比输出 |
| `norm_stats.json` 加载 — V1 推理是否完整内嵌? | 检查 `convert_kai0_to_v1.py` 是否把 stats 也存进 pickle; 否则 serve_policy_v1.py 单独加载 |
| Inference cold start (V1 build + CUDA Graph capture, ~30s) 影响 systemd 启动 | 启动脚本里加 readiness probe; `wait_for_serve.sh` 等 graph capture 完成才进 autonomy |
| ROS2 client chunk 协议与 V1 输出不一致 | 端到端用同 schema, V1 serve 内部 padding/格式补齐 |
| 单 5090 资源争抢 (推理 + 其他后台 GPU 进程) | sim01 部署时锁定 `CUDA_VISIBLE_DEVICES=0` 给 serve, 其他进程禁用 GPU |

#### Out of scope (本阶段不做)

- Layer A 任何项 (kernel fusion / wgmma / FA3) — kernel 32ms 已远 < timer 周期 (100ms @ 10Hz), ROI 低
- 双 5090 利用 — 真机约束
- B3 WebSocket payload trim — 用户排除
- 任何破坏 §6 数值对齐 (rel error 1.42%) 的改动

### 7.6 Step 0 实测 latency profile (2026-05-23, baseline)

V1 真机部署完成后 (5-fix commit `0320aa2`, V1 vs JAX 行为已对齐), 首次跑通 30s observe-mode 真机 stack, 收 `/tmp/kai0_latency_43037.csv` → snapshot `/tmp/step0_latency_snapshot.csv` (477 samples → 跳前 30 warmup → 447 steady).

#### 11 段实测 (P50 / P95 / P99 / max / mean, 单位 ms)

| 段 | P50 | P95 | P99 | max | mean |
|---|---:|---:|---:|---:|---:|
| t_image_age (cam → client lag) | **55.6** | **70.5** | 79.3 | 79.9 | 55.5 |
| t_obs_construct (3× CPU resize_with_pad) | **35.7** | **42.5** | 45.5 | 56.3 | 35.0 |
| t_ws_full_rtt (含 server_total 38) | 43.4 | 47.7 | 49.8 | 51.5 | 43.6 |
| t_ws_overhead (msgpack + loopback) | 6.5 | 10.5 | 11.9 | 12.5 | 6.8 |
| server_preproc (msgpack decode + H2D) | 1.6 | 2.3 | 2.9 | 7.5 | 1.6 |
| server_state_encode | 0.5 | 0.8 | 1.1 | 3.4 | 0.5 |
| **server_infer (V1 forward, Pi05InferenceTuned)** | **34.0** | **35.6** | 36.3 | 40.5 | 34.4 |
| server_postproc (denorm + .cpu()) | 0.2 | 0.4 | 0.7 | 0.9 | 0.2 |
| server_total | 36.7 | 38.4 | 39.7 | 43.7 | 36.9 |
| t_buffer_integrate (RTC merge) | 0.6 | 4.4 | 6.2 | 6.7 | 1.3 |
| **t_loop_total (cycle work)** | **80.5** | **89.1** | 94.1 | 102.3 | 80.0 |

#### Cam → action emit 完整链 (P50 累计 wall-clock)

| # | 段 | side | P50 (ms) | 累计 P50 (ms) |
|:-:|---|---|---:|---:|
| 0 | 传感器曝光 + USB + ROS2 transport + executor lag (image_age) | hardware/ROS | 55.6 | 55.6 |
| 1 | client obs_construct (3× resize_with_pad CPU) | client CPU | 35.7 | 91.3 |
| 2 | client → server msgpack + WS (~半 RTT) | net + CPU | 3.3 | 94.6 |
| 3 | server msgpack decode + H2D | server CPU+PCIe | 1.6 | 96.2 |
| 4 | server state_encode (sentencepiece + embed) | server CPU+GPU | 0.5 | 96.7 |
| 5 | server V1 forward (Pi05InferenceTuned) | server GPU | 34.0 | 130.7 |
| 6 | server postproc + D2H | GPU+PCIe | 0.2 | 130.9 |
| 7 | server → client WS + decode | net + CPU | 3.3 | 134.2 |
| 8 | client RTC buffer_integrate + publish | client CPU | 0.6 | **134.8** |

**真实控制延迟 (cam → emit) P50 = 134.8 ms, P95 ≈ 159 ms.** 当前真机跑 ~11Hz cycle (t_loop_total 80ms vs 10Hz timer 100ms — cycle 撑得很满, idle 仅 20ms).

#### 关键观察

1. **server_infer 已触顶** (32-34ms, §6.5 32.05ms benchmark 与 live 34ms 一致, +2ms inference_mode/sync overhead). 想再快需 Step 11/13 (3-10 天工程换 1-3.5ms).
2. **真正大头不在 forward**: client 端 obs_construct (35.7ms) + image_age (55.6ms 不在 cycle 但决定感知延迟).
3. **cycle 80ms = obs 36 + forward 34 + ws 7 + RTC 1 + 其他 2** — 两个 30+ms 大头, 其他都已经很小.

#### 复跑命令

```bash
# observe-mode (不动机械臂, profile 数据一致)
./start_scripts/start_autonomy_v1.sh > /tmp/v1_step0.log 2>&1 &
LAUNCH_PID=$!
# 等到 /tmp/kai0_latency_<pid>.csv 出现, 累积 ≥400 行 (~40s)
# 然后 kill -INT $LAUNCH_PID
```

### 7.7 P1 image_age root cause: SingleThreadedExecutor 阻塞 callback drain

> 实测 image_age P50=55.6ms / P95=70.5ms, **远超 30fps 相机物理 floor**: sensor exposure (~16ms) + USB readout (~10ms) + ROS2 transport (~5ms) 物理上限 ~31ms — 必有非物理来源.

#### 元凶

`ros2_ws/src/piper/scripts/policy_inference_node.py:2629`:

```python
# Use SingleThreadedExecutor unconditionally (Rerun thread-safety).
executor = rclpy.executors.SingleThreadedExecutor()
```

#### 链路时序

```
T_0          timer fire → inference_loop() 进入 (单线程 executor 阻塞所有 callback)
T_0..T_0+80  80ms inference (image_*_callback 队列 blocked, _img_*_deque 不更新)
T_0+80..T_0+100  20ms idle, callbacks 消费 (最多消化 0-1 帧因 33ms 相机间隔)
T_0+100      下个 cycle 开始
        ...
T_0+180      _record_latency_sample 读 deque[-1].header.stamp
```

`_record_latency_sample` (line 2138-2146) 读 `self._img_front_deque[-1].header.stamp` 即 "最新到达过 client 的帧". 但 callbacks 在 80ms cycle 中**完全不 fire**, deque 没更新, 最新帧的 stamp 是 cycle 开始之前若干 ms 的, 不是相机实际发的最新帧.

#### image_age 来源拆解 (估算)

| 来源 | P50 估 (ms) | 备注 |
|---|---:|---|
| sensor 曝光 (auto-exposure ~16ms) | 8-16 | 物理 floor |
| RealSense USB readout + driver | 5-10 | 物理 floor |
| ROS2 transport (realsense → multi_camera → policy_inference) | 2-5 | 小 |
| **SingleThreadedExecutor cycle 阻塞 callback drain** | **20-30** | **可修 (P1.a)** |
| `_get_synced_frame` `min(latest)` sync 偏向最慢相机 | 5-10 | 可改 (P1.c) |
| **合计** | **40-71** | **匹配 P50 55.6 / P95 70.5** |

#### 次要问题: `_get_synced_frame` (line 1899-1952)

```python
frame_time = min(latest_3_cameras_stamp)  # 等最慢的相机
# 然后 popleft() 拿"最老的 >= frame_time 的那一帧" — 偏向最老
```

= 等最慢相机 + 用其他相机较老帧. 应改 `max(latest)` 或直接 `[-1]` 取最新 (相机间偏移目测对任务无影响).

#### 修复方案

| 方案 | 节省 image_age | 工程量 | 风险 | 备注 |
|---|---|---|---|---|
| **A1. MultiThreadedExecutor + ReentrantCallbackGroup for image subs** | **-20-30ms** | 0.5 天 | 中 | Rerun 注释说需加锁, 实施时验证 |
| A2. image callbacks 单跑 `rclpy.spin` 子线程 (绕过 executor) | -20-30ms | 0.5 天 | 低 | 备选 (Rerun 锁难加时) |
| A3. `_get_synced_frame` 改 max(latest) 或直接 deque[-1] | -5-10ms | 0.3 天 | 低 | 锦上添花 |

---

### 7.8 20Hz cycle 攻关执行优先级 + 跟踪表 (2026-05-23-)

> **目标**: cam → action emit < 50 ms (≥ 20Hz cycle + image_age 砍合理), 真实控制延迟 → 100ms 以下.

#### 优先级表 (按 ROI 排序, 排除 GPU forward 优化 — 见 §6.6/§7.2 现状)

| Pri | 项目 | 预期省 P50 (ms) | 工程量 | 风险 | 备注 |
|:---:|---|---:|---|---|---|
| **P1.a** | MultiThreadedExecutor + ReentrantCallbackGroup | -20-30 (image_age) | 0.5 天 | 中 (Rerun 锁) | image_age 第一锤 |
| **P1.b** | 相机配置 C2+C5+C7+C4 (60fps + 关 D435 depth + raw API 无 sync) | -15-25 (image_age) | 0.5-1 天 | 低 | **本次执行**, 与 P1.a 收益可叠加 |
| P1.c | `_get_synced_frame` 改 max(latest) 或不 sync | -5-10 (image_age) | 0.3 天 | 低 | 锦上添花 |
| **P2** | obs_construct GPU 化 (B2, .venv_5090_trt) | -30-32 (cycle) | 1-3 天 | 中 (preprocess parity 需对齐) | cycle 主攻 |
| **P3** | B2 POSIX SHM transport (替 msgpack+WS) | -4-7 (ws_overhead) | 2-3 天 | 中 | cycle 补刀, 仅 V1 路径, 不影响 JAX |
| P3-alt | B1 Unix Domain Socket 替 TCP localhost | -1-2 | 0.3 天 | 低 | 顺手做 |
| P5 | RTC jitter (P95 4.4ms → 1ms) | -1-3 (jitter only) | 1-2 天 | 中 | 收尾 |

#### 不做 (本阶段)

| 项 | 排除原因 |
|---|---|
| B5 In-process V1 (取消 WS) | `.venv_5090_trt` (cu128) vs `.venv` (cu126) PyTorch 版本隔离, 不能合 |
| B4 CUDA IPC tensor 共享 | 必须 client 先 GPU 化 (P2 完成后才有意义), 独立 5-7 天 ROI 一般 |
| C1 输入分辨率改 424×240 / C8 hardware crop | 训练数据 letterbox 224×224, 改输入 → OOD |
| C3 固定曝光 (8ms) | 训练用 auto-exposure, 数据分布不一致 |
| Layer A (kernel fusion / wgmma / FA3) | 见 §6.6, 3-10 天工程换 1-3.5ms, ROI 极低 |

#### 决策树

```
P1.b 执行 (本次) → 量 → 记
  ├─ image_age 降到 物理 floor (~30ms)         → 跳 P1.a, 直接进 P2
  └─ image_age 还在 45ms+                      → 做 P1.a (executor 改) → 量 → 记

P2 (B2 GPU preprocess)
  ├─ cycle ~48-55ms (P95) → 18-20Hz 达成        → 收工或再补 P3
  └─ cycle ~55ms+ → 进 P3 (SHM transport)       → cycle ~45-50ms = 20Hz

cam → emit 目标
  100 ms (~10Hz) → 验收
  75 ms (~13Hz)  → 优秀
  50 ms          → 超额 (理论可能, 需 P1+P2+P3 全做且 image_age < 25ms)
```

#### 实施进展总览 (2026-05-23, V1 20Hz 攻关单日完成)

```
┌──────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ 节点                 │ baseline │  P1.b    │   P2     │  A.2     │  C.2†    │ C.4+C.3‡ │
├──────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ cycle P50 (ms)       │   80.5   │   79.8   │   62.2   │   44.3   │   43.9   │ ↓ 40.05  │
│ cycle P95 (ms)       │   89.1   │   93.8   │  100.3   │  100.3   │   49.6   │ ↓ 44.25  │
│ cycle freq cap (Hz)  │  ~12     │  ~13     │  ~16     │ ↑ 22.6   │   22.7   │ ↑ 25.0   │
│ ws_overhead P50 (ms) │   6.5    │   ~6.5   │   ~6     │   5.4    │   5.3    │ ↓ 2.2    │
│ obs_construct (ms)   │   35.7   │   35.1   │   14.9   │ ↓ 0.03   │   0.03   │   0.03   │
│ server_infer P50     │   34.0   │   34.2   │   34.9   │   34.6   │   34.2   │   34.5   │ (floor)
│ image_age (ms)       │   55.6   │   55.9   │   58.2   │   42*    │   125.7  │  130.3   │ § 真实总 age
│ 改动                 │ V1 fix   │ depth off│ fast_obs │ pipeline │ stamp fix│ SHM v2   │
│ 真机 execute 验证    │ —        │ —        │   ✓      │   ✓      │   ✓      │   ✓      │
└──────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
   * A.2 旧 metric, 70% 样本 NaN, biased 偏倚 valid 30%
   † C.2 image_age 测量纠错: 旧 deque[-1] 是 "record-time latest 帧" 不是 "实际用的帧"
   ‡ C.4 SHM v2 (替 WS+msgpack TCP loopback) + C.3 buffer_integrate vectorize
   § image_age 130ms = 物理 sensor 50 + worker prefetch 22 + cycle 44 + var 10. RTC chunk
     merge 让等效控制延迟 ≈ cycle_period (now 50ms@20Hz), 不直接由 image_age 决定.

Δ baseline → C.4 (含全部优化):
  cycle P50    80.5 → 40.05 ms  (-50%, +100% throughput)
  cycle P95    89.1 → 44.25 ms  (-50%)
  cycle freq cap  12 → 25 Hz (work-limit, 实际由 inference_rate timer 决定)
  ws_overhead   6.5 → 2.2 ms (-66%)

20Hz timer 启用 (start_autonomy_v1.sh inference_rate:=20.0):
  cycle period 50ms, cycle work P95 44.25ms → headroom 5.75ms 安全运行
  RTC 参数适配: latency_k=2 min_smooth_steps=3 rtc_execute_horizon=4 (M2-C 等比例缩放)
```

#### 跟踪表 — 实施记录 (A 已落地 / B 失败归档 / C 待实施)

每步实施后跑 ≥30s observe-mode profile, 收 `/tmp/kai0_latency_*.csv`. 时间 ms.

##### A. 已落地的有效改动 (改前 → 改后)

| Step | 改前 (P50 ms) | 改后 (P50 ms) | Δ P50 | 改了什么 | 验证 snapshot |
|---|---|---|---|---|---|
| **0 baseline** | — | cycle **80.5** / image_age **55.6** / cam→emit **134.8** | — | V1 5-fix commit `0320aa2`. fps=30, D435 head depth on, SingleThreadedExecutor | `/tmp/step0_latency_snapshot.csv` (447 samples) |
| **P1.b-partial** (C5+C7 only) | cycle 80.5 / image_age 55.6 / cam→emit 134.8 | cycle 79.8 / image_age 55.9 / cam→emit 134.3 | cycle -0.7, image_age +0.3 (噪声内) | `config/camera_depth_flags.py` `ENABLE_DEPTH_TOP_HEAD` 通过 launch arg `enable_head_depth:=false` 关掉. fps 保留 30 (因 C2 失败). | `/tmp/p1b_30fps_depthoff.csv` (448 samples). 结论: depth pipeline 不在 image_age critical path |
| **P1.a 回退到无影响态** | — | (= baseline) | 0 | 撤销 MultiThreadedExecutor 改动 (见 B.2/B.3 失败原因). 保留 `_sensor_cb_group = ReentrantCallbackGroup()` 代码 (SingleThreaded 下 Reentrant 退化为 MutuallyExclusive, bit-identical). JAX 路径完全不动 | — |
| **P2 Step 1+2** ⭐ | cycle 80.5 / image_age 55.6 / cam→emit 134.8 / obs_construct **35.7** | cycle **62.2** / image_age 58.2 / cam→emit **120.4** / obs_construct **14.9** | **cycle -18.3**, obs_construct **-20.7**, image_age +2.6, cam→emit -14.4 | launch arg `fast_obs_pipeline:=true` 启用 3 项跳过: (1) `_jpeg_mapping` cv2 encode+decode roundtrip; (2) `bridge.imgmsg_to_cv2(bgr8)` → `np.frombuffer(msg.data).reshape(H,W,3)` 直 view; (3) `cvtColor(BGR2RGB)` — multi_camera_node publish 已 rgb8, 全程零格式转换. JAX 路径默认 false, bit-identical | `/tmp/p2_step12_snapshot.csv` (414 samples). cycle ~16Hz, 距 20Hz 还差 12ms |
| **A.2 异步流水线** ⭐⭐ | cycle 62.2 / obs_construct 14.9 / image_age 58.2 / cam→emit 120.4 | cycle **44.3** / obs_construct **0.03** / image_age **42.2** (valid 30%, 70% NaN) / cam→emit ~86 | **cycle -17.9** = 22.6Hz **cycle work 达成 20Hz 目标** ✓, obs_construct **-14.9**, image_age (valid) -16.0 | launch arg `pipelined_obs:=true` 启用. 新增 `ObsPrefetchWorker` 类 (~90 行 in policy_inference_node.py), 在背景线程持续 pop `_get_observation()` 放 maxsize=1 queue ("drop old, put new"); main 推理 loop 从 `worker.get_obs(timeout=0.1)` 取 prefetched obs, fallback `_get_observation()` 同步. 等价于把 obs_construct 15ms 藏到 forward 35ms 背后. JAX 路径默认 false, bit-identical. 真机 execute 验证 OK | `/tmp/a2_snapshot.csv` (384 samples). image_age 70% NaN 因 worker pop deque 比 cam push 快, main `_record_latency_sample` 读 deque[-1] 时 deque 偶发空. 测量准确性需 C.2 修复, 但实际 obs 由 worker 准备, 真实 obs_age 不受 NaN 影响 |
| **C.2 image_age 测量修复** | image_age "valid 30%, 70% NaN" 偏倚样本, P50 显 42.2 但样本不可信 | image_age **0% NaN, 100% valid**. P50 = **125.7ms** P95 148.5 (= **本次推理实际用的 head 帧 → action 发出**的真实总 age) | image_age 测量准 (从偏倚 42→真实 126) — **不是性能回归**, 是发现真实控制延迟一直就是这么高, 旧 metric 测错了对象 | `_get_observation_with_stamp()` 新方法返 `(obs, head_stamp_ns)` tuple, stamp 走**侧通道** (不进 obs dict 避免 JAX/V1 transform 链兼容性风险). `ObsPrefetchWorker.queue` 改存 tuple. `_record_latency_sample` 新增 `obs_stamp_ns` 参数, 优先用它而非 deque[-1]. 旧 `_get_observation` 保留为 thin wrapper (兼容). JAX 路径不传 obs_stamp_ns → fallback deque (= legacy 行为, bit-identical) | `/tmp/c2_v2_snapshot.csv` (350 samples). 真实控制延迟揭示: 物理 sensor ~50ms + worker prefetch lag ~22ms + cycle work 44ms + variance 10ms = 126ms. RTC chunk merge 让等效控制延迟 ≈ cycle_period, 不直接由 image_age 决定 |
| **C.4 SHM v2 transport** | cycle 43.87 / 49.64 (WS msgpack+TCP loopback) | cycle **40.05 / 44.25**, ws_overhead 5.30/8.31 → **2.18/5.45** | **cycle -3.8 / -5.4 ms** ⭐, ws_overhead -3.1 / -2.9 | 新增 `kai0/scripts/shm_transport.py` (~250 行) — `ShmServer` 守护线程 + `ShmClient` 双端. 协议: 4MB shm region `/dev/shm/kai0_v1_obs` (header 64B + image 451KB zero-copy + metadata msgpack) + 64KB resp region. 同步: hybrid busy-poll (0-200µs 硬 spin 不释 GIL + 0.2-50ms `time.sleep(0)` yield + 50ms+ soft sleep). 关键 v2 优化: (1) 客户端 `np.frombuffer(shm_buf).reshape(3,3,224,224)` view + numpy assign 单 memcpy/cam, 跳 `np.stack` + `tobytes` 中间 alloc; (2) hybrid poll 0 sleep 在 inference 期间 = 低 detect latency. server `--transport ws\|shm\|both` (default ws = JAX legacy). client `transport=shm` (V1 路径). | `/tmp/c2_v2_snapshot.csv` vs `/tmp/shm_v2b_snapshot.csv` 对比. 真机 1414 cycle execute 验证 OK, infer 37-42ms 稳定, RTC ratio 0.5-0.85 健康. JAX 路径完全不动 (默认 ws transport) |
| **C.3 buffer_integrate vectorize** | buffer_integrate P95 4.07ms, 25% cycle spike 2-8ms (smooth 段 list comprehension) | P95 3.70ms (-0.37), spike count 不变 25% | spike 主因不是 Python loop 而是 **Linux CFS scheduler 抢占 inference thread** (~5ms 时间片) — 矢量化收益弱 | `StreamActionBuffer.integrate_new_chunk` 把 47 次 Python loop `w_old[i]*np.asarray(old_list[i])+w_new[i]*np.asarray(new_list[i])` 改成单次 `np.asarray(old_list[:n]) → broadcast w[:,None]*A + w[:,None]*B` 矢量化运算. 调查实测显示 spike 真凶是 OS 调度而非计算, 真修需 SCHED_FIFO (需 CAP_SYS_NICE / root). Vectorize 仍保留, 数值等价 (broadcast 与 element-wise scalar product 数学等价) | 真机记录 116 个 >2ms spike, 多数仍在 smooth 段位置但其实是被调度 preempt. 留待 C.7 SCHED_FIFO 真修 |
| **20Hz timer + RTC sweep** | `inference_rate:=10.0`, period 100ms (cycle work 40ms 但 sleep 60ms), **实际推理 10Hz**, RTC M2-C (k=3, smooth=3, exec_h=6) | `inference_rate:=20.0`, period 50ms, **实际推理 20Hz**. RTC 按比例缩放: `latency_k:=2 min_smooth_steps:=3 rtc_execute_horizon:=4` | 推理频率 10→20Hz (+100%), timer 节流 100→50ms. RTC params 缩放反映新 cycle 节奏: latency_k=2 (覆盖 ~1.5 publish step), exec_h=4 (保持 2×latency_k 比例) | start_autonomy_v1.sh 4 行改动. RTC k=2/exec_h=4 是教育猜测, 任务可能需要 sweep 微调 (M2-C 验证只在 10Hz). cycle work P95 44.25 < 50ms period, 安全 headroom 5.75ms | **待真机验证 20Hz execute 任务行为**. 若 jitter overrun (P95 触 50ms) 可降到 inference_rate=18 (period 55ms) 留 11ms headroom |

##### B. 失败尝试归档 (现象 + 根因 + 处置)

| # | 尝试 | 失败现象 | 根因 (调查后定位) | 处置 |
|:-:|---|---|---|---|
| **B.1** | **P1.b-C2**: `cam_fps:=60` (所有相机 30→60fps) | 启动时 D405 hand_left 报 `Couldn't resolve requests`, 3/3 attempts fail. 仅 D435 head + D405 hand_right 起来 | `rs-enumerate-devices` 实测**硬件层 mode list 限制**: hand_left S/N 409122273074 在 640×480 Color 只 expose 30/15/5fps (vs hand_right 同型号同 fw 5.17.0.10 expose 90/60/30/15/5; D435 expose 60/30/15/6). 60fps 在 424×240 OK. 物理原因: hand_left 当前接到 **USB 2.0 root hub (Bus 003, 480M)**, librealsense 按协商速率拒了高带宽 mode; 重插拔 + 切端口仍落 USB 2.0 bus, 主板/线材层面 | 接受 30fps 跑 V1 (本质 image_age 物理 floor 限制 ~25-40ms, 60fps 只省 10-15ms 收益小). 未来若需 60fps: 换 USB 3.0 端口/线 或换 hand_left 相机 |
| **B.2** | **P1.a attempt 1**: `MultiThreadedExecutor()` (default `num_threads = os.cpu_count()` ≈ 32) + image+joint subs 入 `ReentrantCallbackGroup` | 第 1 次 infer 跑出 1753ms (cold start), 第 2 次起 ~3s/iter, CSV 几乎不增. 推理 loop alive 但 throughput 崩 | 推理 loop 是独立 `threading.Thread` (line 648). 32 个 executor worker thread 同时跑 Reentrant cb, 在 CPython GIL 下与 inference 线程的 numpy/msgpack/WS pack 工作争 GIL, inference 被打成碎片. cb 工作量虽小但发起非常频繁 (30fps × 3 image + 200Hz × 2 joint), GIL 切换开销淹没 inference | revert. 未尝试 `num_threads` 中间值 (4/8) — 先跳到 attempt 2 试极端 num_threads=2 |
| **B.3** | **P1.a attempt 2**: 同 B.2 但 `MultiThreadedExecutor(num_threads=2)` | 永远卡在 "Waiting for sensor data" log, deque 始终空, inference loop 一次都没进 | num_threads=2 + 6 个 sensor sub 在同一 Reentrant 组 — executor 只有 2 worker, 看似够但实际 ROS2 内部某处 (DDS callback dispatch / msg 反序列化) 在 multi-thread context 下行为变化, 6 cb 之间调度不均. 具体卡在哪未定位 (无 py-spy / gdb-py-bt 权限) | revert 到 SingleThreaded. P1.a 真正解锁需更深 root cause: 选项 = (a) py-spy/gdb attach 量 hang 点; (b) 改设计 (sensor subs 单独跑二级 Node + `rclpy.spin()` 子线程, 物理隔离 ROS2 调度). 标 P1.a-v3 留后续 |

##### C. 待实施 (按优先级)

| 优先级 | Step | 预期 Δ cycle (ms) | 工程量 | 风险 | 触发条件 |
|:-:|---|---|---|---|---|
| **C.0** ✅ | 真机 execute 模式验证 P2 Step 1+2 模型对齐 | 0 (validation) | 30 min | 0 | **已完成** — 真机跑通, 无异常 |
| **C.1** ✅ | A.2 异步流水线实施 + observe 实测 + 真机 execute 验证 | cycle -18ms | 2h | 中 | **已完成** — cycle 62→44ms (22.6Hz), 真机验证通过 |
| **C.2** ✅ | image_age 测量修复 (tuple 侧通道传 head_stamp_ns, 不进 obs dict) | 测量准 (42→真实 126) | 1h | 低 | **已完成** — 揭示真实 cam→emit 126ms 而非 86ms; obs dict bit-identical |
| C.3 | Path D: cv2.resize 替 PIL resize_with_pad | -3-5 ms | 30 min | 中 (数值偏移叠加 P2/A.2) | 收益小, P95 紧时再做 |
| C.4 | Path B1: Unix Domain Socket 替 TCP loopback | -1-2 ms | 30 min | 低 | 顺手做 |
| C.5 | **B.1**: `_get_synced_frame` 改 `max(latest)` 或不 sync | -5-10 ms image_age | 0.3 天 | 低 | image_age 路径 |
| C.6 | **B.2 P1.a-v3**: sensor subs 在二级 Node + 独立 spin 线程 | -20-30 ms image_age | 1-2 天 | 中 | image_age 主攻 (cycle 已超额) |
| C.7 | Path Z2: POSIX SHM ring buffer 替 msgpack+WS | -5-12 ms | 0.5-1 天 | 中 | cycle 已超额, 暂不需 |
| C.8 | server_infer P95=67ms jitter 调查 (V1 forward 偶发) | P95 收紧 | 0.5-1 天 | 低 | P95 cycle 100ms 影响 RTC 稳定性时 |

#### P1.a 实测发现 (2026-05-23) — MultiThreaded executor hang

两轮 attempt 后 V1 路径都跑不通 (见跟踪表 attempt 1/2 行). 现象:
- attempt 1 (`MultiThreadedExecutor()` 默认线程数): first inference 1.7s, 第 2 cycle 起 ~3s/iter, CSV 一行后停滞
- attempt 2 (`MultiThreadedExecutor(num_threads=2)`): 永远卡在 "Waiting for sensor data", deque 没填

未深入定位前已 revert (兼容性优先). 假设走向 (待验证):
1. **GIL 抖动**: image cb 在 multi-thread reentrant 下争抢 GIL, inference 线程 (独立 `threading.Thread`) 被持续打断
2. **`_get_synced_frame` 内部锁顺序问题**: image cb 高频更新 deque 时 inference 线程读 deque[-1] 出现 stale view (虽然 CPython deque.append 原子)
3. **rclpy 内部某个 lock** 在 MultiThreaded 下行为不一致

需要 py-spy / gdb-py-bt 实测 hang 点才能定下根因. 替代设计 (P1.a-v3): 不动主 executor, 把 sensor subs 移到二级 Node + 独立 `rclpy.spin()` 线程, 物理隔离 ROS2 调度.

#### P1.b 实测发现 (2026-05-23)

**C5+C7 (D435 depth off) 在 fps=30 隔离测试下 image_age 无改善** (P50 +0.3ms, 噪声内). 验证了 §7.7 root cause: image_age 主导项是 `SingleThreadedExecutor` 在 cycle 中阻塞 callback drain (20-30ms), 与 depth pipeline 无关.

**C2 (fps 30→60) 被 D405 hand_left 阻挡**: 该设备 (S/N 409122273074, fw 5.17.0.10) 在 BGR8 640×480 @ 60fps 配置 `Couldn't resolve requests`. D435 head 和 D405 hand_right (fw 5.15.1.55) 60fps OK. 怀疑 firmware-specific 限制 (5.17 vs 5.15 行为差) 或 USB hub 仲裁失败. 修复路径:
- 选项 1: `rs-enumerate-devices` 查 hand_left 实际支持的 fps 列表
- 选项 2: 升级/降级 firmware (有副作用风险)
- 选项 3: 让 multi_camera_node 在请求 fps 失败时自动 fallback (mixed fps: head 60 + wrists 30)
- 选项 4: 接受 30fps 物理 floor, 把节省时间投到 P1.a + P2

**结论**: 单独 P1.b 收益不显著 — **下一步主攻 P1.a (executor 改)** 才是真正能砍 image_age 的关键. 跟踪表 baseline 与 P1.b-partial 行差异在噪声内, 视为 "depth-off 无显著代价/收益".

**修改代码留存** (兼容 JAX 路径, 默认 `auto` 走 macros):
- `ros2_ws/src/piper/launch/autonomy_launch.py`: 加 `cam_fps` (默认 '30') + `enable_head_depth` / `enable_left_depth` / `enable_right_depth` (默认 'auto') launch args, `ParameterValue(value_type=...)` 强制类型避免 ROS2 string→bool 自动转
- `ros2_ws/src/piper/scripts/multi_camera_node.py`: 加 3 个 `enable_*_depth_override` param ('auto'/'true'/'false'), `_resolve_depth()` helper 优先 override, fallback macro
- `start_scripts/start_autonomy_v1.sh`: V1 路径加 `cam_fps:=30 enable_head_depth:=false` (临时回退到 30fps, 直到 D405 60fps 问题解决)
- `config/camera_depth_flags.py`: 不动 (JAX legacy 默认 D435 head depth=True)

#### 本次 P1.b 实施清单

1. `ros2_ws/src/piper/launch/autonomy_launch.py` (`multi_cam` Node `fps` 参数 30 → 60)
2. `config/camera_depth_flags.py`: `ENABLE_DEPTH_TOP_HEAD = False` (C7 关 D435 depth)
3. `multi_camera_node.py`: 不改 (depth 开关已由 `_DEPTH_ENABLED_MAP` 门控; aligned_depth 在 depth 关后自动 no-op)
4. C4 (`enable_sync=false`): N/A — 当前用 `pyrealsense2` 原生 pipeline API 而不是 `realsense2_camera_node`, 无 `enable_sync` 参数; 当 depth 关后, `rs.align(rs.stream.color)` 也跳过, 自然没有 cross-stream sync 开销
5. 副作用: rerun_viz_node 订阅的 D435 depth topic 不再有消息 → 仅影响背景 mesh 可视化 (前景仍是 RGB), 不影响 inference
6. 验证: colcon build piper → 重启 stack → 收 30s+ profile → 与 baseline 比

### 7.9 异步流水线设计 (A.2 obs prefetch worker, 2026-05-23-)

> **背景**: P2 后 cycle P50=62ms. 拆解 = obs_construct (15ms) + forward (35ms) + ws_overhead (5ms) + RTC (1ms) + 协调 (~6ms). obs_construct 串行在 forward 前, 占 cycle 24%. 流水线可以**把 obs_construct 藏到 forward 背后**, cycle 砍 ~14ms 直达 20Hz.

#### A. 异步的 3 种含义 (避免概念混淆)

| 形式 | 当前状态 | 砍什么 |
|---|---|---|
| **A.1 帧捕获异步** (相机持续 publish, 主线随时读 deque[-1]) | 已是设计意图, 但被 `SingleThreadedExecutor` 阻塞 callback 破坏 | image_age 20-30ms (P1.a-v3 子 Node 隔离) |
| **A.2 obs_construct 与 forward 流水线** (用 background worker 提前准备 obs) | **未实施 ← 本节设计目标** | cycle 14-15ms (obs_construct 时长) |
| **A.3 action 播放与下一 cycle forward 并行** (chunk buffer + RTC merge) | **已实施** (StreamActionBuffer + Pi0RTC) | (不在 cycle 内, 但让 chunk 边界平滑) |

A.1 和 A.2 互相**正交独立**, 可分别上, 也可叠加.

#### B. 当前同步 vs A.2 流水线 — 时序对比

**当前 (cycle N 同步)**:
```
T=0    pop obs from deque (deque[-1])
T=0    obs_construct (15ms, CPU)         ── 阻塞 forward 起步
T=15   WS send + server_total (38ms)
T=53   WS recv + RTC merge + publish (~1ms)
T=54   cycle work done. sleep until next period.
       cycle = 62ms (含 buffer overhead)
```

**A.2 流水线 (cycle N+1, 假设 worker 在 cycle N 末尾已 prefetch)**:
```
背景 worker (常驻):
  loop:
    obs_N+1 = self._get_observation()    # 15ms
    self._prefetch_queue.put(obs_N+1, block=True)   # 等 main 取走才继续

主线 (cycle):
  T=0     prefetched_obs = self._prefetch_queue.get(timeout=0.1)  # 0-1ms 等
          # worker 立刻开始准备 obs_N+2
  T=0     WS send + server_total (38ms)  ← worker 平行做 obs_N+2 (15ms)
  T=38    WS recv + RTC merge + publish (~1ms)
  T=39    cycle work done. obs_N+2 已在 queue (T=15 完成).
          → next cycle pop 几乎 0 等待
          
  cycle = 39-40ms ≈ 25Hz max throughput
  加 jitter buffer / period 控制 → 实际 ~48-50ms = 20-21Hz
```

#### C. 实施清单 (~2 小时)

##### C.1 新增 `ObsPrefetchWorker` (~50 行)

挂在 policy_inference_node 上的内部类, 持有:
- `_request_event = threading.Event()` — main 取走后置位, worker 启动新一轮
- `_prefetch_queue = queue.Queue(maxsize=1)` — worker put, main get
- `_running = True` flag — destroy_node 时清掉
- `_thread = threading.Thread(target=self._loop, daemon=True)` — daemon 自动随主进程退

worker loop:
```python
def _loop(self):
    while self._running:
        try:
            obs = self._owner._get_observation()  # 共用 _get_observation 路径
            if obs is None:
                time.sleep(0.005)
                continue
            self._prefetch_queue.put(obs, timeout=1.0)  # main 长时间不取就丢弃 retry
        except queue.Full:
            continue  # main 卡了, 丢这帧, 下轮再生
        except Exception as e:
            self._owner.get_logger().warn(f'ObsPrefetchWorker error: {e}')
            time.sleep(0.05)
```

##### C.2 改 `_inference_loop` (~20 行)

```python
# 旧:
obs = self._get_observation()
if obs is None:
    time.sleep(0.01); continue

# 新 (gated by self._pipelined_obs):
if self._pipelined_obs:
    try:
        obs = self._obs_prefetch.get_obs(timeout=0.1)
    except queue.Empty:
        # Fallback: 同步取一次 (worker 卡了, 不阻塞 inference)
        obs = self._get_observation()
        if obs is None:
            time.sleep(0.01); continue
else:
    obs = self._get_observation()
    if obs is None:
        time.sleep(0.01); continue
```

##### C.3 Launch arg + 启动

- `autonomy_launch.py`: `pipelined_obs_arg = DeclareLaunchArgument('pipelined_obs', default_value='false', ...)`, `ParameterValue(value_type=bool)`
- `policy_inference_node.py`: `declare_parameter('pipelined_obs', False)` + init worker
- `start_autonomy_v1.sh`: 传 `pipelined_obs:=true`

##### C.4 启动 / 停止

- `__init__` 尾部 (在 inference_loop 之前): 若 `_pipelined_obs=True` 起 worker
- worker 第一次 put 需要 sensor data ready — 跟 inference_loop 一样会自然等
- `destroy_node()` 添加 worker stop logic

##### C.5 验证

- 同 P2 路径: 跟 baseline 对 chunk 数值 (相同输入序列, chunk diff < 1%)
- cycle 时长应从 62ms → 45-50ms (P50)
- 真机 execute 跑任务一遍, 看动作是否流畅

#### D. 风险分析

| 风险 | 来源 | 缓解 |
|---|---|---|
| **obs 时间漂移 + RTC 不一致** | worker 准备 obs 的时间点比 main 用的时间点早 ~35ms; RTC `prev_chunk` 跟当前 obs 的时序不对齐 | obs 的 `state` 用 worker prep 时的; RTC 用的 `prev_chunk` 用 main 上轮的 — 二者跨 cycle 时间错位最多 ~35ms, RTC inference_delay 已经按 last_infer_ms (35ms) 注入, 误差吸收 |
| **GIL 抖动** | worker 跟 main 都跑 Python | obs_construct 主要在 numpy/cv2/np.frombuffer (释放 GIL), main 同期主要等 WS 网络 (也释放 GIL). GIL 冲突期 < 1ms |
| **Queue full / empty edge case** | worker 比 main 慢 (e.g. sync_frame 偶发 None) → queue empty → main fallback 同步 | fallback 路径就是当前 baseline 行为, 不变 bit-identical |
| **worker 永远 None** (cameras 挂了) | main 仍可同步 fallback, log warn | log 心跳 + 超 N 次 warning |
| **数据不一致** (worker 用旧 obs, main 期望 newer) | worker 永远 put newest available | queue maxsize=1, main 用 timeout=0.1, 自动取 newest |

#### E. 预期收益

| 指标 | 现状 P2 | + A.2 pipeline | + A.2 + A.1 (P1.a-v3) |
|---|---|---|---|
| cycle P50 | 62 ms | **48 ms** (-14) | 48 ms |
| cycle Hz | 16 | **21** | 21 |
| image_age | 58 ms | 58 ms (不影响) | **28 ms** (-30) |
| cam→emit | 120 ms | 106 ms | **76 ms** |
| 感知 Hz | 8.3 | 9.4 | **13** |

A.2 (2h 工程) + A.1/P1.a-v3 (1-2 天工程) 是当前架构能拿到的接近极限. 再之上需要 SHM transport (A.5) 或 V1 kernel fusion (§6.6) 才能突破.

#### F. 兼容性

- launch arg `pipelined_obs` default 'false' — JAX 路径不变
- V1 路径 `start_autonomy_v1.sh` 显式 `pipelined_obs:=true`
- Worker 失败时 fallback 到同步 `_get_observation()` = 当前 P2 路径
- 不破坏向后兼容

---

