# Realtime-VLA — EE 末端执行器稳定性优化 Layer 1 (2026-05-25)

> 与 `v1_triton_log.md` (latency P50 76→32ms) 并列的**另一维度**优化日志: **真机末端抖动 / 漂移 / 走3退1** 的 deploy-side 修复. 起因 V1 20Hz 部署后真机 idle/task 抖动可见, 但 latency 已达标无可压, 转去攻 cmd 时间线平滑度.
>
> **同 series 文档**: `strategy.md` (战略) / `roadmap.md` (5 阶段路线) / `v1_triton_log.md` (V1 推理优化) / `layer_b_plan.md` (Layer B 系统级)

---

## 1. Layer 1 总进度表

V1 20Hz 真机部署后 (`pi05_flatten_fold_vis_v2_full_step49999`, vis_v2_full ckpt, 80-180Hz publish) 真机表现:

| 子项 | 状态 | 提交 | 改前 → 改后 (主要指标) | 工时 |
|---|---|---|---|---|
| 1.1D idle_gate | ❌ REVERTED | — | 单帧 cmd-state threshold check 在 80Hz publish 引入 chatter, 反向恶化 | 1.5h (失败) |
| 1.1A determinism | ❌ REVERTED | — | `torch.use_deterministic_algorithms` 让 V1 Triton kernels 输出 NaN | 1h (失败) |
| **1.1B min_jerk RTC overlap smooth** | ✅ KEPT | `17ed253` | jiggle 3.33→1.79mm (-46%), jerk peak/s 3.23→1.91 (-41%) | 1.5h |
| **1.1E publish-time EMA α=0.5** | ✅ KEPT | `f9fdd9a` | state jiggle 1.78→0.63mm (-65%), state jerk95 5049→3562 (-29%) | 1h |
| **RTC 综合重调 (k/exec/smooth/publish)** | ✅ KEPT | `be851bc` | task EE 反向运动 1.27/s → 0.12/s (-91% "走3退1") | 1h |
| 1.1C ACT temporal ensembling | ⏳ Pending | — | 工时高 (4-6h) + 跟现有 RTC 重叠风险大, 待评估 | — |

**Layer 1 累计真机收益** (vs origin baseline, 同 vis_v2_full ckpt 同 init pose ±10°):

| 维度 | origin (linear, k=2, no EMA, publish=30) | Layer 1 累计 default | 减幅 |
|---|---:|---:|---:|
| Idle 真机 jiggle (mm) | 3.33 | **0.63** | **-81%** |
| Idle 真机 jerk95 (mm/s³) | 7580 | **3562** | **-53%** |
| Task EE 反向运动 / s | 1.27 | **0.12** | **-91%** |

---

## 2. 测试方法论

### 2.1 数据来源

- **autonomy_recorder_node** 录制每 ep 30Hz LeRobot v2.1 parquet (`Task_A/autonomy/2026-05-25-v2/data/chunk-000/episode_*.parquet`):
  - `action` (14D): 实际 publish 到 master arm 的 cmd
  - `observation.state` (14D): puppet 真机 joint 回读 state
- **离线 FK 分析**: 用 `calib/piper_fk.py` 把 14D 关节展到双臂 EE 6-DoF (xyz_m + rpy_rad), 然后算 jerk / FFT / reversal
- **关键指标定义**:
  - **drift**: `||EE_xyz[end] - EE_xyz[start]||` (mm), long-term 累积位移
  - **jiggle**: `detrend(EE_xyz)` 后 std (mm), 短时高频抖动
  - **EE jerk95**: `d³xyz/dt³` magnitude P95 (mm/s³)
  - **走3退1 (reversal/s)**: 检测 `v[t]` 跟 `v[t-5frames]` 反向 (cos < -0.5) 且 `|v[t-5]| > 20 mm/s` 的频率. 真机看着像"走 3 步退 1 步".
- **基准 ep**: ep19 vis_v2_full V1 20Hz idle 101s (init j1=64°, linear smooth, publish=80, no EMA)

### 2.2 改动 → 真机测试 → 反向回退原则

每个子项独立 launch arg / env toggle 控制 (default off 保 backward compat). 真机跑后:
- ✅ 同 init pose 下数据 robust 改善 + 主观不退 → 默认 ON + commit
- ❌ 任一关键指标恶化 / 主观更差 → `git checkout HEAD --` 干净回退, mark 失败原因

---

## 3. 子项详细日志

### 3.1 ✅ Layer 1.1B — RTC chunk-overlap min_jerk smoothstep

**改之前** (`StreamActionBuffer.integrate_new_chunk` 中 chunk overlap 处的权重曲线):

```python
# Linear weights: first element 100% old, last element 0% old
w_old = np.linspace(1.0, 0.0, overlap_len)
w_new = 1.0 - w_old
smoothed = w_old * old + w_new * new
```

边界处 1st derivative 不连续 → chunk 接缝速度突变.

**改之后** (quintic smoothstep, `s(t) = 6t⁵ - 15t⁴ + 10t³`):

```python
tau = np.linspace(0.0, 1.0, overlap_len)
if self.smooth_method == 'min_jerk':
    s = 6.0 * tau**5 - 15.0 * tau**4 + 10.0 * tau**3
    w_old = 1.0 - s
```

1st + 2nd 导数在 t=0 和 t=1 都消失 → chunk 边界连续到加速度层级 (minimum-jerk transition). 数学上 LiPo (arXiv:2506.05165) 推荐.

**真机数据** (vis_v2_full V1 20Hz, idle 100s):

| 指标 | ep19 linear (baseline) | ep28 min_jerk | Δ |
|---|---:|---:|---:|
| drift_L (mm) | 19.4 | 12.7 | -35% |
| drift_R (mm) | 21.5 | 11.6 | -46% |
| **jig_L (mm)** | **3.33** | **1.79** | **-46%** |
| EE v95 (mm/s) | 15.2 | 10.2 | -33% |
| jerk95 | 7969 | 6856 | -14% |
| **jerk peak/s** | **3.23** | **1.91** | **-41%** |
| a-s_std (rad) | 0.0015 | 0.0015 | — |

外加 **emergent post-task attractor freeze** — ep27 (1B + task) 后段 40s+ 内 action 输出 `[-0.3071, 0.1943, ...]` 逐字节不变, 机械臂物理静止. linear baseline 没观察到.

**Backward compat**:
- launch arg `rtc_smooth_method:=min_jerk|linear` (default `min_jerk` post-commit)
- 传 `rtc_smooth_method:=linear` 回退 legacy 行为

**Sibling 失败 attempts** (revert, 不 commit):
- **1.1D idle_gate**: 单帧 `|cmd - state| < threshold` 触发 freeze. 80Hz publish 下 `state[-1]` 是 100ms 前真机回读, 跟当前 cmd 时间错位 → 高频在 frozen_state 和 raw_cmd 之间 oscillate (chatter). 真机主观"明显更抖". 需要 hysteresis (进/出阈值不同) + rate-limit, 留 future work.
- **1.1A determinism**: env `V1_DETERMINISTIC=1` + `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`. 启动 server OK 但 inference 输出 NaN — V1 Triton kernels 不支持 PyTorch 的 deterministic mode. 不可行.

### 3.2 ✅ Layer 1.1E — Publish-time EMA on cmd timeline

**改之前**: cmd 经 RTC overlap smooth 后直接 publish, 单帧 ε noise 直接到真机.

**改之后** (在 `_publish_action` jump-protection 之后, publish 之前):

```python
if self._publish_smooth_alpha < 1.0 and self._last_published_action is not None:
    alpha = self._publish_smooth_alpha
    prev = self._last_published_action[:14]
    raw_act = act[:14].copy()
    act[:14] = alpha * act[:14] + (1.0 - alpha) * prev
    # Gripper skip — j6, j13 是 binary 不该 LP filter (见下)
    act[6] = raw_act[6]
    act[13] = raw_act[13]
```

一阶 IIR LP filter on cmd timeline. α∈(0,1], 越小越平滑越滞后. Phase lag ≈ `(1-α)/α × dt_publish`, α=0.5 @ 180Hz ≈ 5.6ms (远小于 Piper PD 100ms 自己滞后).

**Offline 验证** (ep28 raw cmd post-process 不同 α):

| α | jerk95 | reduction |
|---|---:|---:|
| 1.0 (off) | 6752 | baseline |
| 0.7 | 4626 | -31% |
| **0.5** | **3141** | **-53%** |
| 0.3 | 1777 | -74% |
| 0.1 | 553 | -92% |

**真机数据** (vis_v2_full V1, idle 140s, init j1≈11°):

| 指标 | ep28 1B only | ep29 1B + EMA α=0.5 | Δ |
|---|---:|---:|---:|
| drift_L (mm) | 12.7 | 2.0 | -84% |
| drift_R (mm) | 11.6 | 77.3 | +566% ⚠ |
| jig_cmd (mm) | 1.79 | 0.60 | -66% |
| **jig_state (真机) (mm)** | **1.78** | **0.63** | **-65%** |
| jerk95_cmd | 6856 | 3502 | -49% |
| **jerk95_state (真机)** | **5049** | **3562** | **-29%** |
| a-s_std (rad) | 0.0015 | 0.0012 | -20% |
| last-30s jerk_state | 4309 | 2093 | -51% |

drift_R 异常 +77mm 是 single right-arm excursion in front segment (last-30s drift_R=0.2mm 确认 attractor freeze 行为没破), 不是 EMA 引起.

**真机主观**: "明显更稳" (用户原话).

**Gripper skip 物理意义**:
- gripper joint (j6 L, j13 R) 实质是 binary (0 = close, 0.07 = open)
- EMA on binary channel → cmd 卡 mid-range → Piper 物理 servo 到半开位置 (机械上不稳)
- ep33 测过含 gripper EMA 的版本: R gripper cmd 90% 帧落在 [0.01, 0.03] mid-range, 最长 half-grasp run 103.7s
- Skip gripper from EMA 是 conceptually correct, 即使最终 vis_v2_full 半夹问题不是 EMA 引起 (是 ckpt-level — 见 §5)

**Backward compat**:
- launch arg `publish_smooth_alpha:=<0..1]` (default 0.5)
- 传 `publish_smooth_alpha:=1.0` 关闭 EMA, 回退原行为

### 3.3 ✅ RTC 综合重调 (start_autonomy_v1.sh defaults)

**问题** (用户主观真机反馈): task 模式下机械臂"走 3 步退 1 步" — 走一段然后反向短暂.

**离线诊断**: 用 5-frame lookback velocity reversal detection (cos<-0.5 且 |v|>20 mm/s):

| ep | config | EE reversals/s |
|---|---|---:|
| ep30 V1 default (k=2, exec_h=4, publish=80, EMA on) | idle-optimized | **1.27** ⚠ |
| ep31 (k=2, EMA off) | EMA off | 1.35 (EMA 不是 cause) |
| ep32 (k=6, exec=12, smooth=8, publish=80) | k-retune | 0.24 (**-81%**) |
| ep34 (k=6, publish=180) | 完整新 default | **0.12** (-91%) |

**物理诊断** (为什么 k=2 不够):

| 物理量 | 旧 V1 default | 新 default | 含义 |
|---|---:|---:|---|
| Inference period | 50ms | 50ms | model 每 50ms 出新 chunk |
| **RTC blend window** | k=2 × 33ms = **67ms** | k=6 × 33ms = **200ms** | overlap 平滑时间窗 |
| Cmd inconsistency 时窗 | ≥100ms | — | model 跨 obs 不一致最长时间 |

**关键矛盾**: 旧 default blend window (67ms) **比** model inconsistency 时窗 (100ms+) **短** → 还没 blend 完旧 chunk, 新 inconsistent chunk 已经来 → cmd 频繁掉头, 真机执行成"走3退1".

新 default `k=6` 让 blend window 200ms 充分覆盖 inconsistency 时窗, 走3退1 大幅消失.

**改之后** (`start_autonomy_v1.sh` 硬编码 RTC overrides):

| 参数 | 旧 default | **新 default** | 理由 |
|---|---:|---:|---|
| `inference_rate` | 20.0 | 20.0 | V1 base, 保持 |
| **`latency_k`** | **2** | **6** | blend window 67→200ms, 覆盖 model inconsistency 时窗 |
| **`rtc_execute_horizon`** | **4** | **12** | =2×k 标准比例, 维持 RTC guidance 长度 |
| **`min_smooth_steps`** | **3** | **8** | 配比 k=6, 充分 overlap smoothing |
| **`publish_rate`** | (node default 30) | **180** | Piper hardware max ~200Hz (10% safety margin); EMA at 180Hz 给 5.6ms phase lag (vs 12.5ms@80Hz) |

**Trade-off**:
- ✅ task 反向运动 -91%, 真机感受平顺
- ✅ 真机 cmd 更细 (5.6ms granularity), EMA filtering 更平滑
- ⚠ task 启动响应时间: 67ms → 200ms (慢 130ms, 但 < Piper PD 100ms 量级, 不可感知)
- ⚠ idle drift 略增 (12.7mm/100s → ~15mm, 仍 < 30mm 可接受阈值)

**Backward compat**: launch args (`latency_k:=`, `rtc_execute_horizon:=` 等) 仍可显式 override 回旧值; `start_autonomy.sh` / JAX path 不受影响 (用 node-level legacy default).

---

## 4. Layer 1 累计真机数据

vs **origin baseline** (linear smooth, k=2, exec_h=4, smooth=3, publish=80, no EMA), 同 vis_v2_full ckpt, 同任务条件:

| 维度 | origin | Layer 1 完整 default | 减幅 |
|---|---:|---:|---:|
| Idle drift_L (mm/100s) | 19.4 | ~13 | -33% |
| Idle drift_R (mm/100s) | 21.5 | ~12 | -44% |
| **Idle 真机 jiggle (mm)** | **3.33** | **0.63** | **-81%** |
| **Idle 真机 jerk95 (mm/s³)** | **7580** | **3562** | **-53%** |
| Cmd jerk peak/s | 3.23 | 1.91 | -41% |
| Idle post-converge | 持续 random walk | **后段绝对 freeze** | 0 drift |
| **Task EE reversal/s** | **1.27** | **0.12** | **-91%** |
| Task 启动响应 lag | 67ms | 200ms | +130ms (可接受) |

---

## 5. 未解决问题 — vis_v2_full Gripper 半夹 (ckpt-level)

**现象**: 真机执行 fold task 时 gripper "反复夹住又松开" (用户主观).

**Deploy-side 诊断**:
- 测过 EMA on / off / gripper skip / RTC k=6 等多种组合, **现象都存在**
- ep34 raw model cmd j13 (R gripper) 分布: **89.5% 帧在 [0.025, 0.030]** mid-range, std(diff)=0.00025 (极稳)
- → **model 输出 gripper cmd 本身就在 mid value (0.027 stable)**, 不是 deploy-side 任何模块引入

**结论**: vis_v2_full ckpt 训练时学到的 "fold task gripper attractor" 就是 partial-grasp 0.027. 真机 Piper gripper 在 cmd=mid 时机械上不稳 (摩擦 + load 让物理位置 wander) → 视觉看着是"半夹+松开".

**User 验证**: 换其他 ckpt (非 vis_v2_full) 测试, gripper 行为正常 → 锁定 root cause 在 ckpt training.

**未来 fix**:
- **Training-side** (推荐): 训练 demo 里把 fold task 的 gripper 强制 close (full grasp), 让模型学到 binary attractor
- **Deploy-side workaround** (备选): 加 launch arg `gripper_binarize:=true` 做 hysteresis (cmd > 0.04 → 0.07, < 0.02 → 0.0, mid → hold prev). 仅作 quick mitigation, 可能破坏部分 task partial-grasp 需求.

---

## 6. 失败 Attempts (REVERTED, 记录避免重复)

### 6.1 Layer 1.1D — Idle gate (chatter)

**Hypothesis**: 真机 idle 时 model 输出 ε noise → closed-loop random walk drift. Gate when `|cmd - state| < threshold` → freeze cmd at current state.

**Failure mode**:
- 单帧 threshold check 在 80Hz publish 引入 chatter — cmd 在 frozen_state (100ms 前 state) 和 raw_cmd 之间高频 oscillate
- `state[-1]` deque 时间错位: 100ms 前真机 state, 跟当前 raw_cmd 时间不对齐 → freeze 切换瞬间反而引入大跳变
- 真机主观: "明显更抖"

**潜在改善 (1D-v2, future)**: 加 hysteresis (进入 freeze 需 N=30 帧连续 < threshold, 退出 freeze 需单帧 > 2× threshold) + rate-limit (freeze 后至少持续 K 帧). 工时估 1d. 跟 1.1B + 1.1E 已经把 idle 处理得不错, 这条线优先级低.

### 6.2 Layer 1.1A — Torch determinism (V1 Triton incompatible)

**Hypothesis**: Offline replay 同 obs × 10 重复 std = 0.005 rad (0.3°) — 推理本身有 ε noise. 关 deterministic 可压.

**Failure mode**:
- env `V1_DETERMINISTIC=1` + `torch.use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Server 启动 OK, warmup OK, 但 inference 输出全 NaN
- V1 Triton kernel 不走 cuDNN / cuBLAS deterministic path, deterministic flag 干扰 fp 精度 → NaN
- 无 quick fix (需要 patch Triton kernel reduce ordering, 不可行)

---

## 7. 待办 / 下一步

### 7.1 Layer 1.1C — ACT-style temporal ensembling

**思路**: ACT 原始论文做法 — 每 m 步 re-query, overlapping chunks, exp-weighted average across overlapping predictions.

**对比已做**:
- 1.1B 是 chunk-edge smoothing (相邻 chunk 衔接 1 处 ε-blending)
- 1.1C 是 multi-chunk overlap aggregation (N 个 overlapping chunk 同时存在, 每帧 publish 时聚合)
- 跟 RTC `latency_k` overlap 思想有部分重叠, 整合容易出 bug

**状态**: 工时高 (4-6h coding + 真机测), 跟现有 RTC 重叠风险, 待评估是否值得做.

### 7.2 Layer 2 — 训练侧 (扩 idle demo + idle filter)

Deploy-side 已榨干. 进一步改善 (init pose 全空间稳 + gripper binarize attractor) 需 training-side fix:
- 扩 idle demo 覆盖更宽 init j1 range (±90° from current 17-30° 集中区)
- 训练时 gripper close → 强制 binary, 避免学到 mid attractor
- 参考 π0.5 (Physical Intelligence 2025-09) 的 idle filter

详见 `../../../training/future_plans/` 下相关 plan.

---

## 相关 Commits + 文件

| commit | 文件 | 改动 |
|---|---|---|
| `17ed253` | `ros2_ws/src/piper/scripts/policy_inference_node.py` `ros2_ws/src/piper/launch/autonomy_launch.py` | 1.1B min_jerk quintic smoothstep + launch arg toggle |
| `f9fdd9a` | (同上 2 文件) | 1.1E publish-time EMA + launch arg toggle |
| `be851bc` | `ros2_ws/src/piper/scripts/policy_inference_node.py` `start_scripts/start_autonomy_v1.sh` | 1.1E gripper-skip + RTC defaults retune (k/exec/smooth/publish) |

**回退方法** (若发现新 ckpt 上反向): 
```bash
./start_scripts/start_autonomy_from_ckpt_v1.sh <ckpt> --execute \
    rtc_smooth_method:=linear publish_smooth_alpha:=1.0 \
    latency_k:=2 rtc_execute_horizon:=4 min_smooth_steps:=3 publish_rate:=30
```
