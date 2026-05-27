# Fixed-Noise Inference Fix — vis_v2_full 真机 oscillation 诊断与修复

> **目的**: 为 vis_v2_full 真机部署 "走几步退几步 + 夹爪犹豫" 问题提供诊断结论 + sim01 端代码修复方案.
> **日期**: 2026-05-27
> **诊断 ckpt**:
> - `pi05_flatten_fold_vis_v2_full/49999` (cnbj cluster) — 表现差
> - `task_a_pure200_base_pi05_step49999` (/vePFS) — 表现好
> - `pi05_flatten_fold_vis_v2_full_tac/49999` (cnbj) — 已确认无效 (见 [analysis](../../training/analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md))

---

## 1. 问题描述

vis_v2_full 真机执行表现:
- ✗ 机械臂走几步退几步 (oscillation)
- ✗ 夹爪无法长期闭合
- ✗ 夹爪来回犹豫抓不到衣服

同 RTC 参数 + pure_200 ckpt: ✅ 正常.

---

## 2. 诊断数据 (apples-to-apples, vis_v2_merged_val 30 ep)

> ⚠️ Val 不一致提醒: cnbj 上的 `vis_v2_merged_val` ep 865-874, /vePFS 上的 ep 0-9. 都是 vis_v2 数据集的 hold-out 子集 (各 30 ep), 但物理样本不同. 数据已可看出 trend 但严格对比需要在同 val 重测.

### 2.1 P0 Gripper 分布 (双臂 dim 6 / dim 13)

三模型都呈双峰: 大量 `~0` 闭合 + 一簇 `~0.06` 半开. 单看 gripper 分布无法区分模型质量.

### 2.2 P1 Chunk-to-chunk 连续性 (overlap-region 平均 |diff|)

| Ckpt | Left arm random | Right arm random | Verdict | Left arm fixed-noise | Noise 贡献 ΔL |
|---|---:|---:|---|---:|---:|
| **vis_v2_full** | **0.0265** | **0.0234** | 🟢 HEALTHY | **0.0206** | **+0.0059** |
| pure_200 | 0.0390 | 0.0367 | 🟡 MARGINAL | 0.0389 | +0.0001 |
| TAC v7 | 0.0666 | 0.0554 | 🔴 BAD | (没测) | — |

**关键观察**:
- **vis_v2_full 的 chunk 连续性 actually 比 pure_200 还好** (0.026 vs 0.039)
- pure_200 几乎完全 deterministic (noise 贡献 0.0001), vis_v2_full 受 noise 影响 (贡献 0.006, 22%)
- TAC v7 chunk 连续性最差 — TAC 训练并未改善, 实际更糟 (另有[bug 分析](../../../docs/training/analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md))

### 2.3 P2 多采样方差 (同 obs N=5 个不同 noise sample 的 std)

| Ckpt | mean across frames | **max across (frame, dim)** | Verdict |
|---|---:|---:|---|
| **vis_v2_full** | 0.0234 | **0.6688** | 🟡 MARGINAL (rare-event multi-modal) |
| pure_200 | 0.0117 | **0.1903** | 🟢 HEALTHY |
| TAC v7 | 0.0231 | 0.6525 | 🟡 MARGINAL |

**关键观察**:
- vis_v2_full 在**绝大部分帧**都稳定 (mean 0.023 不算高), 但**某些 rare 关键时刻**多 mode (max 0.67)
- pure_200 全程 deterministic (max 仅 0.19)
- 0.67 std 出现的具体 (dim, frame) 还需要进一步 breakdown — **是否对应 gripper trigger / 翻转时刻是核心待验证假说**

---

## 3. 根因假说 (优先级)

| # | 假说 | 与症状对应 | 与数据吻合 |
|---|---|---|---|
| **H1** | **Rare-event multi-modal collapse** at gripper trigger / state transition. 大部分时间 vis_v2_full deterministic, 但在关键时刻 flow matching ODE 在多个 attractor 间 noise-driven 选择 → 不同 chunk 跳不同 mode → 真机看到 oscillation | "走几步退几步" + "夹爪犹豫" | ✅ P2 max 0.67 显示存在 high-variance 关键帧 |
| H2 | OOD scene drift (真机场景 ≠ val 场景, 抗 OOD 弱) | "夹爪抓不到衣服" | △ 需真机场景对照实验验证 |
| H3 | Chunk discontinuity (我们之前的错误假说) | — | ❌ 已被 P1=0.026 (HEALTHY) 推翻 |

---

## 4. 修复方案 — Fixed-Noise Inference (G0)

### 4.1 原理

`pi0.sample_actions` 与 `pi0_rtc.sample_actions` 都接受可选 `noise` 参数 (shape `(b, action_horizon, action_dim_padded)`). 不传则每次内部随机生成. **如果改成启动时生成一次, 整个 session 复用**, 则:

- P2 (多采样方差) 实质性消失 — 同 obs 始终输出同 action chunk
- 跨 chunk 连续性受**输入变化**驱动 (state@k → state@k+5), 不再叠加 noise variance
- **vis_v2_full fixed-noise P1 = 0.021 (HEALTHY, 比 pure_200 0.039 还低)**

如果 H1 假说成立, fixed-noise 会消除关键时刻的 mode 漂移 → 真机 oscillation 应消失.

### 4.2 sim01 端代码改动

**目标文件**: `/data1/tim/workspace/deepdive_kai0/kai0/scripts/serve_policy_v1.py`

**改动位置**: ShmServer / WebsocketPolicyServer 初始化处 (~ line 786-808).

**改动 patch (示意)**:

```python
# 在 main() 函数体内, policy 创建之后, 服务启动之前添加:

import numpy as np
import os

# G0 fixed-noise inference: 消除 flow matching noise variance.
# Pi05: action_horizon=50, action_dim_padded=32.
# Seed 可通过 env 配置, 部署时切换 seed 不需要改代码.
_seed = int(os.environ.get("OPENPI_FIXED_NOISE_SEED", "0"))
_FIXED_NOISE = np.random.RandomState(_seed).randn(50, 32).astype(np.float32)
logger.info(f"G0 fixed-noise inference enabled (seed={_seed}, shape={_FIXED_NOISE.shape})")

def _infer_fixed_noise(obs):
    return policy.infer(obs, noise=_FIXED_NOISE)

# 然后把:
#   infer_callback=policy.infer,
# 改成:
#   infer_callback=_infer_fixed_noise,
```

如果同时启用 WebsocketPolicyServer, 需要让 server 拿到 wrapped policy. 最简单的做法: **新建一个 wrapper class**:

```python
class _FixedNoisePolicyWrapper:
    """Wraps policy.infer to inject a fixed noise tensor."""
    def __init__(self, inner_policy, noise):
        self._inner = inner_policy
        self._noise = noise
    @property
    def metadata(self):
        return self._inner.metadata
    def infer(self, obs):
        return self._inner.infer(obs, noise=self._noise)

# 然后用 _FixedNoisePolicyWrapper(policy, _FIXED_NOISE) 替换 policy 传入 WebsocketPolicyServer 即可.
```

### 4.3 RTC 兼容性

✅ **完全兼容**. `policy.infer` 已经把 `noise` 参数 forward 到底层 `sample_actions`, `pi0_rtc.sample_actions:240` 也接受 noise 参数. RTC 的 `prev_action_chunk` 与 noise 互不冲突, 同时启用是预期用法.

---

## 5. 验证步骤 (sim01 端)

| Step | 操作 | 期望 |
|---|---|---|
| 1 | apply patch + 重启 serve_policy_v1 (`OPENPI_FIXED_NOISE_SEED=0`) | logger 输出 "G0 fixed-noise inference enabled (seed=0, shape=(50, 32))" |
| 2 | 真机执行原叠衣任务 | 如果 oscillation 消失 → ✅ H1 假说确认, fix 完成 |
| 3 | 如 seed=0 仍 oscillation, 改 `OPENPI_FIXED_NOISE_SEED=1` 重启, 重测 | 不同 noise 起点会到不同 attractor; 找出一个走"好的 mode"的 seed |
| 4 | 试 seed=2,3,4,5 各一次 (每次重启 server) | 找到 best seed |
| 5 | 若全部 seed 都 oscillate, 说明 fix 不是 noise 主因 | 回到 H2/H3 + 训练侧修复路线 |

---

## 6. 回退方案

如果 G0 真机不 work, 不要 revert 这个 patch — env var 未设置时仍按原行为运行 (但当前 patch 总是启用; 可加条件: `if _seed != -1: wrap else use raw policy.infer`).

更保险的 patch:

```python
_seed_env = os.environ.get("OPENPI_FIXED_NOISE_SEED")
if _seed_env is not None and _seed_env != "":
    _seed = int(_seed_env)
    _FIXED_NOISE = np.random.RandomState(_seed).randn(50, 32).astype(np.float32)
    logger.info(f"G0 fixed-noise inference enabled (seed={_seed})")
    infer_fn = lambda obs: policy.infer(obs, noise=_FIXED_NOISE)
else:
    infer_fn = policy.infer

shm_server = ShmServer(
    infer_callback=infer_fn,
    ...
)
```

部署时只需 `export OPENPI_FIXED_NOISE_SEED=0` 启用; `unset OPENPI_FIXED_NOISE_SEED` 关闭, 回到原行为, 不影响 pure_200 等已正常工作的部署.

---

## 7. 上游依据

- 诊断脚本: `kai0` 的 `/tmp/diagnose_vis_v2_full.py` (P0/P1/P2) + `/tmp/diagnose_fixed_noise.py` (P1 fixed vs random)
- 原始 diff 来源: `policy.infer` ([src/openpi/policies/policy.py:68](../../../src/openpi/policies/policy.py)) 与 `pi0_rtc.sample_actions` ([src/openpi/models/pi0_rtc.py:234](../../../src/openpi/models/pi0_rtc.py)) 都支持 noise 参数
- 数据更正: 早期分析中 vis_v2_full P1=0.063 BAD 是错误数据 (跨 session 引用混淆); fresh measurement 给出 0.026 HEALTHY. 见 [analysis 中的 §数据更正](../../training/analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md)
- TAC bug: TAC v7 训练失效的 convention bug 在 `src/openpi/models/pi0.py:335` — `time_per_token = jnp.where(prefix_mask_tac, 1.0, time[..., None])` 应为 `0.0`. 但这与 vis_v2_full 真机问题正交.
