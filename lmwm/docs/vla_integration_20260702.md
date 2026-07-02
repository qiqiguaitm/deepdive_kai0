# LMWM ↔ VLA 集成接口(Phase D)

> 日期:2026-07-02。将真实未来验证的 LMWM(Phase A/B/C)打包成一个在线预测器,供 VLA planning prior 使用。

## 为什么是现在

Phase C 证明独立的单帧世界模型已达天花板:帧历史不改善真实未来预测,剩余杠杆(动作条件、更少抖动的标签)在 VLA/数据侧。因此正确的下一步是将当前经诚实验证的 LMWM 暴露给策略 —— 而非继续调优世界模型本身。

## 配方(对真实观测未来验证)

```
p_cal   = softmax(greedy_logits / T),         T = 1.30
p_prior = transition_probs[current_milestone]
p_fused ∝ p_cal^(1-λ) · p_prior^λ,            λ = 0.30
```

`T=1.30` 和 `λ=0.30` 是 Phase B 的 held-out 最优参数。在 held-out episode 上,对真实观测下一 milestone:top1 ≈ **0.42**,top3 ≈ **0.73**,top5 ≈ **0.86**,NLL ≈ **1.80**,校准 ECE ≈ **0.005**。这些是诚实数字 —— 对现实打分,不是循环图查找指标。

## API

```python
from lmwm.vla_interface import VLALMWMPredictor

predictor = VLALMWMPredictor.from_yaml(
    "lmwm/configs/inference/kai0base_dinov3h_vla_realfuture.yaml"
)
out = predictor.predict(current_features)          # (B, feature_dim) 或 (feature_dim,)
# 或 predictor.predict_one(current_feature)
```

`current_milestones` 可选的;若省略则按最近 DINOv3-H prototype 分配(使用最后 `frame_dim` 维,因此历史增强输入也可用)。

### 返回字段(每样本)

| 字段 | 形状 | 含义 |
|---|---|---|
| `next_milestone` | () | 融合分布 argmax(top-1 下一 milestone) |
| `next_milestone_probs` | (M,) | 带软图先验的帧条件分布 —— **主要输出** |
| `topk_milestones` / `topk_probs` | (k,) | 排序的下一 milestone 候选 + 概率 |
| `subgoal_latent` | (latent_dim,) | 贪心下一 milestone 的 L2 归一 prototype subgoal |
| `confidence` | () | 最大融合概率(已校准;可用于门控) |
| `entropy` | () | 融合分布的熵(任务分支不确定性) |
| `calibrated_probs` | (M,) | 仅神经温度缩放分布(无先验) |
| `current_milestone` | () | 分配的/观测的当前阶段 |

## 建议的 VLA 使用方式

- **Subgoal 条件**:将 `subgoal_latent`(DINOv3-H prototype)作为隐变量视觉目标输入,LaWAM 风格。
- **候选集而非硬目标**:优先使用 `topk_milestones`/概率而非单一下一 milestone —— 问题有 ~13 分支,top-1 常在错而 top-5 覆盖 86%。
- **按 `confidence`/`entropy` 门控**:当 LMWM 不确定(高熵/低置信)时将决策权交给策略自身先验。
- **不要**将 top-1 视为 ground truth,也不要用旧 graph-hybrid `runtime.UnifiedLMWMPredictor` 回退数字(0.997) —— 那些是对定义其标签的图验证的。

## 配置

`configs/inference/kai0base_dinov3h_vla_realfuture.yaml`:
`checkpoint`(单帧 real-future best),`graph_npz`,`temperature: 1.30`,`prior_weight: 0.30`,`topk: 5`。

## 带入 VLA 的诚实边界

- 单 `kai0_base` DINOv3-H,同任务 held-out;`T`/`λ` 按数据集重新拟合。
- 输出是 milestone 分布 + 隐变量 prototype subgoal —— 不是解码图像或机器人动作。
- 绝对 top-1 ≈ 0.42 反映固有 ~13 分支歧义;依靠 top-k + 置信度,不靠点预测。

## 状态

**LMWM 已准备好 VLA 集成。** 该接口通过一个在线预测器,曝露一个经校准、真实未来验证的下一 milestone 分布、排序候选、隐变量子目标和不确定信号。
