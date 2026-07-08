# Phase C — 帧历史有帮助吗?(负结果)

> 日期:2026-07-02。检验当前帧之外的时间上下文是否改善真实未来下一 milestone 预测。
> 所有数字在 held-out episode 划分(40,572 pairs / 611 episodes / 37 milestones)上,对**真实观测**的未来评分。
> 与 Phase A/B 同划分和真实未来目标(历史数据集是同 pair 集的行对齐增强)。

## 设置

`scripts/export_history_pairs.py` 将每对的单个 `current` 帧替换为原始缓存中 `[t-(H-1)·S, ..., t-S, t]` DINOv3-H 帧的拼接;所有其他字段(milestone、真实未来、episode_id)逐字复制。以 `label_source: real_future` 训练了两个跨度(其他同 Phase A 配置):

- H=4,stride=2(跨度约 6 帧)
- H=6,stride=4(跨度约 20 帧)

## 结果(神经贪心头 vs 真实未来)

| 模型 | 输入维度 | top1 | top3 | top5 | NLL | 最优融合 top1/NLL |
|---|---|---|---|---|---|---|
| 单帧 | 1280 | **0.383** | 0.686 | 0.822 | **1.978** | 0.417/1.798 |
| 历史 H4 s2 | 5120 | 0.377 | 0.686 | 0.826 | 1.979 | 0.421/1.782 |
| 历史 H6 s4 | 7680 | 0.367 | 0.671 | 0.815 | 2.024 | 0.417/1.809 |

## 发现

1. **历史没有帮助。** 所有差异在噪声范围内(±0.006 top1,NLL 在 H4 时相同)。更长跨度(H6 s4)在神经头上略差。
2. **更大输入过拟合更快。** 两个历史模型都早峰(val top1 在 ~600 步)后下降,而训练损失持续下降 —— 在相同 200k pair 上 4-6 倍更大的输入过拟合。
3. **单帧已到天花板。** 当前 DINOv3-H 帧已捕获 milestone 状态;近期原始帧不增加任何信息。下一 milestone 的 ~13 分支熵是**固有任务歧义**,不是缺失上下文的产物。

旁注:DINOv3-H milestone 分配大约每 ~2 帧变化一次(平均 54 压缩阶段 / ~110 帧/episode),因此"下一独特 milestone"本质上是抖动的 —— 这正是 CRAVE 需要时序平滑(Viterbi/sym-adaptive-vote)来进行自身读出的原因。在如此抖动的逐帧标签上的历史不增加信号。

## 含义

进一步单帧/短历史 LMWM 建模有边际收益递减。剩余的杠杆是(a)动作条件和(b)更好/更少抖动的标签 —— 两者都在 VLA/数据侧。**自然的下一步是将当前校准好的 LMWM 集成为 VLA 的 planning prior**,而非继续调优独立世界模型。

## 产物

- 数据集:`data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_hist{4s2,6s4}.npz`
- Checkpoint:`checkpoints/stage3_realfuture_hist/*`
- 评估:`outputs/real_future_eval/realfuture_hist{4s2,6s4}/`,
  `outputs/phase_b_eval/realfuture_hist{4s2,6s4}/`
