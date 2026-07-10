# Phase A — 真实未来标签与图无关评估

> 日期:2026-07-02。
> 目标:通过(1)对**真实观测的下一 milestone** 训练和(2)在 held-out episode 上对真实未来评估,打破 LMWM"预测图表"的循环设置。

## 动机

导出的 pair 数据集已存储 `future_milestone` —— 帧 `t` 之后的真实观测下一独特 milestone。但之前所有训练器都忽略了它,将贪心/max-product 头训练在 `greedy_next[current_m]` 上,这是一个确定性**图表查找**,以离散化当前状态为索引。

两个问题:

1. 图查找标签与真实观测未来在**仅 24.2%** 的情况下一致(75.8% 的 pair 不一致)。目标丢弃了真实轨迹。
2. 给定当前状态,真实下一 milestone 的熵 ≈ **2.56 nats(~13 有效分支)** —— 高度多模态。单个 argmax 不可能在大多数时候正确,所以"vs graph top1=0.96"衡量的是表的复现,而非对现实的预测。

## 改动

- `src/lmwm/data.py`:`load_graph_policy_data(..., label_source=...)`,支持 `"graph_lookup"`(默认,向后兼容)和 `"real_future"`(贪心/max-product/prototype 头以 `future_milestone` 为目标)。转移头始终以经验 milestone 级分布 `transition_probs[current_m]` 为目标,使其 real-future NLL 在不同标签源之间可比。
- `scripts/train_unified_lmwm.py`:从配置读取 `label_source`,记录到 run meta。
- `configs/training/kai0base_dinov3h_stage3_realfuture.yaml`:real-future 运行。
- `scripts/eval_real_future.py`:图无关评估器。在 held-out episode 划分上对 `future_milestone` 评分(top-1/3/5,NLL)并附非神经基线。适用于任何 `UnifiedLMWM` checkpoint。

## 结果(held-out:40,572 pairs / 611 episodes / 37 milestones)

| 模型/基线 | vs graph(循环) top1 | vs 真实 top1 | top3 | top5 | NLL |
|---|---|---|---|---|---|
| 均匀分布 | — | 0.024 | 0.057 | 0.109 | 3.61 |
| 图 argmax(greedy) | — | 0.240 | — | — | — |
| 经验分布 `P(next\|cur milestone)` | — | 0.240 | 0.483 | 0.633 | 2.57 |
| 图训练·神经贪心头 | **0.936** | 0.233 | 0.366 | 0.474 | 16.04 |
| 图训练·神经转移头 | — | 0.207 | 0.434 | 0.582 | 2.68 |
| **真实未来训练·神经贪心头** | 0.275 | **0.383** | **0.686** | **0.822** | **1.98** |
| 真实未来训练·神经转移头 | — | 0.201 | 0.438 | 0.594 | 2.67 |

产物:

- 图训练评估:`outputs/real_future_eval/graph_trained/summary.json`
- 真实未来训练 checkpoint:
  `checkpoints/stage3_realfuture/20260702_045756+kai0base_dinov3h_stage3_realfuture/best.pt`
- 真实未来训练评估:`outputs/real_future_eval/realfuture_trained/summary.json`

## 发现

1. **0.94 是循环的。** 图训练贪心头对图表得 0.936,但对**现实仅 0.233** —— 勉强超过非神经基线(图 argmax 0.240,经验 0.240),其 NLL **16.0** 表明一个病态过自信的点预测器。
2. **真实未来训练确实有帮助。** 对现实,贪心头从 0.233 → **0.383** top1,0.474 → **0.822** top5,NLL 16.0 → **1.98**。
3. **帧特征携带当前 milestone id 之外的动态信息。** 真实未来贪心头的 NLL(1.98)现在**反超经验 milestone 级分布基线**(2.57)。这是 LMWM 第一次证明它学到了"当前 milestone 查表"之外的东西 —— 即从"表验证"向真实世界模型迈进。

## 诚实说明

- 真实未来 top1 0.38 绝对值不算"好" —— 但问题有 ~13 分支,所以 top-3/top-5 和 NLL 是有意义的指标,且模型在这些指标上明显有信息。
- 仍为单 `kai0_base` DINOv3-H,同任务 held-out,隐变量 prototype 输出(非解码图像或动作)。

## 下一步(Phase B)

使**分布**帧条件化:从真实未来训练分布性下一 milestone 目标(soft label/直接对观测做 CE),评估 NLL + 校准,并仅将图用作先验 —— 不作为 ground truth。在 real-future 判据下重新推导回退逻辑(旧的 hybrid 0.997 在新判据下不成立)。
