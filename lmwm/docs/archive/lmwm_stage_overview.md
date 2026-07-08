# LMWM 阶段总览

> LMWM 构建过程的单页导览:每个阶段做了什么、产出什么。如需详尽的逐步记录,参见
> [automatic_iteration_log_20260701.md](automatic_iteration_log_20260701.md);
> 设计原理参见
> [recurrence_state_world_model_plan.md](recurrence_state_world_model_plan.md)。
>
> 最后更新:2026-07-02。

## LMWM 是什么

LMWM **不是**像素预测,也**不是**完整 VLA 策略。它是 CRAVE milestone 状态上的紧凑循环世界模型,消费当前 DINOv3-H 帧特征,为 VLA planning-prior 用途输出:

- 下一 milestone(贪心);
- 最高完成度下一 milestone(最大积);
- milestone 上的转移分布;
- 每个头的隐变量 prototype subgoal;
- 用于下游门控的置信度/熵信号。

设计借鉴 LaWAM 在冻结编码器空间(DINOv3)中预测隐变量子目标的思路,但隐变量是 CRAVE 的任务感知 milestone 状态,而非通用视觉未来 token。

**数据基座**:`kai0_base` DINOv3-H —— 334,875 有效帧 / 3,055 episode / 37 个 milestone prototype。

## 四个建模阶段

### Stage-1 — LaWM 形状转移模型(管线验证)

- **做了什么**:`(r_t, r_future) -> 逆向码 u_t -> r_hat_future -> milestone 分类器`。先用 one-hot smoke 运行验证循环,再换真 DINOv3-H 1280D prototype。
- **结果**:val top1 = **1.0**,MSE ≈ 0.002–0.007。
- **诚实解读**:高分是预期的,因为输入和目标都取自 37 个 prototype 的有限表。这验证了数据处置和架构,**不是**世界模型本身。

### Recurrence Latent State Probability Graph(循环隐状态概率图,显式图)

- **做了什么**:每帧→最近 milestone 中心→episode 内排序→压缩连续相同→统计转移→行归一化+平滑。Greedy = `argmax P(next|cur)`;max-product = 向最高 progress 终点 milestone 的有限 horizon DP。
- **结果**:1,232 条非零边,平均压缩 episode 长度 54,终点 milestone #36(progress 0.954)。
- **性质**:一阶 Markov + 有限 horizon → planning prior,不是 ground truth。

### Stage-2 — 图监督策略(蒸馏)

- **做了什么**:将图蒸馏进神经模型:`帧特征 -> 转移行 + 贪心下一 + 最大积下一`。
- **结果**(最佳步 1100):val KL = 0.156,greedy top1 = **0.935**,max-product top1 = **0.935**。

### Stage-3 — Unified LMWM(统一 LMWM,当前最佳产物)

- **做了什么**:将图预测和隐变量 subgoal 预测合并到一个模型(共享 trunk + 5 头)。
- **结果**(最佳步 1100):greedy 0.936 / max-product 0.935 / prototype cosine **0.99**。
- **产物**:`checkpoints/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified/best.pt`。

## 鲁棒性 & 部署迭代(围绕 Stage-3)

| # | 迭代 | 关键结果 |
|---|---|---|
| 5 | 推理 wrapper | 4096 样本:greedy top1 0.965,proto cosine 0.99 |
| 6 | 逐 milestone 误差分析 | 定位弱 milestone:**#34 / #30 / #5 / #2**(边界/歧义阶段) |
| 7 | Hybrid 图回退 | hybrid top1 **0.997**(⚠ 回退用的图先验也是标签定义来源 —— 非独立证明) |
| 8 | Gate 阈值 sweep | 默认 `greedy=0.90 / max=0.92`,top1 ≈ 0.998,fallback ≈ 0.196 |
| 9 | 在线运行时 API | `UnifiedLMWMPredictor`;离线/运行时一致性通过(tol 5e-6) |
| 10 | 置信度校准 | ECE 0.007,AURC 0.002 → 置信度可用于排序可靠预测 |
| 11 | 验证驱动弱集选择 | 自动选 10 个弱 milestone;top1 0.998 但 fallback 升至 ~33% |
| 12 | 学习型 uncertainty fallback | 逻辑回归误差模型替代硬编码弱集;fallback ≈ 16% |
| 13 | 调优学习型策略 | 修好 #34(0.906→1.0);回退率 ≈ 21% |
| 14 | 全量 20 万 pair 评估 | 最终策略对比(见下) |

### 全量 20 万 pair 策略对比(最终结论)

| 策略 | 平均 top1 | 平均回退率 | 定位 |
|---|---|---|---|
| validation_selected_safe | **0.9976** | 0.331 | 最准,重度依赖图先验 |
| **recommended(默认)** | 0.9965 | **0.200** | 平衡默认 |
| learned_tuned | 0.9955 | 0.220 | 最强学习型回退候选;全量仍略逊默认 |

**结论**:`recommended` 保持默认;`learned_tuned` 修好了 #34 但全量 mean top1 仍未反超;`validation_selected_safe` 是高准确率、图先验权重模式。

论文表格快照:`docs/icra_experiments/README.md` 由 `scripts/summarize_icra_experiments.py` 生成,将循环图、训练阶段指标和运行时策略比较汇总为 Markdown/CSV/JSON 表格,供 ICRA 风格实验规划使用。

### 第二个划分管线检查:kai0bd

`kai0bd_feature_stage1` 现在在较小的 base+dagger 缓存特征划分上运行了相同的 LMWM 管线:501 episode(251 base-like / 250 dagger-like),45k 帧,796D 特征状态,64 milestone。它已导出 fixed-horizon 和 next-unique pair、pair 级循环图、Stage-1/2/3 checkpoint 和运行时摘要。这仍是管线验证,因为图标签是确定性表目标,但它证明了代码路径不限于原始的 kai0_base DINOv3-H 缓存。

## 诚实局限(必须知道)

1. **表式标签**:训练目标 `greedy[current_m]` 和 `max_product[current_m]` 是以当前 milestone id 为索引的确定性图表查找,而当前 milestone 本身是最近 prototype。所以 top1 ≈ 0.96 主要反映帧→簇可分性,而非真正的动态预测。
2. **循环回退**:hybrid 回退使用的图先验也是图监督标签的定义来源,因此 hybrid 0.997 **不是**神经模型改进的独立证明。
3. 仅 `kai0_base` DINOv3-H(无 `kai0_dagger` 缓存);主 1280D 产物的评估仅为同任务 held-out。较小的 `kai0bd` base+dagger 缓存特征划分现在验证了相同管线,但尚不是独立的跨任务结果。模型预测隐变量 prototype subgoal,非解码图像或机器人动作。

## Phase A(已完成 2026-07-02) — 真实未来标签 + 图无关评估

脱离循环"预测图表"设置。对真实观测的下一 milestone(`future_milestone`)训练,在 held-out episode 上对现实验证。参见 [phase_a_real_future_20260702.md](phase_a_real_future_20260702.md)。

关键诚实数字(held-out,对真实未来):

| | vs graph(循环) | vs 真实 top1 | top5 | NLL |
|---|---|---|---|---|
| 图训练贪心头 | 0.936 | 0.233 | 0.474 | 16.0 |
| 经验分布基线 | — | 0.240 | 0.633 | 2.57 |
| **真实未来训练贪心头** | 0.275 | **0.383** | **0.822** | **1.98** |

核心:旧 0.94 对现实崩塌到 0.23;真实未来训练将 top1 提升到 0.38/top5 到 0.82 且 NLL 反超非神经经验基线 —— LMWM 第一次证明帧特征携带 milestone-id 查表之外的动态。

## Phase B(已完成 2026-07-02) — 校准 + 图作先验融合

在 held-out episode 上对真实未来回答了两个诚实问题,无须重训。参见 [phase_b_calibration_prior_20260702.md](phase_b_calibration_prior_20260702.md)。

- **校准**:原始 ECE 0.10 → **0.005**(单温度 `T=1.30`);帧条件分布在单参数修复后可信。
- **图作软先验**(对数线性池化,非硬回退):最优 `lam≈0.3` → top1 0.383→**0.417**,top5 0.822→**0.856**,NLL 1.978→**1.798**。图作为 ~30% 先验确实有帮助 —— 旧循环 hybrid 回退的原则性、诚实评估的替代品。

推荐 real-future 配方:`p = softmax(logits/1.30)^0.7 · P(next|cur_milestone)^0.3`;汇报 top-k + NLL。

## Phase C(已完成 2026-07-02) — 帧历史条件:负结果

用短视频历史窗(H=4 s=2 和 H=6 s=4)增强 pair 并重训真实未来模型。结果:**历史没有帮助** —— top1 0.383→0.377→0.367,NLL 持平/更差,更大输入过拟合更快。单帧已到天花板;~13 分支熵是固有任务歧义。剩余杠杆(动作条件、更少抖动的标签)在 VLA/数据侧。参见 [phase_c_history_20260702.md](phase_c_history_20260702.md)。

## Phase D(已完成 2026-07-02) — VLA 集成接口:LMWM 已 VLA-ready

将真实未来配方打包为 `lmwm.vla_interface.VLALMWMPredictor`:`p = softmax(logits/1.30)^0.7 · P(next|cur_milestone)^0.3`,曝露经校准的下一 milestone 分布、top-k 候选、隐变量 prototype subgoal 和 confidence/entropy。Held-out 对真实未来:top1 ≈ 0.42,top5 ≈ 0.86,NLL ≈ 1.80,ECE ≈ 0.005。参见 [vla_integration_20260702.md](vla_integration_20260702.md)。

**状态:已准备好 VLA 集成。** 进一步独立 LMWM 调优有边际收益递减;下一步增益需要 VLA 侧的动作条件。

## 代码地图

- 模型(唯一真源):`src/lmwm/models.py`(`UnifiedLMWM`,`GraphSupervisedLMWM`,`LaWMShapedLMWM`,`MLP`)。
- 数据/配置/划分:`src/lmwm/data.py`。
- 训练脚手架:`src/lmwm/training.py`。
- 在线运行时 API:`src/lmwm/runtime.py`(`UnifiedLMWMPredictor`)。
- VLA 接口:`src/lmwm/vla_interface.py`(`VLALMWMPredictor`)。
- 训练器(薄编排):`scripts/train_state_world_model.py`(Stage-1),`scripts/train_graph_policy_model.py`(Stage-2),`scripts/train_unified_lmwm.py`(Stage-3)。
- 图构建:`scripts/build_recurrence_graph.py`。
- 论文表格汇总:`scripts/summarize_icra_experiments.py`。
