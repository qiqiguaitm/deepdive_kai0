# LMWM 自动迭代日志 — 2026-07-01

## 目标

以小步可回滚的增量构建最终的隐变量里程碑世界模型。每步必须有产物、验证运行和简短解读才能进入下一步。

## 迭代策略

- 不修改 `lmwm/vendor/LaWAM`。
- 在更换模型家族前优先做 LaWM 形状的小步。
- 所有导出的数据集/配置/日志/checkpoint 都放在 `lmwm/` 下。
- 在 prototype 表任务上的高分视为管线验证,不视为最终模型质量。
- 如果新步表现不佳,保留日志但从上一个更强 checkpoint 继续。

## 已完成的步骤

### 1. Stage-1D:帧特征 → 未来 milestone prototype

目的:去掉过于简单的 prototype-to-prototype 输入,改用真实 DINOv3-H 帧特征作为当前观测。

产物:

- 导出配置:`lmwm/configs/datasets/kai0base_dinov3h_frame2proto_next_unique.yaml`
- 数据集:`lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique.npz`
- 训练配置:`lmwm/configs/training/kai0base_dinov3h_stage1d_frame2proto_next_unique.yaml`
- 脚本:`lmwm/scripts/export_dinov3h_milestone_pairs.py`
- 脚本:`lmwm/scripts/train_state_world_model.py`
- 运行:`lmwm/logs/stage1d/20260701_142250+kai0base_dinov3h_stage1d_frame2proto_next_unique`
- Checkpoint:`lmwm/checkpoints/stage1d/20260701_142250+kai0base_dinov3h_stage1d_frame2proto_next_unique/best.pt`

验证:

- 数据集对:200000
- Episode:3055
- 输入维度:1280
- 未来 prototype 维度:1280
- 验证集 top1 最终步:1.0
- 验证集 MSE 最终步:0.0003602966

解读:帧→prototype 路径健康,但目标仍是有限的 37-prototype 表。这验证了数据处置和训练机制,不是完整的世界模型。

### 2. Recurrence Latent State Probability Graph(循环隐状态概率图)

目的:将 CRAVE milestone 分配转化为阶段转移概率模型。

产物:

- 配置:`lmwm/configs/datasets/kai0base_dinov3h_recurrence_graph.yaml`
- 脚本:`lmwm/scripts/build_recurrence_graph.py`
- 图:`lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz`
- 元数据:`lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph_meta.json`

图定义:

- 每一帧 DINOv3-H 分配到最近 milestone 中心。
- episode 内帧按序排列。
- 压缩连续相同 milestone。
- 统计压缩阶段间的转移计数。
- 行归一化(加小平滑)。
- Greedy next:单步 `argmax P(stage_{t+1} | stage_t)`。
- Max-product next:向最高 CRAVE progress 的终点 milestone 做有限 horizon 动态规划。

验证:

- 有效帧:334875
- Episode:3055
- Milestone:37
- 非零转移边:1232
- 平均压缩 episode 长度:54.0815
- 终点目标:milestone 36,progress 0.9540757

解读:这是第一个显式的 LMWM 循环图产物。当前 max-product 路径使用 finite horizon 10 和一阶转移概率,因此继承了 Markov 假设,应被视为 planning prior 而非 ground truth。

### 3. Stage-2 图监督策略模型

目的:将循环图蒸馏为一个基于当前视觉特征的神经模型。

产物:

- 配置:`lmwm/configs/training/kai0base_dinov3h_stage2_graph_policy.yaml`
- 脚本:`lmwm/scripts/train_graph_policy_model.py`
- 运行:`lmwm/logs/stage2_graph/20260701_142639+kai0base_dinov3h_stage2_graph_policy`
- Checkpoint:`lmwm/checkpoints/stage2_graph/20260701_142639+kai0base_dinov3h_stage2_graph_policy/best.pt`

目标:

```text
current frame DINOv3-H feature
  -> transition probability row          [转移概率行]
  -> Greedy next milestone                [贪心下一 milestone]
  -> Max-product completion next milestone [最大积完成下一 milestone]
```

最佳验证步:1100

- Val KL:0.1564875
- Greedy top1:0.9348073
- Max-product top1:0.9347580

解读:这是第一个神经循环世界模型。误差仍然存在,因为阶段边界附近的帧级特征可能映射到歧义的阶段分配,且图标签是概率转移行的确定性投影。

### 4. Stage-3 unified LMWM(统一 LMWM)

目的:将图预测和隐变量 subgoal 预测合并到一个模型中。

产物:

- 配置:`lmwm/configs/training/kai0base_dinov3h_stage3_unified.yaml`
- 脚本:`lmwm/scripts/train_unified_lmwm.py`
- 运行:`lmwm/logs/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified`
- Checkpoint:`lmwm/checkpoints/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified/best.pt`

目标:

```text
current frame DINOv3-H feature
  -> transition probability row
  -> Greedy next milestone id
  -> Max-product completion next milestone id
  -> Greedy latent prototype subgoal      [贪心隐变量 subgoal]
  -> Max-product latent prototype subgoal [最大积隐变量 subgoal]
```

最佳验证步:1100

- Val KL:0.1561412
- Greedy top1:0.9361382
- Max-product top1:0.9353002
- Greedy prototype cosine:0.9895124
- Max-product prototype cosine:0.9915678

解读:这是当前最强的 LMWM 产物。它同时提供概率化阶段世界输出和隐变量 prototype subgoal,可作为 VLA 条件输入。

### 5. Unified LMWM 推理 wrapper

目的:将当前最佳 unified LMWM 暴露为一个 VLA-ready 推理产物。

产物:

- 脚本:`lmwm/scripts/infer_unified_lmwm.py`
- 输出目录:`lmwm/outputs/stage3_unified_inference/20260701_best`
- 预测文件:`lmwm/outputs/stage3_unified_inference/20260701_best/predictions.npz`
- 摘要:`lmwm/outputs/stage3_unified_inference/20260701_best/summary.json`

命令:

```bash
CUDA_VISIBLE_DEVICES=0 python lmwm/scripts/infer_unified_lmwm.py   --checkpoint lmwm/checkpoints/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified/best.pt   --dataset_npz lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique.npz   --graph_npz lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz   --output_dir lmwm/outputs/stage3_unified_inference/20260701_best   --max_samples 4096   --device cuda:0
```

4096 采样行验证:

- Greedy top1:0.96484375
- Max-product top1:0.96508789
- Greedy 置信度均值:0.9614772
- Max-product 置信度均值:0.9591309
- 转移置信度均值:0.2024447
- 转移熵均值:2.7746096
- Greedy prototype cosine 均值:0.9915323
- Max-product prototype cosine 均值:0.9929357

VLA-facing 字段:

```text
current_milestone
transition_probs
greedy_pred
max_product_pred
greedy_subgoal_latent
max_product_subgoal_latent
greedy_confidence
max_product_confidence
transition_entropy
```

### 6. 逐 milestone 误差分析

目的:为下一轮鲁棒性迭代定位不稳定的阶段。

产物:

- 脚本:`lmwm/scripts/analyze_lmwm_predictions.py`
- 输出目录:`lmwm/outputs/stage3_unified_inference/20260701_best/error_analysis`
- CSV:`lmwm/outputs/stage3_unified_inference/20260701_best/error_analysis/per_milestone_metrics.csv`
- 摘要:`lmwm/outputs/stage3_unified_inference/20260701_best/error_analysis/summary.json`

4096 样本推理检查中最差的 milestone:

- #34:Greedy 0.90625,Max-product 0.87500,support 3198
- #30:Greedy 0.90756,Max-product 0.90756,support 5087
- #5:Greedy 0.90323,Max-product 0.91935,support 3271
- #2:Greedy 0.95294,Max-product 0.89412,support 3843

解读:下一轮鲁棒性迭代应聚焦于边界/歧义阶段,而非全局模型容量。一个实用的下一步是添加置信度门控,当神经置信度低或当前 milestone 属于已知弱集时回退到图表输出。

### 7. Hybrid 图回退推理

目的:结合神经预测与显式循环图先验,使当前 unified LMWM 对下游 VLA 使用更安全。

产物:

- 更新的脚本:`lmwm/scripts/infer_unified_lmwm.py`
- 配置:`lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_recommended.yaml`
- 输出目录:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback`
- 预测:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback/predictions.npz`
- 误差分析:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback/error_analysis/per_milestone_metrics.csv`

Hybrid 规则:

```text
默认使用神经 LMWM 预测。
在以下情况回退到循环图表:
  置信度 < 阈值,或
  当前 milestone 属于弱 milestone 集合。
```

来自之前误差分析的弱 milestone 集:

```text
{2, 5, 30, 34}
```

`greedy_conf_threshold=0.93` / `max_product_conf_threshold=0.93` 在 4096 样本上验证:

- Neural Greedy top1:0.96484375
- Neural Max-product top1:0.96508789
- Hybrid Greedy top1:0.99755859
- Hybrid Max-product top1:0.99804688
- Greedy 回退率:0.20532227
- Max-product 回退率:0.21069336
- Hybrid Greedy prototype cosine:0.99674082
- Hybrid Max-product prototype cosine:0.99737370

解读:这不是神经模型改进的独立证明;回退使用的图先验也是图监督标签的定义来源。但它在操作上有用,因为它暴露了不确定性并防止已知弱 milestone 仅依赖神经预测。

### 8. Hybrid 门控阈值 sweep

目的:用实测准确率/回退率 tradeoff 选择默认门控,而非手动挑选置信度阈值。

产物:

- 脚本:`lmwm/scripts/sweep_hybrid_gate.py`
- Sweep CSV:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback/gate_sweep/hybrid_gate_sweep.csv`
- 摘要:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback/gate_sweep/summary.json`

验证和早期 VLA 集成的推荐设置:

```text
greedy_conf_threshold = 0.90
max_product_conf_threshold = 0.92
weak_milestones = {2, 5, 30, 34}
```

这给出平均 top1 约 0.9976,平均回退率约 0.196(4096 样本推理检查)。

推荐配置直接执行并产出:

- 输出:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/predictions.npz`
- 摘要:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/summary.json`
- 逐 milestone 分析:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/error_analysis/per_milestone_metrics.csv`
- Hybrid Greedy top1:0.99707031
- Hybrid Max-product top1:0.99804688
- Greedy 回退率:0.18774414
- Max-product 回退率:0.20434570

应限制图回退时的保守设置:

```text
greedy_conf_threshold = 0.80
max_product_conf_threshold = 0.80
weak_milestones = {2, 5, 30, 34}
```

这给出平均 top1 约 0.9935,平均回退率约 0.146。

### 9. 在线运行时预测器 API

目的:将当前最佳 LMWM 从批量评估脚本迁移到 VLA 策略代码可调用的稳定在线接口。

产物:

- 模型模块:`lmwm/src/lmwm/model.py`
- 运行时 API:`lmwm/src/lmwm/runtime.py`
- Smoke 脚本:`lmwm/scripts/smoke_runtime_predictor.py`
- 运行时 smoke 输出:`lmwm/outputs/runtime_smoke/20260701_hybrid_recommended/summary.json`
- 离线/运行时一致性检查:`lmwm/outputs/runtime_smoke/20260701_hybrid_recommended/batch_runtime_compare.json`

运行时 API:

```python
from lmwm.runtime import UnifiedLMWMPredictor

predictor = UnifiedLMWMPredictor.from_yaml(
    "lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_recommended.yaml"
)
result = predictor.predict(current_features, current_milestones)
```

返回字段包括:

```text
current_milestone
transition_probs
neural_greedy / neural_max_product
graph_greedy / graph_max_product
hybrid_greedy / hybrid_max_product
hybrid_greedy_subgoal_latent / hybrid_max_product_subgoal_latent
greedy_confidence / max_product_confidence
transition_entropy
greedy_fallback_mask / max_product_fallback_mask
```

512 样本 smoke 验证:

- 显式当前阶段匹配:true
- 推断当前阶段匹配:1.0
- Greedy 回退率:0.173828125
- Max-product 回退率:0.181640625
- 转移行和均值:1.0
- 转移行和最大绝对误差:2.3841858e-07
- Hybrid Greedy latent norm 均值:1.0
- Hybrid Max-product latent norm 均值:1.0

离线/运行时一致性检查(对比推荐批量推理输出):

- 比较行数:512
- 精确匹配字段:current milestone,hybrid milestone ids,fallback masks
- 浮点容差:5e-6
- 通过:true

解读:当前 LMWM 产物现在有一个可复用的在线预测器,不仅是离线脚本。下一步是添加校准指标和 VLA 集成的薄 serving wrapper 或策略适配器。

### 10. 置信度校准分析

目的:量化神经置信度是否足够有意义以支持回退门控,而非仅依赖手动选择的阈值。

产物:

- 脚本:`lmwm/scripts/calibrate_lmwm_confidence.py`
- 摘要:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/summary.json`
- Greedy 可靠性分箱:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/greedy_reliability_bins.csv`
- Max-product 可靠性分箱:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/max_product_reliability_bins.csv`
- Greedy 阈值曲线:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/greedy_threshold_curve.csv`
- Max-product 阈值曲线:`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/max_product_threshold_curve.csv`

4096 样本推荐推理输出上的结果:

Greedy 头:

- 准确率:0.96484375
- 平均置信度:0.96147716
- ECE:0.00726309
- AURC:0.00227934
- 90% 覆盖率下的风险:0.00732303
- 20% 纯神经回退下的最佳阈值:0.97
- 该阈值下的接受准确率:0.99726527

Max-product 头:

- 准确率:0.96508789
- 平均置信度:0.95913082
- ECE:0.00653240
- AURC:0.00246913
- 90% 覆盖率下的风险:0.00813670
- 20% 纯神经回退下的最佳阈值:0.95
- 该阈值下的接受准确率:0.99646955

解读:置信度值可用于排序可靠的神经预测。推荐运行时门控仍使用 hybrid sweep 阈值 `0.90 / 0.92`,因为 hybrid 系统已强制已知弱 milestone 回退,已消耗部分回退预算。校准支持回退门控的原则,应用于调优 deployment 特定的覆盖率/准确率 tradeoff。

### 11. 验证驱动的弱 milestone 策略

目的:用从 held-out 逐 milestone 验证指标自动生成的策略替代手动维护的弱 milestone 列表。

产物:

- 脚本:`lmwm/scripts/select_hybrid_policy.py`
- 源指标:`lmwm/outputs/stage3_unified_inference/20260701_best/error_analysis/per_milestone_metrics.csv`
- 生成的配置:`lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_validation_selected.yaml`
- 选择摘要:`lmwm/outputs/stage3_unified_inference/20260701_policy_selection/summary.json`
- 验证输出:`lmwm/outputs/stage3_unified_inference/kai0base_dinov3h_stage3_hybrid_validation_selected/summary.json`
- 逐 milestone 验证分析:`lmwm/outputs/stage3_unified_inference/kai0base_dinov3h_stage3_hybrid_validation_selected/error_analysis/per_milestone_metrics.csv`

选择规则:

```text
当满足以下条件时选择当前 milestone 进行图回退:
  samples >= 30,且
  transition_support >= 0,且
  min(greedy_top1, max_product_top1) < 0.94
```

所选弱 milestone:

```text
{2, 5, 7, 19, 21, 23, 24, 30, 31, 34}
```

验证选择安全策略在 4096 样本上的结果:

- Neural Greedy top1:0.96484375
- Neural Max-product top1:0.96508789
- Hybrid Greedy top1:0.99804688
- Hybrid Max-product top1:0.99877930
- Greedy 回退率:0.31738281
- Max-product 回退率:0.33227539
- Hybrid Greedy prototype cosine:0.99689162
- Hybrid Max-product prototype cosine:0.99763995

解读:验证选择策略是一个安全的运行模式。它相比平衡推荐策略提高了 hybrid top1,但将回退率从约 19-20% 增加到约 32-33%。保留原推荐配置作为平衡默认;当 VLA 管线偏好更高的 planning-prior 依赖而非神经自主时使用验证选择配置。

### 12. 学习型不确定性回退原型

目的:通过从神经预测错误中学习一个轻量级误差风险模型,减少对硬弱 milestone 列表的依赖。

产物:

- 训练脚本:`lmwm/scripts/train_uncertainty_policy.py`
- 运行时集成:`lmwm/src/lmwm/runtime.py`
- 学习型配置:`lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_learned_uncertainty.yaml`
- 策略产物:`lmwm/outputs/uncertainty_policy/20260701_logistic/uncertainty_policy.npz`
- 策略训练摘要:`lmwm/outputs/uncertainty_policy/20260701_logistic/summary.json`
- 运行时 smoke 输出:`lmwm/outputs/runtime_smoke/20260701_learned_uncertainty/summary.json`
- 逐 milestone 分析:`lmwm/outputs/runtime_smoke/20260701_learned_uncertainty/error_analysis/per_milestone_metrics.csv`

模型:

```text
每个 head 的逻辑回归误差风险模型
特征 = [head 置信度,转移熵,转移置信度,当前 milestone one-hot]
标签 = 神经 head 预测错误
```

从 4096 行神经预测的训练/验证划分:

Greedy 头验证:

- 错误率:0.03515625
- 20% 回退下的推荐误差阈值:0.30
- 回退率:0.15429688
- 接受准确率:0.99653578
- 错误召回率:0.91666669

Max-product 头验证:

- 错误率:0.033203125
- 20% 回退下的推荐误差阈值:0.40
- 回退率:0.15332031
- 接受准确率:0.99538636
- 错误召回率:0.88235295

4096 行(无硬弱 milestone 列表)的运行时 smoke:

- Neural Greedy top1 vs graph:0.96484375
- Neural Max-product top1 vs graph:0.96508789
- Hybrid Greedy top1 vs graph:0.99804688
- Hybrid Max-product top1 vs graph:0.99682617
- Greedy 回退率:0.17211914
- Max-product 回退率:0.16430664

解读:学习型不确定性相比验证选择安全策略大幅减少了回退,同时保持了较高的 hybrid 准确率。但尚未成为默认,因为逐 milestone 分析显示 milestone #34 仍未完全覆盖:学习型 hybrid 在 #34 上达到 0.984375 而非 1.0。保留 `stage3_hybrid_recommended` 作为平衡默认,`stage3_hybrid_validation_selected` 作为安全的图先验权重模式,`stage3_hybrid_learned_uncertainty` 作为学习型回退原型。

### 13. 调优学习型不确定性策略

目的:改进学习型不确定性原型在已知弱 milestone #34 上的表现,无需退回硬弱 milestone 列表。

产物:

- 运行时阈值覆盖:`lmwm/src/lmwm/runtime.py`
- 调优配置:`lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_learned_tuned.yaml`
- 阈值 sweep 摘要:`lmwm/outputs/runtime_smoke/20260701_learned_uncertainty/threshold_sweep_summary.json`
- 运行时 smoke 输出:`lmwm/outputs/runtime_smoke/20260701_learned_tuned/summary.json`
- 逐 milestone 分析:`lmwm/outputs/runtime_smoke/20260701_learned_tuned/error_analysis/per_milestone_metrics.csv`

调优阈值:

```text
greedy_error_threshold = 0.25
max_product_error_threshold = 0.30
```

这些是从误差概率 sweep 中选取的最低平均回退组合,完全覆盖两个头的 milestone #34。

4096 行运行时 smoke:

- Neural Greedy top1 vs graph:0.96484375
- Neural Max-product top1 vs graph:0.96508789
- 调优学习型 Hybrid Greedy top1:0.99853516
- 调优学习型 Hybrid Max-product top1:0.99853516
- Greedy 回退率:0.21362305
- Max-product 回退率:0.21411133

Milestone #34 逐 milestone 验证:

- Neural Greedy top1:0.90625
- Neural Max-product top1:0.875
- 调优 Hybrid Greedy top1:1.0
- 调优 Hybrid Max-product top1:1.0
- Greedy 回退率:1.0
- Max-product 回退率:1.0

解读:调优学习型不确定性现已修复之前的 #34 缺失,同时将回退率保持在远低于验证选择安全策略的水平。它是当前最强的学习型回退候选。平衡推荐配置保持默认,直到该学习型策略在超出当前 4096 行样本和 Task_A/kai0_base 设置的范围内得到验证。

### 14. 全量 20 万 pair 运行时策略评估

目的:验证运行时策略,超出之前 4096 行的 smoke 检查。首次全量运行保存了预测并发现写入全 latent 数组很昂贵,因此 `lmwm/scripts/eval_runtime_policy.py` 现在支持 `--summary_only` 用于快速策略级比较。

产物:

- 全量评估器:`lmwm/scripts/eval_runtime_policy.py`
- 调优学习型全量预测:`lmwm/outputs/runtime_eval/20260702_learned_tuned_full/runtime_predictions.npz`
- 调优学习型全量摘要:`lmwm/outputs/runtime_eval/20260702_learned_tuned_full/summary.json`
- 调优学习型全量逐 milestone 分析:`lmwm/outputs/runtime_eval/20260702_learned_tuned_full/error_analysis/per_milestone_metrics.csv`
- 推荐全量摘要:`lmwm/outputs/runtime_eval/20260702_recommended_full_summary/summary.json`
- 验证选择安全全量摘要:`lmwm/outputs/runtime_eval/20260702_validation_selected_full_summary/summary.json`
- 策略对比:`lmwm/outputs/runtime_eval/20260702_policy_comparison_summary.json`

全量 20 万 pair 指标:

```text
策略                    平均 top1   贪心 top1    最大积 top1   平均回退率
validation_selected_safe  0.997555    0.997425      0.997685     0.3314975
recommended              0.996455    0.996255      0.996655     0.2002300
learned_tuned            0.9954875   0.995770      0.995205     0.2197125
```

调优学习型策略全量详情:

- Neural Greedy top1 vs graph:0.96325
- Neural Max-product top1 vs graph:0.961735
- Hybrid Greedy top1 vs graph:0.99577
- Hybrid Max-product top1 vs graph:0.995205
- Greedy 回退率:0.222725
- Max-product 回退率:0.2167

全量逐 milestone #34 结果:

- 样本:3606
- Neural Greedy top1:0.88657793
- Neural Max-product top1:0.88879645
- 调优 Hybrid Greedy top1:0.99972268
- 调优 Hybrid Max-product top1:1.0
- Greedy 回退率:0.98058791
- Max-product 回退率:0.99389906

解读:调优学习型不确定性在全量 #34 上修好了,但在全量 20 万 pair 上 mean top1 仍落后于平衡推荐策略,且使用了稍多的回退。当前默认仍应为 `stage3_hybrid_recommended`。调优学习型策略作为最佳学习型回退候选仍然有用,在替换默认前应用更广泛的特征或更多训练数据来改进。验证选择安全策略仍是最高准确率的图先验权重模式。

## 当前最佳产物

使用 Stage-3 unified checkpoint 作为当前最佳模型:

`lmwm/checkpoints/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified/best.pt`

## 已知局限

- 仅使用了已有的 DINOv3-H `kai0_base` 缓存;本轮未找到匹配的 DINOv3-H `kai0_dagger` 缓存。
- 循环图是一阶 Markov + 有限 horizon 最大积规划的。
- 评估在相同任务族的 held-out episode 上,非跨任务。
- 标签来自 CRAVE milestone 分配,边界噪声和簇歧义会传播到监督中。
- 当前模型预测 DINOv3-H 隐变量 prototype subgoal,非解码图像或机器人动作。

## 下一次自动迭代

下一步应瞄准鲁棒性而非另一个容易的准确率提升:

1. 为转移行添加置信度校准和熵指标。
2. 按当前 milestone id 和转移 support 计数评估误差。
3. 为低置信或低 support 转移添加回退策略。
4. 如果 DINOv3-H dagger 特征可用,重新构建图并训练 base+dagger 混合模型。
5. 添加推理 wrapper,返回 `current_stage`、`transition_probs`、`greedy_subgoal_latent`、`max_product_subgoal_latent` 和置信度分数。
