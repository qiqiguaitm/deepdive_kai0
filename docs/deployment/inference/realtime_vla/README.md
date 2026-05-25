# realtime_vla/ — 实时 VLA 推理优化 Series

> **场景**: deepdive_kai0 推理时延优化路线 — 从 P50 76ms → 32ms 已达成 (V1 Triton), Layer B 系统级优化下一阶段。
> **来源**: 原 `realtime_vla_optimization_analysis.md` (1687 行) 拆分。

## 5 文件的层次

```
战略层    strategy.md            决策摘要 + 上下文 + 优化项目排序 + Fallback (选项 Y)
路线层    roadmap.md             5 阶段实施路线 + 真机测试方案
延时层    v1_triton_log.md       V1 Triton latency 优化日志 (P50 76→32ms 已实施)
平滑层    ee_stability_layer1.md EE 末端稳定性 Layer 1 日志 (jiggle -81%, reversal -91%, 2026-05-25)
未来层    layer_b_plan.md        Layer B 系统级优化 plan (异步流水线 / SHM / multi-rate)
```

## 按需求找文件

| 你想做什么 | 去 |
|---|---|
| 看决策摘要 / 选项 X vs Y 取舍 / 上下文基线 | strategy.md |
| 看 5 阶段具体实施 / 真机测试方案 / pi0_pytorch 改动清单 | roadmap.md |
| 看 V1 Triton FP8 / async / SHM 怎么实现 + 实测延时数据 | v1_triton_log.md |
| 看 EE 抖动 / 走3退1 / RTC 调参怎么修 + 真机 before/after 数据 | ee_stability_layer1.md |
| 看 Layer B (异步 obs pipeline / RTC 平滑 / 速度自适应 / 客户端 MPC) 下一步 | layer_b_plan.md |
| 排查 P50 异常 / 实施 bug 复盘 | v1_triton_log.md (实施过程踩坑记录) |

## 阅读顺序建议

- **第一次了解**: strategy.md → roadmap.md (一遍)
- **要实施新优化**: roadmap.md → layer_b_plan.md
- **要复盘 V1 latency**: v1_triton_log.md
- **要复盘 EE 平滑度调参**: ee_stability_layer1.md
- **要查具体技术点**: 直接 grep 5 文件

## 跨场景跳转

- RTC 算法实现 (与 realtime 优化正交) → `../rtc_implementation.md`
- ROS2 image 推理校验 → `../ros2_image_inference_validation_review.md`
- sim01 部署 (推理 host) → `../sim01_deployment.md`
- PyTorch 原生训练 (R1/R2, 选项 X 训练侧) → `../../../training/future_plans/plans/pytorch_native_vis_v2_full.md`
