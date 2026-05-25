# inference/ — 推理与真机部署

> **场景**: 把训完的 ckpt 部署到真机 / sim01 / 推理时延优化 / RTC / IPC / ROS2 校验 / web 推理服务。

## 目录结构

```
inference/
├── README.md                              ← 你在这里
├── rtc_implementation.md                  RTC (Real-Time Chunking) 算法实现
├── ipc_inference_deployment_review.md     IPC 推理服务架构 review
├── ros2_image_inference_validation_review.md  ROS2 图像处理与推理校验
├── sim01_deployment.md                    sim01 部署文档
├── build_web_venv.md                      web 推理服务通用 venv (多框架解耦)
└── realtime_vla/                          实时推理优化 series (4 文件)
    ├── README.md
    ├── strategy.md                        决策摘要 + 上下文 + 优化排序 + Fallback (选项 Y)
    ├── roadmap.md                         5 阶段实施路线 + 真机测试方案
    ├── v1_triton_log.md                   V1 Triton 已实施日志 (P50 76→32ms)
    └── layer_b_plan.md                    Layer B 系统级优化 未来 plan
```

## 文件清单

| 文件 | 行数 | 用途 |
|---|---|---|
| [`rtc_implementation.md`](rtc_implementation.md) | ~249 | RTC (Inference Real-Time Chunking) 算法实现, 4 schedules + jax.vjp guidance |
| [`ipc_inference_deployment_review.md`](ipc_inference_deployment_review.md) | ~178 | IPC 推理服务架构 — 与原版差异分析 |
| [`ros2_image_inference_validation_review.md`](ros2_image_inference_validation_review.md) | ~430 | ROS2 图像处理 + 推理结果校验策略 review |
| [`sim01_deployment.md`](sim01_deployment.md) | ~611 | sim01 仿真机器人部署 / 完整 step-by-step |
| [`build_web_venv.md`](build_web_venv.md) | ~137 | 通用 web 推理服务 venv — 支持多框架, 与代码解耦, 可用于 data_manager + 推理 host |
| [`realtime_vla/`](realtime_vla/README.md) | series | 实时 VLA 推理优化 (P50 76→32ms 已达成, Layer B 下一阶段) |

## 按需求找文件

| 你想做什么 | 去 |
|---|---|
| 看实时推理优化路线 / 选项 X 战略 | realtime_vla/strategy.md |
| 看 5 阶段实施路线图 | realtime_vla/roadmap.md |
| 看 V1 Triton 已实现日志 (异步流水线 / FP8 / SHM) | realtime_vla/v1_triton_log.md |
| 看下一阶段 Layer B 系统级优化 | realtime_vla/layer_b_plan.md |
| RTC 算法 (chunk overlap / inpainting / guidance) 怎么实现 | rtc_implementation.md |
| IPC 推理服务 (kai0-style) 架构 | ipc_inference_deployment_review.md |
| ROS2 image topic / 推理 input 校验 | ros2_image_inference_validation_review.md |
| sim01 上完整部署 ckpt 推理 | sim01_deployment.md |
| 起一个 web 服务接口供其他框架/真机调用推理 | build_web_venv.md |

## 跨场景跳转

- 部署前需要先拿 ckpt → `../training_ops/data_sync_tos.md` (ckpt 训练→TOS→sim01)
- 推理时实时可视化 → `../visualization/inference_visualization.md`
- 推理时硬件故障/相机问题 → `../incidents/` (查历史)
- 跨本体推理 routing (Soft Prompt domain_id) → `../strategy/cross_embodiment_strategy.md` §5.2
