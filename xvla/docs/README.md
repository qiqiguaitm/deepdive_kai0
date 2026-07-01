# xvla/docs/ — X-VLA 文档集散地

> 2026-07-01: 所有 XVLA 相关文档从 `docs/` 各子目录收归此处。旧路径保留 symlink 向后兼容。

## 目录

```
xvla/docs/
├── README.md                                          ← 本文档
├── deployment/
│   └── inference/
│       ├── xvla_inference_bringup.md                  部署架构说明
│       ├── xvla_rtc_design.md                         RTC 设计文档
│       └── xvla_upstream_vs_local_consistency.md      upstream vs 本地一致性
├── training/
│   ├── analysis/
│   │   ├── xvla_dataset_vs_official.md                数据 vs 官方对照
│   │   ├── xvla_innovation_directions.md              创新方向分析
│   │   └── xvla_vs_official_gap_rootcause.md          gap 根因分析
│   ├── plans/
│   │   ├── xvla_camera_robust_grasp_final.md          视觉鲁棒抓取计划
│   │   ├── xvla_domain_slot_init_ablation.md          domain slot init 消融
│   │   ├── xvla_proprio_shortcut_openloop_fix.md      proprio 开环修复
│   │   └── xvla_track_x_curriculum.md                 Track X 课程学习
│   ├── experiments/
│   │   ├── xvla_conditioning_methods_results.md        conditioning 方法实验
│   │   ├── xvla_e0_v1_official_fixedcam_results.md    E0 fixedcam 官方配方
│   │   ├── xvla_e0_v1_official_results.md             E0 v1 官方配方
│   │   ├── xvla_taskp_local_5090_results.md           TaskP binary 本地训练
│   │   ├── xvla_taskp_continuous_results.md           TaskP 连续夹爪训练 ⭐
│   │   ├── xvla_track_x_x3_ablation_results.md        Track X X3 消融
│   │   └── xvla_x3_controlled_a0423_results.md        X3 对照实验
│   └── xvla_blackimage_dataloader_lesson.md           黑图数据加载教训
```

## 快速入口

| 场景 | 文档 |
|---|---|
| 部署 X-VLA 到真机 | `deployment/inference/xvla_inference_bringup.md` |
| 训练 X-VLA 本地 | `training/experiments/xvla_taskp_local_5090_results.md` |
| 连续夹爪训练 | `training/experiments/xvla_taskp_continuous_results.md` |
| E0 折叠训练 | `training/experiments/xvla_e0_v1_official_fixedcam_results.md` |
| X3 对照实验 | `training/experiments/xvla_track_x_x3_ablation_results.md` |
| 数据质量教训 | `training/xvla_blackimage_dataloader_lesson.md` |
