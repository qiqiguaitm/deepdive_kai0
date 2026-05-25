# visualization/ — 可视化

> **场景**: 在线推理可视化 / 点云 → Mesh / Rerun 工具使用与坑点。

## 文件清单

| 文件 | 用途 |
|---|---|
| [`inference_visualization.md`](inference_visualization.md) | 在线推理可视化 + 交互执行控制 |
| [`inference_visualization_mesh.md`](inference_visualization_mesh.md) | 在线推理可视化:点云 → Mesh 化升级方案 |
| [`rerun_mesh_transparency_lesson.md`](rerun_mesh_transparency_lesson.md) | Rerun Mesh 透明度 — Debugging Postmortem (lesson, 避坑) |

## 按需求找文件

| 你想做什么 | 去 |
|---|---|
| 在线看推理输出 + 控制执行 | inference_visualization.md |
| 把点云 viz 升级成 mesh (光滑面 + 法线) | inference_visualization_mesh.md |
| Rerun mesh 透明度设错了 / debug 经验 | rerun_mesh_transparency_lesson.md |

## 跨场景跳转

- 推理服务本身 → `../inference/`
- 录数据时的 viz (不同于推理 viz) → `../data_collection/`
