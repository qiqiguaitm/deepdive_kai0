# LMWM 文档体系(索引)

> LMWM = Latent Milestone World Model,面向 kai0 π0.5 VLA 的 milestone+1 价值层子目标预测器。
> 本文件是**唯一入口**:先看这里,按需下钻。**当前版**在顶层,**历史版**在 `archive/`。
> 最近整理:2026-07-08。

---

## 🎯 想快速了解 → 按这个顺序读

| # | 文档 | 一句话 |
|---|---|---|
| 0 | [`LMWM2_FINAL_ARCHITECTURE.md`](LMWM2_FINAL_ARCHITECTURE.md) | ⭐ **最终定档架构**(2026-07-08 P1/P2 数据收敛;9设计赌注逐项裁决表;保留/降级/砍掉全表;组件I/O规范;跨任务机制;编码器耦合;降级阶梯;待SR裁决清单) |
| 1 | [`REDESIGN_LMWM2_2026-07.md`](REDESIGN_LMWM2_2026-07.md) | 重设计方案 LMWM-2(三路文献调研;优雅性假设部分被实测证伪→定档见 #0) |
| 2 | [`ARCHITECTURE_AND_BASELINE.md`](ARCHITECTURE_AND_BASELINE.md) | 现 LMWM 架构速查(参数量/I-O/vs LaWM)——作为参照基线 |
| 3 | [`FINAL_CROSSTASK_PREDICTOR.md`](FINAL_CROSSTASK_PREDICTOR.md) | proto teacher 定案 + 交付 ckpt + 跨任务指标 |
| 4 | [`PITFALLS_AND_HISTORY.md`](PITFALLS_AND_HISTORY.md) | **版本演进史 + 踩坑/错误路径**(避免重复犯错,踩坑前必看) |

## 📚 专题(深挖时看)

| 文档 | 范围 |
|---|---|
| [`ABLATION_CONVERGENCE_2026-07.md`](ABLATION_CONVERGENCE_2026-07.md) | 所有控制变量消融全表(teacher/anchor/fwd_arch/lift/code_dim/center_w/pred_input/proto) |
| [`RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md`](RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md) | 研究方向:普适 milestone、身份多峰、命名与模块、融合调研 |
| [`PROGRESS_lawm_comparison_2026-07.md`](PROGRESS_lawm_comparison_2026-07.md) | LaWM 官方 ckpt 在我们数据上的实测对比 |

## 🗄️ 历史 / 已归档

`archive/` 下是被上面的当前版**取代**的旧文档(阶段计划、早期架构、单点诊断、迭代日志)。**内容已提炼进 `PITFALLS_AND_HISTORY.md` 的时间线与踩坑表**,除考古外不必读。见 [`archive/README.md`](archive/README.md)。

---

## 🧭 一句话现状(2026-07-08,最终定档)

- **最终架构(via P1/P2 数据收敛)**:π0.5 SigLIP 冻结共享塔 → 预测器(proto teacher/MDN K=4,prev_ẑ 条件化,密度弃权)+ 生成器(AdaLN)→ milestone+1 子目标 grid。部署 ~34M,零 CRAVE/零 DINO/零 bank。
- **teacher = proto(簇中心码,定案)**:开放词表,与 inv 打平但去 InverseEnc+CE 锚;ID 活在 SigLIP 连续流形。
- **跨任务**:多任务联合训练(kai0/coffee/xvla),prev_ẑ 自递归条件 = 主杠杆,proto 连续码 = 开放词表。
- **vs LaWM**:reach 1.67s > 1.48s、参数轻 ~10×、同塔嵌入;唯一未覆盖 = 下游 SR → P0。
- **P1/P2 关键裁决**:价值几何(u·z)证伪/熵门证伪/DINO 沙箱否决/密度弃权成立/proto+prev_ẑ 保留。
- **下一步:P0 oracle-SR**(子目标条件对 π0.5 是否有正增量;milestone-horizon vs 固定 1.2s 论文主张)。

## 🔧 关键产物(代码/模型/数据)

- 训练:`lmwm/scripts/train_multitask.py`(多任务 3锚 3teacher)、`train_twomodel_v2.py`(单任务)、`train_ablation.py`(消融)
- 模型:`lmwm/checkpoints/teach_proto_3task.pt`(⭐推荐)、`teach_proto_4task.pt`、`final_crosstask_{3,4}task.pt`(inv 对照)
- bank:`temp/{crave_full_dinov3h(kai0),coffee_dinov3h,xvla_dinov3h,vis_dinov3h}` + `lmwm/data/recurrence_graphs/*`
- bank 构建:`make_visbase_dinov3h_index.py`(通用)、xreb 缓存捷径(coffee/xvla)
- 网站 showcase:`web/showcase/reports/lmwm_final/`
