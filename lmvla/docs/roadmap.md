# LMVLA 阶段路线（E0→E3）

> 本文是**父层摘要**；权威计划 + 每步细化 kill criteria 见 [`../lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md`](../lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md)（2026-07-08 整合更新）。
> 裁决铁律：**唯一指标 = 下游 SR / action-MAE**；GT milestone 先行隔离注入机制，再换真预测器。

---

## 主线阶段

| 阶段 | 目标 | kill criteria（不过就停/回退） | 状态 |
|---|---|---|---|
| **E0** 环境搭通 + baseline | 一条命令在 RoboTwin 2.0 上跑 pi05 rollout 出 SR（后续一切注入实验的地基） | pi05 在 RoboTwin 上跑不出非零 SR（harness 不忠实）→ 先修环境不推进 | ✅ 已通（closed-loop eval 可跑；见 robotwin_sim_env_setup.md） |
| **E1** 辅助监督验证 | 验证"milestone 辅助监督"对策略**到底有没有用**（先 GT milestone） | GT milestone 辅助监督相对 baseline 无 SR/MAE 增益 → milestone 信号对该策略无价值，止损 | ⏳ |
| **E2** SigLIP 原生 prefix 注入（主方案 P） | milestone 作虚拟图像 token 进 prefix + KI，测下游 SR | 注入相对 E1 最佳辅助方案无增益，或 KI 破坏预训练能力 → 回退注入设计 | ⏳ |
| **E3** 真预测器 + 集群 + 真机 | 用 LMWM 真 next-milestone 预测器替 GT，上集群规模 + kai0 叠衣真机 | 真预测器（top1~0.4）注入后 SR 显著低于 GT 上界 → 需先提预测器质量 | ⏳ |

> E0→E3 的依赖是硬的：**GT 上界不成立就不做真预测器**（否则分不清是注入机制差还是预测误差污染）。

---

## 两条子项目内的进行中工作（并行支撑主线）

**CRAVE**（详见 [`../crave/docs/STATUS.md`](../crave/docs/STATUS.md)）
- 🔴 P0 决定性：Tier3 sim01 rollout（A/B/C 三臂）——AB_plan 唯一决定性判据，等 A 臂集群训完（本地无 sim 不可验）。
- 🟠 P1 已跑实：A1 切分 / A2 keyframe·OOD·dedup 均本地验证过；剩 A1-VLM 段命名（需 API）→ 接 AWBC `prompt_from_task`。
- ❌ 已否决别再走：二值化 advantage、EM-HMM 统一框架、B2 弱成败信号（CRAVE 无细微失败信号）。

**LMWM**（详见 [`../lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md`](../lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md)）
- world-model intrinsic 指标已达标（top1≈0.459，部署单模型蒸馏 0.449）；当前最优 `stage3_unified/.../best.pt`。
- ❌ 负结果：帧历史条件无增益（Phase C）；7B 均值+方差目标被否；frame-only 天花板已论证。

---

## 资产盘点（E0 起点，来自 MASTER_PLAN §2）

| 资产 | 位置 | 状态 |
|---|---|---|
| RoboTwin 2.0 全套 | `/vePFS/shock/vla/RoboTwin` | ✅ 可复用 |
| RoboTwin conda env | `/vePFS/HuanQian/conda_envs/RoboTwin`（py3.10） | ✅ |
| pi05 base / fold-awbc ckpt | `kai0/checkpoints/pi05_*` | ✅ 本地 |
| LMWM provider | `lmwm/checkpoints/teach_proto_3task.pt` | ✅ |
| 仿真评测双机落地 | [`../../docs/deployment/robotwin_sim_env_setup.md`](../../docs/deployment/robotwin_sim_env_setup.md) | ✅ 本地 + gf3 |
