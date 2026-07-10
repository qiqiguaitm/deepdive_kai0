# LMVLA 全景架构 — CRAVE → LMWM → VLA

> 本文只讲**三段如何拼成一条流水线**；每段内部算法见各子项目 docs（本文末给单一事实源）。
> 最终产出:**LMWAM = LMWM × kai0 π0.5**，在 RoboTwin 2.0(sim)+ kai0 叠衣真机上以 **SR / action-MAE** 裁决。

---

## 0. 为什么是这三段

几乎所有"自我改进 / RL / world-model"的 VLA 路线，瓶颈都是**价值/子目标信号的获取成本**（RECAP 要数千次真机 rollout + 失败标注；HIL-SERL 要人标 reward classifier）。
LMVLA 的赌注是：**用零标签零训练的重复几何（CRAVE）先拿到离散 milestone 结构 → 用一个轻量世界模型（LMWM）把它变成可预测的子目标 → 以 π 原生的 prefix token 通路注入策略**。三段各自可独立验证、可独立替换。

---

## 1. 段一 · CRAVE — 零训练里程碑/价值引擎

**输入**：同任务 demo 视频集（`frames`）。
**产出**：milestone 图（KMeans 簇 + 顺序化）+ 每帧 progress/value + 技能切分。

```text
frames ─► encoder(默认 DINOv3-H, frozen) ─► KMeans 簇 ─► 顺序化(precedence/isotonic) ─► readout(Viterbi-DP / 在线 DC+vote) ─► milestone id / value
```

关键事实（来自 crave STATUS/positioning）：
- milestone 比时间多解释 **2×** 动作方差（R² 0.43 vs 0.22）→ milestone = **动作相关技能相位**，不只是计时器。
- 跨数据集泛化 corr 0.94–1.00（XVLA/coffee）；对真 `stage_progress_gt` corr 0.865（监督 pi0-AE 为 0.897），**零标注**。
- 在线可因果化：30Hz 原生 + DC(α=2.0) + sym adaptive vote + boxcar smooth，仅 3 参数、完全在线，corr 0.974。
- **诚实短板**：需同任务 demo 集（非零样本）；无结果信号（抓不到"自信但错"）；跨任务/跨本体零样本弱。

接口：`crave.value.DiscreteValue` / `crave.value.ContinuousValue`；编码器 `crave.encoders.load_encoder("dinov3-h")`。

---

## 2. 段二 · LMWM — 递归里程碑世界模型

**输入**：CRAVE milestone id / prototype 序列（+ 当前 DINOv3-H 帧特征）。
**产出**：给定"当前状态"预测"下一步子目标"，供策略消费。

一次前向输出（`lmwm.runtime.UnifiedLMWMPredictor`）：
- recurrence transition probability row；
- **Greedy** next milestone：一步 `argmax P(stage_{t+1} | stage_t)`；
- **Max-product** next milestone：有限视野 DP / max-product 搜到终点里程碑，取路径上下一步；
- Greedy / Max-product **latent prototype 子目标**（可直接作视觉子目标）；
- confidence / entropy → 下游 VLA gating。

hybrid 接口在神经预测上加图 fallback（`hybrid_greedy`、`hybrid_max_product`、`*_subgoal_latent`、`*_fallback_mask`）——图 fallback 是**规划先验**，非独立学习修正。
当前最优 artifact：`lmwm/checkpoints/stage3_unified/.../best.pt`（DINOv3-H frame → 上述全部信号）。

> 建模负结果（避免重复踩坑）：帧历史条件无增益（Phase C）；7B 编码器在均值+方差目标上被否；EM-HMM 在 768D 塌缩。详见 lmwm `PITFALLS_AND_HISTORY.md` / crave `em_hmm_negative_result.md`。

---

## 3. 段三 · 注入 VLA — SigLIP 虚拟图像 token + KI

**主方案 P**（`INJECTION_DESIGN_2026-07.md` 定稿，取代旧"4 输入 Alternate-DiT"）：

- milestone 子目标**就在 SigLIP = PaliGemma 视觉空间** → 当作"虚拟未来图像 token"经**原图像投影**进 π0.5 的 **VLM prefix**，近零 distribution-shift（不用 cross-attn、不用 adaRMS）。
- 保预训练靠 **KI（Knowledge Insulation）stop-grad**：连续 action expert → backbone 的 K/V 套 `sg()`，**不是冻 backbone**。
- **GT-first 铁律**：先用 GT milestone 隔离"注入机制本身有没有用"，再换真预测器（top1~0.4 的预测误差会污染结论）。

**裁决**：唯一指标 = 下游 **SR / action-MAE**（intrinsic 指标已达标）；sim 用 RoboTwin 2.0（双臂，匹配 kai0 叠衣本体），域用 kai0 叠衣真机，和 LaWM 98.6% 摆一起对比。

---

## 4. 依赖与边界（谁不依赖谁）

- CRAVE **不依赖** LMWM / VLA：可独立作为零标签数据工具（keyframe / 失败定位 / dedup / AWBC 子任务切分）。
- LMWM **依赖** CRAVE 的 milestone 定义，但**不依赖**具体 VLA；产出是通用子目标接口。
- VLA 注入 **依赖** LMWM 的子目标，但注入机制（prefix token + KI）与 milestone 来源解耦——可换任何"SigLIP 空间子目标"来源。

这条解耦是刻意的：任一段可单独替换/消融，而不推翻其余两段。

---

## 单一事实源

- CRAVE 方法：[`../crave/docs/cross_episode_recurrence_value_METHOD.md`](../crave/docs/cross_episode_recurrence_value_METHOD.md)；定位：[`../crave/docs/CRAVE_positioning_and_roadmap.md`](../crave/docs/CRAVE_positioning_and_roadmap.md)
- LMWM 架构：[`../lmwm/docs/LMWM2_FINAL_ARCHITECTURE.md`](../lmwm/docs/LMWM2_FINAL_ARCHITECTURE.md)；注入：[`../lmwm/docs/INJECTION_DESIGN_2026-07.md`](../lmwm/docs/INJECTION_DESIGN_2026-07.md)；总纲：[`../lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md`](../lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md)
- 阶段路线与 kill criteria：[`roadmap.md`](roadmap.md)
