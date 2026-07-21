# LMWM 文档体系（索引）

> LMWM = Latent Milestone World Model，面向 kai0 π0.5 / starVLA VLA 的 milestone+1 价值层子目标预测器。
> 本文件是**唯一入口**。当前最终版在顶层，历史 / 被取代 / 未来阶段的文档在 [`history/`](history/) 且由 [`HISTORY.md`](HISTORY.md) 索引。
> 最近整理：**2026-07-15**（集成线痛点/报告三件套去重 → 单一报告；已执行计划与快照归档到 `history/`）。

---

## 🧭 先读这两个

| 文档 | 一句话 |
|---|---|
| ⭐ [`ARCH_LMWM_final_2026-07-20.md`](ARCH_LMWM_final_2026-07-20.md) | **LMWM 架构定版单一事实源**:r 场信号 → r-谷分段/r-脊目标 → 生成器+MDN 预测器 → **并联双尺度**注入(非替换)· 每层选了什么/淘汰了什么 · ⚠️ §4 证据状态(架构定版≠效果证成:P1 打平未超,唯一稳健机理=t6 +8.5pt) |
| 📖 [`RECURRENCE_UNIVERSAL_goals_and_roadmap.md`](RECURRENCE_UNIVERSAL_goals_and_roadmap.md) | **唯一详源**:目标/路线图/全部实验日志 §4。**动手前先读这个**,不要拿 `crave/docs/final_architecture.md`(v1)当最新方案 |
| 🔬 [`ARCH_predictor_vs_single_2026-07-21.md`](ARCH_predictor_vs_single_2026-07-21.md) | **为何两模型而非单预测器**(受控 A/B,So400m,8+3 seed):单发单模型反而更准且不坍缩→历史"持久坍缩"叙事不复现;两模型价值=**多模态** best-of-K 0.779>单模型 0.765。⚠️内在指标非 SR |

## ⭐ 当前最终方案（两条线，顶层 = 单一事实源）

### 线 A · 独立 LMWM（milestone 世界模型本体）

| 文档 | 一句话 |
|---|---|
| [`RESULT_newcrave_final_arch_2026-07-11.md`](RESULT_newcrave_final_arch_2026-07-11.md) | **统一 DINOv3-base LMWM**：全链路一个编码器（挖矿/预测器/生成器/teacher/解码同空间）· proto teacher 码=簇中心 shared PCA128 · deploy 0.910/id3 0.940 · 闭环可视化 · teacher 码三方消融 §4.6 |
| [`DECODER_dinov3base_video_2026-07-11.md`](DECODER_dinov3base_video_2026-07-11.md) | 可视化解码器：**pooled（软）+ grid（锐）** 两法 + 视频流时序一致 |

**运行**：`train_multitask.py --encoder dinov3base --teacher proto --teacher_code shared_pca`
**交付 ckpt**：`lmwm/checkpoints/dinov3base_lmwm_sharedpca_kaicoffee.pt` + `dinov3base_decoder/kai_grid_dec.pt`

### 线 B · LMWM × LaWAM 集成（把 LMWM 换进 starVLA 世界模型槽，对比 SR）

| 文档 | 一句话 |
|---|---|
| ⭐ [`LMWM_report_full_draft.md`](LMWM_report_full_draft.md) | **集成线单一报告**：§1 痛点（3 个已验证 + 本地/论文双证据）· §2 方案设计（Path A hook / Path B 原生 · swap 契约表）· §3 实验与结果对比（Arm M milestone vs Arm B baseline） |
| 📊 [`RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md`](RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md) | **对比结果数据源**（§3 backing）：M 92.20% vs B 96.40% @20000 · 逐任务 + 失败模式 + 未训练尾巴分析 · 原始 JSON `data/lmwm_vs_lawam_libero10_20k.json` |
| ⚠️ [`RESULTS_p1_vs_final_terminal_2026-07-20.md`](RESULTS_p1_vs_final_terminal_2026-07-20.md) | **修正上一行的尾巴叙事**：P1 支线未 import `build_pairs_abl` → 丢弃最终 milestone(缺陷源);终版同任务同空间 lift **+0.0655 全场最高**(P1 −0.098)· 证否「Viterbi 根治大尾巴」与「尾巴越大越差」(Spearman **+0.538**)· 瓶颈在 generator 非 predictor |
| ⛔ [`RESULTS_crave_dualanchor_lmwm_2026-07-20.md`](RESULTS_crave_dualanchor_lmwm_2026-07-20.md) | **作废**:测的是已淘汰的 CRAVE v1 双锚(roadmap §5 已排除)· 仅「最后一个非锚段」度量陷阱与共同口径设计可打捞,**数值勿引用** |
| 🧪 [`DESIGN_progress_value_head_2026-07-15.md`](DESIGN_progress_value_head_2026-07-15.md) | **连续进度 value 头设计**：治 task 6 类弥散子目标（LMWM milestone 盲区）· 用途 A 监督头（无需历史）/ 用途 B 进度条件输入（发射+转移，需历史治别名）· 实现路径 + 混合目标 |
| [`LAM_starVLA_contract_2026-07-12.md`](LAM_starVLA_contract_2026-07-12.md) | 支撑参考：LAM↔starVLA I/O 契约（P0 产出，替换 decoder 的接口基线） |
| [`BUG_AUDIT_2026-07-12.md`](BUG_AUDIT_2026-07-12.md) | 支撑参考：框架 bug 审查 + Path A/B 修复方案（当前 `h_t1_gt`→milestone+1 override 的设计依据） |

**对比臂**：Arm M（LMWM decoder + milestone+1 目标 + swap_teacher）vs Arm B（released LaWM decoder + t+7 目标，同 recipe no-swap 自训）。结果填入报告 §3。

---

## 🗄️ 历史 / 被取代 / 未来阶段 → 全在 [`history/`](history/)，索引见 [`HISTORY.md`](HISTORY.md)

⚠️ **动手前先读 [`HISTORY.md`](HISTORY.md)**，防照搬旧方案/旧代码（SigLIP-era 的 reach 1.67s、deploy 0.753 等都是旧口径）。HISTORY 现分五类：
- 🔴 **已被取代**（SigLIP-era 旧最终，标注了各自"可打捞"内容）
- 🟡 **未来阶段参考**（SigLIP 融合注入，尚未做）
- 📌 **仍有价值**（PITFALLS_AND_HISTORY 踩坑表 · RESEARCH_DIRECTION）——**踩坑前必查**
- 🧩 **集成线已执行计划 / 快照**（2026-07-12~13 的替换计划、复现计划、P1 快照、痛点分析完整版——报告已吸收验证子集）
- 📦 **更早期归档**（[`history/archive/`](history/archive/) 27 docs，2026-07-01~04 探索期）

CRAVE 侧对应索引：[`../../crave/docs/HISTORY.md`](../../crave/docs/HISTORY.md)。
