# CRAVE 定位与可行方案 roadmap

> 把前期分析汇总成可执行方案。三层组织:**(1) 能做什么 ·(2) 怎么做 ·(3) 对标领域 SOTA 的优势**,再给**分阶段安排 + 决定性实验**。
> 日期 2026-06-16。配套:[METHOD](cross_episode_recurrence_value_METHOD.md) · [方法对比](value_advantage_methods_comparison.md) · [AB_plan](awbc_milestone_value_AB_plan.md)。

## 0. 重新定位(一切方案的前提)

**CRAVE 的不可替代资产 = milestone 图 + 零标签技能切分;value 是副产品。**
- 离散 value 阶梯**紧对齐机械臂动作基元**(milestone = 基元交界/子目标)→ CRAVE 实际在做**无监督跨 episode 技能切分**,离散让"这一阶 = 这个基元完成"可读。
- 因此**不要**把 CRAVE 当 RECAP 的标量 value 平替(那条路 RECAP 严格更强);把它当**结构/技能引擎 + RL 冷启 + 零标签数据工具**。
- 对标:VIP/LIV/TOPReward 给的是**连续标量** progress/reward,**都没有**可检视的离散 milestone/技能结构;RECAP 给真回报但**黑箱 + 贵**。CRAVE 独占"零训练 + 离散结构 + 跨 episode 重复 grounding"。

---

## 1. 能做什么 ×2. 怎么做 ×3. 对标优势(四组工作流)

### A 组 · 立即可做(低成本,与现有 AWBC 闭环)

**A1 自动子任务切分 → AWBC `prompt_from_task`**
- 怎么做:milestone 边界切每条 demo 成基元段 → VLM 描述边界帧命名("grasp corner"/"fold left")→ 写 task 串 → 接现成 `prompt_from_task`。
- 对标优势:vs 人工子任务标注/LLM 凭空分解——CRAVE 的切分**grounded 在本机器人数据的真实重复结构**,零人工、与该本体动作一致。

**A2 零标签数据工具(keyframe / 失败定位 / dedup)**
- 怎么做:milestone 边界帧 = keyframe 导出;帧到最近簇残差 = OOD/异常;value 卡在 milestone k = 卡在基元 k 的失败定位。
- 对标优势:VIP/LIV 标量 reward **定位不了 WHERE**;监督切分要标签。CRAVE 免费给可归因结构。

### B 组 · 核心研究(补最大软肋:无 action/无结果信号)

**B1 基元→milestone 转移挖掘 → RL-free 基元级 advantage【决定性实验】**
- 怎么做:用现成 milestone 模型 + parquet `action` 列。每条 episode 在每个 milestone 转移 k→k+1 处取动作段 → 聚类/刻画 → 算"该动作类推进 milestone 的可靠度" → **基元级 advantage(action-aware,零 RL)**。
- 对标优势:RECAP 要数千次真机自主 rollout + 失败标注才拿到 action-aware 优势;此法**零 RL 零标签**拿到(相关非因果的)action 级信号,且在**可解释的基元粒度**(RECAP 帧级黑箱)。
- ⚠️ 边界:相关非因果(milestone 在动作 A 后推进 ≠ A 好);它给"基元级 credit 定位",不是真因果优势——但比 AE 帧级噪声 advantage 可操作得多。

**B2 终点可达性 + OOD 残差 → 弱成败信号** ❌**已验·降级**(见 [验证审查](CRAVE_doable_items_verification.md))
- 怎么做:没到终止 milestone 簇的 episode = 弱失败;`CRAVE_value × 到达终点指示` → 能区分成败的粗 advantage。
- **实测否决**:细微 dagger 失败 on-manifold,OOD 残差不分 neg/pos(0.981 vs 0.979),终点 corr 仅 0.13、仅 2% 未完成。→ **只对粗失败/OOD 场景有效**(那些 CRAVE value 自己就掉),廉价补 neg 洞不成立。真 neg 仍需 RL/结果信号(C2)。

### C 组 · 模型化 / 规模化

**C1 蒸馏成在线分布式离散 value 头**
- 怎么做:小 frozen-feature MLP + **分布式 201-bin CE 头(RECAP 式,非 scalar+MSE)**,用 CRAVE 标签训。
- 对标优势:vs kai0-AE(scalar+MSE,OOD 欠读压到 0.27)——分布式更校准、零人工标;vs RECAP——无需 RL。去掉 cache 依赖、对未见状态平滑泛化。

**C2 CRAVE 冷启 V + 少量真机 rollout RL 微调(exceed-demonstrator 路径)**
- 怎么做:CRAVE value 当 RECAP pipeline 的冷启 baseline V(替代昂贵 MC value 预训练)→ 少量自主 rollout 成败做 advantage 修正 → 优势条件 BC。
- 对标优势:RECAP value 预训练贵;CRAVE 零成本替代 → **低成本走通"从经验改进/超越示教"**,这是 CRAVE 唯一能碰到"超越示教"的途径。

### D 组 · 持续性

**D1 增量挖矿 + 漂移监控 + 域自适应**
- 怎么做:增量 KMeans 吸收新 episode、milestone split/merge;覆盖率/漂移指标告警"该重挖";自动按目标域选/加权挖矿集。
- 对标优势:把 vis0526-vs-dagger 那种**手动挖矿域错误制度化消灭**;自我维护 vs 一次性挖矿。

---

## 2. 可行方案安排(按 杠杆×成本×依赖 排序)

| 阶段 | 工作 | 周期 | 成本 | 产出/判据 |
|---|---|---|---|---|
| **Phase 0(决定性)** | **B1 pilot** | ~3-5 天 | 极低(有 action+milestone) | 验证"基元动作↔milestone 转移"**可挖且稳定**→ 整个重定位成立。**Go/No-Go**:转移可靠度是否显著 > 随机、跨 episode 一致 |
| **Phase 1(快赢,可并行)** | A1(子任务→AWBC prompt)+ A2(数据工具) | ~1-2 周 | 低 | A1 直接喂现有 AWBC;A2 出 keyframe/失败定位工具。与 Phase0 并行 |
| **Phase 2** | C1(分布式离散头蒸馏)+ B1 全量 + B2 | ~2-4 周 | 中(单机 GPU) | 一个**action-aware 的在线 advantage labeler**;喂 [AB_plan](awbc_milestone_value_AB_plan.md) 的 B 臂(升级版) |
| **Phase 3(高天花板)** | C2(CRAVE 冷启 + RL 微调) | ~1+ 月 | 高(真机 rollout) | 验证"低成本超越示教";仅在 B1 验证 action-aware 后启动 |
| **持续** | D1(增量挖矿/漂移/域自适应) | 长期 | 低-中 | 自我维护,治挖矿脆弱 |

**关键路径**:Phase 0 的 B1 pilot 是**整套重定位的最小决定性实验**——它若成立(基元动作可靠驱动 milestone 转移),则 A1/B2/C1/C2 全部有地基;若不成立(action↔转移太噪/多模),CRAVE 退回"纯结构 + 数据工具",仍有 A 组 + D 组价值,但放弃 action-aware 野心。

**与现有工作的接口**:A1/C1 直接接 AWBC `prompt_from_task` 与 [AB_plan](awbc_milestone_value_AB_plan.md);C2 接 RECAP 式 pipeline;全部复用现成 milestone 模型 + kai0 数据,无新数据采集(Phase 3 除外)。

---

## 3. 对标领域 SOTA 一览(CRAVE 的位置)

| 子领域 | SOTA 代表 | 它们给什么 | CRAVE 差异化优势 |
|---|---|---|---|
| 经验型 RL value/优势 | **RECAP / π*0.6** | 真回报、action-aware、能超示教 | 零训练零标签、可解释结构、可当其**冷启 V**(C2) |
| 视频 progress/reward | **VIP · LIV · TOPReward** | 连续标量 progress(训练/VLM-logit) | **离散 milestone 结构 + 技能切分**(标量给不了);零训练 |
| 监督进度回归 | **kai0-AE(stage_progress)** | scalar 进度(需 ~1 周标注) | 零标注、in-dist 打平且更平滑、OOD 更稳 |
| 技能/动作切分 | 一般需训练/逐任务调 | 切分 | 跨 episode 重复一次挖出、零训练、且**与 progress value 耦合** |

**一句话**:CRAVE 不在"谁的标量 value 更准"上竞争(那是 RECAP/VIP 的主场),它在"**零标签拿到可解释任务结构 + 把结构变成 RL 冷启与 action 级信号**"上独占——这是 B1 决定性实验要锁死的命题。
