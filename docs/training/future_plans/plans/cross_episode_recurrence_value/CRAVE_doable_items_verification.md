# CRAVE 可做项验证审查(快速/零大GPU 项,数据+文献核验)

> 回顾 roadmap 的可做项,把**能快速验证、无需大量 GPU**的逐个用数据+文献核验,判定可做/不可做,详述"怎么对比/怎么实现/如何处理",落盘。
> 日期 2026-06-16。配套:[roadmap](CRAVE_roadmap_and_positioning.md) · [方法对比](value_advantage_methods_comparison.md) · [前沿定位](CRAVE_frontier_positioning_and_scenarios.md)。状态持续更新。

## 总览(成本 × 可验证性 × 状态)

| 项 | 成本 | 可快速验证 | 状态 | 一句话结论 |
|---|---|---|---|---|
| B1 基元↔milestone(action) | 低(有action列) | ✅ | ✅**已验** | milestone 比时间多解释 2× 动作方差(R²0.43 vs 0.22)→ 是技能相位;但转移非动作切分 |
| B2 终点可达性+OOD残差→失败信号 | 低(现成模型) | ✅ | ❌**已验·否决** | 对**细微 dagger 失败无效**(残差不分 neg/pos, 终点 corr0.13);只抓粗失败(value已自带) |
| A2 keyframe 提取 | 极低 | ✅ | ✅**可做(平凡)** | milestone 边界=keyframe, 直接导出 |
| A2 失败定位/异常 | 低 | ✅ | ⚠️**部分** | 同 B2: 粗异常可(value掉), 细微不可 |
| A1 子任务切分→prompt | 低(切分)+VLM(命名) | 切分✅/命名需API | ⏳待跑 | 切分零成本; 命名~20次VLM调用 |
| C1 蒸馏分布式头 | **高(GPU训)** | ❌ | ⏸搁置 | 需训练; 且因果-DP已够(corr0.94) |
| C2 冷启+RL | **高(真机rollout)** | ❌ | ⏸搁置 | 要真机/RL |
| D1 增量挖矿 | 中(原型可低) | 部分 | ⏳待设计 | 增量KMeans+漂移监控 |

---

## 已验项详述

### B1 · 基元↔milestone 的动作相关性 ✅(已验)
- **怎么对比**:R²(action | milestone) vs R²(action | 时间分桶,同桶数)。若前者>后者,milestone 捕捉动作相关技能结构而非计时器。
- **怎么实现**:现成 milestone 模型 + kai0_base `action` 列;每帧最近 milestone / 时间桶;组均值预测 action 的解释方差比。脚本 `crave_milestone_action_pilot.py`。
- **结果**:R²(milestone)=**0.43** vs R²(time)=**0.22**(2×);但转移帧动作变化仅 1.10× 非转移 → milestone 是**视觉状态(技能相位)边界,非动作不连续点**。
- **如何处理/含义**:支持"按 milestone 条件化动作"(分层/基元信号);**不支持** keyframe-硬动作切分。→ roadmap C 组(相位条件化 BC)有据;斜坡/动作切分用途弱。

### B2 · 终点可达性 + OOD 残差 → 弱失败信号 ❌(已验·否决)
- **怎么对比**:① 终点可达性=CRAVE 末值 vs AE-neg 率(per-ep);② OOD 残差=帧到最近 milestone 距离,看 AE-neg 帧残差是否显著高于 AE-pos。文献:失败=OOD偏离训练流形([2503.08558](https://arxiv.org/abs/2503.08558)/[2509.26308](https://arxiv.org/abs/2509.26308))。
- **怎么实现**:`mv_value_full`(1117ep 末值)+ 挖 smooth800_dagger 模型算残差。脚本 `crave_b2_failure_signal.py`,图 `crave_b2_failure_signal.png`。
- **结果(否决)**:① 末值<0.7 仅 **2%**,corr(未完成度, AE-neg)=**0.13**(弱);② AE-neg 帧残差 **0.981** vs AE-pos **0.979**(**几乎相同,不可分**),高残差帧里 AE-neg 仅 6%。
- **如何处理/含义**:**细微 dagger 失败是 on-manifold(像 demo 的合理布料态),残差/终点都抓不到**。CRAVE 只能抓**粗失败**(布料被拿走/全脱轨 → value 自己掉,无需 B2)。→ **"廉价补 neg 洞"基本不成立**,确认 CRAVE 无失败信号的根本局限;真 neg 仍需 RL/结果信号(C2)或人标。**roadmap B2 降级为"仅粗失败/OOD场景成功检测"**。

### A2 · keyframe 提取 ✅(平凡可做)
- **怎么实现**:milestone 边界帧(value 阶/最近 milestone 变化处)= keyframe,直接导出,零成本。**可做**,无需专门验证。
- **失败定位/异常**:同 B2——粗异常(value 掉)可,细微不可。⚠️ 部分可做。

---

## 待验/搁置项处理建议
- **A1 子任务切分**:切分零成本(B1 已证 milestone=技能相位);命名需 VLM(~20 次/任务,廉价)。**下一步快速可验**:导出 milestone 段 + 边界帧,接 VLM 命名,喂 AWBC `prompt_from_task`。
- **D1 增量挖矿**:增量 KMeans + milestone 覆盖/漂移监控,原型可低成本。**下一步可设计小实验**:新数据流入时 milestone 中心漂移量度。
- **C1/C2(高 GPU)搁置**:C1 蒸馏因 causal-DP 已 corr0.94 而无必要;C2 要真机 RL。两者非快速可验,留待资源到位。

## 关键诚实结论(本轮)
**两个核验把 roadmap 收窄得更扎实**:① B1 证 milestone=动作相关技能相位(条件化有据);② **B2 否决了"廉价补 neg 洞"**——细微失败 on-manifold,CRAVE 廉价手段抓不到,根本局限确认。→ CRAVE 的落地重心应是**进度/结构/相位条件化 + 成功检测(粗)**,而**不要**指望它替代 AWBC/RL 的细粒度失败判别。
