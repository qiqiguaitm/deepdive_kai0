# milestone 概念再审视 + 普适 "milestone+1" 探索

> 跳出 CRAVE,从本质上问两件事:① "milestone = 跨 episode 反复出现的状态" 这个定义合不合理?② 有没有不依赖 CRAVE 的、更普适的 milestone+1 发现方式?结合文献 + 叠衣场景。日期:2026-07-04。

## 1. "recurrence = milestone" 是**频率定义**,不是**功能定义** —— 这是根源问题

你的理解(跨 episode 反复出现的场景 = milestone)是一个**便宜的启发式**(零训练、不需要 reward/标签),且和真 milestone 相关。但它把**频率**当**功能**,这正是我们一路踩的坑的根:

- **aliasing**:反复出现的**视觉**态 ≠ 任务阶段。手臂划过同一区域、"平摊"≈"折好"→ 同一 milestone 落在多个任务相位(greedy≠maxprod 分析:50% 横向/后退)。
- **频率 ≠ 重要性**:一个态反复出现,可能只因它是**常见过渡/停留态**(手悬停),不因它是**任务关键子目标**。
- **描述性 ≠ 功能性**:recurrence 只告诉你"哪些态常见",不告诉你"哪些态是决策点/瓶颈/该当子目标"。

## 2. 更合理的 milestone 定义:三个**功能性** notion(文献)

| notion | 定义 | 抓住的本质 | 相对 recurrence 的升级 |
|---|---|---|---|
| **① 瓶颈 bottleneck** | 出现在**很多成功轨迹、却很少在失败轨迹**里的态(betweenness 中心性;McGovern-Barto) | **任务必经 + 判别成功/失败** | recurrence≈频率;bottleneck≈**必要性**。常见停留态频率高但不是瓶颈;真子目标是瓶颈 |
| **② 变点 change-point / 事件边界** | **动力学 regime 突变处**(接触建立/断开、prediction-error 尖峰);认知科学的"事件分割" | milestone 是**边界**不是态 | recurrence 找"态",change-point 找"**相位转换点**" |
| **③ 承诺/不可逆 commitment** | 越过后**几乎不回退**的点(转移概率不可逆性) | **已提交的进度里程** | recurrence 无方向性;commitment 编码"**过了就回不去**" |

**综合定义(比 recurrence 强)**:milestone = 一个**(a) 对成功必要(瓶颈、判别失败)+ (b) regime 变点/不可逆承诺 + (c) 对剩余任务有预测力** 的状态/边界。**recurrence 只是弱代理**——它碰巧和这三者相关,但也会误收常见非 milestone 态。这一句话直接解释了我们所有 aliasing 现象。

## 3. 落到叠衣场景:最物理的 milestone = **接触/拓扑事件**

双臂叠衣的最本质 milestone = **contact / cloth-topology 事件**(抓取 grasp、提起 lift、压出折痕 crease-made、松手 release):
- 是**瓶颈**(每条成功都经过)、是**变点**(接触 regime 变)、是**不可逆**(成功里折好的很少再摊开)。
- recurrence 只是近似它们,还会在"手悬停/整理"这些常见非事件态上误触发。
- **杠杆**:给特征加 **proprio + 接触/触觉**信号 → milestone 从"视觉反复态"变成"接触事件",一举打掉视觉 aliasing(这也是 §6.1 proprio 修起末别名、ceiling 分析建"相位唯一 milestone"的同一方向)。

## 4. 普适 "milestone+1":**离散、功能化的 latent world model**(subsume CRAVE)

CRAVE = recurrence-KMeans(码本)+ 经验转移图(下一步)——两处都是**频率**选择。普适化 = 把这两处换成**学习出来、按功能优化**的版本:

| CRAVE(频率) | 普适(功能,学习) | 文献 |
|---|---|---|
| KMeans 聚"反复态" | **VQ 离散瓶颈**:码本按"**可预测未来 + 对 value/目标充分**"学出,码=milestone | VQ-VAE / 离散 latent world model;HiMaCon(无标注挖层级操作概念) |
| 经验转移图取 next | **学习的高层转移/下一码模型**(skill-token 的 next-token / latent world-model 前向) | latent world model 的 next-latent 预测(zero-shot 控制) |
| 连续段压缩 | **变点/termination 分割**:按 prediction-error/终止条件切段,段=功能技能 | change-point HRL(2025)、NBDI termination(2025)、BUDS/LOTUS 无分割挖技能 |
| 覆盖率选簇 | **瓶颈 + 成功/失败判别加权**(需要一点失败数据或成功信号) | bottleneck / betweenness |

**收敛的普适框架**:一个**带离散 skill/milestone 瓶颈的层级 latent world model** ——
- **milestone = 离散码**,学来"可预测 + 必要"(非"反复");
- **milestone+1 = 高层转移模型预测的下一码**(next-token / world-model rollout);
- 用**变点分割 + 瓶颈/失败判别**兜功能性。

**这把 CRAVE 收编为"零训练特例"**(KMeans≈未训练的 VQ、经验图≈未训练的转移),而 **LMWM 已是第一步**(它学了转移 + latent subgoal)。真正的普适升级 = 让**离散化本身也是学习+功能化的**(VQ world model),而非 recurrence 聚类。

## 5. 便宜首验(按 ROI 排)
1. **加 proprio/接触信号重定义 milestone**(最便宜、最对症):看 aliasing(greedy≠maxprod 分歧率、50% 后退)是否骤降 → 验证"接触事件 > 视觉反复"。
2. **瓶颈打分**:用成功(+少量失败/OOD)算每个候选 milestone 的"成功轨迹经过率 − 失败经过率",替代覆盖率选簇 → 看是否剔掉常见非 milestone 态。
3. **VQ 离散瓶颈 vs KMeans-recurrence**:同数据训一个小 VQ(码本按预测未来+progress 充分学),比 milestone 的相位唯一性 / greedy≈maxprod 一致率 → 验证"学习码 > 频率簇"。
4. milestone+1:在 2/3 的码上训小转移模型(next-code),对真实未来评(top-k/NLL),对照 LMWM。

## 6. 一句话
- **Q2**:"反复出现 = milestone" 是**能用的便宜代理,但不是本质**;本质是**必要性(瓶颈)+ 变点/不可逆 + 预测力**。叠衣里最本质的 milestone 是**接触/拓扑事件**。
- **Q1**:普适 milestone+1 = **带离散功能化瓶颈的层级 latent world model**(VQ 码=milestone、学习转移=milestone+1、变点分割、瓶颈加权),CRAVE 是其零训练特例,LMWM 是第一步。

## 参考
- 瓶颈/技能发现:[Bottom-Up Skill Discovery (BUDS)](https://arxiv.org/pdf/2109.13841) · [LOTUS](https://arxiv.org/html/2311.02058v3) · [Open-World Skill Discovery](https://arxiv.org/pdf/2503.10684)
- 变点/终止:[Change-Point HRL (2025)](https://arxiv.org/pdf/2510.24988) · [NBDI termination (2025)](https://arxiv.org/pdf/2501.12668) · [Free-Energy 子目标](https://arxiv.org/pdf/2412.16687)
- 层级概念/latent world model:[HiMaCon (2025)](https://arxiv.org/pdf/2510.11321) · [Hierarchical Planning with Latent World Models (2026)](https://arxiv.org/html/2604.03208v1) · [World Models for Manipulation 综述 (2026)](https://arxiv.org/pdf/2606.00113)
