# 跨 Episode 重复度挖掘 → 自动 Milestone / Value(AWBC 标签升级)— 调研 + 方案

> **核心假说(用户,2026-06-11)**: 同任务多条 episode 中**反复出现的图像/状态 = 任务必经过程(milestone/bottleneck)**;**低重复度的图像 = 非必要操作甚至 error/negative 样本**。据此从跨 episode 结构挖 value,替代/增强现有 AWBC 的逐帧进度回归。
> **状态**: 📋 调研完成(2026-06-11,deep-research 104 agents / 22 源 / 25 claims 三票核验,21 confirmed / 4 killed)→ **假说前半成立(强先例),后半须软化**;V0 探针脚本已建(`train_scripts/kai/data/recurrence_v0_probe.py`),待跑。
> **上游**: AWBC pipeline([`../../../deployment/strategy/awbc_implementation_plan.md`](../../../deployment/strategy/awbc_implementation_plan.md));ViVa 对比([`awbc_viva_value_comparison_plan.md`](awbc_viva_value_comparison_plan.md),其 DSM-r30 变体**手标** milestone——本方案目标之一是自动挖出来)。
> **动机(现有 pipeline 的病根)**: pi0-AE 是单帧视觉回归器,`absolute_advantage = V(t+50)−V(t)` 二阶差分把噪声放大(corr 0.896→0.3-0.4);且完全**不利用跨 episode 结构**。另:AE 训练数据(kai0_advantage)在完成瞬间截止、无收尾段 → vis episode 尾段 value 系统性下坠(已实证,见 end-drop 分析)。

---

## 1. 调研结论(全部三票核验,引用见 §6)

### 1.1 假说前半("重复 = 必经")— ✅ 有 25 年直接先例

- **McGovern & Barto (ICML 2001)** 字面形式化了这个假说:bottleneck = "在成功路径上频繁经过、失败路径上不经过的观测区域",目标概念"在**每条**成功轨迹上都出现"。diverse density(多示例学习)挖掘,gridworld(找到门口)+ 连续状态机器人验证有效。
- 后续脉络:L-Cut(2004,统计化 recurrence 判定)→ betweenness centrality(NeurIPS 2008)→ 2025 HRL survey 确认为公认 subgoal 发现准则。

### 1.2 假说后半("稀有 = negative")— ⚠️ 文献明示的脆弱半边

- **McGovern & Barto 2001 原文就警告**:有用的子目标也会出现在稀有/失败路径上 → 负证据必须**软化**(Gaussian 宽度 / 按 bag 分级),不能硬性"出现在负包即排除"。
- 经典 diverse density **需要失败轨迹作负包**;我们 800-3000 条全成功 → "稀有=negative"在最强先例里**没有形式化对应**。
- 现代侧唯一先例 = TCC 论文的异常检测提议("嵌入轨迹偏离典型轨迹的帧标为异常")——**仅 1 个定性例子(卧推视频),无定量基准,从未当 negative 标签用于 BC**。
- **自家数据里的反例**:抓角失败后的 **regrasp 恢复动作**是低频的,但正是我们最想要的能力(XVLA 抓不到角问题)。硬标 negative = 删掉恢复能力。
- → **结论:低 recurrence 段只能软降权 / 第三档 "uncertain" prompt 标签 + 人工审计,绝不自动硬标 "Advantage: negative"。**

### 1.3 四条方法学教训(决定实现细节)

| # | 教训 | 出处 | 对我们的含义 |
|---|---|---|---|
| 1 | **裸帧频计数不行**:every-visit 被停留时长主导("agent 大部分时间在房间里,极少在门口") | McGovern 2001 | 30Hz 视频慢段会霸占帧数,抓角/对折等关键瞬间反而帧少 → **first-visit:每 episode 每状态簇只计一次** |
| 2 | recurrence 判定要**双阈值统计**(Binomial:出现 episode 数 > t_o 且 hit 比例 > t_p) | L-Cut 2004 | 抗噪的 milestone 接受准则 |
| 3 | milestone = recurrence 的**局部峰值**(相对时间邻域),非全局阈值 | Betweenness 2008 | Rooms 域峰值在门口"两侧"而非门口本身 — 按邻域比较选峰 |
| 4 | 离散图方法在连续 RGB 上**不 scale** | HRL survey 2025 | 学习的 embedding 层是前提,裸像素/哈希不可行 |

### 1.4 现代机器:TCC → XIRL → GraphIRL(推荐采用线)

| 工作 | 提供什么 | 关键证据 |
|---|---|---|
| **TCC** (CVPR'19) | 逐帧"共性分数"现成机制:cycle-consistency(帧的软最近邻映射回自己 = 公共路径;误差大 = 稀有/绕路候选)。进度信号 Kendall τ **0.75 vs TCN 0.66**(from scratch) | 3-0 |
| **XIRL** (CoRL'21) | **端到端配方**:TCC 跨 episode 对齐(零标注)→ **value = 嵌入空间到 goal 帧的负距离**。明确消除"对单条参考轨迹对齐"(= 我们逐帧回归器的病)。代码开源 | 3-0 |
| **GraphIRL** (CoRL'22) | **治布颜色 nuisance**:先抽象掉外观(纹理)再在抽象空间对齐 → 对"同任务、外观多样视频"鲁棒。借**原则**(先抽象再对齐)不借实现(它是刚体物体图;布用分割 mask 形态/DINO 特征) | 3-0 |

**对现有 AWBC 的核心收益**:对齐相位 value **天然单调** → advantage = 相位推进速率,**结构上消除二阶差分崩塌**(0.896→0.3 那个),比省标注更有价值。

### 1.5 被否的捷径与覆盖缺口(诚实标注)

- ❌ 0-3 否决:"标 1 条参考 episode 经 TCC 传播 ≈ 50 条全标视频" — **别按此预算**。
- ❌ 1-2:betweenness 加速效果的"随机子目标对照归因" — 加速是真的,归因到"共性子目标"未坐实。
- **覆盖缺口**:VIP/LIV/R3M 视频 value 预训练、AWE waypoint、ILEED 示范加权三块**无幸存核验 claim**,本结论不依赖它们(自读时留意)。
- 所有实证来自刚体/仿真/gridworld,**无可变形双臂布操作先例**;"recurrence→自动milestone→AWBC标签"完整链**没人发表过** = 风险 + 可发表贡献点。

---

## 2. 失败模式与缓解(预注册)

| 失败模式 | 机理 | 缓解 |
|---|---|---|
| **多策略叠法** | 两种合法折法把 recurrence 劈成两半,各自都"不常见" | 先按整体轨迹嵌入聚类成"策略模式",**按模式分别对齐/挖掘** |
| **稀有恢复动作误杀** | regrasp/纠错低频但宝贵 | 软负 + 人工审计 bottom-decile 段(V0 必做项) |
| **布外观多样性**(白/蓝/米) | 视觉聚类被颜色/纹理主导 | GraphIRL 原则:先抽象(布分割 mask 形态描述子 / DINOv2 语义特征);固定相机视角是优势 |
| **TCC 单调相位假设** | 重复子动作/非单调顺序使对齐失真(LAV/GTCC 已证) | 先 V0 验证;必要时换 soft-DTW 类对齐或分段对齐 |

---

## 3. 方案

### 3.1 V0 探针(1-2 天,先证伪/证实再投入)— 脚本已建

`train_scripts/kai/data/recurrence_v0_probe.py`:

1. 抽 ~50 episode(默认 `A_new_smooth_800/base`,top_head 相机)× 3Hz 降采样;
2. 冻结 **DINOv2-small** 抽帧特征(L2 归一);
3. 全库 KMeans(k≈48)→ 每簇 **episode 覆盖率(first-visit)**;
4. 输出:
   - `coverage_curve.png`:簇覆盖率 vs 簇平均时间位置 → **看峰值是否对上直觉 milestone**(抓角/第一折/第二折/完成);
   - `milestone_clusters.png`:高覆盖簇的代表帧网格(肉眼判 milestone 语义);
   - `low_coverage_segments.md` + 缩略图:**bottom-decile 低覆盖段清单 → 人工审计:真错误还是 regrasp 恢复?**(决定假说后半生死);
   - `per_episode_timeline.png`:每 episode 时间线按所属簇覆盖率着色(低覆盖段一眼可见)。

**V0 判据**:
- 覆盖率峰对上直觉 milestone → 前半成立,进 V1;
- 低覆盖段多为恢复动作 → 后半只能做软降权(预期如此);多为真错误 → 可更激进;
- 簇被布颜色主导(同色聚一起而非同阶段聚一起)→ 先解决抽象层再谈对齐。

### 3.2 V1 正式 pipeline(~2-3 周,V0 通过后)

| 步 | 内容 | 依据 |
|---|---|---|
| a | 全量 2-4Hz 降采样(1000ep×3cam ≈ 0.2-0.4M 帧,数 GPU 时) | — |
| b | 外观鲁棒特征:DINOv2 + 布分割 mask 形态 | GraphIRL 原则 |
| b' | **策略模式预聚类**(整体轨迹嵌入 → k-means),按模式分别处理 | §2 失败模式 1 |
| c | 改 XIRL 开源码训 TCC head(按相机) | XIRL |
| d | recurrence(帧)= 对 K≈20 条随机参考 ep 的软对齐一致性;first-visit 计数;Binomial 双阈值 | 教训 1/2 |
| e | **自动 milestone = recurrence 局部峰值** → 替代 ViVa-DSM 手标 | 教训 3 |
| f | **V(t) = milestone-index / 连续对齐相位**(单调)→ 喂回现有 `discretize_advantage.py`(AWBC 训练侧零改动) | XIRL |
| g | 低 recurrence 段 → 软降权 / "uncertain" 第三 prompt 标签 + 审计 | §1.2 |

### 3.3 评估(沿用项目铁律)

- offline:新 value 与 `stage_progress_gt` 的 corr(kai0_advantage 上可直接验,有 GT);差分后的 advantage corr 是否优于 0.3-0.4 基线;
- **真机为终判**:新标签训出的 AWBC vs 现 pi0-AE 标签版(同 init/同配方,单变量=标签来源,沿用 awbc_viva 对比框架 §1)。

---

## 4. 参考文献(按阅读优先级;全部经 3 票核验引用)

**第一梯队(必读)**
1. Dwibedi et al., *Temporal Cycle-Consistency Learning*, CVPR 2019 — [arXiv:1904.07846](https://arxiv.org/abs/1904.07846)(§3 cycle-consistency、Table 6 进度信号、Fig.7 异常检测)
2. Zakka et al., *XIRL: Cross-embodiment Inverse RL*, CoRL 2021 — [arXiv:2106.03911](https://arxiv.org/abs/2106.03911)(TCC 对齐 → 负距离 value;开源)
3. McGovern & Barto, *Automatic Discovery of Subgoals using Diverse Density*, ICML 2001 — [PDF](https://mcgovern-fagg.org/amy_html/old/pubs/mcgovern_barto_isairs2001.pdf)(假说原始形式化 + first-visit + 软负警告)

**第二梯队(实现前)**
4. Kumar et al., *GraphIRL*, CoRL 2022 — [arXiv:2207.14299](https://arxiv.org/abs/2207.14299)(先抽象再对齐,治外观 nuisance)
5. Şimşek, Wolfe & Barto, *L-Cut*, UMass TR 2004 — [PDF](http://all.cs.umass.edu/pubs/2004/simsek_wb_TECH04.pdf)(Binomial 双阈值 recurrence 判定)
6. Şimşek & Barto, *Skill Characterization Based on Betweenness*, NeurIPS 2008 — [PDF](https://proceedings.neurips.cc/paper/2008/file/934815ad542a4a7c5e8a2dfa04fea9f5-Paper.pdf)(milestone = 局部峰值)

**第三梯队(背景)**
7. Klissarov, Bagaria, Konidaris, Precup, Machado et al., *HRL Survey*, 2025 — [arXiv:2506.14045](https://arxiv.org/abs/2506.14045)
8. VIP — [arXiv:2210.00030](https://arxiv.org/abs/2210.00030);LIV — [arXiv:2306.00958](https://arxiv.org/abs/2306.00958)(视频 value 对照线,未经本轮核验)
9. AWE — [arXiv:2307.14326](https://arxiv.org/abs/2307.14326);Keyframe-Focused IL — [arXiv:2106.06452](https://arxiv.org/abs/2106.06452)

**开放问题(也是贡献点)**
- 有没有任何已发表工作把对齐误差当 negative/降权标签用于 weighted BC?(本轮调研:没有 → 空白)
- 我们数据里有几个策略模式?低覆盖段错误 vs 恢复的真实占比?(V0 回答)
