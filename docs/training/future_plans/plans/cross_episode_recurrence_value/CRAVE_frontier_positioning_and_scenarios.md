# CRAVE 在 VLA / 世界模型前沿中的定位、适用场景与 SOTA 优缺点

> 深度调研 robot VLA + 世界模型 + 价值/奖励学习前沿(2024–2026 论文 + GitHub),定位 CRAVE 的不可替代优势,给场景 × SOTA × 实施方案。
> 日期 2026-06-16。本文档自成体系;CRAVE 方法见 [METHOD](cross_episode_recurrence_value_METHOD.md),与监督/RL value 的机理对比见 [方法对比](value_advantage_methods_comparison.md)。

## 1. 前沿地图(与"价值/奖励"相关的部分)

| 方向 | 代表工作 | 与 value/reward 的关系 |
|---|---|---|
| **VLA 策略 + 自我改进** | π0/π0.5/**π\*0.6-RECAP** ([2511.14759](https://arxiv.org/abs/2511.14759))· OpenVLA · Octo · GR00T · Gemini Robotics · 仓库 [openpi](https://github.com/Physical-Intelligence/openpi) / [LeRobot](https://github.com/huggingface/lerobot) | RECAP 用**分布式 MC 回报 value** 算 advantage 做优势条件 BC,闭合"从经验改进"环 |
| **零训练 value(VLM)** | **GVL** ([2411.04549](https://arxiv.org/abs/2411.04549))· TOPReward ([2602.19313](https://arxiv.org/abs/2602.19313)) | GVL = VLM(Gemini)对**打乱帧做时序排序** → 逐帧完成度;零样本仅需任务文本 |
| **零训练 value(对比/生成)** | VIP ([2210.00030](https://arxiv.org/abs/2210.00030))· LIV ([2306.00958](https://arxiv.org/abs/2306.00958))· ViVa ([2604.08168](https://arxiv.org/abs/2604.08168)) | 预训练一次→零样本逐帧标量进度;ViVa 用视频生成模型 |
| **世界模型(RL-in-imagination)** | Genie Envisioner · Cosmos Policy · iVideoGPT · UniSim · WorldVLA · DiWA / World4RL · Ctrl-World ([2510.10125](https://arxiv.org/abs/2510.10125))· 综述 [2605.00080](https://arxiv.org/abs/2605.00080) | 学到的模拟器里做 RL **必须有 reward/value 信号**(Cosmos Policy 把 value 当 latent frame 编进去) |
| **真机 RL(样本高效)** | **HIL-SERL** / SERL ([2401.16013](https://arxiv.org/abs/2401.16013))· RL-100 | 用少量人标训 reward classifier + SAC,25–50min 收敛;**瓶颈是 reward 来源** |
| **离线 RL / 优势条件 BC** | CRR · AWAC · RECAP | 先有 advantage 再筛/加权数据 |

**共性洞察**:几乎所有"自我改进 / RL / 世界模型"路线的**真正瓶颈都是 value/reward 信号的获取成本**——RECAP 要数千次真机 rollout + 失败标注;HIL-SERL 要人标 reward classifier;world-model RL 要 reward 头。**CRAVE 的全部价值就压在"零标签零训练地提供这个信号"上**。

## 2. CRAVE 的结构性优势(别人都没同时具备的组合)

1. **零训练 + 零标签 + 零 VLM/API**:frozen DINOv2 + KMeans + Viterbi-DP,纯 CPU 可跑,可扩到百万级离线数据。
2. **离散技能结构(milestone 图),非仅标量**:实测 milestone 比时间多解释 **2×** 的动作方差(R² 0.43 vs 0.22)→ 是动作相关的技能相位,不只是计时器。VIP/LIV/TOPReward/GVL **都只给连续标量**。
3. **重复性是信号而非噪声**:GVL 明确在"重复/次优轨迹"上失效;CRAVE 恰恰**靠跨 episode 重复**挖结构 → 在大批同任务 demo 上是优势区。
4. **确定性、可复现、可检视**:无 VLM 随机性;milestone 簇可视化/可审计(簇间流转 2D/3D)。
5. **在线可因果化**:固定滞后 Viterbi,零训练拿到 corr 0.94 的在线 value(频率窗按 fps 标定)。
6. **质量已逼近监督**:kai0_base ep2047 对真 stage_progress_gt corr **0.865**(监督 pi0-AE 0.897),零标注。

**结构性短板(诚实)**:① 需**同任务 demo 集**(不能像 GVL 单视频/纯文本零样本);② 无结果信号(不能区分自信但错的动作,不能超越示教);③ 跨任务/跨本体零样本弱(GVL 强);④ value 是"demo 流形进度"代理,非真回报。

## 3. 最直接对标:CRAVE vs GVL(两个零训练 value,反向取舍)

| 维度 | **GVL**(VLM 在上下文) | **CRAVE**(重复几何) |
|---|---|---|
| 信号来源 | VLM 世界知识 + 帧时序排序 | 跨 episode 统计重复 |
| 输入需求 | **仅任务文本**(可单视频/零样本) | **需同任务 demo 集** |
| 跨任务/跨本体零样本 | **强**(300+ 任务) | 弱(需各域挖矿) |
| 成本 | 大 VLM(Gemini)逐视频 API,贵 | frozen 小模型 + 聚类,**极廉,可扩百万级** |
| 输出结构 | 连续标量完成度 | **离散 milestone 技能结构 + 标量** |
| 重复/次优轨迹 | **失效**(官方列为局限) | **优势区**(重复=信号) |
| 确定性/可审计 | VLM 随机、黑箱 | 确定、可视化 |
| 失败/退步 | 概率式成功检测 | 结构性退步(value 回落) |

→ **不是竞争是互补**:**少样本/零样本/跨任务/语义** 用 GVL;**大批同任务 demo + 要便宜 + 要离散结构 + 要确定可复现** 用 CRAVE。二者可串联(GVL 跨任务冷启 → CRAVE 在目标任务大数据上精修结构)。

## 4. 场景 × SOTA × CRAVE 优缺点(核心)

### 场景 A · 大规模离线数据集 value/advantage 打标(AWBC / 离线 RL)
- **SOTA**:GVL(VLM-VOC + advantage-weighted)、RECAP-value(需 RL)、监督 pi0-AE(需人标)。
- **CRAVE 优**:零标注零训练、可扩百万级(GVL 的 VLM API 成本在大数据上爆炸)、离散结构可直接喂 AWBC `prompt_from_task`、in-dist 质量已逼近监督(0.865 vs 0.897)、对重复 demo 是优势区。
- **CRAVE 劣**:需同任务 demo 集;无失败结果信号(neg advantage 弱)。
- **判定**:**大批同任务离线打标 = CRAVE 主场**(我们的 [AB_plan](awbc_milestone_value_AB_plan.md) 正在验)。

### 场景 B · RL 冷启 baseline V / 世界模型 imagination 的 reward
- **SOTA**:RECAP 分布式 MC value(贵)、world-model 内置 reward 头(Cosmos Policy 把 value 编进 latent)、HIL-SERL 的人标 reward classifier。
- **CRAVE 优**:零成本给一条 dense 进度 value 当**冷启 baseline V**(替代昂贵 MC 预训练 / 省人标 classifier),再用少量真机成败做 advantage 修正 = "低成本可超越示教"路径;在 world-model imagined rollout 里当**零标签 dense reward**。
- **CRAVE 劣**:本身不含结果信号,需叠加稀疏成败(终点可达性/OOD 残差)才完整。
- **判定**:**当 RL/世界模型的廉价 reward 前端**——高杠杆,但需配弱成败信号。

### 场景 C · 子任务/技能分割(分层策略 / 世界模型子目标 / AWBC 子任务 prompt)
- **SOTA**:GVL 进度(连续,无边界)、UVD/AWE 类切分(多需训练或逐任务调)。
- **CRAVE 优**:milestone = 跨 episode 一致的离散技能相位(R² 证明动作相关),**帧精确 + 跨 episode 一致**(VLM 时序定位弱、逐次不一致);零训练免费切分 → 喂分层 / world-model 子目标 / AWBC prompt。
- **CRAVE 劣**:边界是**视觉状态**边界,非尖锐动作不连续(转移仅 1.10× 更 eventful)→ "keyframe/硬基元"用途弱,"技能相位条件化"用途强。
- **判定**:**零标签一致技能相位划分 = CRAVE 独占**(GVL 给不了离散一致边界)。

### 场景 D · 数据 curation / 质量过滤 / 去重
- **SOTA**:GVL 的 VOC(Value-Order Correlation)分数过滤数据集。
- **CRAVE 优**:milestone 覆盖/到达终点率 = 免费质量分,且**极廉可全量扫**(GVL VOC 要逐集 VLM);milestone 表示天然支持去重/检索。
- **CRAVE 劣**:语义判断不如 VLM(如"这条 demo 是否做对了任务"GVL 更懂)。
- **判定**:**大数据廉价初筛用 CRAVE,语义精筛用 GVL**(互补)。

### 场景 E · 成功检测 / OOD / 失败定位
- **SOTA**:GVL-SD(0.75 一致检测)、专用 success classifier、HIL-SERL reward classifier。
- **CRAVE 优**:到终止 milestone 簇 = 成功代理;到最近簇残差大 = OOD/异常;value 卡在 milestone k = **卡在第 k 个技能相位的失败定位**(可归因,GVL/classifier 只给"分低")。
- **CRAVE 劣**:无语义,细粒度成功判定(如"叠得齐不齐")不如 VLM。
- **判定**:**可归因的相位级失败定位 = CRAVE 强**;语义成功判定用 GVL。

### 场景 F · 真机 RL 的实时进度监控 / reward shaping(HIL-SERL 类)
- **SOTA**:HIL-SERL 的人标 reward classifier。
- **CRAVE 优**:在线固定滞后-DP 零训练给 dense 进度 reward(替代人标 classifier 的标注),corr 0.94、零滞后、可因果。
- **CRAVE 劣**:dense 进度 reward 是"接近示教流形"代理,纯 sparse 成败仍需另给。
- **判定**:**省掉 reward classifier 的人标**——直接接 HIL-SERL/SERL 当 dense shaping。

## 5. 实施方案(按杠杆排序)

1. **[最高]离线打标管线(场景 A)**:CRAVE 全量挖 milestone + value/advantage → 喂 AWBC discretize → `prompt_from_task`。已就绪([AB_plan](awbc_milestone_value_AB_plan.md))。脚本 `smooth800_v24_full.py` / `hdf5_v24_eval.py`。
2. **RL 冷启 + 弱成败(场景 B)**:CRAVE value 当 baseline V + 终点可达性/OOD 残差当弱 reward → 接 HIL-SERL / world-model imagination。需补"终点可达性"小模块(零训练)。
3. **技能相位 → 分层/子任务(场景 C)**:milestone 边界切段 → VLM 命名 → 分层策略 / world-model 子目标 / AWBC prompt。需一个 VLM 命名小步(每任务 ~20 次调用,极廉)。
4. **在线监控/shaping(场景 F)**:固定滞后-DP(已验证)→ HIL-SERL dense reward。
5. **数据 curation(场景 D)**:milestone 覆盖/到达率全量扫 → 廉价初筛,GVL 精筛。

## 6. 一句话定位
**在"价值/奖励信号获取成本"这个全行业瓶颈上,CRAVE 占住的是「大批同任务 demo + 要极廉 + 要离散技能结构 + 要确定可复现」这一格**——GVL 占「少样本/零样本/跨任务/语义」,RECAP 占「能超越示教但最贵」。CRAVE 不与它们正面竞争标量精度(in-dist 已 0.865 逼近监督),而是做**别人贵到做不起的大规模零标签结构化前端**,并可作 RL/世界模型的廉价冷启 reward。其最该补的洞是**结果信号**(终点可达性/OOD),补上即从"离线 labeler"升级为"能给 RL 冷启的进度引擎"。
