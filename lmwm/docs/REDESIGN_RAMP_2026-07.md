# RAMP:milestone 预测为 VLA 赋能 —— 跳出 LMWM 的重设计方案

> 日期 2026-07-08。定位:**跳出 LMWM 现架构**,从基础思想 + milestone 预测任务的本质特性 + 三路深度文献调研(subgoal/milestone 预测、WM-for-VLA、value/进度表征,2022→2026)出发的重设计。
> RAMP = **R**etrieval-**A**nchored **M**ilestone **P**rediction(检索锚定的 milestone 预测)。
> 与现架构的对比与定案见 §5,实验计划见 §6。

---

## 1. 回归基础思想:我们在赌什么,2026 文献怎么裁决

**赌注(2026-07-01 原始规划)**:给 VLA 的提示应是**价值层的 milestone 计划**,而非通用未来帧 —— milestone 是低熵瓶颈态,比固定时距未来帧对行动更有指导性。

### 1a. 文献对赌注本身的裁决:✅ 方向被独立验证,且生态位仍空白

| 证据 | 结论 |
|---|---|
| **VISTA** (2602.10983) | 世界模型只出 **keyframe 级 milestone 序列**(目标图⊕文本子任务),OOD SR 14%→69%;理由与我们同源:"瓶颈态低熵、状态不变性抑幻觉" |
| **DreamVLA** (2507.04447) | 预测**压缩的任务相关抽象**(动态区/深度/语义)> 预测整帧 |
| **LBP** (2505.06861) | 在**冻结 SigLIP 空间**做隐子目标;粗任务对齐子目标 > 密集短时预测;**反向**(从终局倒推)预测误差不随 horizon 累积 |
| **LaWAM** (2606.15768) | 冻结 VFM 特征上的 latent subgoal 条件 VLA = LIBERO 98.6%、24× 快于像素 WM —— 我们的大方向已是 SOTA 配方 |
| **生态位空白** | 所有 latent-WM 竞品(LaWAM τ=1.2s、FLARE、V-JEPA-2-AC、VPP)全是**固定时距**;"以语义 milestone 定义 horizon 的 WM"**无任何已发表系统** —— 这是我们真正的差异化 |

### 1b. 文献对我们**实现方式**的裁决:⚠️ 三张黄牌

1. **纯生成式 latent 回归是最弱的子目标接口**(HIQL 2307.11949 / OpenHelix 2505.03912 / GHIL-Glue 2410.20018 三方证据):无 grounding 辅助损失会塌缩成静态指令码;生成物 off-manifold 需要过滤器兜底。
2. **层级系统的瓶颈常在"接口"而非"生成质量"**(GHIL-Glue):采 K 候选 + 进度分类器过滤(负例含**时间反序对**)+ 当前帧/目标帧**去同步增广**,+25~53% SR。我们把全部力气花在生成侧,接口侧为零。
3. **唯一裁决是下游 SR,我们从未测**。所有 intrinsic 定案(center_w=0.1、teacher=proto、MDN K=4)都可能被 SR 重排。

### 1c. 考古发现的两处内伤(重设计的直接动机)

- **静默漂移**:原始规划是"离散 milestone 计划 + 转移图 + 检索 prototype 接口",逐步漂成"连续生成 SigLIP grid"—— **越来越像 LaWM 本身**,把 milestone 特有的离散/价值/检索结构丢掉了。
- **搁置的裁决**:`archive/next_milestone_vla_validation_plan.md`(2026-07-03)早已写好 GT-first + kill criteria + "检索真实帧规避 GHIL-Glue 失败模式" 的决定性验证 plan,**与今天三路文献调研的结论高度吻合,但被搁置从未执行**。

**重设计 = 把漂移拉回本真(离散+价值+检索)+ 把搁置的 SR 裁决放回第一优先级。**

---

## 2. milestone 预测任务的本质特性 → 设计原则

这是本文档的核心推理:**架构必须由任务特性决定**,而不是沿用"世界模型=预测未来特征图"的默认形状。

| # | 任务特性 | → 设计原则 | 依据(文献 + 自家实验) |
|---|---|---|---|
| **T1** | **假设空间有限且已知**:milestone 是挖出来的 bank(M≈15–50/任务),不是开放的未来 | 用**分类/检索 over bank**,不做自由生成 —— 有界假设空间**构造性消除幻觉** | B2FF 2606.09258(milestone bank+选择);GHIL-Glue 的核心失败模式(逼真但不可行的生成)对检索**天然免疫**;自家检索解码器经验 |
| **T2** | **未来多模态但模式数少**:单帧条件下真实分支 ~2 峰,top1 天花板 ~0.28–0.42(kNN 探针=信息论上限,非容量问题) | **校准的 categorical top-K + 外部过滤**,不堆容量硬拟合;不确定时把决策权交还策略 | 自家容量消融全平(260M deploy +0.00);QueST 2407.15840 / VQ-BeT 2403.03181(VQ 分类>MDN,免组件塌缩);GHIL-Glue sample-K-then-filter |
| **T3** | **价值单调是 milestone 的定义性质**(CRAVE 挖矿的构造) | value 必须**上到测试时**:候选过滤(value-forward 门)+ 进度停滞才重发 —— 而不只是训练标签 | GHIL-Glue 过滤器(时间反序负例 ≡ 我们的单调性);TaKSIE 2410.11013 / Anticipation-VLA 2605.01772(进度门控重发);V-GPS 2410.13816(value 测试时重排) |
| **T4** | **消费者是 VLA,不是人眼** | payload 必须 **on-manifold + 空间结构化 + 与策略同塔**;评估只认 **action-MAE/SR**,grid-cos 降级 | LaWAM 消融(空间结构化子目标>pooled token);2605.06388(语义空间>重建空间);OpenHelix(grounding 辅助损失 3.45→4.01 是最大单杠杆);grid-cos 已被自家判"不敏感" |
| **T5** | **身份与外观解耦**:milestone 身份跨 episode 共享,外观属于当前 episode | 身份=分类头;外观=**以当前状态为 key 的 episode-上下文检索**,不用全局簇中心 | 自家:episode-medoid 目标 > 全局中心(0.864>0.832,kNN 上界 0.877);forward-from-current > absolute(oracle 0.935>0.82) |
| **T6** | **horizon 是事件不是时距**:milestone 达成才该推进 | **事件驱动调度**:价值越过锚点→推进 m+2;进度停滞 τ 秒→重发/重选 | 生态位空白(§1a);LBP 反向一致性;B2FF 失败恢复选锚 |

一句话:**milestone 预测本质上是"带价值序的有限集合上的检索问题",不是"连续特征场的生成问题"。** 现 LMWM 用生成范式解检索问题 —— 这是从任务特性出发得出的最深一刀批评。

---

## 3. RAMP 架构

### 3a. 数据流

```
在线(全部在 π0.5 SigLIP 同塔空间,冻结 ~400M,零新增编码开销):
  帧 224² → SigLIP grid G_t (16×16×1152) + gist g_t (1152)

  ① 身份头(~0.3M MLP):categorical p(next_ms | g_t, prev_ms, proprio, t_bin)
       — 弃 MDN;prev_ms/proprio/t_bin 是自家探针实测 +11/+5.2/+3.5pt 的正交信息
  ② 价值头(~0.1M):v(g_t) ∈ [0,1] 单调进度(CRAVE 蒸馏,SARM 式 stage-classify-then-regress)
  ③ 门控(零参数):top-K 候选 × value-forward 过滤(v(候选)>v(g_t)+δ)× 熵门
       — 高熵/低置信 → 不注入子目标,交还 π0.5 语言通道(不硬塞错误提示)
  ④ 检索 payload(零参数,bank 索引):
       对选中的 m*,在其簇内以 g_t 为 key 做 episode-上下文最近邻 → 取真实帧
       → 该帧的 SigLIP grid G_sub(16×16×1152) —— on-manifold 由构造保证
  ⑤ (可选)残差 adapter(~5M,零初始化):G̃_sub = G_sub + f(G_t, G_sub)
       — 仅当 Phase 0 显示"跨 episode 外观错配"真的伤 SR 才加;这是现生成器的降级续命位
  ⑥ 注入 π0.5(LMWAM v2 plan 的接线,升级):
       G_sub(可 4× 下采样 → 64 token)进 prefix 新视觉槽
       + type-embedding 区分 + 零初始化投影 + KI stop-grad + 去同步增广 + subgoal-dropout
       对照臂 = FLARE 式(2505.15659):无显式 token,只加"未来 token 对齐 milestone 嵌入"辅助损失

离线(label 工厂,不在线):
  CRAVE 挖矿(per-task)
  + UVD(2310.08581)反向单调性交叉验收 milestone 边界
  + bank 质量验收标准(新增,治 vis bank id3=0.11 无验收就上桌的教训):
      簇内纯度 / value 跨度 / 检索键区分度 三关卡

调度(事件驱动,T6):
  v(g_t) 越过 m* 锚点 → 推进预测 m+2;进度停滞 > τ → 重发或重选候选
```

### 3b. 部署参数对比

| | LMWM(现) | RAMP |
|---|---|---|
| 在线新增参数 | 34M(MDN 3.3M + 生成器 30.3M) | **~0.5M**(+可选 adapter 5M) |
| bank 依赖 | 中(teacher 查表) | 高(payload 直接来自 bank;索引=内存非参数) |
| 注入 π0.5 | 未接 | 设计核心(双臂:prefix-KV / FLARE-aux) |

### 3c. 明确不采纳的文献建议(基于自家已有实验,防止重复踩坑)

- **"预测器换 ViT"**(DINO-WM 等全用 ViT):自家 H7 决定性实验已证**空间 token 对身份分类无增益**(grid vs pooled A≈B),gist vs grid 预测器输入打平(Δ0.0003)。ViT 只在"输出必须是 grid"时才需要 —— RAMP 的 grid 来自检索,预测器只需出 categorical,**小 MLP 即可**。文献建议在此被自家证据否决。
- **"上 diffusion/flow 子目标头"**:自家实测 flow best-of-16 0.834 < 回归 0.872,MHP/MCL deploy 反降。多模态上限来自单帧条件的信息论天花板,不来自头的表达力;top-K+门控是对的处理方式。
- **"换 V-JEPA/DINO 目标空间"**(2605.06388 说 V-JEPA 2.1 最强):同塔 SigLIP 的 FLARE 逻辑(子目标活在策略正在读的空间)+ 自家 E2 实测(SigLIP 同塔 deploy 0.716 > DINOv3-H 0.694)优先;可留一个 LaDi-WM 式双空间辅助损失作远期消融,不换塔。

---

## 4. 三路调研浓缩(20 篇核心证据索引)

| 主题 | 论文(arXiv) | 对 RAMP 的贡献 |
|---|---|---|
| 检索>生成的接口 | GHIL-Glue 2410.20018 | 过滤器设计(时间反序负例)、去同步增广、"接口是瓶颈" |
| bank+选择 | B2FF 2606.09258 | milestone bank 预生成 + 可恢复性选择 |
| milestone 序列 WM | VISTA 2602.10983 | keyframe 级预测抑幻觉、⊕文本子任务 |
| 冻结空间隐子目标 | LBP 2505.06861 | SigLIP 空间验证、反向规划、粗>密 |
| 零成本对照臂 | FLARE 2505.15659 | 辅助对齐损失,无推理开销,+26% |
| 隐子目标要 grounding | OpenHelix 2505.03912 / HIQL 2307.11949 / LCB 2405.04798 | 辅助损失防塌缩;latent>语言的度量精度 |
| 进度门控重发 | TaKSIE 2410.11013 / Anticipation-VLA 2605.01772 | 事件驱动调度 |
| value 测试时用 | V-GPS 2410.13816 / GVL 2411.04549 | 重排/过滤;GVL=CRAVE 的零训 baseline |
| 校准多模态头 | QueST 2407.15840 / VQ-BeT 2403.03181 | categorical > MDN |
| 挖矿对标 | UVD 2310.08581 / InfoCon 2404.10606 / SARM 2509.25358 | 边界交叉验收 / 信息量筛选 / 阶段-回归结构 |
| 注入方式 | DreamVLA 2507.04447 / DUST 2510.27607 / LaWAM 2606.15768 | 分块注意力保护 action token;空间化>pooled;KI/零初始化 |
| 视觉 CoT / 子目标图 | CoT-VLA 2503.22020 / SuSIE 2310.10639 / Seer 2412.15109 | 长程 +10~17%;GT goal-image OOD +40% |

---

## 5. RAMP vs LMWM 现架构:逐维对比与定案

### 5a. 逐维对比(诚实版:LMWM 赢的维度照写)

| 维度 | LMWM(现) | RAMP | 谁优 / 证据 |
|---|---|---|---|
| 子目标**身份** | MDN K=4 连续码(proto 蒸馏) | categorical top-K + 熵门 | **RAMP**:校准、免 MDN 塌缩;union_ce 本就是自家 in-dist 实测最强锚(0.770) |
| 子目标**外观 payload** | 生成(AdaLN 渲染到当前画布) | 检索真实帧 grid(episode-上下文 NN) | **各有胜场**:LMWM 保当前画布外观(0.935>0.82 是自家最硬的生成侧证据);RAMP 保 on-manifold(GHIL-Glue)。**只能 SR 裁决 → Phase 0 双臂** |
| off-manifold 风险 | 有(blob 伪影靠"仅可视化"豁免,未经消费端验证) | **构造性为零** | **RAMP** |
| value 的用法 | 仅训练标签(价值单调 milestone) | 训练 + **测试时门控/过滤/重发** | **RAMP**;这才兑现"价值层 WM"的原始承诺 |
| horizon | milestone(名义),推进无门控 | 事件驱动 + 进度门 | **RAMP**(也是论文生态位主张) |
| VLA 注入 | 未接(lewm_vision_encoder.py 半成品) | 设计核心,双臂 + kill criteria | **RAMP** 把它放回第一优先级 |
| 开放词表 / scaling | proto 连续码(已定案,LOO 小胜) | 检索天然开放:新任务=挖 bank 即可,身份头可换簇中心 NN(零训) | **RAMP** 略优,且直接继承 proto 的"簇中心=身份"思想 |
| 部署参数 | 34M | ~0.5M(+索引内存) | **RAMP** |
| bank 质量依赖 | 中 | **高**(payload 直出 bank) | **LMWM**;故 RAMP 必须配 bank 验收三关卡(§3a) |
| 多模态上限 | 单帧条件天花板 ~0.28 | 同天花板(信息论,不是架构能解的) | 平;RAMP 的差异是**承认它**并把不确定性交给策略 |
| 已验证程度 | intrinsic 指标充分(reach 1.67>LaWM 1.48 等) | 组件各有自家实验支撑,整体链路未跑 | **LMWM** |

### 5b. 继承关系(RAMP 不是推倒重来)

- ①身份头 = predm 谱系 + union_ce 锚(自家最强 in-dist 结果)直接升级;
- proto teacher 的"簇中心=身份"思想 → 变成检索键与开放词表路径;
- ⑤可选 adapter = 现生成器的降级续命位(若 SR 证明画布保持真有用,30M 生成器以 5M 残差形式回归);
- ⑥注入 = `archive/lmwam_v2_plan_20260704.md` 的 prefix-KV 接线方案(KI/零初始化/token 预算已想清楚)直接复用;
- Phase 0 = `archive/next_milestone_vla_validation_plan.md` 的 GT-first kill criteria 复活升级(当时的 D2 风险"DINOv3-H≠π0.5 编码器"已被 E2 同塔迁移**解决**——检索到真实帧后用 π0.5 自己的 SigLIP 编码,空间完美匹配)。

### 5c. 定案

**采纳 RAMP 为主方案,但裁决权交给 Phase 0 的 oracle-SR 实验**:
- 若 A1(oracle 检索帧)>A0:RAMP 主链路成立,进 Phase 2 换真预测器;
- 若 A2(oracle 生成 grid)显著>A1:说明画布保持是真需求 → RAMP 加回 adapter⑤/保留生成器,两案融合;
- 若 A1/A2/A3 全≈A0:**子目标条件对 π0.5+语言无正增量** → 诚实收口,转 FLARE 辅助损失路线(它不要求"注入有用",只要求"预测未来这个训练信号有用")。
- 无论哪个分支,**milestone-horizon vs 固定 1.2s** 的消融都保留 —— 这是对基础思想本身的直接检验,也是相对 LaWAM 的论文主张。

---

## 6. 实验计划(P0 决定性优先,便宜优先,每步带 kill criteria)

### P0 —— oracle 裁决(最高信息量,不训任何新 WM)
| 臂 | 内容 |
|---|---|
| A0 | π0.5 baseline(语言条件,无子目标)|
| A1 | + **oracle 检索真实帧** G_sub(GT next-milestone,episode-上下文 NN)= RAMP payload 上界 |
| A2 | + **oracle LMWM 生成 grid**(现生成器,GT 码)= 生成 payload 上界 |
| A3 | + oracle milestone **文本标签**(最便宜通道)|

- 训法:LoRA(gemma_300m_lora r32)+ KI 全冻结 provider + 零初始化投影 + 去同步增广 + subgoal-dropout;单 top-head 视图先行。
- 评估阶梯:**kai0 离线 action-MAE**(便宜、天级,先出信号;注意 v3-PTS 教训——离线 MAE 对数据错位盲,只作排序不作绝对判断)→ **LIBERO-Long 闭环 SR**(sim 决定性;`fastwam/experiments/libero/` 已有 π0 闭环,需 CRAVE 套 LIBERO demo 挖 bank)→ 真机叠衣(域)。
- **kill criteria**:A1/A2/A3 全≈A0(噪声内)→ 停止注入路线,转 FLARE-aux;A*>A0 → 进 P2。
- 算力:LoRA 微调走集群(H20 队列/gf3 恢复后);本地 2 卡只做接线 smoke + 离线 eval。

### P1 —— RAMP 组件落地(与 P0 并行,本地可做)
分类头+价值头(复用 train_multitask.py 骨架)、episode-上下文检索索引、value/熵门、bank 验收三关卡 + UVD 交叉验收脚本。产出:`RAMPProvider`(纯前向、全冻结,对齐 lmwam_v2 的 Provider 契约)。

### P2 —— 换真预测器(仅 A*>A0 才做)
oracle→预测,量化预测误差吃掉多少增量(top-K+门控 vs top1 硬目标消融;GHIL-Glue 过滤器消融)。判据:剩余增量>0。

### P3 —— 论文主张与 scaling
milestone-horizon vs 固定 1.2s 消融(生态位);GVL 零训 value baseline;UVD 对标(聚类步增益);跨任务(3 任务联合)+ 开放词表(LOO)复测,这次以 action-MAE 为准。

### 指标纪律
主指标 = action-MAE → 闭环 SR;grid-cos 降级为 sanity;新增:milestone top-K acc、value-forward 率、熵门触发率、bank 验收三关卡。

---

## 7. 风险与止损

| 风险 | 概率 | 止损 |
|---|---|---|
| 子目标条件对 π0.5+语言无正增量(H0) | 中(短任务大概率,长程文献支持有) | 只在长程测;kill criteria 明确;FLARE-aux 是有尊严的退路 |
| 检索跨 episode 外观错配伤 SR | 中 | A1 vs A2 直接测;伤则加 adapter⑤或回生成器 |
| bank 质量不过关(vis 教训) | 中 | 验收三关卡前置,不过关的任务不进联合训练 |
| LIBERO 需重挖 bank,周期长 | 高 | 离线 action-MAE 先行出信号;LIBERO 并行准备 |
| 预测误差吃光 oracle 增量 | 中 | top-K+门控;P2 有单独判据;身份头还有 prev-ms/proprio 增量没用满 |
| 集群/gf3 不可用 | 现状 | P1 本地先行;P0 微调排 H20 队列 |

---

## 8. 参考

三路调研原文(subgoal 预测 / WM-for-VLA / value-进度表征)由 2026-07-08 深度调研产出,核心 20 篇见 §4 表。自家证据全部来自 `PITFALLS_AND_HISTORY.md` 与 `ABLATION_CONVERGENCE_2026-07.md` 的实测编号。
前置文档:`ARCHITECTURE_AND_BASELINE.md`(现架构)、`archive/lmwam_v2_plan_20260704.md`(注入接线)、`archive/next_milestone_vla_validation_plan.md`(GT-first 验证原案)。
