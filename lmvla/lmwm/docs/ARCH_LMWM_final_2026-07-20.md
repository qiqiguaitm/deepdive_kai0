# LMWM 最终架构方案(2026-07-20 落盘)

> **定位**:本文 = LMWM **架构定版**的单一事实源(各层选了什么 / 为什么 / 淘汰了什么)。
> **实验日志与判据**仍以 [`RECURRENCE_UNIVERSAL_goals_and_roadmap.md`](RECURRENCE_UNIVERSAL_goals_and_roadmap.md) 为**唯一详源**;
> per-task SR 数字以 [`RESULTS_libero10_all_variants_matrix_2026-07-19.md`](RESULTS_libero10_all_variants_matrix_2026-07-19.md) §6 为准。
>
> ⚠️ **架构已定版 ≠ 效果已证成**。当前 P1(只换 WM 超 LaWAM)**打平未超**;
> 已确立的只有**总方向**(dual2q > armB,两次独立评测一致,Δ=+1.2~+2.0);
> **per-task 机理归因目前无法判定**(两次研究给出相反答案,见 §4 与 roadmap §4.18)。

---

## 0. 一句话

**冻结 DINOv3-base 空间里逐帧算跨-episode 复现密度 `r(o)` → 低谷分段、高脊取 canonical 代表帧 →
「帧 t → 下一段 r-脊」训一个 (生成器 + MDN 预测器) 的子目标世界模型 →
以【并联双尺度】而非替换的方式注入 VLA(局部 t+7 通道 always-on 守精度,全局 milestone 通道带 hint-dropout 守指引)。**

无聚类 · 无阈值 · 无 K · 无锚 —— 一套超参跨本体。

---

## 1. 全栈数据流

```
① 信号层 · 复现密度场 r(o)            [CRAVE v2, 零训练]
   r(o_t) = 1/(N_ep−1) · Σ_{j≠ep(t)} exp( −dmin(o_t, E_j)² / 2σ² )
     dmin  = o_t 到 episode j 的最近帧距离(冻结 DINOv3-base pooled, L2)
     σ     = 所有跨-ep dmin 的中位数(median-heuristic, 尺度无关)
   三个正交读法: 幅值=OOD监控/训练加权 · 低谷=子任务边界 · 高脊=canonical 代表帧

② 标签层 · r-谷分段 + r-脊目标          [p1_libero_rvalley_pairs.py]
   分段: find_peaks(−gaussian_filter1d(r, 1.4), prominence=0.03, distance=n/12)
   代表帧: 每段 argmax r = r-脊
   训练对: 帧 t → 【下一段的 r-脊】; 末段 → 锚末帧
   => 100% 帧覆盖(137154 对)

③ 模型层 · 子目标世界模型               [p1_train_lmwm_libero.py]
   InverseEnc(g_t, g_f) → code z        teacher(逆动力学), 看未来 milestone+1
   MilestoneGenerator(g_t, z) → ĝ_next  AdaLN 调制, = LaWM decoder 替身
   MilestonePredictorGrid(g_t) → MDN(K=4) over z   部署头(蒸馏 teacher code)
   损失: smooth_L1(ĝ, g_f) + lift_w·relu(cos(ĝ,g_t) − cos(ĝ,g_f)) + MDN nll

   ⭐ 为何两模型而非单预测器直接回归目标 latent: 见 [ARCH_predictor_vs_single_2026-07-21.md]
      受控实验(So400m, 8+3 seed): 单发时单模型反而更准(+0.021, 8/8)且【不坍缩】——
      历史"持久坍缩"叙事在此设定不复现; 两模型的真正价值 = 【多模态】:
      MDN 分叉处提议 K 候选, best-of-K 0.779 > 单模型 0.765(+0.014, 3/3), 确定性单回归器做不到。
      ⚠️ 内在指标非下游 SR。

④ 注入层 · 并联双尺度 LMWAM-DS          [lawam.py, flowmatching_expert]
   局部通道 (LaWM 原样, always-on):  decoder → h_t7_pred, aux MSE → features[:,-1]
   全局通道 (新增, 不替换):          LMWM generator → h_ms, hint-dropout p=0.15 只作用于此
   DiT 条件 = [ h_t | h_t7 | h_ms | vlm ]
```

---

## 2. 每层的定版选择与被淘汰项

| 层 | **定版** | 淘汰项 → 为什么 |
|---|---|---|
| 编码器 | 冻结 **DINOv3-base** pooled | DINOv3-H(相位更脏)· Wan-VAE / SigLIP2(外观/对齐空间≠任务相位) |
| 信号 | **连续 r 场** | **BGMM 视觉聚类**:LIBERO 低视觉方差塌成 M=1,γ 四数量级无效 → 不普适 |
| 分段 | **r-低谷**,单一全局 prominence=0.03 | 固定 K / K=0.55√N(非自适应/按长度错轴)· per-mode coverage≥0.5(kai0 专属,LIBERO 塌 M=2) |
| 代表帧 | **r-脊**(段内 argmax r) | **段边界**:谷=跨-ep 分歧点,拿边界当目标 = 重蹈 milestone+1 覆辙 |
| 终端 | **末段锚末帧** | v1 的「丢弃最终 milestone」· **双锚 Viterbi**(kai0 专用,LIBERO 吸尾) |
| 输入 | image-only pooled | proprio:probe 0.96→0.97,几乎不加分 |
| 降依赖 | **hint-dropout p=0.15**(保目标 + 正则依赖) | **自适应视界目标**(改目标):插值成"既非 t+7 又非脊"的含混视界,t8 −20 / t9 −10 |
| 注入 | **并联双尺度** | **通道替换**(`torch.where` 覆盖 `h_t1_gt`):必丢局部监督,见 §3 |

### 三条已被数据推翻的直觉(勿重犯)

1. **「降依赖要改目标」→ 错**。粗暴的 hint-dropout 胜、优雅的 adaptive 败。
   教训:**保目标 + 正则依赖**,不能改目标。目标区"模糊"时反而**更**需远端指引,收缩是致命的(t9 遮挡 −10)。
2. **「milestone 包含 t+7」→ 错**。`align = dcos(g_ms−g_t, g_{t+7}−g_t)` 全任务仅 **0.46-0.54**(early 低至 0.34)。
   两通道信息**互不包含** → 替换必丢局部监督 → 必须**并联**。
3. **「t8 是诊断任务」→ 错**。t8 是全表方差最大的任务(std 5.5~12,armB 范围 [68,98])。
   **不得再用 t8 作判据**;per-task 只用低方差任务(t6 std 1.7~3.0)。

---

## 3. 为什么是「并联」不是「替换」(三级因果)

| 级 | 证据 |
|---|---|
| **代码级** | `lawam.py:670-678` 用 `torch.where` **覆盖** `h_t1_gt` → 辅助损失目标与 DiT 的 256-token 条件**同时**从 t+7 换成 milestone;梯度打进 VLM latent action query → 整个"latent action"抽象被重塑 |
| **特征级** | 两通道 align 仅 0.46-0.54;局部动态真实存在(locMag 58-89)但远 hint 指不到 |
| **行为级** | 盈亏由**任务瓶颈类型**决定,不由特征统计:t8 毫米级放置需局部通道 vs t6 选错分支需全局通道。align 不能预测 per-task Δ(t8 .491 ≈ t6 .457) |

**推论**:局部信息在**推理期根本不存在** → 解释了 CFG 权重扫 / t-sched 在推理侧全灭(与 CFG 文献一致:guidance 训练烙定,采样端 reweight 救不了)。

---

## 4. ⚠️ 证据状态(架构定版 ≠ 效果证成)

LIBERO-10,变 seed 重复评测(n=4~8):

| 方案 | n | 聚合 | **t6 弥散** | t8 双壶 |
|---|---|---|---|---|
| 机制② tsched | 8 | 95.22±0.91 | **85.0**±2.4 | 90.0±8.1 |
| **dual2q 并联双 query** | 4 | 94.80±0.71 | **85.0**±1.7 | 88.5±9.8 |
| no-WM 纯 VLA | 4 | 94.30±0.54 | 76.5±1.7 | 93.5±1.7 |
| hintdrop(替换式) | 4 | 94.25±0.84 | 82.5±2.2 | 87.5±5.5 |
| armB LaWM(t+7) | 4 | 93.60±1.12 | 78.5±3.0 | 88.5±12.0 |

> ⚠️ **2026-07-20 本机独立复核修正了下表的 per-task 归因**(roadmap §4.18,3 seed×2 臂×500ep):
> **总方向复现**(dual2q 95.47±1.21 vs armB 93.47±0.58,Δ=+2.00, t=2.59),
> 但 **§6「t6 是唯一稳健机理、p<0.001」不成立** —— 本机 t6 Δ=+7.33 却 **t=1.15**(dual2q std **10.26**,逐路 70/90/84);
> 且 §6 报的 t6 std 1.7 **低于 n=50 时的二项下限 5.51**(欠散无物理机制,P=0.037)。
> 反而 **t8 在本机效应最大**(Δ=+9.33, t=2.65),与 §6「不得用 t8 作判据」相反。
> **两次独立研究给出相反的 per-task 归因 → n=3~8 不足以支撑 per-task 归因。**

| | 状态 |
|---|---|
| ⚠️ **机理归因不稳定** | ~~t6 是唯一稳健机理 p<0.001~~ → **已被本机复核推翻**(见上方横幅)。效应量方向复现(+7.3 vs +8.5)但显著性被严重高估;哪个 per-task 是真信号**目前无法判定** |
| ⚠️ **P1 未达成** | §6 聚合 94.8 vs 94.4;本机复核 95.47 vs 93.47(Δ=+2.00, t=2.59, n=3)。方向一致但仍未过 §6.6 自设判据 |
| ❌ **机制② 未证明有效** | vs 自身基座 dual2q 仅 +0.42(t=0.79, ns) |
| ⚠️ **LaWM 可能是负贡献** | armB 93.60 < no-WM 94.30;t9 82.5 vs 89.5(t≈2.8)。WM 的正贡献来自 **LMWM 的 t6,不是 LaWM** |
| 🚧 **零和张力(封顶原因)** | precision↔guidance 是**一根张力**:t8 要 0 指引、t6 要满指引。任何混合只能设**一个全局平衡** → 必零和重分配 → **聚合封顶 ~94.8** |
| 🚧 **LIBERO 已无分辨力** | 聚合饱和 + t6 单任务样本量太小 → **必须转未饱和基准(RoboTwin 积木族)** |

**评测硬约束**(违反则结论无效):① 重复评测**必须变 seed**(同 seed 误差棒小 5 倍);
② 聚合差 <1.5pt 不可声称;③ per-task 只用低方差任务;④ ckpt 跨集群必须带 `config.yaml` + `dataset_statistics.json`。

---

## 5. 与 CRAVE v1 的分工(勿混用)

| | v1 BGMM 离散管线 | **v2 r 场(本架构)** |
|---|---|---|
| scope | kai0 任务 A **value/AWBC 读出**(corr 0.943,**未被推翻**) | **普适信号**:边界/代表帧/WM 目标/OOD,跨本体 |
| 阶段代表 | coverage 筛选的簇 mode | **r-脊帧**(无聚类) |
| 已知失效 | LIBERO 塌 M=1;coverage 门槛塌 M=2 | t8 型"同物二次操作"别名(V7/V8 在解) |
| 后续 | 维持现状,不再迭代 | 主线 |

**终端问题在 v2 已消解**:v1 的两个病根(丢最终段 / 目标取分歧点)都已修;
且 kai0「回 home 污染」是 v1 **最近原型分配**的失效模式,**r 连续密度天然免疫**(roadmap §4.17 实测:kai0 末段 r-脊在 97% 位置,未塌到 home)。

---

## 6. 关键文件

| 环节 | 文件 |
|---|---|
| 信号+标签 | `lmwm/scripts/p1_libero_rvalley_pairs.py`(robotwin 版 `p1_robotwin_rvalley_pairs.py`) |
| 紧凑目标存储 | `lmwm/scripts/p1_build_target_compact.py` → `target_compact.npz` |
| 模型训练 | `lmwm/scripts/p1_train_lmwm_libero.py` |
| 产物 | `lmwm/data/libero_rvalley/pairs.npz`(137154 对 = 100% 帧覆盖) |
| 注入 | `lawam.py`(双目标并存)· `flowmatching_expert`(第三段 future tokens + mask) |
| V8 执行计划 | `PLAN_V8_lmwam_ds_dual_scale_2026-07-18.md` |
| eval 模板 | `train_scripts/kai/volc/libero_eval_2ckpt_x4_8h20.yaml` · `libero_eval_mech2_x8_seeds.yaml` |

---

## 7. 未决(按优先级)

1. **P2 · RoboTwin 下游重训 + eval** —— LIBERO 聚合已饱和无分辨力,这是唯一能继续判优的场。特征已抽(V4),缺下游。
2. **零和张力**:是否存在"per-task 自适应平衡"而非单一全局平衡?当前所有混合方式(共享 query / 双 query / CFG)都只能设一个。
3. **P3 终局**:把 LMWM 搬进 **VLA 自身编码器空间**,去掉 frozen-DINOv3 外挂 —— 别名塌缩与跨空间翻译都源于此割裂。张力:frozen-DINOv3 本是为跨本体普适选的。
4. **V7 循环 belief**:失败全是"超时磨蹭无发散" → 无状态逐帧预测器是根因。主线实现 = **喂 value/progress 作 belief**(V7.1′ 标量 → V7.2′ value-GRU 隐状态),同时闭合 CRAVE(value 读法)与 LMWM(WM 读法)。
