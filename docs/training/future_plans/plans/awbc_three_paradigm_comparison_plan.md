# AWBC 三范式对比实验 (Conditioning / Loss-Weighting / Resampling)

> **建立**: 2026-07-21 · **状态**: 📋 设计定档, 待提交 · **资源**: cnsh robot-task, 每臂 16 卡 (2×8 A100)
> **一句话**: 同一份 chunk-001 数据, 用 AWBC 的三种范式各训一版, 对比"优势信号以何种机制进训练"对真机效果(尤其抓取/不冻)的影响。
> **上游**: 优势信号调研见 [`velocity_dataset_speed_gap_findings`](velocity_dataset_speed_gap_findings.md) · dagger 用法见 `docs/training/analysis/chunk001_schema.md` · 速度门控杀抓取教训见 [`dagger_launchpoint_trim_freeze_fix_plan`](dagger_launchpoint_trim_freeze_fix_plan.md)

---

## 1. 动机与假设

**背景**: 现有 AWBC(`pi05_v4_awbc_chunk001_dagger_crave` = `_labeled`)只用了"①条件化"(把 `Advantage: positive/negative` 放进 prompt, 普通 flow-matching loss)。而 AWBC 的"W"(Weighted)本指"②损失加权 / ③采样加权"(AWR/CRR/AWAC/Sirius/IWR)——本仓库从未用过。且我们已实测:
- 条件化的 pos/neg 对速度/动作幅度影响 ≈0(信号弱), 且靠泛化到"高优势 token"→ 有 OOD/冻结风险;
- `_dprog` 的速度门控把抓取(臂静止+夹爪合拢)误标 negative → 真机死循环。

**新条件**: chunk-001 已带真三分类 `dagger_frame_class ∈ {0=robot, 1=intv, 2=preintv}`(2026-07-20 同步), 是干净的逐帧优势代理(class1=人控纠错含抓取, class2=失败先兆)。

**假设**:
- **H1**: ②/③(在梯度/采样上直接区分好坏动作, 带 KL 约束贴行为分布)比 ①(靠条件泛化)更能治冻结/死循环, 且 ② 天生保住抓取(class1 无论臂速都上权)。
- **H2**: ② 与 ③ 期望等价(同一 class 权重), 真机差异应小 → 若成立, 选实现更简的 ③。
- **H3**: 三者叠加(cond+weight)≈ RECAP 完整形态, 上限最高(留作后续, 本实验先测单范式)。

---

## 2. 变量控制:共享的一切 vs 唯一变量

**共享(所有臂逐字段一致, 复用 `_labeled` 配方)**:
- 数据: chunk-001 (387 base + 387 dagger), **保留 `dagger_frame_class`**(build 改动, 见 §4.1)
- init `pi05_base` · arch pi0.5 无 DCT · bs256 · fsdp16 · 50k · cosine lr 1.5e-5→1.5e-6 · ema 0.9999
- **唯一变量 = 优势/class 信号以何种机制进训练**

**共享的逐帧质量标签**(全部从 `dagger_frame_class` + base 派生, 三臂同源):

| 来源 | 语义 | 质量 | ① prompt | ②③ 权重 w |
|---|---|---|---|---|
| base | 专家示范 | 好 | positive | 1.0 |
| class 0 robot | 自主-正常 | 好(残余) | positive | 1.0 |
| class 1 intv | 人控-纠错(含抓取) | **最好** | positive | **2.0** |
| class 2 preintv | 自主-临失败 | **坏** | **negative** | **0.0** |

> ⚠️ **范式固有的不可完全单变量**: 条件化(①)需要**平衡的**二值信号才不退化(class2 仅 1% → 若 pos/neg 按 class2 切则 99% positive 退化); 而加权(②③)天生处理不平衡。故 ① 采用"人控 vs 机器人"平衡二值(positive={base,class1}≈人控质量, negative={class0,class2}=机器人), ②③ 采用分级权重 {0:1,1:2,2:0}。**三臂共同点**: 同源自 class, 均视 class1 最好 / class2 最坏 / 保住抓取; **残余混淆**: ① 把 class0 当 negative, ②③ 当 w=1 —— 已在 §7 结论中显式标注, 不隐藏。

---

## 3. 四个臂

| 臂 | 范式 | 优势怎么进 | prompt | loss | sampler | 代表论文 |
|---|---|---|---|---|---|---|
| **B0 基线** | ①(progress) | `_labeled`(top-30% 进度) | pos/neg | 普通 | 均匀 | (现有, 已部署OK) |
| **A1 COND** | ① 条件化 | prompt(class 派生 pos/neg) | pos/neg | 普通 | 均匀 | DT 2106.01345 · RvS 2112.10751 · RECAP 2511.14759 |
| **A2 WEIGHT** | ② 损失加权 | `per_token_loss × w(class)` | 中性(无 advantage) | **加权** | 均匀 | AWR 1910.00177 · CRR 2006.15134 · AWAC 2006.09359 · Sirius |
| **A3 RESAMPLE** | ③ 采样加权 | 采样频率 ∝ w(class) | 中性 | 普通 | **加权** | IWR 2012.06733 · Sirius |

- **B0 已存在**(`pi05_v4_awbc_chunk001_dagger_crave/49999`), 作参照, **不重训**。
- A1/A2/A3 各一个 16 卡任务。推理: A1 喂 positive; A2/A3 喂中性 task prompt。

---

## 4. 实现清单(按依赖排序)

### 4.1 数据(共享前提, 一次构建)
- 复用 `build_chunk001_dagger_crave_labeled.py`(已含改动1: 保留 `intervention`/`dagger_frame_class`, base 填默认)。
- **新增**: 落盘时同时写一列 `sample_weight`(= §2 表的 w, 供 ③ 采样器 / ② loss 读)与逐帧 `adv_bin`(供 ① prompt 的 pos/neg / discretize)。base 无 class → class=demo, w=1, bin=positive。
- 产出 `A_v4_chunk001_dagger_crave_cls`(带 class+weight+bin 三列), norm_stats 重算。cnsh 可访问路径 `/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/`。
- ⚠️ **cnsh 数据可达性**: `_labeled` 已证 `/vePFS/...` 在 cnsh 挂载可读(B0 就在跑), 直接同位构建即可, 无需跨集群 cp。

### 4.2 A1 COND(仅 config, 零代码)
- config `pi05_v4_awbc_chunk001_dagger_crave_human`: 同 `_labeled`, 换 repo_id → `_human`, prompt_from_task(pos/neg)。
- 打标 = `relabel_human_awbc.py`: `task_index = intervention`(positive⟺人控: base + class1;negative: class0/class2)。**无代码改动**; 即上一轮 `_human` 配方的正式化。
- **✅ 构建产出(2026-07-21)**: `/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/`**`A_v4_chunk001_dagger_crave_human`**
  (774 ep / 1,378,292 frames; 387 base + 387 dagger; `dagger_frame_class{0,1,2}`+`intervention` 已保留; norm_stats 已算)。
  ⏳ **待 relabel**(task_index 仍为占位 0)+ 提交 YAML `pi05_v4_awbc_human_cond_cnsh_16gpu.yaml`。

### 4.3 A2 WEIGHT(代码: loss 加权)
- `pi0.py compute_loss`: `per_token_loss`(§confirmed 逐样本存在)→ `× weight[:,None]` 再平均。weight 从 batch 取。
- data pipeline: `sample_weight` 列透传到 model 输入(参 AdvantageEstimator 透传 `progress` 的做法, `config.py:600` 附近)。
- config `_weight`: 新增 `awbc_loss_weight: bool`; prompt 中性(default_prompt=task, 不插 advantage)。
- **必须 smoke**: 3 步本机 smoke 验证 loss 不 NaN、weight 生效(class2 帧梯度=0)。

### 4.4 A3 RESAMPLE(代码: 加权采样器)
- data_loader: 现有 `domain_sample_weights` 是 **domain 级**; 需 **frame 级** 按 `sample_weight` 的加权采样(或近似: 用 class 当 domain, 3 domain 权重 {0:1,1:2,2:0}, class2→0 等效丢弃)。
- **优先用 domain 复用路径**(class→task_index/domain, 走已有 `DomainWeightedSampler`), 免写新采样器。
- config `_resample`: 中性 prompt + domain_weights=(class0:1, class1:2)(class2 单独排除)。
- **必须 smoke**: 验证 batch 里 class 分布≈目标权重、class2 ≈0。

### 4.5 提交(每臂一个 YAML, 基于 `pi05_v4_awbc_crave_dagger_cnsh_16gpu.yaml`)
- 改 `TaskName` / `Config` / preflight 检查列(`dagger_frame_class`/`sample_weight` 存在)。
- 提交: `source ~/.volc_creds && kai0/.venv/bin/python train_scripts/kai/volc/submit_yaml.py <yaml>`(经 gsy)。

---

## 5. 评估协议

| Tier | 指标 | 说明 | 判据 |
|---|---|---|---|
| T1 离线(自动) | inline-eval MAE@{1,10,25,50} | AWBC 不敏感, 仅 sanity/收敛 | 不作主判据 |
| T2 离线探针(自动) | 速度保真度 + pos/neg 条件位移 + **抓取帧 positive%** | 复用 `velocity_{fidelity,condition_shift}.py`; **新增抓取帧标注比**(§dagger_launchpoint 口径) | A2/A3 抓取帧不被压 |
| T3 **真机(决定性)** | 成功率 / throughput / **抓取成功率(死循环率)** / 回折不冻 | positive/中性 prompt 推理 | **主判据**: 谁最少死循环 + 最高抓取成功 |

**归因**: 三臂 vs B0 的 Δ 干净归因到"AWBC 范式"。重点回答: ②/③ 是否消除 `_dprog` 的死循环、是否保住抓取、是否比 ① 更稳。

---

## 6. 资源与排期

- **算力**: cnsh robot-task, 3 臂 × 16 卡 (2×8 A100) × ≤72h。可并行(队列容量足)或串行。
- **数据构建**: 本机(gf0)约 30–60 min(复用现成 build + 加列)。
- **代码+smoke**: A2/A3 各约半天(loss/sampler + 3 步 smoke)。A1 即时。
- **门禁**: A2/A3 **必须本机 3 步 smoke 通过**再提交 16 卡(train_scripts/CLAUDE.md 规范; 不提交未测代码到 72h 任务)。

---

## 7. 风险与结论模板

**风险**:
- R1 ②③ 权重 {0:1,1:2,2:0} 里 class2 仅 1% → 影响可能小; 缓解: 同时报"若把 class0 也降权"的敏感性(留作后续)。
- R2 单变量不纯(§2 ⚠️): ① 的 class0=neg vs ②③ 的 class0=w1 → 归因时**显式区分**"范式效应"与"class0 处理差异"。
- R3 cnsh 数据/环境: 复用 `_labeled` 已验证路径与镜像, 风险低。

**结论回填**(实验后):
- 范式排序(真机死循环率/抓取成功): ① vs ② vs ③ vs B0
- H1/H2/H3 证实/证否
- 是否定为新默认 AWBC 训练法

---

## 8. 关联
- 调研: 本轮对话"AWBC 三范式"综述(会话记录) · [`velocity_dataset_speed_gap_findings`](velocity_dataset_speed_gap_findings.md)
- 数据格式: `docs/training/analysis/chunk001_schema.md`(class 编码 + Sirius/IWR 合约)
- 教训: [`dagger_launchpoint_trim_freeze_fix_plan`](dagger_launchpoint_trim_freeze_fix_plan.md)(速度门控杀抓取) · 记忆 `project_velocity_gate_kills_grasp`
