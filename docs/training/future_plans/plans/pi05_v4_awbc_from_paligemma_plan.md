# pi05 × v4(base+dagger)KAI0 AE AWBC × **from-PaliGemma-base** — 训练 plan

> **建立**: 2026-06-27
> **目的**: 用**全部 v4 base + dagger** 跑完整 **KAI0 AE AWBC 流程**(Advantage Estimator → 打标 → discretize → AWBC 训练),但 init **从 PaliGemma VLM base 冷启动**(不 warm-start PI 的 pi05_base / 不用 KAI0 mixed_1_clean),步数 **100k**,在 **cnbj Robot-North-H20 8 卡闲时**提交。验证"**自有 v4 数据 + AWBC + 从 base 自训**"三者叠加能否训出可部署叠衣策略。
> **状态**: 🚀 **已提交训练**(2026-06-27)— `t-20260627104413-qcb92`(cnbj Robot-North-H20 8×H20 闲时,Queueing)。config `pi05_v4_awbc_from_paligemma` 已注册+push(HEAD e150b06);labeled `A_v4_base_dagger`(1824ep)已 staged 至 cnbj `/vePFS-North-E`(video symlink retarget,0 broken);discretize 复用姊妹 plan 的 binary top-30%(reuse 同一 labeled 集 = 与 cnsh mmdhx 完全同源)。详见 §8。
> **依据**: from-PaliGemma 冷启动路线已被 [`pi05_from_paligemma_base_training_plan.md`](pi05_from_paligemma_base_training_plan.md) §7.5 **实证收敛**(148k @1=0.0066/@50=0.0181,offline 全 horizon 反超 warm-start)。本 plan = 把那条已验证的冷启动路线套到 **v4 AWBC** 数据上。
>
> ### ⚠️⚠️ v4 两个关键要点(实施时务必,与基线 plan 一致)
> 1. **norm_stats 必须对 v4 重算** —— v4 动作分布变了(夹爪 action≠state,取主臂指令)→ 绝不能复用旧 v2/v3 的 norm,否则夹爪维归一化错位 → 静默训坏(offline MAE 也看不出)。`compute_norm_states_fast.py` 对 merged v4 集重算。
> 2. **夹爪不裁(原始 v4 action)** —— v4 已是"主臂意图指令",裁剪(≤5mm→0)对 v4 无意义/有害。
> **上游/姊妹 plan**: [`pi05_v4_awbc_validation_plan.md`](pi05_v4_awbc_validation_plan.md)(同流程,init=pi05_base / 50k 的 warm-start 版)· AWBC 总纲 [`../../deployment/strategy/awbc_implementation_plan.md`](../../deployment/strategy/awbc_implementation_plan.md)。
> ⚠️ **铁律**: 真机为终判;VLA 报告先看 val MAE(不是 train loss);idle 轨迹 MAE 反指。

---

## 0. 这是什么 / 与姊妹 plan 的唯一差异

本 plan 与 [`pi05_v4_awbc_validation_plan.md`](pi05_v4_awbc_validation_plan.md) **数据、AE、AWBC 流程、v4 铁律完全相同**,只在**训练规格 §3** 上分叉:

| 项 | 姊妹 plan(warm-start)| **本 plan(from-base)** |
|---|---|---|
| init | pi05_base(`CheckpointWeightLoader`)| **from PaliGemma base(`PaliGemmaLocalWeightLoader` + `pt_224.npz`,action expert 随机)** |
| 步数 | 50,000 | **100,000** |
| LR | warm-start(1.5e-5 / warmup1k)| **冷启动(peak 3e-5 / warmup 3k / decay100k / end 3e-6)** |
| 集群 | 待定 | **cnbj Robot-North-H20 8 卡,闲时(idle/抢占)** |
| 价值 | 快、稳(对照基线)| 自主可控、不依赖 PI 预训练权重;科研意义 |

→ 两 plan 可作 **"warm-start vs from-base" 在 v4 AWBC 上的受控对照**(同数据/同 AE/同 eval)。

---

## 1. 数据(全部 v4,已落地校验)

| 域 | 来源 | 日期 | ep | 帧 |
|---|---|---|---:|---:|
| **base**(demo)| `vis_base/v4` | 13(4-23~6-04)| **1207** | **1,348,869** |
| **dagger**(纠错,含 intervention)| `vis_dagger/v4` | 12(5-29~6-23)| **789** | **1,020,660** |
| **merged** | `self_built/A_v4_base_dagger`(新建)| | **≈1996** | **≈2,369,529** |

- merge 时**删 intervention 列**(base v4 无、dagger v4 有);两者均 action≠state、14D、前裁+尾裁、3 相机 → 可直接混。
- **单本体**(仅 v4 Task_A vis)→ 不涉及双本体 domain conditioning。

---

## 2. KAI0 AE AWBC 流程(复用现有 pipeline,与姊妹 plan 同)

| Stage | 做什么 | v4 注意 |
|---|---|---|
| **0–1 AE** | ✅ 复用最早 kai0-trained AE `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1`(2026-04 训,eval.py 默认 step **100000**)| AE 从图像+state 预测 stage_progress;v4 state 与旧一致 → 复用应可;⚠️ Stage 2 后**核验 advantage 与 GT 进度正相关**,不行再考虑在 v4 上重训 AE |
| **2 打标** | `stage_advantage/annotation/eval.py Flatten-Fold KAI0 <A_v4_base_dagger>` → 每帧加 `absolute_advantage` 列 | 多卡 `--num-workers/--worker-id` 切片 |
| **3 discretize** | `discretize_advantage.py --discretion-type binary --advantage-source absolute_advantage [--stage-nums 2]` → `task_index∈{0,1}` + tasks.jsonl(`Advantage: positive/negative`)| 默认建议 top-30% + stage-aware |
| **4 AWBC 训练** | 克隆 `pi05_flatten_fold_awbc`(config.py:2095)→ `repo_id`=v4 labeled 集,`prompt_from_task=True`;**init 换成 from-base(见 §3)** | init/LR/步数/集群见 §3 |

---

## 3. 训练规格(克隆 `pi05_flatten_fold_awbc` → 换 from-base init)

- **config** 新建 `pi05_v4_awbc_from_paligemma`(克隆 config.py:2095):
  - `repo_id` → `A_v4_base_dagger`(Stage 3 labeled);`base_config=DataConfig(prompt_from_task=True)`;`use_delta_joint_actions=False`(absolute)。
  - ⚠️⚠️ **norm_stats 必须对 v4 重算**(`compute_norm_states_fast.py`)—— 绝不复用旧 v2/v3 norm。
  - **夹爪 = v4 原始 action(不裁)**。
  - ✅ **init = from PaliGemma base(冷启动)** = `weight_loader = PaliGemmaLocalWeightLoader(npz_path=…/paligemma_weights/pt_224.npz)`(载 SigLIP-So400m + Gemma-2B VLM;**action expert + 动作投影 + time MLP 随机 init**)。⚠️ **不是 pi05_base、不是 mixed_1_clean** 任何 warm-start。已被 [`pi05_from_paligemma_base_training_plan.md`](pi05_from_paligemma_base_training_plan.md) §7.5 实证收敛。
  - **LR/warmup(冷启动配方,沿用 from_paligemma 实证值)**:CosineDecay warmup **3,000** / peak **3e-5** / decay 到 **100k** / end **3e-6**(比 warm-start 1.5e-5/warmup1k 高一档 —— 随机 action expert 需更激进起步)。⚠️ 不要套 warm-start 的 LR。
  - ✅ **100,000 step**;batch 128;fsdp 8;EMA 0.9999;save 每 2k / keep 10k;`inline_eval_val_root` → v4 留出 val。
    - 步数理由:冷启动 from base 需的步数 > warm-start 50k;from_paligemma 双本体 ~148k 才收敛 → 单域 v4 AWBC 取 **100k anchor**,**未 plateau 则续训**。
  - **单 domain**:仅 v4 Task_A vis → 沿用 AWBC config 单 domain,**不用** from_paligemma 的 num_domains=2 域条件。
  - **推理永远喂 positive prompt** `"Flatten and fold the cloth. Advantage: positive"`(train==deploy)。
- **集群**:**cnbj Robot-North-H20 单节点 8 卡,闲时任务(idle/抢占,空闲即跑,可被高优打断/续跑)**(`submit-training-job` skill;from_paligemma 实证 run 同集群 `t-20260619152024-pthmf`,PaliGemma `pt_224.npz` 权重该集群已缓存)。

---

## 4. 评估(真机为终判)

| Tier | 做法 |
|---|---|
| Tier 1 offline | v4 留出 val 逐 ckpt **val MAE**(整体 + **夹爪维单列**)+ loss → 收敛 + 选 best。⚠️ AWBC 对 MAE 不敏感,只 sanity。冷启动早期 MAE 会高(参 from_paligemma 6k @1≈0.47)。 |
| Tier 2 标注核验 | Stage 2 后:advantage vs GT 进度 corr + 抽检高/低 advantage 帧合理性(确认 AE 在 v4 上没失效)。 |
| Tier 3 真机(决定性)| 部署 best ckpt 跑叠衣:成功率 / 各 sub-phase 通过 / **夹持稳定性(松手/脱落)= v4 主验证点**。 |

**对照**:
- 旧 AWBC `pi05_flatten_fold_awbc`(smooth800+全dagger,action==state)→ 比"v4 新夹爪 vs 旧 action==state"。
- **姊妹 plan**(v4 AWBC warm-start,若也跑)→ 隔离"from-base vs warm-start"在 v4 AWBC 上的差距。
- from_paligemma 纯 BC(§7.5,148k 双本体)→ 隔离"AWBC vs 纯 BC"贡献(注:数据/本体不同,仅定性参考)。

**判据(v4 + from-base 可用性)**:
- ✅ 可用 = 训练收敛 + 真机能叠衣 + 夹持比旧 v3 明显更稳。
- ⚠️ ≈ 旧 = 数据/路线无真机增量(但仍可用)。
- ❌ 更差 / 不收敛 = 查 norm / 动作语义 / AE 失配 / **冷启动+AWBC 叠加是否需更多步**。

---

## 5. 落地步骤
1. **build** `A_v4_base_dagger`(合并 v4 base 13 + dagger 12,删 intervention,episode_index 重排,视频 symlink)。
2. **重算 norm_stats**(v4 动作分布)。
3. **Stage 2** AE 打 advantage(复用 adv_est_v1)+ **核验对齐**。
4. **Stage 3** discretize → labeled 集。
5. **注册 config** `pi05_v4_awbc_from_paligemma`(克隆 `pi05_flatten_fold_awbc` → 换 `weight_loader=PaliGemmaLocalWeightLoader(pt_224.npz)` + LR warmup3k/peak3e-5 + 100k step + 单 domain),commit/push。
6. **提交 cnbj Robot-North-H20 8 卡 100k 闲时训练**(`submit-training-job`;先确认该集群已有 `pt_224.npz` PaliGemma 权重 + v4 labeled 数据)。
7. **eval**:val MAE → 选 ckpt → **真机**(对照旧 AWBC + 姊妹 plan warm-start),落 §4 判据。
8. 回填 results.md + 更新 master history。

---

## 6. 风险 / 注意
- **冷启动 × AWBC 叠加(本实验首次)**:from_paligemma §7.5 实证的是**纯 BC co-train**(default_prompt、双本体);本实验是"**冷启动 + AWBC advantage-prompt + 单域 v4**"三者首次叠加 —— 随机 action expert 早期与 advantage 条件 prompt 同时学,可能需比纯 BC / 比 warm-start 更多步。**100k 若 val MAE 未 plateau 则续训**(参 from_paligemma 148k)。
- **AE 在 v4 上是否失效**:Stage 2 后必须核验(Tier 2),不过关则在 v4 上重训 AE(工作量大)。
- **norm 复用陷阱**:v4 动作分布 ≠ 旧 → 必须重算 norm。
- **action_dim padding**:pi05 action_dim=32,v4 是 14D → padding 逻辑沿用 flatten-fold config。
- **PaliGemma 权重就位**:`PaliGemmaLocalWeightLoader` 读本地 `pt_224.npz`(cnbj 集群 from_paligemma run 已缓存;提交前确认路径)。
- **base 早期段**:v4 base 是 4-23~6-04 → 偏干净;若要更全可等 TOS 补 v4 后期日期。

---

## 7. 决策定档(✅ 2026-06-27 用户确认)
1. ✅ **数据 = 全部 v4** base(13)+ dagger(12)= ~1996ep/2.37M 帧。
2. ✅ **AE = 最早 kai0-trained AE** `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1`(复用,step 100000)。Stage 2 后核验对齐。
3. ✅ **init = from PaliGemma base**(`PaliGemmaLocalWeightLoader` + `pt_224.npz`;action expert 随机)。⚠️ 不是 pi05_base、不是 mixed_1_clean。
4. ✅ **步数 = 100,000**。LR = warmup3k / peak3e-5 / decay100k / end3e-6。
5. 🔲 **discretize**(待定,默认建议 binary top-30% + stage-aware `--stage-nums 2`)。
6. ✅ **集群 = cnbj Robot-North-H20 单节点 8 卡,闲时任务**。

> 主配置已定;仅 ⑤ discretize 阈值待定 + "开始实施" → ① build A_v4_base_dagger + 重算 norm → ② Stage 2 打标(adv_est_v1)+ 核验 → ③ Stage 3 discretize → ④ 注册 config `pi05_v4_awbc_from_paligemma` → ⑤ cnbj H20 8 卡 100k **闲时**训练 → ⑥ eval(真机对照旧 AWBC + 姊妹 plan)。

---

## 8. 实施记录(2026-06-27 提交)

| 项 | 落地 |
|---|---|
| **数据** | 复用姊妹 plan 已 labeled 的 `A_v4_base_dagger`(AE adv_est_v1 打标 + discretize binary top-30%,1824ep,与 cnsh `mmdhx` 完全同源)。无需重跑 Stage 2/3。 |
| **跨集群 staging** | cnsh `/vePFS` → cnbj `/vePFS-North-E`:rsync labeled parquets(269M)+ meta + `norm_stats.json`(v4 重算版)+ video symlink 树(排除 `data_unlabeled_bak`);v4 源视频 25 日期 cnbj 已有(TOS 同步),symlink 前缀 `/vePFS/tim/workspace`→`/vePFS-North-E/vis_robot/workspace` 全量 retarget,**5982 symlink 0 broken**。 |
| **config** | `pi05_v4_awbc_from_paligemma`(config.py)= clone `pi05_v4_awbc` + `PaliGemmaLocalWeightLoader(pt_224.npz)` + cold LR(warmup3k/peak3e-5/decay100k/end3e-6)+ 100k + 单 domain;cnbj venv 解析通过。 |
| **yaml** | `train_scripts/kai/volc/v4_awbc_from_paligemma_cnbj_8gpu.yaml`(Robot-North-H20 8×H20,`Preemptible:true` 闲时,入口 preflight 验数据/norm/tasks/npz/val/symlink + 自动 `--resume`,save 2k,MaxRetry 3)。 |
| **提交** | `t-20260627104413-qcb92`(cn-beijing,Queueing,闲时等碎片整理);commit `e150b06` 已 push,cnbj repo 已 pull。 |

**对照锚点**:姊妹 warm-start `pi05_v4_awbc`(cnsh `t-20260624233509-mmdhx`,50k,~step38k/Running)→ 同 labeled 数据/AE/eval,隔离 **from-base vs warm-start**。

### 后续(待训练产出)
- 监控:闲时被抢占 → 入口 `--resume` 自续;若长期拿不到碎片可考虑提 Priority 或转独占。
- eval:val MAE(整体+夹爪维)逐 ckpt → 选 best → 真机叠衣(对照旧 AWBC + 姊妹 warm-start)。
- 回填 `results.md` + master `00_training_history.md`。

---

## 关联
- 姊妹 plan(warm-start 版,同流程):[`pi05_v4_awbc_validation_plan.md`](pi05_v4_awbc_validation_plan.md)
- init 路线(from PaliGemma base,已实证):[`pi05_from_paligemma_base_training_plan.md`](pi05_from_paligemma_base_training_plan.md) §7.5(config `pi05_kaivis_from_paligemma`,148k 收敛 @1=0.0066)· loader `kai0/src/openpi/training/weight_loaders.py:64`(`PaliGemma(Local)WeightLoader`)· 权重 `…/paligemma_weights/pt_224.npz`(cnbj 已缓存)
- AWBC 总纲 + 复用 AE 路线:`docs/deployment/strategy/awbc_implementation_plan.md`
- config 克隆源:`kai0/src/openpi/training/config.py:2095`(`pi05_flatten_fold_awbc`)· AE config :1027(`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`)
- AE ckpt:`kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/`
- stage_advantage 脚本:`kai0/stage_advantage/annotation/{eval.py, evaluator.py, discretize_advantage.py}`
- 数据:`kai0/data/Task_A/vis_base/v4`(13日期)+ `vis_dagger/v4`(12日期)
- v4 夹爪约定背景:[`gripper_action_clip_experiment.md`](gripper_action_clip_experiment.md)
