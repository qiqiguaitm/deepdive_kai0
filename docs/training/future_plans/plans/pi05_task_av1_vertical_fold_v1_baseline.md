# pi05 × Task_AV1 (Vertical Fold v1 新 SOP) — 首次基线训练 plan

> **建立**: 2026-06-12
> **目的**: 用**我们自己新设计的"竖向折叠 (Vertical Fold v1)"流程**采集的数据集 `Task_AV1`,取 **200 episode** 做 pi05 的**首次基线训练**(50k step),验证新 SOP 数据能否训出可部署的叠衣策略,并**正式记录新 SOP**(本文档 §1)。
> **状态**: 📋 **规划草稿** — 本次仅文档;**待用户补全 SOP 细节(§1)+ 确认配置**后再 build + 训练。
> ⚠️ **铁律**: 真机为终判;VLA 训练报告先看 **val MAE**(不是 train loss)。

---

## 1. ⭐ 新 SOP 记录:Vertical Fold v1(竖向折叠 v1)

> **本节用于正式记录新叠衣 SOP。** 以下"已知"项从数据集自动提取;**"待补充"项请用户填写**(只有采集方知道物理流程)。

### 1.1 已知(从 `Task_AV1` 数据提取)
| 项 | 值 |
|---|---|
| 任务 prompt | **"Flatten and fold the cloth. Vertical Flod v1."**(meta/tasks.jsonl)|
| 本体 | vis(Agilex 双臂 Piper),3 相机 `top_head/hand_left/hand_right`,14D 关节 state/action,30Hz |
| 单条 episode 时长 | 中位 **~47s**(1409 帧),范围 1017–1855 帧 |
| 采集日期 | 2026-06-11-v2、2026-06-12-v2(持续采集中,TOS→本地自动同步) |
| 数据规模(截至 2026-06-12) | **245 ep**(06-11: 133 + 06-12: 112),仍在增长 |
| ⚠️ action 语义 | **action ≡ state 逐维精确相等**(含夹爪)→ 夹爪记录的是"被物体限位的实际位置"非"意图闭合指令"(同 Task_A,见 [`gripper_action_clip_experiment.md`](gripper_action_clip_experiment.md))|

### 1.2 待补充(请用户填写 — 这是"记录 SOP"的核心)
- **新流程相对旧流程改了什么 / 为什么改**(动机:旧 horizontal/其它折法的什么问题促成竖向折?)
- **竖向折叠的分步动作**(step-by-step:展平 → 第 1 折 → … → 完成,每步双臂/夹爪做什么、抓哪个角)
- **关键阶段 / 成功判据**(哪些 sub-phase 最易失败,如抓角/对折/压平)
- **采集约定**(初始布料摆放、相机视角、单/双布、是否含投放等待段、idle 处理)
- **与已有数据集的关系**(Task_AV1 vs Task_A smooth800/dagger:同本体不同折法?能否混训?)

> 我先把框架搭好,你把 1.2 的物理流程补进来,这份就是新 SOP 的正式文档。

---

## 2. 实验设计:首次基线(单 run)

| 项 | 配置 |
|---|---|
| 模型 | **pi05**(`Pi0Config(pi05=True)`)|
| 数据 | `Task_AV1` 取 **200 ep** 子集(见 §3)|
| init | **warm-start `mixed_1_clean/params`**(沿用现有 flatten-fold pi05 配方;❓ 待确认:warm-start 还是别的)|
| steps | **50,000** |
| 目标 | 新 SOP 数据的可部署基线 + 摸清新折法的 offline/真机表现 |

---

## 3. 数据准备:取 200 ep

- **来源**: `kai0/data/Task_AV1/base/{2026-06-11-v2, 2026-06-12-v2}/`(245 ep 可选)。
- **选 200 ep(❓ 待确认方式)**:
  - **默认推荐**:**随机 200**(固定 seed,跨两日期混合,代表性好)→ 合并成单数据集 `Task_AV1_200`(lerobot v2.1,episode_index 重排)。
  - 备选:按日期取(如全 06-11 的 133 + 06-12 的 67)。
- **val(留出)**:剩余 **45 ep 作 held-out val**(新 SOP 是新分布,**用 Task_AV1 自己的留出集**做 in-distribution eval,比用旧 `vis_v2_merged_val` 更对口)。
- **norm_stats**:对 `Task_AV1_200` **重算**(`compute_norm_states_fast.py`)。
- **build 脚本**:复用 `build_no_release.py --mode raw --merge-*` 或写薄脚本合并指定 200 个 ep + 重排 + 视频 symlink。
- ⚠️ **夹爪(决策点)**:本数据 action≡state、夹爪抓取停在布厚(给不到力,见 §1.1)。**首次基线先用原始 action**(不裁),把夹爪裁剪(≤5mm→0)作为**后续对照**(已有独立 plan)——除非你现在就想基线即用裁剪版。

---

## 4. 训练规格(克隆现有 flatten-fold pi05 config)
- **config** 新建 `pi05_task_av1_vfold_v1_200`(克隆 `pi05_flatten_fold_A_smooth800_dagger_full` config.py:1798):
  - `repo_id` → `Task_AV1_200`;`default_prompt="Flatten and fold the cloth. Vertical Flod v1."`(对齐数据 prompt);`use_delta_joint_actions=False`。
  - init `CheckpointWeightLoader("mixed_1_clean/params")`;cosine **warmup 1k / peak 1.5e-5 / decay 50k → 1.5e-6**;EMA **0.9999**;**50k step**;batch **128**;**fsdp 8**;save 每 2k / keep 10k;`inline_eval_val_root` → Task_AV1 留出 val。
- **资源**:单节点 **8 卡**(cnsh A100 / cnbj H20)。

---

## 5. 评估(真机为终判)
- **Tier 1 offline**:Task_AV1 留出 val(45 ep)逐 ckpt **val MAE**(整体 + 夹爪维单列)+ loss → 收敛 + 选 best ckpt。
- **Tier 3 真机**:部署 best ckpt 跑**竖向折叠**,看成功率 / 完成率 / 各 sub-phase 通过率 + 夹持稳定性(松手/脱落)。
- **判据**:新 SOP 能否训出真机可用的竖向折策略;与旧 SOP(horizontal smooth800)基线对比(可选)。

---

## 6. 落地步骤
1. **补全 §1.2 SOP**(用户)。
2. **build `Task_AV1_200`**(随机 200 ep + 重排 + 视频 symlink)+ 留出 45 val + **重算 norm_stats**。
3. **注册 config** `pi05_task_av1_vfold_v1_200`(§4),git commit/push。
4. **提交 8 卡训练**(50k,每 2k save)。
5. **eval**:val MAE 曲线 → 选 ckpt → 真机竖向折叠 rollout。
6. **回填**结论 + 写 results.md + 更新 master history。

---

## 7. 待确认(动手前)
1. **选 200 ep 方式**:随机 200(默认推荐)还是按日期?
2. **init**:warm-start `mixed_1_clean`(默认)还是别的(如从 PaliGemma base / 别的 ckpt)?
3. **夹爪**:首次基线用原始 action(默认)还是直接用裁剪版(≤5mm→0)?
4. **val**:用 Task_AV1 留出 45 ep(默认推荐)还是旧 `vis_v2_merged_val`?
5. **prompt**:训练用数据原 prompt "Flatten and fold the cloth. Vertical Flod v1."(默认对齐)还是规范化(去掉 typo "Flod"→"Fold"、去尾空格)?

---

## 关联
- 数据: `kai0/data/Task_AV1/base/`(TOS `tos://transfer-shanghai/KAI0/Task_AV1`,watchdog 每 10min 自动同步)
- 夹爪处理对照: [`gripper_action_clip_experiment.md`](gripper_action_clip_experiment.md)(action≡state / 夹爪给力)
- config 克隆源: `kai0/src/openpi/training/config.py:1798`(`pi05_flatten_fold_A_smooth800_dagger_full`)
- 同步脚本: `train_scripts/kai/data/sync_task_av1_from_tos.sh` + `~/task_av1_sync_watchdog.sh`
