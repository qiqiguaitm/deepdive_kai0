# 从 PaliGemma base 自训 pi0.5(不用 PI 机器人预训练)— 训练 Plan

> **建立**: 2026-06-10
> **目标(用户定档)**: 用**自己的数据**(kai0 base+dagger + vis base+dagger)、从 **PaliGemma VLM base** 起(**不 warm-start PI 的 pi05_base 机器人预训练 ckpt**)、单节点 **8×A100、几天**,训出**可部署的叠衣策略**(kai0 + vis 两本体,**非**跨本体泛化研究)。
> **依据**: 深度调研(101 agents / 19 源 / 25 claim 3-票核验,primary = pi0 `2410.24164` / pi0.5 `2504.16054` / Knowledge-Insulation `2505.23705` / openpi repo)+ kai0 代码事实(本仓 openpi fork)。
> ⚠️ **方法学铁律**(本项目): **真机为终判,offline MAE 系统性反指**。VLA 训练报告先看 **val MAE**(不是 train loss)。

---

## 0. ⭐ 裁决:可收敛吗?

**可以收敛到可部署单任务策略 —— 但有两个硬条件,且真实算力 ≫ "几天"。**

| 维度 | 结论 |
|---|---|
| **一般性命题**(从 PaliGemma base、跳过机器人预训练能否收敛) | 🟢 **YES(高置信)**。PI 自己的消融:from-scratch(PaliGemma init + 随机 action expert)在 LIBERO 拿 96.6/97.2/94.6/84.8/92.7 ≈ from-generalist 98.0/...;pi0 论文:from-scratch 在所有测试任务都 work,预训练"有时快 2×"但非必需;open-pi-zero 从 PaliGemma 训到 SIMPLER 87.9%/97.8%,单机 8 卡 1-2 天 |
| **你的精确场景**(真机/双臂/可变形布/长程/百万帧/2本体) | 🟡 **marginal-to-YES**。最强 from-scratch 证据全来自**仿真 + 单臂/单本体 + 短程 + ~50 demos**;你是**真机双臂可变形长程**。**大 in-domain 数据(7.7M帧)是有利因素**,但没有 primary 源直接证过"从 base 训真机双臂叠衣" → 谨慎乐观 |
| **必须用对配方** | 🔴 **naive flow-matching from base 是错的**(见 §2):慢 7.5× + 毁 VLM。需用 **Knowledge Insulation (KI)** 或接受 naive 的慢/风险 |
| **算力现实** | 🔴 **从 base 约 240k step(planning anchor)**,8×A100 实测 ≈ **1-2 周**,**远超"几天"**(见 §5)。这是最大冲突点 |

> **一句话**: 技术上能收敛、有 primary 证据支撑;但 (a) 必须 KI-style 不能 naive,(b) 8×A100 跑满配方要 ~1-2 周不是几天,(c) 真机双臂叠衣这个具体域没人直接证过。**如果"可部署"是硬目标 + 时间紧,warm-start 仍是最稳的(但你已排除)。**

---

## 1. pi0 / pi0.5 训练逻辑 + 阶段顺序(调研核验)

**pi0**(`2410.24164`,3-0):两阶段
1. **机器人预训练** >10k 小时(7 robot configs / 68 tasks + 全 OXE)→ 建广域知识 base
2. **后训练** 在窄高质量 per-task 数据(~5–100+ 小时)上 finetune

**pi0.5**(`2504.16054`,3-0):两阶段,stage-1 目标不同
1. **stage-1 = 280k step**,广域机器人+非机器人数据,**离散 FAST-token 的 next-token-prediction**(预测文本/物体位置/动作 token)。⚠️ **97.6% 数据来自其它机器人/跨本体/web**,只 ~400h(2.4%)是目标任务
2. **stage-2 = 80k step**,特化到目标域 + **新增随机初始化的 flow-matching action expert**(NTP + flow-matching 联合);action expert "post-training 开始时随机初始化"

**关键**: 两者都**从 web 预训练 VLM(PaliGemma)初始化**,机器人预训练是"广域知识"层,不是收敛前提。**web/跨本体数据驱动的是泛化(尤其 OOD)**,对单一 in-distribution 目标任务**不是收敛必需**(`2504.16054`:no-WD 在 in-distribution mock-home 差异"不显著")。

---

## 2. 三个必须避开的失败模式(决定配方)

| # | 失败模式 | 证据 | 对策 |
|---|---|---|---|
| **F1** | **naive flow-matching from base**(pi0 配方:随机 action expert 直接 + VLM 联合训)| 随机 action expert 梯度**污染预训练 VLM backbone** → **训练慢 7.5×** + **语言跟随退化**(`2505.23705`,3-0;Fig6b) | 用 KI(F2 对策),或接受慢 + 单 prompt 场景下语言退化影响小(见 §4 路径 B1) |
| **F2** | **冻结 VLM**(以为冻住就不污染) | 冻结的 web-VLM 表示**对机器人不够** → **0%**(叠衫 ~10% vs pi0.5+KI ~55%,KI 论文 Fig4a/8) | **绝不冻 VLM**,backbone 必须 finetune |
| **F3** | **VLM grounding 灾难性遗忘** | action expert 梯度降 backbone 表示 → 模型忽略指令(去抓垃圾而非勺子)| **web/VL co-train**(泛化提升最大) **或** KI stop-gradient(无 web 数据时尤其有用) |

→ **正解 = Knowledge Insulation (KI)**(PI 当前 SOTA 配方,`2505.23705`):
1. **stop-gradient** 在 action-expert↔VLM backbone 的 attention KV 路径 → 随机 action expert 权重**永不改 VLM**
2. backbone 上加 **FAST 离散动作的 NTP loss**(仅训练期当表示学习信号)+ 轻量 VL/web 数据
3. action expert 用 flow-matching 训连续动作(**α=1**,因为现在作用在独立权重上)
→ 训得和 pi0-FAST 一样快、稳定收敛。

---

## 3. kai0 代码现状(本仓事实)

| 项 | 现状 |
|---|---|
| **你要的 init 路径** | `weight_loaders.PaliGemmaWeightLoader`(`kai0/src/openpi/training/weight_loaders.py:64`)**已实现**:加载官方 PaliGemma 权重 + action expert 保持随机。**但全仓 25+ config 没一个用它**(全用 `CheckpointWeightLoader` 从 `pi05_base` warm-start)→ **你需要手接** |
| 三个 loader | `NoOpWeightLoader`(全随机,不要)/ `CheckpointWeightLoader`(pi05_base warm-start,你排除)/ **`PaliGemmaWeightLoader`(你的路径)** |
| 模型 | `pi05=True`:PaliGemma **gemma_2b** + action expert **gemma_300m** + SigLIP So400m/14 ≈ **3.3B**;discrete_state_input + adaRMSNorm 注入 flow 时间步 + max_token 200;action_dim 32 / **horizon 50**(~1.67s@30Hz) |
| 现有(warm-start)超参 | AdamW · cosine warmup1k / peak **1.5e-5** / decay 50k→1.5e-6 · EMA **0.9999** · batch **128** · fsdp **8** · **50k step** · resize224 · 无 quantile norm |
| **双本体机制(可复用)** | `pi05_kaivis_perdsnorm_cond`:**domain conditioning(2 域 kai/vis)+ domain_weights (1.0, 3.97)** 做帧级 1:1 平衡(kai 5.78M / vis 1.46M 帧)→ 你的 co-train 直接套 |
| **KI 训练路径** | 🔴 **kai0/openpi 都没有**。stop-grad + FAST-NTP co-train 头**需自己实现**(openpi issue #365/#814 也是用户在问)→ **非平凡工程** |
| 数据 | kai0_base 3055ep/3.36M + kai0_dagger 3457ep/2.42M + vis(vis_v2_full)1406ep/1.93M + vis_dagger(待补)≈ **8.5k ep / 7.7M+ 帧** |

---

## 4. 推荐配方(两条路径,按工程量/风险)

> 都满足:`PaliGemmaWeightLoader` init、**不冻 VLM**、双本体 domain-cond co-train、per-embodiment norm。

### 路径 B1 — naive-from-base(工程最小,慢但可行)⭐ 给你这个目标的务实首选
**理由**: 你是**单任务、单一固定 prompt("Flatten and fold the cloth")、纯部署**——F3(语言遗忘)对你**影响小**(不需要开放世界语言跟随)。所以可以接受 naive flow-matching 的"语言退化",只需吞下"慢 7.5×"= 多跑 step。**几乎零额外工程**(只换 weight_loader + 加 step + 调 LR)。
- **init**: `weight_loader = PaliGemmaWeightLoader()`(action expert 随机)
- **不冻 VLM**;optimizer AdamW
- **LR**: 比 finetune 高一档 + 更长 warmup —— **peak 3e-5~5e-5 / warmup 2k-5k / cosine decay → 1/10**(随机 action expert 需更激进起步;open-pi-zero 用 5e-5/global-batch-1024 作参考)
- **steps**: **planning anchor ~240k**(openpi DROID:from-PaliGemma 240k vs from-robot-ckpt 100k);可先订 **150k 看曲线再续**
- EMA 0.9999 · batch 128(显存够可 192-256)· fsdp 8 · horizon 50
- **双本体**: domain conditioning(套 `kaivis_perdsnorm_cond`),domain_weights 按帧级平衡 + **可上调部署目标本体权重**
- **per-embodiment norm**: kai0 / vis **各自算 norm_stats**(关节 scale/DoF 不同,绝不共用)
- **数据配比**: 4 个数据集(kai base/dagger + vis base/dagger)co-train,大致按帧数比 + 部署目标本体适度上权

### 路径 B2 — KI-from-base(正解,工程重)
若要**又快又保 VLM grounding**(或将来要做语言条件/多任务),实现 KI:
- stop-gradient(action-expert→backbone attention KV)+ **FAST 离散动作 NTP 头**(训练期)+ flow-matching action expert(α=1)+ 轻量 VL co-train
- 同样 PaliGemmaWeightLoader init、不冻 VLM、双本体 co-train
- 收敛快(≈ pi0-FAST)、稳;但 **kai0/openpi 无现成实现 → 需 ~1-2 周工程**(改 model forward 加 stop-grad + 加 FAST tokenizer/NTP loss + co-train 数据管线)

### 收敛判据(两路径通用)
- **val action MAE**(held-out,**先看这个**,不是 train loss)逐 ckpt 曲线 → 单调降 + plateau
- 周期性 **sim/真机 rollout 成功率**(终判)
- **语言跟随 spot-check**(B1 尤其:换个无关 prompt 看动作是否变,检测 grounding 是否还在/已塌)

---

## 5. ⚠️ 算力现实(必须正视的冲突)

| 配置 | step | 8×H100 wall-clock | **8×A100 估算** |
|---|---|---|---|
| from **robot ckpt**(warm-start,你排除) | ~100k | ~2 天 | ~3-4 天 |
| from **PaliGemma base**(你的路径) | **~240k** | ~5 天 | **~1-2 周** |

→ **"几天"装不下 from-base 满配方(~240k step)。** 选项:
1. **接受 ~1-2 周**(订满 240k);
2. **砍 step 到 ~120-150k**(欠收敛风险,先出 ckpt 看 val MAE 曲线决定续不续);
3. **只训部署目标单本体**(数据少一半,可能更快收敛,但丢另一本体——与"2 robots 部署"目标冲突);
4. **(诚实)若可部署是硬约束 + 时间紧 → warm-start 仍最稳**(你已排除,但保留在桌面)。

---

## 6. 诚实权衡 + 最终建议

| | from-base(你的选择) | warm-start pi05_base(对照) |
|---|---|---|
| 收敛到可部署 | 🟡 能,但慢(240k)+ 真机双臂域未被直接证过 | 🟢 最稳(你既有 work 锚点 smooth_800 MAE@1=0.0089) |
| 算力 | 🔴 ~1-2 周 8×A100 | 🟢 ~10-20k step 续训即可 |
| 工程 | B1 小 / B2 重(KI 需自实现) | 零(现成 config) |
| 价值 | 自主可控、不依赖 PI 权重;科研意义 | 快、稳、省 |

**建议(按你"可部署 + 几天 + 从 base"的约束排序)**:
1. **先跑 B1(naive-from-base)的小规模 sanity**:`PaliGemmaWeightLoader` + 150k step + LR 3e-5/warmup3k,双本体 co-train。**~3-5 天出第一个 ckpt 扫 val MAE 曲线** → 判断是否在收敛轨道上。单任务单 prompt 下 naive 的语言退化不致命,这是性价比最高的验证。
2. **若 val MAE 明显下降**(进入收敛轨道)→ 续到 240k + 真机 rollout 终判。
3. **若想要快/保 grounding 且愿投工程** → 上 B2(KI)。
4. **全程保留 warm-start 作对照基线**(同数据、同 eval),量化"自训 vs PI 预训练"的真实差距 —— 这也回答了你科研层面的问题。

---

## 7. 落地步骤

1. **建 config** `pi05_kaivis_from_paligemma`(克隆 `pi05_kaivis_perdsnorm_cond` → 改 `weight_loader=PaliGemmaWeightLoader()`、LR/warmup/steps 按 §4 B1)。注册到 `config.py`。
2. **数据**: 合并 kai base/dagger + vis base/dagger(复用 `kai_vis_merged` build),**per-embodiment norm_stats 各自算**(`compute_norm_states_fast.py`)。
3. **PaliGemma 权重就位**: `PaliGemmaWeightLoader` 默认拉 `gs://vertex-model-garden-paligemma/.../pt_224.npz` → 离线环境需先下到本地 cache,改 loader 路径。
4. **提交 8 卡训练**(cnsh/cnbj 8×A100/H20,fsdp8,batch128,150k step,每 5-10k save)。
5. **监控**: val MAE 曲线(`vis_v2_merged_val` 同协议)+ loss;~50k 起看是否在降。
6. **判定**: 150k 出 ckpt → val MAE vs warm-start 基线 → 决定续 240k / 真机 rollout / 转 B2 / 回 warm-start。

---

## 8. 开放问题(调研未能直接 de-risk)
1. 有没有人发表过**真机双臂可变形/叠衣、百万帧、from-PaliGemma-base(无机器人预训练)**的收敛结果?(目前证据全是仿真单臂短程)
2. 双本体 co-train 的最优配比 + per-embodiment norm:部署目标本体该上权多少?混训 2 本体 vs 每本体单独训一个模型,哪个更好?
3. PI "from scratch" LIBERO run 的完整超参表(LR/warmup/wd/EMA/horizon/FAST vocab/web 配比)论文未给全 → §4 数字是起点需调。
4. kai0/openpi 里 KI(stop-grad + FAST-NTP + flow expert)的最小正确实现?有无社区 fork 现成?

---

## 关联
- 数据/已有 work 锚点: `task_a_new_smooth_800_new_norm_results.md`(warm-start MAE@1=0.0089 基线)
- 双本体已有: `pi05_kaivis_perdsnorm_cond`(config.py:1036)domain conditioning + domain_weights
- 代码: `kai0/src/openpi/training/weight_loaders.py:64`(PaliGemmaWeightLoader)· `config.py`(新建 config)
- 调研 primary: pi0 `2410.24164` · pi0.5 `2504.16054` · Knowledge Insulation `2505.23705` / `pi05_KI.pdf` · open-pi-zero(github.com/allenzren/open-pi-zero)· openpi DROID `README_train.md`
