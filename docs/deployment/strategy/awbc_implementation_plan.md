# AWBC / RECAP Advantage 升级实施方案

> **目的**: 把当前 deepdive_kai0 简化版的 "全 dagger 标 positive" 升级为完整 RECAP 4-step pipeline — 用 advantage estimator 给每帧打 ground-truth advantage 值, 再做 advantage-weighted behavior cloning.
> **现状**: 简化版已在用 ([`../../train_scripts/data/label_dagger_positive.py`](../../train_scripts/data/label_dagger_positive.py)); 完整版**未启动**, 触发条件未达成 (val MAE@1 仍在下降).
> **关键依赖**: 需要 DAgger 走 Form C (双 dataset 分离, inference + dagger), 见 [`dagger_implementation_plan.md`](dagger_implementation_plan.md) §4.5.
> **上游参考**:
> - 论文: [RECAP (Physical Intelligence 2025-11, arXiv 2511.14759)](https://arxiv.org/abs/2511.14759)
> - KAI0 上游实现: `/data1/tim/workspace/kai0/stage_advantage/` (已开源, 完整 4-step)
> - 预标注数据集: `Task_A/advantage/` (KAI0 官方 HuggingFace / ModelScope, 跳过 Stage 0-3 直接 Stage 4 训练可用)

---

## 1. 算法定位 (一句话)

VLA 训练时不再"每帧等权 BC", 而是**按每帧 advantage 值加权 BC** — 高 advantage 帧 (人类示范 / 策略成功段) 权重大, 低 advantage 帧 (策略失败段) 权重小或负. Advantage 由一个独立训练的 estimator (基于 pi0 fine-tune) 学习得到, 监督信号是手动标的 stage_progress_gt.

**与 DAgger 简化版的差别**: 简化版按 dataset 来源 (dagger=positive, inference=negative) 二值标签; 完整版按 frame 级 advantage 连续值标签.

---

## 2. AWBC 适配 / 不适配场景

| 场景 | 是否适合 | 理由 |
|---|---|---|
| **DAgger 已迭代 N 轮, val MAE@1 平台 (不再下降)** | ✅ 首选 | 数据增加边际收益递减, 需要更细致的 frame-level 加权 |
| **失败模式仍是"分布外 / 不会"类 (DAgger 适配)** | ✅ | RECAP 设计目的就是 boost DAgger 数据的训练效率 |
| **多 sub-phase 任务** (Task_A flatten-fold 多步骤) | ✅ | stage_progress_gt 天然按 subtask 标注 |
| **失败是 action edit 类 (精度差 mm)** | ❌ | 走 RLT (论文实证 RLT 解 action edit 远优于 BC 类) |
| **新任务从零建立能力** | ❌ | AWBC 假定已有 baseline policy, bootstrap 阶段用纯 DAgger |
| **单 sub-phase 任务 (无明显 stage 划分)** | ⚠️ | stage_progress_gt 标注难, 退化为按 progress 单调递增标 |

**适配的失败模式特征**: (a) DAgger 加 demo 效果递减, (b) 任务有 ≥ 2 个清晰 sub-phase 边界 (易于人工标 stage_progress_gt), (c) policy 已能完成 30%+ 任务 (有 inference rollout 可用作 negative samples).

---

## 3. 4-Step Pipeline 详解

### Stage 0 — 手动标 `stage_progress_gt` (一次性)

**目标**: 为每帧打 stage progress 值, 作为 Stage 1 estimator 的监督信号.

**流程**:
1. 对每个 episode, 人工标:
   - Episode 起止时间戳
   - Subtask 边界时间戳 (例如 Task_A 2 阶段任务: Stage 1 = 抓 → Stage 2 = 对折)
2. 脚本根据时间戳自动算每帧的 `stage_progress_gt` (0-1 浮点, 阶段内单调递增)
3. 写入 parquet 列

**工具**: KAI0 上游 `stage_advantage/annotation/README.md` 提供脚本 + 标注规范.

**输入**: Form C dataset (推荐用 inference + dagger 混合, 至少 200 episode)
**输出**: 同 dataset, parquet 多一列 `stage_progress_gt`
**工作量**: 标 500 episode (Task_A 当前体量) ~ 1 周 (操作员 + 标注员协作), 或者**用 KAI0 预标注的 `Task_A/advantage/`** 跳过此步.

### Stage 1 — 训 Advantage Estimator

**目标**: Fine-tune pi0 模型, 用 stage_progress_gt 监督回归 advantage 值.

**命令** (KAI0 上游):
```
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
    --exp_name=run1 --save_interval 10000
```

**关键 config 项** (来自 `kai0/src/openpi/training/config.py` ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD):
- 基础模型: pi0 (不是 pi0.5)
- 数据集: Form C inference + dagger 混合
- Loss: 2-step relative advantage regression (frame n vs frame n+50)
- `skip_norm_stats=True` — estimator 不需要 norm
- 通常 50k-100k step, 1-2 天 (8×A100)

**关键原理**:
- 输入: (images, state) — 与训 policy 一致
- 输出: advantage 标量 (取代 action 输出 head)
- 训练时见到 inference 段的"失败状态" + dagger 段的"成功状态", 学会区分

**为什么需要 inference 段**: 这一步是 Form C 的核心依据. 没有 inference 段 → estimator 没法学到"什么算低 advantage" → 跳过这步会让 Stage 2 输出全部 positive (失去意义).

**输出**: trained estimator ckpt (~7GB)
**工作量**: 1-2 天集群训练

### Stage 2 — 预测 Advantage 标到 dataset

**目标**: 用 Stage 1 的 estimator 给整个 dataset 每帧打 advantage 值.

**命令**:
```
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset
```

**输出**: 新 parquet (路径如 `data_KAI0_100000/data/chunk-000/`), 含两列额外:
- `relative_advantage`: frame n 与 frame n+50 的 progress 差值
- `absolute_advantage`: frame n+50 absolute value 与 frame n 的差值, 截到 [-1, 1]

**实现细节** (`kai0/stage_advantage/annotation/evaluator.py`):
- 批处理 GPU 推理 + 并行视频解码
- 整个 Task_A 500 episode ~ 几小时

**工作量**: 半天

### Stage 3 — Discretize Advantage → task_index + tasks.jsonl

**目标**: 把连续 advantage 值二值化 / N-bin, 转成 `task_index` (int64), 配合 `meta/tasks.jsonl` 给每个 task_index 一个 prompt 字符串.

**命令**:
```
bash stage_advantage/annotation/discretize_advantage.sh
```

**典型 binning** (`discretize_advantage.py`):
- 二值: advantage > median → "Advantage: positive"; ≤ median → "Advantage: negative"
- 也支持 N-slice 分箱 (例如 top-25% / mid-50% / bot-25%)

**输出**: 同 dataset, 每帧 `task_index` 指向 `tasks.jsonl` 中的 prompt
**工作量**: 半天

**与 `label_dagger_positive.py` 对比**:
- `label_dagger_positive.py`: 把所有 dagger 帧标 positive, 不看 advantage 值 — 等价于 "用 dataset 路径粗暴当 advantage"
- Stage 3 完整版: 按真实 ground-truth advantage 值打标, frame-level 精度

### Stage 4 — AWBC 训练

**目标**: 用 advantage-weighted BC 训 policy.

**命令**:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc \
    --exp_name=run1
```

**核心机制 — `prompt_from_task`**:
- 训练时不用固定 prompt, 而是从每帧的 `task_index` 查 `tasks.jsonl` 得到 prompt
- 即 "Advantage: positive" 帧用 positive prompt 学 (强 BC)
- "Advantage: negative" 帧用 negative prompt 学 (相当于 "学到的关联是: negative prompt → 这类动作")
- 推理时永远 prompt = "Advantage: positive" → 模型只生成 positive 风格动作

**与现有 `pi05_flatten_fold_awbc` config** ([`../../train_scripts/launch/run_awbc_daggeronly_gf0.sh`](../../train_scripts/launch/run_awbc_daggeronly_gf0.sh)):
- 现有: 数据来源 `Task_A/dagger_labeled` (label_dagger_positive 输出), 所有帧 positive
- 升级后: 数据来源 `Task_A/advantage` (Stage 3 输出), 按真实 advantage 区分

**工作量**: 与现有 SFT 训练相同, 30k-60k step, 2-3 天 (gf1 8×A100)

---

## 4. 现有简化版 vs 完整版对比

| 维度 | `label_dagger_positive.py` (现有简化版) | RECAP 完整版 (本方案) |
|---|---|---|
| Advantage 来源 | 数据集路径 (`Task_A/dagger/*` 全标 positive) | Estimator 预测的 frame-level 值 |
| 是否需要 inference 段 | ❌ 否 (当前 Form A 没有) | ✅ 是 (训 estimator 需要) |
| 是否需要 Stage 0 标注 | ❌ 否 | ✅ 是 (~1 周人工; 或用 KAI0 预标注) |
| Advantage 分辨率 | 二值 (整段 positive / 不录 negative) | 连续值 → 二值或 N-bin |
| Negative 样本 | 无 | inference 段失败帧 |
| 训练 config | `pi05_flatten_fold_awbc` (现成) | 同 config, 不同数据集路径 |
| 工作量 | 0 (已实现) | 4-5 天 (跳过 Stage 0 用预标注) / 2 周 (含 Stage 0) |
| 训练效果 (论文) | baseline | +显著提升 (RECAP 论文实测 ~2× throughput on hard tasks) |

---

## 5. Phase 拆分

| Phase | 任务 | 工作量 | 关键文件 / 命令 | 前置依赖 |
|---|---|---|---|---|
| **A0** | **决策点检查**: DAgger 简化版 val MAE 是否平台? 失败模式是否仍是 DAgger 适配? | 0.5 天 | 看 [`../../training/history/00_action_only_finetune_history.md`](../../training/history/00_action_only_finetune_history.md) | DAgger 已迭代 ≥ 3 轮 |
| **A1** | **数据准备**: 确认 Form C dataset (inference + dagger) 已积累 ≥ 200 ep | 0.5 天 | DAgger Phase D5 完成 | [`dagger_implementation_plan.md`](dagger_implementation_plan.md) Phase D1-D5 |
| **A2a (路径 a)** | 用 KAI0 预标注 `Task_A/advantage/` 直接 Stage 4 | 0 天数据准备 | `scripts/download_dataset.py` 拉 HuggingFace dataset | 任务结构与 Task_A 兼容 |
| **A2b (路径 b)** | 自有数据 Stage 0 手动标 stage_progress_gt | 1 周 | `stage_advantage/annotation/README.md` SOP | 操作员 + 标注员 |
| **A3** | Stage 1 训 Advantage Estimator (gf1 集群) | 1-2 天 | `scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` | A2a 或 A2b 完成 |
| **A4** | Stage 2 Predict advantage on 自有 dataset | 0.5 天 | `stage_advantage/annotation/eval.py Task-A KAI0 <dataset>` | A3 完成 |
| **A5** | Stage 3 Discretize → task_index | 0.5 天 | `discretize_advantage.sh` | A4 完成 |
| **A6** | Stage 4 AWBC 训练 + sim01 部署评估 | 2-3 天 | `scripts/train.py pi05_flatten_fold_awbc` (config 改 repo_id) | A5 完成 |

**两条路径总工作量**:
- **路径 a (用 KAI0 预标注)**: A0 + A1 + A2a + A3 + A4 + A5 + A6 = **5-6 天**
- **路径 b (完整自标)**: A0 + A1 + A2b + A3 + A4 + A5 + A6 = **2-3 周**

**推荐**: 先走路径 a (用 KAI0 预标注 estimator 测试效果). 若效果好 → 路径 b 投入自标 stage_progress_gt 给自有数据.

---

## 6. 资源与时间预算

| 资源 | 路径 a | 路径 b |
|---|---|---|
| **工程时间** | 5-6 天 | 2-3 周 |
| **人力 (操作员/标注员)** | 0 | ~1 周 (Stage 0 标 500 ep × subtask 边界) |
| **GPU 时间** | Stage 1 ~1-2 天 (8×A100) + Stage 4 ~2-3 天 (8×A100) | 同 |
| **存储** | Estimator ckpt ~7GB + AWBC ckpt ~7GB + dataset 重复 ~10GB | 同 |
| **数据集体量** | ≥ 200 ep Form C (inference + dagger) | 同 |
| **真机时间** | 0 (训练 + 离线 eval) + sim01 部署评估 ~2h | 同 |

---

## 7. 风险与兜底

| 风险 | 概率 | 影响 | 兜底 |
|---|---|---|---|
| KAI0 预标注 `Task_A/advantage/` 与自有 Task_A 任务结构不一致 | 中 | 大 | 路径 b 自标 stage_progress_gt |
| Estimator 在 inference 段失败状态拟合不充分 (因为 inference ep 比 dagger 少) | 中 | 中 | 多收一轮 DAgger 增加 inference ep 比例; 或者从 RLT replay buffer 抽 transition 反向喂 estimator |
| Stage 3 discretize 阈值不合适 (advantage 分布偏) | 低 | 中 | 改 binning 阈值 (top-25% vs median), 多试几个 config |
| AWBC config `prompt_from_task` 与 deepdive_kai0 现有 pi0.5 sidecar 不兼容 | 中 | 中 | 复用 sim01 部署的 sidecar 注册机制 (已有模板) |
| 训完 ckpt 效果不如简化版 | 低 | 中 | 退回 `label_dagger_positive.py` 简化版; 留 RECAP 数据归档备用 |

---

## 8. 与 DAgger / RLT 的关系 (重要)

| 阶段 | DAgger 主轨 | AWBC 升级 (本方案) | RLT 升级 |
|---|---|---|---|
| **数据来源** | 持续录 dagger (人示范) | DAgger Form C 累积的 inference + dagger | DAgger ckpt 当 RLT 的 frozen VLA |
| **修改对象** | VLA 全模型 (~5B) | VLA 全模型 (~5B) | actor 1M, VLA 冻结 |
| **训练成本** | 30-60k step / 实验 | Stage 1 + Stage 4 双训 | 几小时真机 + 极小算力 |
| **适合的失败** | 分布外 / 模型不会 | DAgger 已平台, 同类失败 | action edit / 单 channel offset |
| **触发顺序** | 第 1 轮 | DAgger 平台后 | DAgger 或 AWBC 平台后, 走 critical phase |

**完整升级路径**:
```
DAgger (Form C) → DAgger 加 demo 持续 → val MAE 平台 → AWBC 升级 → val MAE 再平台 → RLT 修关键 phase
```

每一步都是 **independent + composable** — RLT 用 AWBC 训出的 ckpt, AWBC 用 DAgger 录的数据, DAgger 不变.

---

## 9. 与上游 / 相关文档跳转

- 主轨 DAgger 实施 → [`dagger_implementation_plan.md`](dagger_implementation_plan.md) (Form C 决策 + Phase D1-D5)
- 另一支升级 (RLT) → [`rlt_implementation_plan.md`](rlt_implementation_plan.md)
- KAI0 上游 advantage pipeline → `/data1/tim/workspace/kai0/stage_advantage/README.md`
- RECAP 论文 → [arXiv 2511.14759](https://arxiv.org/abs/2511.14759)
- 现有简化版实现 → [`../../train_scripts/data/label_dagger_positive.py`](../../train_scripts/data/label_dagger_positive.py)
- 现有 AWBC 训练入口 → [`../../train_scripts/launch/run_awbc_daggeronly_gf0.sh`](../../train_scripts/launch/run_awbc_daggeronly_gf0.sh)
- KAI0 预标注数据集 → <https://huggingface.co/datasets/OpenDriveLab-org/Kai0> / <https://www.modelscope.cn/datasets/OpenDriveLab/Kai0>
- 跨本体战略主文档 → [`cross_embodiment_strategy.md`](cross_embodiment_strategy.md)
