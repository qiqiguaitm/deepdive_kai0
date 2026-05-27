# X-VLA Track X — X3.A / X3.B / X3.C Stage A 对照实验结果

> **作用**: 系统记录 Track X (X-VLA 官方架构 native 训练) 三个对照实验的 Stage A ckpt eval 结果, 为 Track X 路线决策提供数据支撑。
>
> **背景**: Track X 用 `lerobot/xvla-base` (0.9B Florence2 + 24-layer SoftPromptedTransformer + EE6D 20D action) 在 uc01/02 8 A800 单节点训练。三个对照量化 (a) XVLA-Soft-Fold 第三方数据是否帮 vis 部署 (X3.A vs X3.B) 以及 (b) Stage A multi-domain continual pretrain 是否必要 (X3.B vs X3.C)。
>
> **🔥 主要结论 (2026-05-25)**: **X3.C (跳过 Stage A, 直接 vis-only fine-tune `xvla-base`) 完胜**:
> - MAE@1 0.0200 vs X3.B 0.0343 (**-42%**) vs X3.A 0.0380 (**-47%**)
> - **Stage A multi-domain continual pretrain 是 net-negative**, 不应该做
> - Track X 终态用 X3.C 路线, 节省 ~60 GPU-h 训练 + 跳过整个 Stage B
>
> **最近更新**: 2026-05-25 (X3.A/B/C Stage A 全部 eval 完成, 路线重大调整)
>
> **关联文档**:
> - `docs/deployment/strategy/cross_embodiment_strategy.md` §1 (3 robots) + §5.2 (Soft Prompt) + §7 (Tri-track)
> - `../../future_plans/plans/xvla_track_x_curriculum.md` — Track X 完整计划 + curriculum 设计
> - `00_training_history.md` — 全量训练历史索引
> - `xvla_conditioning_methods_results.md` — pi0.5 + 不同 conditioning 方式对照 (Hard / Soft / Action Head)

---

## 1. 实验设计

### 1.1 三个 ablation cell

| 实验 | Stage A 数据 | Domain weights | 目的 |
|---|---|---|---|
| **X3.A** | A (KAI0) + B (vis) + **C (XVLA-Soft-Fold)** | kai 1× : vis 7× : xvla 2× | 3-domain, 含第三方 |
| **X3.B** | A (KAI0) + B (vis) | kai 1× : vis 7× | 2-domain, 无 XVLA — 对照 X3.A 测 XVLA 贡献 |
| **X3.C** | **B (vis) only** | vis 1× | 跳过 Stage A multi-domain pretrain, 直接 vis-only finetune — 对照 X3.B 测 Stage A 必要性 |

### 1.2 共享配置

| 项 | 值 |
|---|---|
| Init | `lerobot/xvla-base` (HF 官方 0.9B Phase I ckpt) |
| Architecture | Florence2-Large + 24-layer SoftPromptedTransformer (1024D, 16 heads) |
| Action | **EE6D 20D** chunk (xyz + Rot6D + grip per arm × 2 arms × 30 step horizon) |
| Vision input | 3 cameras: top_head + hand_left + hand_right (resize+pad) |
| Language | bart-large tokenized fixed prompt: `"Flatten and fold the cloth."` |
| Backbone freeze | First `freeze_steps` step VLM frozen, then unfroze for joint optim |
| Optimizer | AdamW (β=0.9/0.95), weight_decay=1e-4 |
| LR schedule | Cosine warmup + decay |
| Resource | uc01 / uc02 single-node 8 A800 |
| Rate | ~0.86 it/s |
| Save interval | every 2000 step + step_final |

### 1.3 仅四个变量差异

| 实验 | Steps | LR (peak) | warmup | freeze_steps | Stage A skipped? |
|---|---:|---:|---:|---:|---|
| X3.A Stage A | 20k | 1e-4 | 1000 | 1000 | no — multi-domain pretrain |
| X3.B Stage A | 20k | 1e-4 | 1000 | 1000 | no — multi-domain pretrain (no XVLA) |
| **X3.C** | 20k | 5e-5 | 500 | 1000 | **yes** — direct vis-only finetune |

> ⚠️ X3.C 用 lower lr (5e-5 vs 1e-4) 因为它从 raw `xvla-base` 出发的 finetune scenario, 不是 continual pretrain. 这是 plan §X3 design 的标准做法。

---

## 2. 训练状态

| 实验 | Host | 启动 | 完成 | 训练时长 | step_final ckpt 路径 |
|---|---|---|---|---|---|
| X3.A Stage A | uc02 | 2026-05-23 16:41 UTC | 2026-05-23 23:08 UTC | ~6.5h | `uc02:/data/shared/ubuntu/local_ckpts/xvla_x3a_stage_a/step_final/state_dict.pt` (3.3 GB) |
| X3.B Stage A | uc01 | 2026-05-23 16:41 UTC | 2026-05-23 23:09 UTC | ~6.5h | `uc01:/data/shared/ubuntu/local_ckpts/xvla_x3b_stage_a/step_final/state_dict.pt` (3.3 GB) |
| X3.C | uc02 | 2026-05-25 15:55 UTC | 2026-05-25 22:18 UTC | ~6.3h | `uc02:/data/shared/ubuntu/local_ckpts/xvla_x3c_vis_only_direct/step_final/state_dict.pt` (3.3 GB) |

> X3.A/B 同时启动并行 (uc02 / uc01 各一台). X3.C 在 X3.A 完成后接着用 uc02 跑.

---

## 3. Eval 结果 (vis_v2_merged last 50 ep × 20 queries)

> **协议**: 用 `LeRobotEE6DDataset` 加载 last 50 episodes of vis_v2_merged (与训练 vis 数据同源, 但 model 没 update 这部分 ep — 用作 in-domain val). 每 ep 取 20 个 query frame (uniformly spread, exclude last 50 frames). 单 ckpt eval 在 1 GPU 上 ~3-4 min.
>
> **Eval 脚本**: `/tmp/eval_xvla_x3.py` (X-VLA EE6D 20D action MAE), `XVLAPolicy.predict_action_chunk()` API.
>
> **Horizon limit**: X-VLA 输出 30-step action chunk, 所以 `MAE@50` 不可用 (chunk 不够长).

### 3.1 完整 MAE 表 (EE6D 20D 空间, 越低越好)

| 实验 | n_ep | n_query | MAE@1 | MAE@10 | MAE@25 | MAE@50 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| **X3.A** (3-domain A+B+C) | 50 | 20 | 0.0380 | 0.0424 | 0.0574 | – | 含 XVLA 第三方数据 |
| **X3.B** (2-domain A+B) | 50 | 20 | 0.0343 | 0.0396 | 0.0561 | – | kai+vis (无 XVLA) |
| **X3.C** (vis-only direct) | 50 | 20 | **0.0200** ⭐⭐ | **0.0282** ⭐⭐ | **0.0475** ⭐⭐ | – | **全 horizon best — 跳过 Stage A 反而最好** |

### 3.2 X3.A vs X3.B Delta (量化 XVLA 数据贡献)

| Metric | X3.A | **X3.B (no XVLA)** | Δ vs X3.A |
|---|---:|---:|---:|
| MAE@1 | 0.0380 | **0.0343** | **-9.7%** |
| MAE@10 | 0.0424 | **0.0396** | **-6.6%** |
| MAE@25 | 0.0574 | **0.0561** | **-2.3%** |

→ X3.B 全 horizon 优于 X3.A. XVLA-Soft-Fold (C domain) 对 vis 部署是 **net-negative**: dilute vis prior, 反而损害最终 model 在 vis 上的精度.

### 3.3 X3.B vs X3.C Delta (量化 Stage A multi-domain pretrain 价值)

| Metric | X3.B (Stage A multi-domain pretrain) | **X3.C (skip Stage A, direct vis-only)** | Δ vs X3.B |
|---|---:|---:|---:|
| MAE@1 | 0.0343 | **0.0200** | **-42%** |
| MAE@10 | 0.0396 | **0.0282** | **-29%** |
| MAE@25 | 0.0561 | **0.0475** | **-15%** |

→ ⭐⭐ **X3.C 完爆 X3.B 全 horizon**. **Stage A multi-domain continual pretrain 是 net-negative**, kai 数据 dilute model 学习 vis 分布的能力。Direct vis-only fine-tune `xvla-base` 才是最佳路径。

### 3.4 三方对比可视化

```
MAE@1:    X3.A 0.0380  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
          X3.B 0.0343  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
          X3.C 0.0200  ▰▰▰▰▰▰▰▰▰▰ ⭐⭐ -42% vs B / -47% vs A

MAE@10:   X3.A 0.0424  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
          X3.B 0.0396  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
          X3.C 0.0282  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰ ⭐⭐ -29% vs B / -33% vs A

MAE@25:   X3.A 0.0574  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
          X3.B 0.0561  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
          X3.C 0.0475  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰ ⭐ -15% vs B / -17% vs A
```

---

## 4. 核心结论

### 4.1 ⭐⭐⭐ X3.C 完胜: 跳过 Stage A 直接 vis-only fine-tune 是最佳路线

| 排名 | 实验 | 路线 | MAE@1 | 训练成本 |
|---|---|---|---:|---:|
| 🥇 | **X3.C** | 跳过 Stage A, direct vis-only finetune | **0.0200** | 50 GPU-h |
| 🥈 | X3.B | Stage A (kai+vis) | 0.0343 | 52 GPU-h (+ Stage B 64h = 116 total 若全跑) |
| 🥉 | X3.A | Stage A (kai+vis+XVLA) | 0.0380 | 同 X3.B |

**X3.C 比 X3.B 好 42% (@1)**, 比 X3.A 好 47% (@1). **训练成本只有 X3.B+Stage B 总和的 43%**。

### 4.2 实际原因 (推测)

`lerobot/xvla-base` (290k ep × 30 domain) 已经是强 generalist VLA base. 加 6512 kai + 6265 vis (×7) "continual pretrain extension" 实际**覆盖** / **dilute** 了 base 中丰富的 generalist prior. **Direct vis-only fine-tune 反而最 surgically 地把 base 调整到 vis-specific 模式, 同时保留 base 的丰富知识**.

这与 X-VLA 论文 §3.3 的 Phase II adaptation 描述吻合 — 论文也是 "Phase I (pretrain 在 30 domain) + Phase II (direct task adaptation 不需要再 multi-domain pretrain)". 我们之前理解错了, 以为需要再做一次 multi-domain "extension pretrain".

**修正认知**:
- ❌ ~~Stage A continual pretrain on (A+B+C) → Stage B vis-only adapt~~ (X3.A/B 路线)
- ✅ **Direct vis-only fine-tune from `xvla-base`** (X3.C 路线, X-VLA 官方 Phase II 风格)

### 4.3 XVLA-Soft-Fold C-domain 也是 net-negative (X3.A vs X3.B)

X3.B 优于 X3.A (-9.7% @1). 这与 plan.md §8.8.3.2 中的假设 "X3.A < X3.B → XVLA 过于不同, dilute vis prior → 弃 XVLA" 一致.

X3.B vs X3.C 的更大 gap (-42%) 进一步说明: **任何加 kai/XVLA 的 Stage A pretrain 都损害 vis 部署精度**. 不是 XVLA 特别坏, 而是 multi-domain Stage A 这个步骤本身就有问题.

### 4.4 Track X 终态 ckpt 决策 (修订)

| 候选 ckpt | 用途 | 决策 |
|---|---|---|
| ✅ **X3.C step_final (vis-only direct)** | **新 Track X 终态** | 主推, 直接进入真机评估 |
| ❌ ~~X3.A step_final~~ | 已弃 | (被 X3.B/C 全 horizon 超越) |
| ❌ ~~X3.B step_final~~ | 已弃 | (被 X3.C 大幅超越 42%) |
| ❌ ~~Stage B vis-only adaptation~~ | **不需要做** | X3.C 已是更高效的等价方法 |

### 4.5 节省时间 / 计算 / 复杂度

- 不需跑 X3.B Stage B (10k step × 6h = ~60 GPU-h saved)
- 不需要复杂 multi-domain curriculum (Stage A + Stage B)
- 不需要 EE6D 转换 kai 数据 (kai parquet 仅用于 Stage A 现在弃用 — 但作为 paper 对照数据点保留)
- vis 真机评估直接用 X3.C ckpt 即可

---

## 5. 资源 + 时间消耗对比

| 实验 | Stage A 数据 ep | Stage A GPU-h | Stage B GPU-h (若做) | 总 |
|---|---:|---:|---:|---:|
| X3.A | A 6512 + B 6265 (vis ×7) + C 3458 (×2) = 16,235 | 8 × 6.5 = **52 GPU-h** | 8 × 8 (待启) = 64 | 116 |
| X3.B | A 6512 + B 6265 (vis ×7) = 12,777 | 8 × 6.5 = **52 GPU-h** | 8 × 8 (待启) = 64 | 116 |
| **X3.C** | B 895 (×1) = 895 | 8 × 6.3 = **50 GPU-h** | (无 Stage B 概念) | **50** |

→ X3.C 训练成本仅 X3.A/B 总和的 **43%** (50 vs 116 GPU-h). 如果 X3.C ≈ X3.B, **首推 X3.C 简化路线**.

---

## 6. 时间线

| 日期 | 事件 |
|---|---|
| 2026-05-22 (晚 战略转向) | Track X 启动 (X-VLA 官方架构 native 训练), 加 X3.A/X3.B 两组对照 |
| 2026-05-23 08:40 UTC | X3.A (uc02) + X3.B (uc01) 同时启动 Stage A |
| 2026-05-23 23:08-23:09 UTC | X3.A + X3.B Stage A 完成 (~6.5h each) |
| 2026-05-25 09:55 UTC | X3.C (vis-only direct) 启动 on uc02 |
| 2026-05-25 ~13:00 UTC | X3.A/B Stage A eval 完成 — **X3.B 全 horizon 完胜 X3.A** |
| 2026-05-25 22:18 UTC | X3.C 训练完成 |
| 2026-05-25 22:?? UTC | X3.C eval 完成 |

---

## 7. Eval 命令速查

### 7.1 X-VLA eval pipeline (custom, 不是 pi0.5 eval)

```bash
ssh uc02  # 或 uc01 — 看 ckpt 在哪
source /data/shared/ubuntu/workspace/X-VLA-env/.venv/bin/activate
cd /data/shared/ubuntu/workspace/xvla_scripts
python /tmp/eval_xvla_x3.py \
    --ckpt /data/shared/ubuntu/local_ckpts/xvla_x3<a/b/c>_stage_a/step_final/state_dict.pt \
    --n-episodes 50 \
    --n-per-ep 20 \
    --out /tmp/eval_x3<a/b/c>_stage_a_final.json
```

### 7.2 关键 dependencies

- `lerobot.policies.xvla.modeling_xvla.XVLAPolicy.predict_action_chunk(batch)` — inference API (返回 EE6D 20D × chunk_size)
- `multi_domain_dataset.LeRobotEE6DDataset` — load val 数据 (同训练 pipeline)
- `XVLAPolicy.from_pretrained(/data/shared/ubuntu/workspace/xvla_ckpts)` 加载 base architecture, 再 `load_state_dict()` 我们的 ckpt

### 7.3 注意事项

- X-VLA chunk size = 30, 所以 MAE@50 不可计算
- prompt 必须是 "Flatten and fold the cloth." (与训练一致)
- domain_id 在推理时不强制传 (model.predict_action_chunk 内部处理)

---

## 8. 后续 (待办)

- [ ] X3.B Stage B 10k vis-only adaptation (若 D2 真机评估认为必要)
- [ ] 真机评估 X3.B vs X3.C (验证 offline EE6D MAE → 实际操作平滑度的相关性)
- [ ] X3.B vs C3.0 (Action Head Cond pi0.5) 对比 — 不同 conditioning 方案 vs X-VLA Soft Prompt 的 paper ablation
