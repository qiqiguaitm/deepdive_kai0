# Track X — X-VLA 官方架构 Native 训练 (X3.A + X3.B + X3.C)

> **状态**: ✅ Stage A 全部完成 (2026-05-23). Eval done (2026-05-25). X3.C ablation 加跑.
> **关联 task**: `#17 Track X X-VLA 官方架构训练`。
> **战略上下文**: [cross_embodiment_strategy.md](../../../deployment/strategy/cross_embodiment_strategy.md) §1 (3 robots) + §5.2 (Soft Prompt) + §7 (Tri-track)。

## ⭐ 主要结论 (2026-05-25 update)

**X3.A vs X3.B Stage A eval (vis_v2_merged last 50 ep × 20 q, EE6D 20D MAE)**:

| Metric | X3.A (3-domain A+B+C) | **X3.B (2-domain A+B, no XVLA)** | Δ |
|---|---:|---:|---|
| MAE@1 | 0.0380 | **0.0343** ✓ | -9.7% |
| MAE@10 | 0.0424 | **0.0396** ✓ | -6.6% |
| MAE@25 | 0.0574 | **0.0561** ✓ | -2.3% |

→ **X3.B 全 horizon 完胜 X3.A**, 确认 H1 假设: XVLA-Soft-Fold (第三方 C domain) **dilute vis prior**, 对 vis 部署是 net-negative. **Track X 终态用 X3.B 路线 (kai+vis only, 弃 XVLA)**.

## 1. 核心思路

用 LeRobot's `lerobot/xvla-base` 0.9B ckpt + custom multi-domain wrapper (`xvla_scripts/multi_domain_dataset.py` + `xvla_train.py`) 在 uc01/02 各 8 A800 上跑。EE6D 20D action (kai+vis 用 PiperFK + Rot6D 编码, XVLA-Soft-Fold 用预计算 `observation/eef_6d`)。

与论文 paper-faithful 不同点: 用 lerobot port 不是原 X-VLA repo (LeRobot wrapper 实现更简洁)。

**Curriculum**: continual pretrain (Stage A, multi-domain mixed) → vis-only adaptation (Stage B), 对齐 X-VLA Phase I' + Phase II 框架。

## 2. 数据状态 (全部就绪)

| 数据集 | EE6D 格式 | 路径 |
|---|---|---|
| kai0_base 20D EE6D parquet | 3055 ep / 3.36M frames | uc01/02 NFS |
| kai0_dagger 20D EE6D parquet | 3457 ep / 2.42M frames | 同 |
| vis_v2_merged 20D EE6D parquet | 895 ep / 1.06M frames | 同 |
| xvla_soft_fold action FK cache | 1542 files / 2.85M frames | 同 |

## 3. Prep ✅ 完成

| 项 | 状态 |
|---|---|
| HF ckpt `lerobot/xvla-base` (3.3GB) | ✅ uc01 NFS `/data/shared/ubuntu/workspace/xvla_ckpts/` |
| X-VLA env (lerobot + torch+cu121 + 全依赖) | ✅ uc01 NFS `/data/shared/ubuntu/workspace/X-VLA-env/.venv` |
| EE6D 转换 (kai/vis joint→EE6D 20D, PiperFK + Rot6D) | ✅ |
| XVLA-Soft-Fold action FK 缓存 | ✅ |
| Multi-domain dataset wrapper + DDP training script | ✅ |

## 4. X3.A — 3-domain (A + B + C) ✅ DONE

Balanced sampling: kai+vis×7+xvla×2 (vis ×7 上采样确保部署 prior 占优)。

| 阶段 | 状态 | Job ID | 完成 | Step | MAE@1 | @10 | @25 | 备注 |
|---|---|---|---|---|---:|---:|---:|---|
| **X3.A Stage A** Continual Pretrain | ✅ done | uc02 | 2026-05-23 23:08 | 20k step_final | **0.0380** | 0.0424 | 0.0574 | uc02 8 A800, ckpt 3.3GB at `/data/shared/ubuntu/local_ckpts/xvla_x3a_stage_a/step_final/state_dict.pt`. EE6D 20D MAE on vis_v2_merged last 50 ep × 20 q |
| **X3.A Stage B** vis-only Adapt | ❌ skipped | — | — | — | — | — | — | X3.B 已胜出, Stage B 不必走 X3.A |

## 5. X3.B — 2-domain (A + B, no XVLA) ✅ DONE — **新 Track X 终态**

Balanced sampling: kai+vis×7 (无 XVLA, 用于对照 XVLA 数据贡献)。

| 阶段 | 状态 | Job ID | 完成 | Step | MAE@1 | @10 | @25 | 备注 |
|---|---|---|---|---|---:|---:|---:|---|
| **X3.B Stage A** Continual Pretrain | ✅ done | uc01 | 2026-05-23 23:09 | 20k step_final | **0.0343** ⭐ | **0.0396** | **0.0561** | uc01 8 A800. ckpt at `/data/shared/ubuntu/local_ckpts/xvla_x3b_stage_a/step_final/state_dict.pt`. **全 horizon 优于 X3.A** |
| **X3.B Stage B** vis-only Adapt | ⏳ pending | — | — | — / 10k | — | — | — | 待启 (LR 5e-5, freeze 500, val MAE 选 best). 或直接用 Stage A 部署 |

## 5.5 X3.C — vis-only direct fine-tune (跳过 Stage A) — 2026-05-25 加跑

Ablation: 量化 Stage A continual pretrain 的实际增益。从 `lerobot/xvla-base` 官方直接 vis 微调 (无 multi-domain 预训练)。

| 阶段 | 状态 | Job ID | Step | 备注 |
|---|---|---|---|---|
| **X3.C** direct vis-only | ✅ done (2026-05-25) | uc02 PID 2739605 | 20k | 20k step, lr=5e-5, freeze 1000. Output `/data/shared/ubuntu/local_ckpts/xvla_x3c_vis_only_direct/step_final/` |
| **X3.C eval** | ⏳ pending | — | — | 同 X3.A/B 协议 eval, 对比 Stage A 价值 |

## 6. domain_id slot 分配

base ckpt 中未占用 slot:
- 19 = A (KAI0)
- 20 = B (vis) ⭐ 部署目标
- 21 = C (XVLA-Soft-Fold)

推理时 force `domain_id=20` (vis)。

## 7. 决策点

- ✅ **D1 (Stage A 完成后)**: X3.A vs X3.B mixed val MAE 对比 — **结果: X3.B 全 horizon 完胜 X3.A (MAE@1 -9.7%)**, XVLA 不值得加。**Track X 终态用 X3.B (kai+vis only)**.
- **D1.5 (X3.C eval 后)**: 量化 Stage A multi-domain pretrain 的价值. 若 X3.C ≈ X3.B, Stage A 是浪费; 若 X3.B < X3.C, Stage A 有效.
- **D2 (X3.B Stage B 后, 可选)**: vis B 真机评估 vs X-VLA SoftFold (同硬件) 100% baseline 对照
- **D3**: 若 X3.B 都打不过 baseline → Track X 主线降权, Track C (Action Head Cond) 提优先级 (但 Track C 已知 collapse, 见 `conditioning_vs_action_representation_ablation.md`)

## 8. 关联 paper ablation

(完整 Phase 3 ablation 设计见 [`cross_embodiment_strategy.md`](../../../deployment/strategy/cross_embodiment_strategy.md) §9 决策点 + §6 RTC/TAC 集成)

Phase 3 table 中:
- **X3.A** Track X (3-domain ⭐) — Florence2 + Soft Prompt, 全数据
- **X3.B** Track X (2-domain) — Florence2 + Soft Prompt, 无 XVLA
- 对照 **C3.0** Track C (Action Head Cond only) — 同 π0.5, 不同 conditioning 注入点
