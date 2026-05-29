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

## ⚠️ 数据管线 bug 修复 (2026-05-29) — 上述 X3.A/B/C 结论需重新验证

X3.A/B/C 用的 EE6D 转换器 + dataset wrapper 发现 3 个 bug, 均已修复。脚本同时从 uc `workspace/xvla_scripts/` (repo sibling, 未版本管理) **归位到 `train_scripts/xvla/`** (data/ + launch/)。详见 [`../../../../train_scripts/xvla/data/README.md`](../../../../train_scripts/xvla/data/README.md)。

| Bug | 影响 | 修复 commit |
|---|---|---|
| **Rot6D 排布** `R[:,:2].T.flatten()` (block `[r00,r10,r20,r01,r11,r21]`) ≠ 上游 `quat_to_rotate6d` (interleaved `[r00,r01,r10,r11,r20,r21]`) | 6 个旋转通道 4 个与预训练 base 错位; 部署用上游 `rotation_6d_to_matrix` 解码会 garble 旋转 | `2a01c85` |
| **Gripper 未二值化** (灌原始米值 ~0–0.08) | action_hub 对 gripper(9,19) 用 BCEWithLogitsLoss 要 {0,1}, 原始值近 0 → gripper 永不学闭合 | `5d5d0a4` (`raw*50<1.0→1`, 匹配上游 AIRAgilex) |
| **decode_frame `frame.index`** (当前 PyAV VideoFrame 无此属性) | 每帧解码抛 AttributeError → except 返回全 0 → **所有 vis/parquet 域为黑图** | `9633e2a` (改 pts 推算帧号) |

→ **X3.A/B/C 全部用此 buggy 管线训练**: rot6d 错排 + gripper 失效是**确定**的; 黑图取决于训练时 PyAV 版本 (若与现在同版本, 则 vis/kai parquet 域全黑, 仅 xvla_soft_fold 的 hdf5 cv2 解码不受影响)。**因此 "X3.B 全 horizon 完胜 X3.A" 等结论建立在 buggy 数据上, 必须用修复版重训后重新验证, 暂不作为定论。**

**官方一致性核对** (2026-05-29, 对照实际训练用的 `lerobot.policies.xvla.modeling_xvla.XVLAPolicy`, 非 upstream `xvla/X-VLA` repo): `forward` 内**无任何 Normalize/Unnormalize** (config 的 `ACTION:MEAN_STD`/`VISUAL:IDENTITY` 被自定义 forward 绕过) → **不需要 norm_stats 也不需要 ImageNet 归一**; `chunk_size=n_action_steps=30`; 图像 dataset 出 256/256/224 = `input_features` 声明, policy `resize_imgs_with_padding=[224,224]` 内部统一; EE6D 路径用 **absolute xyz** (upstream real_world handler 同, lerobotv21 的 delta 仅 joint 域)。

## 1. 核心思路

用 LeRobot's `lerobot/xvla-base` 0.9B ckpt + custom multi-domain wrapper (`train_scripts/xvla/data/multi_domain_dataset.py` + `train_scripts/xvla/launch/xvla_train.py`, 2026-05-29 从 uc `xvla_scripts/` 归位) 在 uc01/02 各 8 A800 上跑。EE6D 20D action (kai+vis 用 PiperFK + Rot6D 编码, XVLA-Soft-Fold 用预计算 `observation/eef_6d`)。

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

## 5.6 A_0423_0527 单数据集 finetune (**fixed pipeline**) — 2026-05-29 加跑

首个用**修复版管线** (rot6d interleaved + gripper 二值化 + decode 修复) 的 X-VLA run。单数据集直接从 `xvla-base` finetune (类 X3.C), 既验证修复后效果, 也作为 A_0423_0527 在 X-VLA 架构上的 baseline (对照同数据集的 JAX pi05 Run-A/B)。

| 项 | 值 |
|---|---|
| 数据集 | `xvla/data/self_built/A_0423_0527` (1085 ep, 1.40M frames, 1.37M chunk-samples, EE6D 20D fixed) |
| 来源 | `kai0/data/Task_A/self_built/A_0423_0527` (Run-A/B 同数据集) joint→EE6D, cnsh→uc TOS 传 8GB deref |
| Config | `A_0423_0527` (`train_scripts/xvla/launch/xvla_train.py`) |
| Steps | **30k** (≈1.40 epoch @ eff batch 64; A_0423_0527 比 vis_v2_merged 大 32%, 30k 匹配/超过 X3.C 1.23-epoch 曝光) |
| LR/freeze | 5e-5, warmup 500, freeze 1000 (同 X3.C) |
| 集群 | uc01 8 GPU, torchrun (port 29534, workers 4) |
| Ckpt | `/data/shared/ubuntu/local_ckpts/xvla_A_0423_0527/` 每 2k step |
| 状态 | ⏳ 运行中 (2026-05-29, step0 loss 102.9, GPU ~96%, ETA ~6h) |

> **数据集存放规范**: 自建 X-VLA EE6D 数据集一律放 `xvla/data/self_built/<name>/` (文件夹经 `self_built/.gitignore` 保留、内容忽略, 不入 git)。转换脚本: `train_scripts/xvla/data/joint_to_ee6d.py` (LeRobot parquet) / `convert_xvla_action.py` (hdf5)。

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
