# SSL Pretraining Experiment Plan — Cross-Embodiment Cloth Folding

> **Created**: 2026-05-21
> **Status**: Phase 0 ready to start
> **Owner**: Tim
> **Related**: [cross_embodiment_data_reuse_plan.md](./cross_embodiment_data_reuse_plan.md), [training_servers_knowledge_base.md](./training_servers_knowledge_base.md)

---

## 1. 背景 + 动机

### 1.1 问题
我们有 4 个核心痛点:
1. **B (vis_base D405 + 新双臂) 数据量小** (~800 ep), 真机表现不够稳
2. **A (kai0_base + dagger, D435 + 老双臂) 数据多** (6512 ep), 但 naive 混训 → 真机抖动 (`mixed_pure2_1800_6000` 验证)
3. **XVLA-Soft-Fold** (1729 ep), 另一套软折叠分布, 没法直接用
4. **cross-embodiment gap** (硬件清单见 §1.2) 无法在 supervised policy 层 absorb

### 1.2 Embodiment Gap (按影响从大到小)

| Gap 类型 | A | B | 影响 |
|---|---|---|---|
| **Wrist 相机** | D435 (FOV 69×42°, rolling, min depth 28cm) | D405 (FOV 87×58°, global, min depth 7cm) | 🔴 **最硬** — 近距 RGB-D 行为完全不同 |
| **Wrist extrinsics** | 老 flange | 新 flange | 🟠 第二严重 — wrist view 物体 pos/scale 受影响 |
| **双臂间距** | 老 setup | 新 setup | 🟡 absolute EE 时 systematic bias |
| **Top 相机** | 老高度/角度 | 新高度/角度 | 🟢 < 5cm/5° 时 augmentation cover |

### 1.3 核心 Insight (定义路线)

> **A 的数据价值不在 "直接帮 B 做 task", 而在 representation / dynamics / prior 这些更上游的层次。**

按 ROI 分四层 (越高越值得做):
| Layer | 描述 | 是否依赖 action 对齐? | A 数据可用程度 |
|---|---|---|---|
| **L1: SSL Visual Pretrain** ⭐ | V-JEPA / track / flow / xview | 否 | **Full power** |
| L2: Embodiment-cond Policy | Co-train with embodiment token | 部分 (need conditioning) | 谨慎 |
| L3: Auxiliary tasks | Inverse dyn, future frame pred | 部分 | 谨慎 |
| L4: Data engine | Replay/retarget, sim2real | 否 | Full power 但需 retargeting |

**本文档专注 L1 (+ L2 dynamics, + L3 policy) 端到端实验。**

---

## 2. 假设 + 验证

| ID | 假说 | 关键实验 | 成功标准 |
|---|---|---|---|
| **H1** | SSL pretrain on A+XVLA+B 提供的 visual repr 在 cloth 任务上 > π0.5 default | E3.1 vs E3.0 | B finetune val MAE 降低 ≥10%, 真机抖动减少 |
| **H2** | Multi-objective (V-JEPA + track + flow + xview) > 单 V-JEPA | E1.5 vs E1.1 | Downstream val MAE 进一步降低 |
| **H3** | Embodiment-conditioned dynamics 让 A 的物理 prior 不污染 B policy | E3.3 vs E3.2 | 真机平滑度 + 复杂场景成功率 提升 |
| **H4** | Motion-residual decomposition: cloth_residual 部分 embodiment-invariant | E2.3 latent 分析 | A vs B cloth_residual latent 的 MMD < 0.1 |

---

## 3. 数据 + 资源

### 3.1 训练数据池

| 来源 | 规模 | 路径 | 包含视角 |
|---|---:|---|---|
| **A** (kai0_base + dagger) | 6512 ep | `/data/shared/ubuntu/.../kai0/data/Task_A/kai0_base/` + `kai0_dagger/` | top + 双 D435 wrist |
| **XVLA-Soft-Fold** | 1729 ep | `/data/shared/ubuntu/.../xvla/data/xvla_soft_fold/` (NFS); `/vePFS/tim/xvla/data/xvla_soft_fold/` (gf0) | TBD (soft fold dataset, 待检查) |
| **B** (vis_base_clean_v2) | 800-895 ep | `/data/shared/ubuntu/.../Task_A/vis_base_clean_v2/` | top + 双 D405 wrist |
| **合计** | ~9000 ep, ~430k frames | — | — |

### 3.2 GPU 资源 (2026-05-21 实测可用)

| 资源 | GPU | 状态 | 分配 |
|---|---:|---|---|
| **Robot-North-H20** (cn-beijing) | 39 H20 free / 56 total | active | SSL 主战场 (单 exp 16 GPU, 可 2-3 并发) |
| **uc02** | 8 A800 | free (XVLA dl 完成) | 数据预处理 + dev / smoke |
| **gf3** | 1 H20 | active (smoke 用) | 单卡 smoke / debug |
| uc01, uc03 | busy | 其他 exp 占用 | 不动 |

### 3.3 预算

- SSL pretrain (Phase 1): ~80 GPU-day
- Dynamics (Phase 2): ~30 GPU-day
- Policy + ablation (Phase 3): ~30 GPU-day
- **总计**: ~140 GPU-day on Robot-North-H20

---

## 4. 实验阶段 (4 个 Phase, 14 个实验)

### Phase 0: 数据预处理 + 伪标签生成 (Week 1)

**目标**: 把 9000 ep 视频跑过 CoTracker3 / RAFT / SAM, 生成离线 pseudo-labels 给 Phase 1 SSL 用。

> **资源分配 (并行)**: uc02 跑 CoTracker + flow (CPU/GPU 混合任务), Robot-North-H20 1 节点跑 SAM (重 GPU)

| Exp | Tool | Input | Output | Resource | ETA |
|---|---|---|---|---|---|
| **E0.1** Pseudo-track | CoTracker3 | 9000 ep × 3 view | `tracks/{ep_id}/{view}.npz` (T, N=32 points, xyc) | uc02 8 GPU | 4-5 day |
| **E0.2** Optical flow | RAFT-Large | consecutive frames | `flow/{ep_id}/{view}.npz` (T-1, H/4, W/4, 2) | uc02 并行 | 同上 |
| **E0.3** Cloth mask | SAM2/SAM3 | 1 frame per second × 9000 ep | `mask/{ep_id}/{view}.npz` (T_sub, H, W, K) | Robot-North-H20 8 GPU | 1-2 day |
| **E0.4** FOV align | OpenCV | D405 wrist RGB | `rgb_aligned/{ep_id}/wrist_*.npy` (D405 crop 到 D435 equiv FOV) | CPU | 几小时 |
| **E0.5** EE-relative action | Python (PiperFK) | A + B + XVLA actions | `action_ee_relative/{ep_id}.npz` | CPU | 几小时 |

**质量检查**: 跑完抽样 50 ep 人工 inspect tracks/flow/mask 是否合理; 不合格的 ep 记入 `skip_list.txt`。

**Phase 0 状态**: ⏳ 待启动

---

### Phase 1: SSL Visual Encoder Pretrain (Week 2-4)

**核心**: 从 π0.5 SigLIP/PaliGemma vision tower **continual-pretrain** (不 from scratch), 输出 cloth-fold-specific encoder。

> **并行调度 (per user 决策)**: 跑 3 jobs in parallel — E1.1 (baseline) / E1.4 (xview) / E1.5 (full multi-objective)。Skip E1.2/E1.3 中间点 (从 E1.5 ablation 反推单项贡献)。

#### E1.1 — V-JEPA Baseline (单目标)
```yaml
backbone:     π0.5 SigLIP (continual-pretrain)
input:        T=16 frames, 3 views, 224×224
objective:    masked latent prediction (V-JEPA 2.1)
mask:         tube 30%, edge-saliency 2× boost on cloth edges (用 E0.3 mask)
lr:           5e-5 layer-wise decay (peak), warmup 2k, cosine to 5e-7
batch:        128 total (16 H20 × 8/gpu)
steps:        50k
data:         A + XVLA + B 全量 (~9000 ep)
embodiment:   不区分 (统一 backbone)
output_path:  /vePFS-North-E/vis_robot/.../ssl_ckpts/E1.1_vjepa_base/
```

#### E1.4 — V-JEPA + Track + Flow + Cross-view (no Phase-2 weight switch)
```yaml
继承 E1.1, 加 3 个 head:
  - track_head: 8M param, predict 32 keypoint tracks over T=16
                loss = L2(xy) + BCE(visibility), w=0.5
  - flow_head:  predict dense flow from latent
                loss = EPE vs RAFT pseudo, masked by cloth mask, w=0.3
  - xview_head: top latent → wrist latent (autoregressive)
                loss = cosine + L2, w=0.2
weights:      固定 (w_vjepa=1.0, w_track=0.5, w_flow=0.3, w_xview=0.2)
```

#### E1.5 — Full Multi-objective + Phase Weights + Saliency + Multi-scale
```yaml
继承 E1.4, 进一步:
  - Phase 1 (step 0-25k):  w_vjepa=1.0, w_track=0.5, w_flow=0.3, w_xview=0.2
  - Phase 2 (step 25k-50k): w_vjepa=0.5, w_track=1.0, w_flow=0.5, w_xview=0.3
  - Saliency mask: edges 2×, interior 0.5×
  - Multi-scale temporal: 一半 batch T=8 (short), 一半 T=48 (long)
  - Anchor loss: 1% batch on LAION subset (防 catastrophic forget)
```

**Phase 1 验收 (downstream micro-eval)**:
- 每个 E1.x 跑完 → 小规模 B-only policy finetune: 3k step, batch 32, 8 GPU on uc02
- 比较 val MAE on B val set
- E1.5 应该明显胜 E1.1 (h2 验证)

**Phase 1 状态**: ⏳ 待 Phase 0 完成

---

### Phase 2: Dynamics Pretrain (Week 5-6)

**核心**: 在 frozen visual encoder 上训 latent dynamics model, 引入 embodiment conditioning + motion-residual decomposition。

#### E2.1 — Latent Dynamics Baseline (无 embodiment)
```yaml
backbone:     E1.5 visual encoder (frozen)
dynamics_net: 6-layer Transformer (512 dim, 50M param)
input:        (z_t, action_t) → predict z_{t+1}
loss:         L2 on latent
data:         A + XVLA + B 全量
embodiment:   不区分 (baseline)
batch:        256, lr 5e-5 cosine 30k step
```

#### E2.2 — + Embodiment Conditioning
```diff
继承 E2.1:
- input: (z_t, action_t)
+ input: (z_t, embodiment_emb, action_t)
+ embedding_emb: 3 个 learnable vec (A_emb, XVLA_emb, B_emb), dim=128
inference: 部署 B 时只用 B_emb
```

#### E2.3 — + Motion-Residual Decomposition ⭐ (paper 原创)
```yaml
分两个 head:
  ego_motion_head:    (action_t, emb_e) → ego_motion_latent
                      (受 embodiment 影响, A/B 各自学)
  cloth_residual_head: (z_t, action_t)   → cloth_residual_latent
                      (NOT 依赖 embodiment, 共享物理 prior)
update: z_{t+1} = z_t + ego_motion_latent + cloth_residual_latent

paper claim: cloth_residual 部分在 A vs B 上分布相同 (用 MMD < 0.1 验证)
```

#### E2.4 — + Inverse Dynamics Auxiliary
```diff
+ Aux head: predict a_t | (z_t, z_{t+1}, emb_e)
  loss: L2 on action, weight 0.2
  作用: 给 visual + dynamics backbone 提供 action-grounded gradient
```

**Phase 2 验收**:
- E2.3 latent 上做 t-SNE / MMD: A vs B 在 cloth_residual 部分应近, 在 ego_motion 部分应远
- 如 motion-residual 分不开 → 回退 E2.2 全联合

**Phase 2 状态**: ⏳ 待 Phase 1 完成

---

### Phase 3: Downstream Policy + Ablation (Week 7-8)

**核心**: 把 Phase 1+2 产出接到 π0.5 policy, B-only finetune, 真机评估。

| Exp | Visual | Dynamics | Action | Data | 用途 |
|---|---|---|---|---|---|
| **E3.0** baseline | π0.5 default | × | absolute | B (smooth_800) | 当前 SOTA, baseline |
| **E3.1** | E1.5 frozen | × | absolute | B | 测 H1 (visual repr 单独 value) |
| **E3.2** | E1.5 LoRA | × | absolute | B | 测 fine-tunable 是否更好 |
| **E3.3** | E1.5 LoRA | E2.3 frozen | absolute | B | 测 H3 (dynamics 额外贡献) |
| **E3.4** Full Stack | E1.5 LoRA | E2.3 frozen | EE-relative | B + A weighted (embodiment cond) | 终态最强 |

**Phase 3 训练设置**:
- 16 H20 × 50k step ≈ 35h/exp
- batch 128, lr 1.5e-5 → 1.5e-6, num-workers 64 (uc cluster style)
- EMA 0.999

**真机测试 protocol**:
- 30 episode per exp, 固定场景 + 3 OOD 场景 (布料/姿态/光线)
- 指标: 抓衣角成功率, 完整折叠成功率, 平均执行时长, 抖动 metric (action diff p99)

**Phase 3 状态**: ⏳ 待 Phase 1+2 完成

---

### Phase 4: Real Machine Evaluation + Paper Tables (Week 9)

- 真机大规模测试 (60-100 episodes/exp)
- Ablation table (见 §6)
- Failure case analysis
- Paper figures (architecture diagram, latent t-SNE, ablation curve)

---

## 5. 时间线 (Gantt)

```
Week 1   ┌────[Phase 0] 数据预处理 (uc02 + Robot-North-H20 并行)
         │       ↓ Phase 0 完成
Week 2-4 │   ┌──[Phase 1] SSL (3 并发: E1.1 + E1.4 + E1.5) on Robot-North-H20
         │   │
Week 5-6 │   │  ┌──[Phase 2] Dynamics (4 串行: E2.1→E2.2→E2.3→E2.4)
         │   │  │
Week 7-8 │   │  │  ┌──[Phase 3] Policy + Ablation (5 jobs)
         │   │  │  │
Week 9   │   │  │  │  ┌──[Phase 4] 真机 + Paper
         └───┴──┴──┴──┘
```

**总 9 周** (并行可压缩到 7-8 周)。

---

## 6. Final Ablation Table (核心 paper 表)

| Variant | Visual | Dynamics | Embodiment cond | Motion-residual | Inverse Dyn | Val MAE | 真机平滑度 | 真机成功率 |
|---|---|---|---|---|---|---:|---:|---:|
| **E3.0** baseline (π0.5 default) | — | — | — | — | — | TBD | TBD | TBD |
| **E3.1** + Visual SSL | E1.5 frozen | — | — | — | — | ? | ? | ? |
| **E3.2** + LoRA tune | E1.5 LoRA | — | — | — | — | ? | ? | ? |
| **E3.3** + Dynamics | E1.5 LoRA | E2.3 | ✓ | ✓ | — | ? | ? | ? |
| **E3.4** Full Stack | E1.5 LoRA | E2.3 | ✓ | ✓ | ✓ | ? | ? | ? |

(待填)

---

## 7. 关键陷阱 + 应对

| 陷阱 | 应对 |
|---|---|
| CoTracker 在 heavy occlusion 失败 | Pseudo-track 加 confidence filter; track loss 按 mask 加权 |
| **D435 FOV (69°) < D405 (87°)** | **把 D405 crop 到 D435 等效 FOV** (而不是反向 — 原方案有 bug) |
| π0.5 SigLIP catastrophic forget | Layer-wise lr decay + anchor loss on 1% LAION batch |
| Multi-objective loss 不收敛 | Phase 1 先单 V-JEPA 5k step 预热, 再逐项加 |
| Phase 0 (CoTracker) 慢 | 9000 ep × 3 view × T=200 frames ≈ 5.4M tracker calls. 用 batch=64 + temporal stride 2 减半 |
| Embodiment cond 在 visual 还是 dynamics? | **Phase 1 visual 不区分 view 来源**, Phase 2 dynamics 才区分。原因: visual repr 想要 invariant, dynamics 必须 partition |

---

## 8. 状态跟踪 (持续更新)

> 每次推进一步, 更新这一节。

### 8.1 Phase 0 — 数据预处理 ⏳

| Sub-task | 状态 | 启动时间 | 完成时间 | 备注 |
|---|---|---|---|---|
| E0.1 CoTracker3 pseudo-tracks | — | — | — | 待启动 |
| E0.2 RAFT optical flow | — | — | — | 待启动 |
| E0.3 SAM2/SAM3 cloth mask | — | — | — | 待启动 |
| E0.4 FOV alignment script | — | — | — | 待启动 |
| E0.5 EE-relative action conversion | — | — | — | 待启动 |
| **Phase 0 整体** | ⏳ pending | — | — | — |

### 8.2 Phase 1 — SSL Pretrain ⏳

| Exp | 状态 | Job ID | Val Loss | Downstream MAE | 备注 |
|---|---|---|---|---|---|
| E1.1 V-JEPA baseline | — | — | — | — | 待 Phase 0 |
| E1.4 + track + flow + xview | — | — | — | — | 待 Phase 0 |
| E1.5 Full multi-objective | — | — | — | — | 待 Phase 0 |

### 8.3 Phase 2 — Dynamics ⏳

| Exp | 状态 | Job ID | Val Loss | MMD A↔B | 备注 |
|---|---|---|---|---|---|
| E2.1 Latent dyn baseline | — | — | — | — | 待 Phase 1 |
| E2.2 + Embodiment cond | — | — | — | — | 待 Phase 1 |
| E2.3 + Motion-residual | — | — | — | — | 待 Phase 1 |
| E2.4 + Inverse dyn aux | — | — | — | — | 待 Phase 1 |

### 8.4 Phase 3 — Policy + Ablation ⏳

| Exp | 状态 | Job ID | Val MAE | 真机 | 备注 |
|---|---|---|---|---|---|
| E3.0 baseline | — | — | — | — | 用现有 smooth_800 结果 |
| E3.1 + SSL frozen | — | — | — | — | 待 Phase 1 |
| E3.2 + SSL LoRA | — | — | — | — | 待 Phase 1 |
| E3.3 + Dynamics | — | — | — | — | 待 Phase 2 |
| E3.4 Full Stack | — | — | — | — | 待 Phase 2 |

---

## 9. 修订历史

| 日期 | 内容 |
|---|---|
| 2026-05-21 | 初版: Phase 0-4 (~9 周) 计划; 数据池 (A 6512 + XVLA 1729 + B 800); 资源 (Robot-North-H20 + uc02); 用户决策: 完整 9 周, Phase 0 并行, Phase 1 3-job 并发 |
