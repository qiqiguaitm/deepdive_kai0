# LMWM 架构框架(收紧版,2026-07-03)

> 本文是 LMWM **当前架构的单一事实源**(取代技术报告 §2.1 的旧描述)。
> 汇总 Phase A–D + 均值/方差研究(L1–L7)+ decode-loss / LaWM 对比(L8–L9)后的收敛形态。
> 数据:`kai0_base` DINOv3-H(334,875 帧 / 3,055 episode / 37 milestone)。指标均为 held-out episode、对**真实未来**。

## 1. 定位(一句话)

**LMWM:当前视觉帧 → (① 离散 milestone 计划 + ② on-manifold subgoal 特征) 供 VLA 做 planning-prior。** 不是像素预测,不是策略;隐变量是 CRAVE 的 milestone 状态,非通用视觉未来 token。

## 2. 数据流水线(离线,一次性)

```
CRAVE demo 视频 → DINOv3-H 编码 → milestone 分配(37 prototype)
              → 循环图(转移先验) + episode-medoid subgoal 目标
              → augin pair 数据集(pairs_next_unique_augin.npz)
```

## 3. 输入表示 = augin(1332-D,已锁定为最优)

| 段 | 维度 | 说明 |
|---|---|---|
| DINOv3-H pooled | 1280 | 当前帧全局特征 |
| prev-milestone one-hot | 38 | 上一 milestone(正交信息,非帧可得) |
| proprio state | 14 | 机器人本体状态(裸拼,非 z-score) |

> 研究结论:7B 编码器否决;多帧历史无增益(Phase C);CRAVE 式 L2 融合不如裸拼。augin 已到编码器侧天花板。

## 4. 核心模型 `UnifiedLMWM`(`src/lmwm/models.py`)

```
augin (1332-D)
  └─ 共享 MLP trunk  [hidden 512 × depth 2, 含 LayerNorm]   (~1.18M)
       ├─ 转移头        → P(下一 milestone)         [分布, 37]
       ├─ greedy 头     → 下一 milestone logits      [37]
       ├─ max-product 头 → 完成路径下一步 logits      [37]
       ├─ greedy proto  → subgoal 隐变量 (1280, L2=1)
       └─ max-product proto → subgoal 隐变量 (1280, L2=1)
```
单模型 ~3.11M 参数(augin)。trunk 容量已验证最优(L3:1024×3 更大反过拟合,单独更差)。

## 5. 训练框架(收紧的核心:两个输出 = 两套损失)

- **标签**:`label_source: real_future`(实际观测的下一独特 milestone,非图查找;二者 75.8% 不一致)。
- **subgoal 目标**:`proto_target_source: episode_medoid`(该阶段内与簇心最相似帧的 latent;实测 0.877 > 簇心 0.836)。

### 5.1 离散头损失
`CE`(greedy + max-product)。方差敏感场景可切 **CVaR-CE 尾部损失**(`ce_tail_mode: cvar`,最差 10% 加权 → NLL std −38%)。

### 5.2 subgoal 头损失 —— **由消费方决定(L8/L9 关键结论)**

| 消费方 | 用哪个 loss | 原因 | 指标 |
|---|---|---|---|
| **喂 VLA 当特征(默认)** | **特征空间**:`1−cos` 或 LaWM `smooth_l1+（1−逐token余弦)` | 必须 **on-manifold**,VLA 才能消费/再编码 | subgoal cos 0.874;grid feat_cos 0.71 |
| **仅渲染"未来长什么样"图** | **decode-space**:`L1(Decode(pred), 真实帧)` | 直接优化解码保真 | pooled decoded L1 0.145(−7.4% vs 特征loss) |

> ⚠️ **两套 loss 近乎正交甚至反相关**(L9):decode-loss 的输出 feat_cos 塌到 0.015(off-manifold,VLA 不可用);特征 loss 的输出解码 L1 更差。**同一个头不能两用** —— 按 LMWM 输出的去向选 loss。默认走特征空间(输出喂 VLA)。

## 6. 推理框架:集成 + 图先验融合 + 蒸馏部署

```
p_cal   = softmax(greedy_logits / T)          T=1.30(校准)
p_prior = transition_probs[current_milestone]  (图作软贝叶斯先验)
p_fused ∝ p_cal^(1-λ) · p_prior^λ              λ=0.30
```
ensemble = 成员概率平均(同 split 不同 init)。

| 配置 | top1 | NLL(std) | subgoal cos | 成本 | 定位 |
|---|---|---|---|---|---|
| big3+mixed6+fuse | **0.459** | 1.715(1.03) | 0.872 | 9×+图 | 均值冠军 |
| mixed_ens_6+fuse | 0.453 | 1.740(**1.00**) | 0.873 | 6×+图 | 平衡 |
| cvar_ens_3+fuse | 0.434 | 1.855(**0.86**) | 0.874 | 3×+图 | 方差冠军 |
| **蒸馏 student(1024×3)** | 0.449 | 1.782 | 0.871 | **1×,无图** | **⭐部署接 VLA** |

蒸馏:9 成员融合 teacher → 单 forward student(图先验烘焙进网络),`checkpoints/stage3_distilled/student.pt`。

## 7. 输出接口(→ VLA)

`lmwm.vla_interface.VLALMWMPredictor.predict(current_features)`:
- `next_milestone_probs` / `topk_milestones`+`probs`(top-5 覆盖 86%)
- `subgoal_latent`(on-manifold,1280-D,LaWAM 风格,可作 code 或直接条件)
- `confidence` / `entropy`(门控用)

用法:以 subgoal_latent + topk 为条件,按 confidence/entropy 门控;**不要**把 top-1 当 ground truth(~13 分支固有歧义,top1 有硬上限)。

## 8. 解码/可视化(旁路,**不在** VLA 推理路径)

仅用于人看"预测长什么样",三条解码路径:

| 解码器 | held-out 保真(L1) | 用途 |
|---|---|---|
| **patch-grid 解码器** | **0.043**(自重建天花板) | 最忠实;`checkpoints/patch_decoder/patch_dec.pt` |
| pooled 解码器 | 0.13 | 平滑可读原型;`dinov3h_decoder/dec.pt` |
| 检索解码器 | — | 预测 latent → 最近真实帧(锐利);`retrieval_decoder.py` |

> L9 可视化(`assets/grid_pred_vs_ceiling.png`):patch 解码器几乎完美(medoid 编码→解码 L1 0.043 ≈ 真实),**模糊全来自预测端**(grid 预测解码 L1 0.171)——预测 256 token 比 pooled 难,是当前瓶颈,非解码器。

## 9. 诚实边界

- 单 `kai0_base` DINOv3-H;`T`/`λ`/成员数换数据集需重估。
- top1 ≈ 0.46 是 frame-only 天花板(7 lever 一致证明);subgoal cos 0.874 是结构性上限,越过需**未来/历史信息**(VLA 闭环反馈或 LaWM `inverse(当前,未来)→code`)。
- Milestone 标签逐帧抖动(~每 2 帧变),限制下一帧监督质量。

## 10. 产物索引

- 模型/数据/训练/运行时/VLA:`src/lmwm/{models,data,training,runtime,vla_interface,retrieval_decoder}.py`
- 部署 ckpt:`checkpoints/stage3_distilled/student.pt`;集成成员 `stage3_augin{,_ens,_tail,_big,_div}/`
- 解码器:`checkpoints/{patch_decoder/patch_dec.pt, dinov3h_decoder/dec.pt}`;decode-优化 subgoal 头 `stage3_decode_subgoal/head_decode.pt`
- 迭代日志:`docs/optimization_plan_20260702.md`(L1–L9 全记录)
- 图:`assets/{grid_pred_vs_ceiling, decode_loss_compare, grid_decode_loss_compare}.png`
```
