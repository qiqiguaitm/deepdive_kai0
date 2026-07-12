# LMWM 最终架构 + Baseline 指标对比(单页速查)

> 权威单页版。详细推导见 `FINAL_REPORT.md`,跨任务见 `FINAL_CROSSTASK_PREDICTOR.md`。更新 2026-07-08。

---

## 1. 数据流

```
帧 224²×3 ─[① π0.5 SigLIP 冻结共享塔]→ grid G_t (16×16×1152) + gist (1152)

训练(teacher=proto 定案):
   下一 milestone SigLIP 中心 c[next] ─(固定投影 W)→ 码 z(128)          ← teacher(查表,0 参数)
   ② 预测器 MDN(gist) ──蒸馏 z──> ẑ                                     ← predm 学"预测中心码"
   ③ 生成器 AdaLN(G_t 画布, z) → 下一 grid Ĝ ≈ G_未来                   ← smooth-L1 监督
部署:
   gist → ② predm → ẑ(128) → ③ 生成器(G_t, ẑ) → 子目标 Ĝ(16×16×1152) → 喂 π0.5 action expert

离线 label 工厂(不在线):DINOv3-H(840M) + CRAVE Viterbi → milestone+1 目标 + 簇中心
```

## 2. 各部分:参数 / 结构 / 输入→输出

| 模块 | 参数 | 结构 | 输入 → 输出 |
|---|---|---|---|
| ① 编码器 π0.5 SigLIP-So400m/14 | **~400M 冻结** | ViT,与 VLA 同塔 | 帧 224²×3 → grid 16×16×1152 + gist 1152 |
| ② 预测器·部署 MDN `MilestonePredictor` | **3.28M** | Linear(1152→1024)·GELU·Linear(1024→1024)·GELU + pi/mu/ls 头 | gist 1152 → K=4 混合 over 码:pi(4)+mu(4×128)+ls(4×128) → ẑ(128) |
| ③ 生成器 AdaLN `MilestoneGenerator` | **30.29M** | proj Conv(1152→512) + 4 残差块[Conv·GELU·Conv] + mod Linear(128→4·3·512) zero-init + out Conv(512→1152) | (G_t 1152×16×16, 码 128) → grid 1152×16×16 |
| teacher(proto) | **0** | 簇中心固定投影查表(无网络) | next_ms → 中心码 z(128) |
| teacher(inv 备选) `InverseEnc` | 5.93M(仅训练) | 2 层 conv 逆向 + Linear | (G_t, G_未来) → z(128) |
| 锚头(仅 inv 用) | 0.013M | Linear(128→99) | z → union milestone logits |

**部署在线主体(predm + 生成器)= 33.58M CNN**。proto 训练图无 InverseEnc / 无锚头(码即中心=身份)。

## 3. vs LaWM(baseline)—— 我们数据、同 reach 协议实测

| 指标 | LaWM | LMWM | 谁好 |
|---|---|---|---|
| reach / model lag(秒,同口径) | 1.48 | **1.67** | **LMWM**(且目标更远 2.64s) |
| 负滞后率(<0) | 10.7% | **6.5%** | **LMWM** |
| 世界模型参数 | ~230M Transformer | **~34M CNN** | **LMWM** 轻 ~10× |
| 与 VLA 嵌入 | 独立 DINO 双塔,latent 跨空间蒸馏 | **同塔复用 KV** | **LMWM** |
| 码调制 | AdaLN-DiT | AdaLN-CNN | 均收敛到 AdaLN |
| 目标语义 | 固定 1.6s 未来(无价值约束) | **CRAVE 价值单调 milestone** | **LMWM** 有②价值前向 |
| oracle grid-cos | 0.770 | 0.715 | ⚠️ 编码器空间不同,不硬比 |
| lift(oracle−persist) | +0.143 | +0.116 | 趋势可比 |
| 下游 SR(extrinsic) | **LIBERO 98.6%** | 待测 | **LaWM**(我们唯一短板) |

⚠️ grid-cos 空间不同(ViT-B 768 vs SigLIP 1152)不硬比;reach(秒)严格同口径。

## 4. 跨任务指标(多任务联合,teach_proto_3task)

| task(簇数) | deploy | id_top3 |
|---|---|---|
| kai0 叠衣(37) | 0.703 | 0.473 |
| coffee 咖啡(15) | 0.784 | 0.977 |
| xvla 叠衣变体(47) | 0.773 | 0.679 |
| **mean** | **0.753** | **0.710** |

一个模型同时干三种不同流程/簇数的任务。teacher=proto 与 inv 打平但更简/更轻/开放词表。

## 5. 一句话
**冻结共享 SigLIP → 预测器(簇中心码,MDN 部署)+ 生成器(AdaLN 当前画布)→ milestone+1 子目标;~34M,价值单调、跨任务、与 VLA 同塔;reach 反超 LaWM,唯一缺 SR。**
