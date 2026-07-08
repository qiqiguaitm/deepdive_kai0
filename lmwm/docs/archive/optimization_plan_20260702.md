# LMWM 自动迭代优化规划(2026-07-02 起,滚动更新)

> 本文件是**执行中的活文档**:每完成一个 lever 就回填「进度日志」并更新 baseline。
> 目标:①模型 predict 解码后效果较好;②降低 loss **均值**(↑top1/cos ↓NLL);③降低 loss **方差**(↓NLL std / CVaR / subgoal 尾部)。自动迭代直到达标或 lever 用尽。

## 目标与度量(held-out,对真实未来 / episode-medoid subgoal)

| 维度 | 度量 | 当前(baseline) | 目标(stretch) |
|---|---|---|---|
| 均值·离散 | fused top1 / NLL | 0.459 / 1.715 (v1) | ≥0.47 / ≤1.68 |
| 方差·离散 | NLL std / CVaR10 | 1.00 / 3.83 | ≤0.90 / ≤3.5 |
| 均值·subgoal | cos mean | 0.874 | ≥0.885 |
| 方差·subgoal | frac<0.7 | 0.029 | ≤0.022 |
| 解码 | 预测→检索最近真实帧命中率 / decode L1 | 检索 2.7% 真帧;subgoal cos 决定命中 | subgoal cos ↑ 即改善 |

> 诚实边界:任务固有 ~13 分支多模态,**top1 有硬上限**;真正可拉的是 top-k / NLL / 方差 / subgoal cos。解码"效果好"= subgoal 更接近真实 medoid → 检索/解码更准,故折算到 subgoal cos。

## 锁定 baseline(v0)

**`mixed_ens_6 + fuse0.3`**(4 普通 + 2 CVaR 成员,输入 = frame+prev-milestone+state):
top1 **0.453** / NLL **1.740**(std **1.00** / p90 3.05 / CVaR10 **3.83**)/ subgoal cos **0.874**(±0.068,<0.7 **2.9%**)。
评估协议:`scripts/eval_mean_variance.py`(均值+方差一起报)。

## Lever 清单(按潜力排序,逐个 A/B vs baseline)

| # | Lever | 目标维度 | 机制 | 状态 |
|---|---|---|---|---|
| L1 | **patch-token 输入 + 小 transformer** | 均值(容量/表示) | pooled 丢空间;patch token 当输入让 transformer 用回来(避开高维输出坑) | 待跑 |
| L2 | **VQ / 混合密度 subgoal 头** | 均值+方差(多模态) | ~13 分支;回归头预测均值丢模态;VQ/MDN 建模多峰、可采样 | 待跑 |
| L3 | **size/深度 sweep**(MLP hidden/depth、transformer trunk) | 均值(容量) | 现用 2 层 512;扫 hidden/depth | 待跑 |
| L4 | **Ensemble 蒸馏** | 部署成本(保均值) | 6-ens 蒸馏进单模型,砍 6× 推理 | 待跑 |
| L5 | **Conformal / 温度再校准** | 方差(尾部保证) | 压最差样本 | 待跑 |
| L6 | **更多 ensemble 成员 / 架构多样性** | 均值+方差 | bagging/异构 | 待跑 |
| L7 | **组合赢家 → 终态** | 全部 | 融合各 lever 的最优 | 待跑 |

## 迭代协议

1. 取当前 baseline。2. 实现 lever,训练(背景,GPU)。3. `eval_mean_variance` 评估。4. 与 baseline 比:任一目标改善且其它不显著退化 → 采纳,更新 baseline;否则记录负结果。5. 回填进度日志。6. 下一个 lever。达标或 lever 用尽则收口。

## 进度日志

- **v0 (2026-07-02)**:baseline 锁定 = mixed_ens_6+fuse(见上)。此前研究见 `mean_variance_research_20260702.md`。
- **L1 patch-token transformer(2026-07-02)**:同 18k 子集 A/B。**subgoal 更好**(cos 0.831→0.856,std 0.093→0.072,<0.7 8.8%→3.5%——空间信息利好未来帧回归);但**离散头过拟合更差**(top1 0.336→0.292,NLL 3.93→6.15,transformer 喂不饱)+ 全量 patch-grid 缓存 130GB 难扩 → **不采纳**。记:patch 空间信息对 subgoal 有真实增益,若 subgoal 优先可 on-the-fly 编码重试。脚本 `lever_patch_token.py`。
- **L2 多假设(MCL)subgoal 头(2026-07-02)**:全量数据。oracle-best 随 K↑ 单调升(K2/4/8 → 0.888/0.896/**0.900**)证实**多模态真实存在**;但 **deploy(top-weight gate)反而降**(0.872→0.852),因当前帧对"走哪分支"固有歧义,gate 只能猜。→ **不采纳**。**结论:frame-only subgoal ≈0.874 是结构性天花板;0.90 需未来信息**(= LaWM inverse(当前,未来)训练态,纯部署够不到)。脚本 `lever_mhp_subgoal.py`,结果 `outputs/lever_mhp/`。
- **L3 容量 sweep(2026-07-02)**:标准 sweep 显示容量↑→离散 top1↑(512×2→1024×4:0.359→0.388)但 NLL/方差变差、subgoal 降。**生产级验证翻案**:1024×3 单模型峰值仅 0.398(<512×2 的 0.408,且 ~step400 后剧烈过拟合),subgoal 0.862<0.874 → **大 trunk 单独不采纳**。**但** big3+mixed6(9 成员异构集成)→ **top1 0.4591(新高,+0.6pt)**、NLL 1.715、方差≈持平、subgoal 保住 → **采纳为均值冠军 v1**(1.5× 推理成本)。脚本 `lever_size_sweep.py` + 配置 `*_augin_big_e{1,2,3}.yaml`,eval `outputs/mean_variance/with_big.json`。
- **v1 baseline 更新(2026-07-02)**:均值冠军 = **big3+mixed6+fuse**(top1 **0.4591** / NLL **1.715** std 1.026 / CVaR10 3.887 / subgoal cos 0.872 <0.7 2.8%);平衡默认仍 = mixed_ens_6+fuse(方差更紧 std 0.995 / CVaR 3.834)。
- **L5 温度重校准(2026-07-02)**:诚实 calib/test(按 episode 划分,`lever_recalibrate.py`)。拟合 T=0.7/λ=0.3 → NLL 均值 1.709→**1.652**(−3.4%,集成略欠自信),但**方差反增**(std 1.029→1.314,CVaR 3.88→4.52)。**Pareto trade-off,双目标不采纳**;保持 T=1。仅要 NLL 均值时可用 T=0.7。结果 `outputs/lever_recal/`。
- **L6 更多异构成员(2026-07-02)**:再加 3 个异构 shape(768×3/640×2/896×2)成 12 成员 → top1 0.4591→0.4599(+0.08pt)、NLL 1.715→1.705、方差微升。**异构集成在 ~9 成员饱和** → 不采纳,保留 big3+mixed6(v1)。配置 `*_augin_div_d{a,b,c}.yaml`,eval `outputs/mean_variance/with_hetero.json`。
- **L4 集成蒸馏(2026-07-02,部署赢家)**:把 9 成员融合 teacher 蒸馏进**单个 1024×3(9.4M)student**(KL soft-prob + proto cos,**图先验烘焙进网络**)。student:top1 **0.4487** / NLL 1.782 / subgoal cos **0.8714**(≈全保留)。**vs 原单模型 0.408 → +4.1pt,单 forward,部署无需运行时图查表** → **采纳为部署 artifact**。ckpt `lmwm/checkpoints/stage3_distilled/student.pt`,脚本 `lever_distill.py`。

## L7 收口:最终 Pareto 前沿与采纳(2026-07-02)

| 定位 | 配置 | top1 | NLL(std) | CVaR10 | subgoal cos(<0.7) | 成本 |
|---|---|---|---|---|---|---|
| **均值冠军** | big3+mixed6+fuse | **0.459** | 1.715(1.03) | 3.89 | 0.872(2.8%) | 9× + 图 |
| **方差冠军** | cvar_ens_3+fuse | 0.434 | 1.855(**0.86**) | **3.65** | 0.874(2.9%) | 3× + 图 |
| **平衡** | mixed_ens_6+fuse | 0.453 | 1.740(1.00) | 3.83 | 0.873(2.8%) | 6× + 图 |
| **⭐部署(推荐接 VLA)** | distilled student | 0.449 | 1.782(1.08) | 4.08 | 0.871(2.9%) | **1×,无图** |
| 基线(研究前单模型) | augin single+fuse | 0.408/0.434 | 1.78 | 4.64 | 0.864 | 1× + 图 |

**采纳**:部署用蒸馏 student(单模型 +4.1pt top1,subgoal 全保留,图烘焙);要极限精度用 big3+mixed6;要极限稳健用 cvar_ens_3。

## 结论:达标情况与天花板(诚实)

| 目标 | 达成 | 说明 |
|---|---|---|
| top1 ≥0.47 | ✗(0.459) | 近天花板;任务固有 ~13 分支多模态,top1 有硬上限 |
| NLL ≤1.68 | ~(平衡 1.705 / 锐化 1.652) | 锐化达标但方差反增(L5 Pareto) |
| 方差 std ≤0.90 | ~(平衡 1.00 / cvar 0.86) | cvar 成员达标但均值略低 |
| subgoal cos ≥0.885 | ✗(0.874) | **L2 证明是结构性天花板**:frame-only oracle 0.90 需未来信息 |
| 解码效果 | ✓ | subgoal 0.874 → 检索最近真实帧忠实可视(见 CRAVE 解码文档) |

**7 个 lever 一致证明触到 frame-only 天花板**;本 session 净增益 = **top1 0.453→0.459(集成)+ 部署单模型 0.408→0.449(蒸馏)**,subgoal/方差在各自 frame-only 上界。

**唯一能越过 subgoal 天花板的路 = 引入未来/历史信息**(L2 oracle 0.90 vs deploy 0.874 的 gap 只有未来态可及):即 (a) 多帧运动历史消歧分支,或 (b) VLA 闭环反馈把"实际走了哪个分支"喂回 —— 这正是 LaWM `inverse(当前,未来)→code` 的机制,属下一阶段架构升级,非当前 frame-only 模型可达。

## L8 关键修正:decode-space loss 才是正确目标(2026-07-02,用户提出)

**用户洞察**:训练目标应是"**预测 latent 解码后与 TRUE next-medoid 图像最像**",latent cosine 只是代理。实测(`lever_decode_loss.py`,pooled 解码器 dec.pt,held-out):

| 训练 loss | decoded img L1(↓) | decoded img cos(↑) | latent cos |
|---|---|---|---|
| latent cos(旧基线) | 0.1667 | 0.704 | **0.868** |
| **decode-space** `L1(D(pred),D(medoid))` | **0.1546(−7.3%)** | **0.731** | 0.854 |

**结论(重要)**:decode-loss 模型 **latent cos 反而更低,但 decoded 图像 L1 好 7.3%** → **证明 latent cosine 与解码保真相悖,之前全程优化的 cos 0.874 不是正确指标**。凡以"解码效果"为目标,subgoal 头必须用 decode-space(感知)loss 训练。图 `assets/decode_loss_compare.png`。
- 边界:pooled 解码器自身有 ~0.13 L1 底噪(模糊),decode-loss 降的是预测的**超出部分**;要绝对更清晰需 patch-grid 解码器(自重建 2.7% vs pooled 13%)。

### Track A 落地(2026-07-02):decode-loss 进生产 subgoal 头
全量数据复现:decoded img L1 latent 0.1566 → **decode 0.1451(−7.4%)**,img cos 0.725→0.750,latent cos 0.880→0.862。产出生产头 `lmwm/checkpoints/stage3_decode_subgoal/head_{decode,latent}.pt`(head_decode = decode 优化版)。**采纳**:凡以解码保真为目标,subgoal 用 head_decode。

### Track B 进行中:patch-grid decode-loss(用户要的最终形态)
- B1:持久化 patch 解码器 `lmwm/checkpoints/patch_decoder/patch_dec.pt`(`track_b1_patch_decoder.py`,存 state+mu/sd,可反传)。
- B2:grid 生成头(augin pooled → 16×16×1280 grid)+ decode-space loss(`L1(D(grid), 真实medoid帧)`)vs grid+latent-loss 基线,`track_b2_grid_predict.py`。

### L9 LaWM loss 对比(2026-07-03,用户要求)
LaWM 重建 loss 在**特征空间**(`lam_lightinng.py:626-661`):recon/target = 特征 token `[B,K,dim]`,`loss = smooth_l1(recon,target,β=0.1) + (1 − 逐token余弦)`(+VQ+entropy);上报指标 = **逐 token 特征余弦** + 特征 L1。同数据 3 个 grid 预测头 × 2 把尺子:

| grid 头(loss) | **LaWM 特征余弦**↑ | 特征 L1↓ | **解码 img L1**↓ | 解码 img cos |
|---|---|---|---|---|
| latent (特征 MSE) | **0.694** | 0.093 | 0.192 | 0.656 |
| LaWM (smooth_l1+cos) | 0.694 | 0.093 | 0.185 | 0.668 |
| decode-space (图像 L1) | **0.015** ⚠️off-manifold | 0.401 | **0.182** | 0.663 |

**结论**:①两把尺子**近乎正交甚至反相关** → 证实 latent/特征余弦 ≠ 解码保真;decode-loss 的 grid 偏离特征流形(cos 0.015),只"骗解码器"。②**patch-grid 预测解码保真(0.182)反不如 pooled+decode-loss(0.145)** —— 预测 256 token 比 pooled 向量难太多,更保真的 patch 解码器补不回;图上三列 grid 全模糊+棋盘 artifact。**分叉**:输出喂 VLA 当特征 → 用 LaWM 特征空间 loss(保 on-manifold);产物是"未来图像" → 用 decode-loss 且当前 **pooled 优于 grid**。图 `assets/grid_decode_loss_compare.png`。
