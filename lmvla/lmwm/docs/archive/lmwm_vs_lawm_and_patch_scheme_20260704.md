# LMWM vs LaWM 最终指标对比 + patch-grid 预测/解码方案 (2026-07-04)

> baseline = **LaWM (LaWAM, arXiv 2606.15768)** 的发表指标。本文把"我们提供给 VLA 的提示" 与 "LaWM 提供给 VLA 的提示" 做维度对齐,补齐跨数据集泛化维度,并给出 patch-grid 预测+解码方案。

## 0. 关键前提:两者度量"提示→VLA 效果"的层不同

- **LaWM 用 extrinsic 度量**:提示接进 VLA 后的**下游策略成功率 (SR)**(LIBERO/RoboTwin/真机)。latent forecast 只定性给了 rollout cos "remains high"(Fig 10)。
- **LMWM 用 intrinsic 度量**:提示 latent 与真值的 **forecast cos_sim**(未接 VLA)。
- ⚠️ **因此严格对齐需要我们跑一个 VLA 测 SR —— 这是当前缺口(最关键的未做实验)。** 下表分 4 层,已注明哪层可比、哪层是缺口。

## 1. LaWM (LaWAM) 发表指标(arXiv 2606.15768,全文表格)

| 项 | 值 |
|---|---|
| LIBERO 平均 SR | **98.6%**(Long 97.0 / Goal 98.4 / Object 99.6 / Spatial 99.4) |
| RoboTwin SR | **92.64% clean / 89.80% randomized** |
| 真机 SR(30 trials/任务) | **平均 90.0%**:Pick-Place 93.3 / Open-Drawer 86.7 / **Fold-Towel 90.0** |
| 世界模型参数 | **230M**(总栈 2.3B;DINOv3 ViT-B/16 蒸馏 + 24 层 transformer) |
| 推理延迟 | 187ms/action-chunk(A100),比像素空间 WAM 快 24× |
| 训练数据 | ~3000h 机器人 + 1500h 第一视角人类视频 |
| latent forecast | 仅定性(Fig 10 rollout cos "high") |
| 跨本体泛化 | 定性(Fig 5:同一 latent-action 轨迹跨未见环境/本体产生连贯 latent 变化) |

## 2. 最终指标对比(4 层)

| 维度 | LaWM (LaWAM) | LMWM (ours) | 可比性 / 结论 |
|---|---|---|---|
| **L1 提示准确度 · intrinsic (latent forecast cos)** | 仅定性 "high" | **0.89–0.90**(对齐协议 §3)/ subgoal in-dist **0.882** | ✅ 我们**定量**,LaWM 只定性;同协议在我们数据复刻 |
| **L2 提示→VLA · extrinsic (下游 SR)** | 98.6 / 92.6 / 90.0% | **— 未接 VLA** | ❌ **缺口**:需跑 VLA 才能直接比 |
| — 其中 Fold-Towel(与我们折叠域最近) | 90.0% | — | 我们也是折叠域,但未测 SR |
| **L3 效率 (WM 参数 / 延迟)** | 230M / 187ms | **21.4M**(11× 小)/ 未测(更小更快) | ✅ 我们显著更轻 |
| **L4 跨数据集/本体泛化** | 3 benchmark + 跨本体(定性,3000h 数据) | 跨本体 vis_base **forward 0.935 vs absolute 0.82**(定量) | 不同范围:LaWM 更广(多 benchmark),我们该轴**有定量数**,LaWM 只定性 |

### 详解 L4 跨数据集泛化(用户关注的关键维度)

| | forward+oracle | absolute | forward+predicted(部署) |
|---|---|---|---|
| kai0 in-dist | 0.971 | 0.897 | 0.90 |
| **vis_base 跨本体·未见外观** | **0.935** | 0.820 | ~0.82–0.90 |

- vis_base 是**不同采集/本体**(D435→D405 相机),对我们是**跨数据集+跨外观**测试。
- **forward-from-current 机制外观无关**:子目标继承当前观测外观(红衣→红衣),不漂到 demo 色 → oracle 未见 0.935 ≫ absolute 0.82。
- 与 LaWM 对比:LaWM 跨本体是**定性**(Fig 5),靠海量多源数据;我们是**定量**(0.935),靠机制(forward-from-current)。**但 LaWM 覆盖 benchmark 更广,我们只测了 kai0→vis 折叠域**(未做 LIBERO/RoboTwin,其数据本地不可复现)。

## 3. 同协议对齐(把 LaWM 的方法在我们数据上复刻,`align_lawm_forecast.py`)

固定 ~1.7s horizon、回归未来帧特征、smooth_l1 β=0.1、inverse+forward(code_dim=32+LN)、metric=cos_sim(LaWM 协议):

| horizon | persistence | forward-only | **inverse+forward(LaWM 式)** |
|---|---|---|---|
| ~1.0s | 0.785 | 0.832 | **0.900** |
| ~1.7s(≈LaWM dt) | 0.742 | 0.819 | **0.890** |

→ 同协议下我们 cos_sim **0.89–0.90**,处 DINO 特征未来回归健康区;inverse+forward ≫ forward-only 印证转移码携带未来信息(与 LaWM 机理一致)。**caveat**:对齐的是协议非域,LaWM 论文确切 latent 数无(仅定性)。

## 4. 诚实结论(提示→VLA 对比)

1. **intrinsic 层**(latent forecast cos):我们**有定量 0.89–0.90**,LaWM 只定性 —— 该层我们不落后且更透明。
2. **extrinsic 层**(下游 SR):**LaWM 有 (90–98%),我们没有** —— 这是要补的最关键实验(接 VLA 测 SR)。在此之前"提示效果"不能直接比 SR。
3. **效率**:我们 WM **11× 小**(21.4M vs 230M)。
4. **泛化**:LaWM 广度赢(多 benchmark + 3000h),我们在**跨本体折叠域有定量泛化数**(0.935);两者不同范围,不能简单谁赢。

---

# 5. patch-grid 预测 + 解码方案(基础设施已存在,Track B1/B2)

我们的特征可用 **DINOv3-H patch-grid**(空间 token,16×16×1280)替代 pooled 1280。这让我们**与 LaWM 架构对齐**(LaWM 用 patch tokens + inverse/forward),且**解码显著更锐**。

## 5.1 patch-grid 解码器(`track_b1_patch_decoder.py` → `checkpoints/patch_decoder/patch_dec.pt`)

| 解码路径 | 像素 L1(越低越好) | 说明 |
|---|---|---|
| **直接 patch-grid 解码** | **0.027** | 空间分辨,衣物位置/机械臂对(见 `assets/ep793_lawm_patch_decode.png` 第3列) |
| pooled 合成解码(dec_v2/flow) | 0.062 | 空间坍缩成 blob |
| unpool→解码 | 0.069 | 从 pooled 反推空间失败 |

→ **patch 解码 L1 0.027 ≈ pooled 的 2.3× 保真**,是当前最锐的非幻觉解码(空间条件,像素级对齐)。

## 5.2 patch-grid 预测器(`train_lawm_patch.py`,LaWM 式 inverse/forward on grids)

- **recon(带 future)**:grid cos 0.775–0.783(code_dim 32–128),image L1 0.053–0.056。
- **TRUE 部署(仅从当前预测未来 grid,held-out,`predict_deploy_patch.py`)**:predict_grid_cos **0.653** vs persistence 0.614;predict_decode_L1 0.075 vs persistence 0.090;天花板(真 grid 解码)0.025。

## 5.3 patch vs pooled 权衡(`lever_patch_token.py`)

| | milestone top1 | subgoal cos | 解码 L1 | 成本 |
|---|---|---|---|---|
| pooled(主线) | **0.336** | 0.831 | 0.062 | 低 |
| patch-grid | 0.292（↓,故主线否决分类头） | **0.856** | **0.027** | 全量缓存 ~130GB |

## 5.4 当前:并行支持三种解码方案(暂不收紧)

按需保留三种解码方案,后续需要时再收紧:

| 方案 | 输入 | 特性 | ckpt |
|---|---|---|---|
| **pooled → flow fixed-noise** | pooled 1280 | 外观最锐(folds/纹理),但空间会漂 | `dec_best.pt`+seed=0 |
| **pooled → dec_v2 (L1)** | pooled 1280 | 稳、糊、确定性兜底 | `dec_v2.pt` |
| **patch-grid decode** | patch-grid 16×16×1280 | **空间保真**(臂/衣/夹子位置对),LaWM 对齐 | `patch_decoder/patch_dec.pt` |

milestone-ID 分类头仍建议保持 pooled(top1 0.336 > patch 0.292);patch 特征即时编码即可(单帧无 130GB 问题,仅全量缓存才有)。

### 三方实测(kai0_base 留出测试集 ep8,self-recon,`make_decoder_compare3_vis.py`)

| 解码器 | 像素 L1(↓保真) | 锐度 | 视觉(见 `assets/decoder_compare3_kai0base_testep8.{png,mp4}`) |
|---|---|---|---|
| **patch-grid** | **0.028** | 429 | **空间最保真**:机械臂/衣物/导轨橙夹位置都对 |
| pooled→flow fixed | 0.097 | **638** | 外观最锐(纹理/褶皱),但空间布局漂(pooled 丢空间) |
| pooled→dec_v2 | 0.059 | 450 | 最糊最稳 |
| *real* | — | *1327* | — |

**要点**:pooling 丢掉空间信息 → flow/dec_v2 只能靠 pooled 向量,空间会漂/糊;**patch-grid 保留 16×16 空间 token → 知道"东西在哪"**,像素 L1 最低(0.028)、空间最保真。flow 的高锐度是合成纹理(不像素对齐,L1 反高 0.097)。**三者各有所长,按用途选**:要空间保真→patch;要外观锐利→flow;要绝对稳/省→dec_v2。

**可选后续**:把 patch-grid 预测(`predict_deploy_patch.py`,deploy grid-cos 0.653)接主线 milestone 视频,做预测态(非 self-recon)三方对照。

## 6. 产物索引
- LaWM 对齐:`scripts/align_lawm_forecast.py` → `outputs/lawm_align/summary.json`;`scripts/lawm_adjacent_baseline.py`
- patch 预测:`scripts/train_lawm_patch.py`,`predict_deploy_patch.py` → `outputs/lawm_patch/{summary,deploy}.json`
- patch 解码:`scripts/track_b1_patch_decoder.py` → `checkpoints/patch_decoder/patch_dec.pt`;`unpool_vs_patch.py` → `outputs/patch_decoder/unpool_vs_patch.json`
- patch vs pooled:`scripts/lever_patch_token.py` → `outputs/lever_patch_token/summary.json`
- 可视化:`assets/ep793_lawm_patch_decode.png`,`assets/patch_recon_compare.png`,`assets/unpool_vs_patch.png`
