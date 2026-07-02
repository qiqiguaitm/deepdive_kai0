# 均值+方差双降研究:编码器侧 × LMWM 侧(2026-07-02)

> 目标:同时降低 loss **均值**与 loss **方差**(减少"错得离谱"的预测次数)。
> 双线并行:编码器侧(表示)与 LMWM 侧(模型/损失/集成),并参考 LaWAM 的 code 因子化机制。
> 全部 held-out episode,对**真实观测**的下一 milestone / episode-medoid subgoal 评估。
> 方差度量:逐样本 NLL 的 std / p90 / CVaR@10%(最差 10% 均值);subgoal 为 cos 的 std / p05 / frac<0.7。

## 一、编码器侧(已收口,两个负结果一个确认)

学习型 MLP 探针,162k 阶段转移,held-out(`probe_encoder_7b.py`):

| 输入表示 | top1 | NLL | NLL std |
|---|---|---|---|
| **H 1280(现用)** | 0.380 | 2.377 | 2.408 |
| 7B-int8 4096 | 0.351 ❌ | 3.090 | 3.287 |
| **H + state 裸拼(现用)** | **0.406** | **2.121** | **2.157** |
| H + state CRAVE式(z-score+L2) | 0.398 | 2.299 | 2.386 |
| H + 7B 拼接 | 0.375 | 3.205 | 3.548 |

结论:
1. **7B 编码器放弃**(用户判断 + 数据双重支持):int8 7B 特征对本任务全面更差,连拼接都被拖累。不实惠且无收益。
2. **CRAVE 式 L2 融合不如裸 z-score 拼接**(0.398 vs 0.406)——保持现有 augin 输入。
3. **编码器侧最优 = H 1280 + prev-milestone + state 裸拼(现状)**,该侧已到头(剩余未测:多相机,成本高、暂缓)。

## 二、LMWM 侧(主战场,三个杠杆全部有效)

### 杠杆 1:Ensemble(降均值为主,方差顺带降)

同 split(seed 2026)、不同 init(`init_seed` knob),3-4 成员概率平均:

| | top1 | NLL | NLL std |
|---|---|---|---|
| 单模型 | 0.408 | 1.953 | 1.652 |
| **ensemble_4** | **0.436** | **1.787** | 1.432 |

**纯神经 ensemble(0.436)已超过之前"单模型+图融合"的 0.434。**

### 杠杆 2:CVaR-CE 尾部损失(降方差为主,均值持平)

离散头 CE 换 `(1-w)·mean + w·mean(最差10%)`(`ce_tail_mode: cvar`,w=0.5):

| | top1 | NLL | **NLL std** | **CVaR10** |
|---|---|---|---|---|
| 普通 CE | 0.408 | 1.953 | 1.652 | 5.55 |
| **CVaR-CE** | 0.383 | 1.947 | **1.013(−38%)** | **4.08(−26%)** |

(focal γ=2 也降尾但幅度小:std 1.56。CVaR 明显更强。)

### 杠杆 3:LaWM 式 code 因子化(subgoal 侧,降均值)

Teacher `inverse(x,g_future)→32维code` + `forward(frame,code)→medoid`;student 只预测 code,部署 `forward(frame, student(x))`(`code_factorized_subgoal.py`,即 LaWAM 的 policy-predicts-code 机制):

| subgoal 头 | cos mean | std | frac<0.7 |
|---|---|---|---|
| 直接回归(基线) | 0.864 | 0.069 | 3.5% |
| variance 尾部损失 | 0.858 | 0.059 | 2.5% |
| **code 因子化 student** | **0.874** | 0.076 | 3.9% |
| **cvar_ensemble(3成员平均)** | **0.874** | **0.068** | **2.9%** |

- code 因子化验证了 LaWM 假设:**预测 32 维 code 比预测 1280 维 latent 容易**(teacher oracle 0.958 证明 code 容量足),部署均值 0.874 = 触到 kNN 上界 0.877。
- 但它不降方差(+tail 也无效);**cvar_ensemble 的 subgoal 平均**同样到 0.874 且尾部更好 → subgoal 侧的平衡赢家。

## 三、组合:最终 Pareto 前沿(全部 +图先验融合 λ=0.3)

| 候选 | top1 | NLL | NLL std | CVaR10 | 定位 |
|---|---|---|---|---|---|
| 基线(augin 单模型) | 0.408 | 1.953 | 1.652 | 5.55 | 之前的最佳 |
| **ensemble_4 + fuse** | **0.453** | **1.698** | 1.189 | 4.27 | **均值冠军** |
| **mixed_ens_6 + fuse**(4普通+2cvar) | 0.453 | 1.740 | **0.995(−40%)** | **3.83(−31%)** | **平衡冠军 ⭐推荐** |
| cvar_ens_3 + fuse | 0.434 | 1.855 | **0.859(−48%)** | **3.65(−34%)** | 方差冠军 |

**推荐配置 = mixed_ens_6 + fuse0.3**:top1 与均值冠军打平(0.453),方差 −40%、最差 10% 样本 −31% —— **均值与方差同时大幅改善,正是研究目标**。

## 四、总提升账(vs 研究前)

| 指标 | Phase A 单帧 | 研究前最佳 | **研究后(mixed_ens_6+fuse)** |
|---|---|---|---|
| top1(真实未来) | 0.383 | 0.434(augin+fuse) | **0.453** |
| NLL | 1.98 | 1.78 | **1.74**(std 1.65→**1.00**) |
| CVaR10(最差10%) | — | 5.55 | **3.83** |
| subgoal cos | 0.832(簇心头) | 0.864 | **0.874**(±0.068,<0.7=2.9%) |

## 五、代码产物

- trainer 新 knob(默认关,向后兼容):`init_seed`(ensemble 用,split 不变仅 init 变)、`ce_tail_mode: focal|cvar|variance` + `ce_tail_weight/ce_cvar_q/ce_focal_gamma`
- `scripts/probe_encoder_7b.py` → `outputs/ceiling_diag/encoder_7b.json`
- `scripts/code_factorized_subgoal.py` → `outputs/code_factorized/`
- `scripts/eval_mean_variance.py`(统一均值+方差评估器)→ `outputs/mean_variance/final.json`
- 成员 checkpoints:`stage3_augin_ens/`(×3)、`stage3_augin_tail/`(focal+cvar×3)
- 配置:`..._augin_e{1,2,3}.yaml`、`..._augin_{focal,cecvar}.yaml`、`..._cecvar_e{1,2}.yaml`

## 六、诚实边界与后续

- Ensemble 推理成本 = 成员数×(每成员 3M 参数,总 ~18M,单帧毫秒级,VLA 可接受;不可接受时用 cvar 单模型 + fuse:0.414/std 0.90)。
- code 因子化 student 与 LaWAM 部署机制同构 —— 接 VLA 时可让**策略直接预测 32 维 code**(LaWAM 原生做法),而非消费我们预测好的 subgoal;两者都已具备。
- 未测(编码器侧仅剩):多相机(hand_left/right)特征 —— 成本高,若需再提可作下一步。
- 数字均为 kai0_base held-out;换数据集需重估 λ/T 与成员数。
