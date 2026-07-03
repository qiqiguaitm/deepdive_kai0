# CRAVE → KAI0-AE 蒸馏:两种标签方法对照 plan

> **目标(收紧)**:用 CRAVE 自动生成的 `stage_progress_gt` **替代人工 Step-0 标注**,各训一个 KAI0-AE,**离线看效果**。**暂不做 VLA / 真机**。
> 这是 [awbc_milestone_value_AB_plan](awbc_milestone_value_AB_plan.md) 的 **B 臂**(CRAVE→AE 蒸馏)。日期:2026-07-03。

## 0. 命题
KAI0-AE 现在的监督信号 `stage_progress_gt` 靠**人工分段 + 线性插值**(Step-0,痛点:人工+循环)。用 **CRAVE 逐帧 value** 顶替它 → 得到一个**零人工标注**的 value 模型。比两种 CRAVE 标签构造法训出的 AE 谁更干净。

## 1. 两种标签方法(都产出逐帧 `stage_progress_gt` 0→1)

| | **A · anchor-linear** | **B · production readout** |
|---|---|---|
| 构造 | 段内**离簇心最相似帧=锚点**(峰值成员,非边界),value=Pord;+ start(0,0)/end(末,1.0);isotonic 兜单调;线性连接 | `crave_value.py::DiscreteValue.value()` —— 三路特征(raw⊕armmask⊕**proprio**)+ coverage 修正 + startK/endK 锚 + Viterbi-DP |
| 形状 | 分段线性(锚间线性) | Viterbi 平滑(处理 dwell/loop/真回退) |
| 鲁棒性 | **高**(start/end 锚 + isotonic 强制 0→1) | 高(proprio 消起末别名) |

**⚠️ 关键教训(sanity 已验证)**:**naive DINOv3-H-only Viterbi 不行** —— 无 proprio → 起末视觉别名(折好布≈摊平布)→ ep763/1527 起点误吸到晚期 milestone、卡住(corr 0.07 / −0.80)。**B 必须走 `crave_value.py` 生产读出**(proprio 是关键),不能自己裸写 Viterbi。A 因为强制 start0/end1+isotonic,天然规避,故 A 用 DINOv3-H 即可。

![两种标签 vs 人工](visualization/ae_distill/stage_label_compare.png)

*(sanity:A mono 1.00 / corr 0.86–0.96 全稳;naive-Viterbi 在 ep763/1527 崩 → 换生产读出。)*

## 2. 数据处理(生成两个可用数据集)

**帧映射**:native 30Hz,DINOv3-H/3路 cache 3Hz(FR=native 索引 ×10)→ 生成 3Hz value → 线性插值到 30Hz → 写 `stage_progress_gt`。

**步骤**:
1. **A 标签(已就绪,全 3055 ep)**:`crave/experiments/gen_ae_stage_labels.py --full` → `temp/crave_ae_labels/anchor/ep*.npy`(native-fps)。DINOv3-H cache 覆盖全 3055 ep。
2. **B 标签**:走 `crave_value.py` 三路生产读出。**覆盖:三路 cache 现有 550 ep**;要全量需先补三路特征提取(或用 DINOv3-H⊕proprio 重建生产读出)。**先按 550 ep 交叉集跑 pilot**,通过再补全量。
3. **写数据集**:以 `kai0_base` 为底(observation.state/action/video/meta),各加 `stage_progress_gt` 列 →
   - `data/Task_A/self_built/crave_stage_A/`(A 标签)
   - `data/Task_A/self_built/crave_stage_B/`(B 标签)
   videos/meta 共享(symlink),仅 parquet 增列。

## 3. AE 训练(各一个,集群)

- **配置**:`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`(pi0-AE,`AdvantageEstimator.value_head`),Step-1 `train_pytorch.py`,50k step(与现有 C 同参,唯一变量=标签)。
- **三臂**:
  - **AE-A** = 训在 crave_stage_A;
  - **AE-B** = 训在 crave_stage_B;
  - **AE-C(现成 baseline)** = 现有 `adv_est_v1/100000`(人工 `stage_progress_gt`)。
- 长训走集群(`submit-training-job`),本地只 sanity。

## 4. 离线看效果(判据 —— 不用"对人工 GT 的 MAE",会循环)

三个 AE 各自打 `absolute_value` / `relative_advantage`,在 held-out 成功 episode 上比(全是成功叠衣,理应几乎全 POSITIVE/NORMAL):

| 指标 | 含义 | 期望 |
|---|---|---|
| **P/N 翻转次数 / NEG 帧占比** | 痛点① 抖动 | **越低越好**(报告里 AE-C 有 256 次翻转) |
| **单调率 mono** | value 平滑单调 | 越高越好 |
| **relative_advantage 噪声(std / 过零率)** | AWBC 标签稳定性 | 越低越好 |
| **完成态 value** | 末帧是否到高值 | 接近 1 |
| 与人工 GT 在**清晰帧**上的一致性 | 合理性(非全局 MAE) | 参考,不作唯一判据 |

**看效果 = AE-A / AE-B 是否比 AE-C 的 P/N 更干净、更单调、advantage 更稳**(正是报告痛点①③要解的)。A vs B 谁更好 = 两种标签形状哪个更利于蒸馏。

## 5. 诚实边界 / 风险
- **天花板 = CRAVE 标签质量**:蒸出的 AE 继承 CRAVE 的"完成态偏弱 + 只抓粗失败"(B2 已否)。换掉的是"人工分段"痛点,不修细粒度盲区。
- **circular MAE 陷阱**:AE-B/C 各拟合自己的标签,别用"对谁的 MAE"评判 → 用 §4 的 P/N 干净度 / 单调 / advantage 稳定度。
- **B 覆盖**:三路 cache 550 ep;pilot 先跑 550,全量待补特征。
- **downstream 决定性判据(AWBC rollout)本轮不做**,留作后续(见 AB_plan Tier3 sim)。

## 6. 里程碑
| P | 内容 | 状态 |
|---|---|---|
| P0a | A 标签全 3055 ep 生成 | 脚本就绪,`--full` 即出 |
| P0b | B 标签(crave_value.py 三路,550 ep pilot) | 待接线 |
| P0c | 写 crave_stage_A / crave_stage_B 两数据集(加列) | 待 |
| P1 | 集群训 AE-A / AE-B(50k),对照 AE-C | 集群 |
| P2 | 离线 P/N 干净度 / 单调 / advantage 对照(§4) | 出结论 |

复现:`crave/experiments/gen_ae_stage_labels.py`(A 标签 + sanity 图);B 标签接 `train_scripts/kai/data/crave_value.py`。
