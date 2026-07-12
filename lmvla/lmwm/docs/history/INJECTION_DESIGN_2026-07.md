# latent milestone 注入 π0.5:方案设计(2026-07-08 深调研综合)

> 目标:把 LMWM 的 latent milestone(16×16×1152 SigLIP-space grid 或 pooled code)注入 kai0 π0.5,**不毁预训练**。
> 两路深调研:① π*0.6(PI 真实后继)+ Knowledge Insulation(PI 的保预训练教科书);② 六种注入范式的文献级对比。
> 结论:**把 milestone 当"虚拟未来图像 token"走 prefix(SigLIP 原通路)+ KI stop-grad 隔离 + 动作前摆位 + CFG dropout + FLARE 式辅助监督**。

---

## 0. 决定性事实(它改变了排序)

**milestone ∈ SigLIP-So400m 空间(1152)= PaliGemma 视觉 embedding 空间。** 最大杠杆:
- 通用文献(调研②)警告"prefix 注入=最高污染"——**那是针对任意新空间的 token**;
- SigLIP 同分布 token 经**原图像投影**进 backbone,几乎零 distribution-shift → prefix 注入从"最危险"变"最原生、最安全";
- 且这正是 **PI 自己(π*0.6)的真实做法**(见 §1)。

---

## 1. PI 的真实做法(π*0.6 arXiv:2511.14759 + KI arXiv:2505.23705)

**澄清**:没有 "π0.7"。π0.5 正式后继 = **π*0.6**(RECAP 方法),底座 π0.6 = π0.5 + KI 训练法。

PI 注入**新高层条件**(advantage 指示 I_t)的四个动作,全部可直接抄:

| PI 做法 | 机制 | 对我们 |
|---|---|---|
| **① 新条件走 VLM prefix 的 token 化通路** | I_t 当文本 token 塞进 prefix,action 经共享注意力读它;**不改 adaRMS、不加 cross-attn** | milestone 走 prefix(虚拟图像 token) |
| **② KI stop-gradient 隔离** | 对连续 flow expert → backbone 的注意力 K/V 套 `sg()`(读得到、梯度不回流);backbone 由**离散 FAST token + VLM next-token loss** 干净驱动 | 新 milestone 投影用 KI 隔离,不腐蚀 backbone |
| **③ 序列摆位:条件 token 放 language 之后、action 之前** | "only the action log-likelihoods are affected"——影响局部化到动作,不污染 VLM 语言/视觉 loss | milestone token 摆在 lang 后 action 前 |
| **④ CFG 式条件 dropout** | 训练随机丢 I_t → conditional+unconditional 双模型,推理可 guidance、可无条件回退 | milestone 随机 dropout,推理鲁棒+可调强度 |

**两个反直觉纠正(推翻我早先的猜测):**
- **adaRMS 不是好注入点**:PI 新增高层条件时**明确不用 adaRMS**;adaRMS 只用一个向量调制归一化,**带宽太低,喂不下 16×16 网格**。adaRMS 只留给 pooled code 作轻量补充,**无 PI 背书**。
- **KI 不是"冻结 backbone"**:论文明确"Freezing the backbone is not viable——预训练表征不足以支撑机器人"。KI = backbone 照常更新,但**只由离散 token + VLM 数据的干净梯度驱动**,被 sg 切断的是"连续 expert → backbone"这条脏路。→ 接线不能简单冻 PaliGemma,要走 KI。

**PI 没有 latent world-model**:π*0.6 的"未来"只是**离散子任务文本 ℓ̂ + 独立 value 模型**(670M,辅助,不进 policy 前向)。与我们"检索/离散 > pooled 合成"教训一致——但我们已定学习式 latent milestone,故取其**注入机制**,不取其"离散化"。

---

## 2. 六范式对比(调研②)保预训练排序

威胁 ∝ 对预训练自注意力流的扰动。三种最安全:

| 范式 | init 恒等? | grid 保真? | PI 实证? | 落点 |
|---|---|---|---|---|
| **虚拟图像 token 进 prefix**(SigLIP 原通路) | 近似(同分布) | ✅ 全 256 token | ✅ π*0.6 + π0.5 | prefix |
| **零初始化门控 cross-attn**(adaLN-Zero / Flamingo tanh-gate) | ✅ **位级恒等** | ✅ 全 grid,不进自注意力 | Act2Goal/LaWAM | action expert 新层 |
| **FLARE 辅助对齐**(推理可丢) | ✅ 推理零改动 | ✗ 仅 pooled | FLARE/DIAL | 纯辅助 loss |
| adaRMS 调制 | ✅ | ✗ 低带宽 | ❌ PI 不用 | action expert(仅 pooled) |
| 纯 action-expert token | ✅ 零风险 | ✅ | 部分 | action expert(不进 VLM) |
| LoRA/adapter/prompt-tuning | — | — | OpenHelix:prompt>全FT | 正交,包住上面 |

---

## 3. 最终推荐(三档,主推第一)

### 主推 · P — SigLIP 原生虚拟图像 token + KI(命中两路调研交集)
```
milestone grid (16×16×1152)
  → 复用 π0.5 SigLIP 图像投影(同分布,近零 distribution-shift)
  → 256 token(或 4× 池化到 64 token 省算力)
  → 摆在 prefix:[图像 768 | 语言 | ★milestone token★]  ← language 之后
  → + type/segment embedding 区分(vs 真实图像)
  → KI stop-grad:milestone 投影(若新初始化)对 backbone 的 K/V 套 sg()
  → CFG dropout:训练随机丢 milestone
  → 冻 LMWM provider(纯前向出 milestone),backbone 走 KI 训练法(不冻)
可选:milestone token 用 DreamVLA 分块掩码,禁其改写真实图像/语言 KV
```
**为什么最优**:唯一同时满足"全空间保真 + 同分布近零污染 + PI 双实证(π*0.6 prefix token + KI)+ 用足 π0.5 共享注意力让 VLM 也能语义推理 milestone"。这**正是你最初直觉("从 VLM 侧注入 latent feature")的正确形态**——加上"SigLIP 同空间"关键条件后,从危险变最优。

### 备选 · A — 零初始化门控 cross-attn(若 P 出现漂移)
action expert 每(隔)块加一个 cross-attn 子层,Q=action hidden,K/V=milestone grid 投影,tanh 门/adaLN-Zero **初始为 0**。grid **永不进自注意力** → 结构性保证不污染预训练 KV,init 位级恒等,保全空间保真。只训投影+门+可选 LoRA。**当 P 的 prefix 注入实测掉语言跟随/掉点时切到这条。**

### 备选 · F — FLARE 辅助对齐(最省首实验 + 正则)
加 K≈4-8 个可学习 milestone query token,只用辅助损失对齐 pooled milestone 嵌入(cosine/smooth-L1),**推理可丢**。最防遗忘、零推理开销,丢空间细节。**先用它一天出信号验证"milestone 条件对 SR 有没有正增量",再上 P。** 也可作 P 的零成本正则叠加。

---

## 4. 对你原三方案的逐条裁决

| 你的原想法 | 裁决 |
|---|---|
| "从 VLM 侧注入 latent feature" | ✅ **正确且最优**——但必须以"SigLIP 同空间虚拟图像 token + KI + 动作前摆位"的形态(主推 P) |
| "从 VLM 侧增加 feature query" | ✅ 可行(FLARE 式备选 F),纯 query+辅助监督丢空间保真;适合首实验/正则,非主通路 |
| "用 latent feature 来监督也行" | ✅ **最保预训练**(备选 F 辅助损失);建议**与 P 叠加**(既注入又监督) |
| "参考 π0.7/LaWAM" | π0.7 不存在→π*0.6:走 **prefix token + KI**,不用 cross-attn/adaRMS;LaWAM 用 AdaLN cross-attn(=备选 A 谱系) |

---

## 5. 实验计划(便宜优先,每步 kill criteria)

| 阶段 | 做法 | 判据 |
|---|---|---|
| **I0 · 辅助 only**(备选 F) | 加 milestone query + FLARE 辅助对齐,推理不注入 | offline action-MAE:有 vs 无辅助;≈0 → milestone 对策略无信息,止 |
| **I1 · 主推 P 接线** | 虚拟图像 token 进 prefix + KI stop-grad + 动作前摆位 + CFG dropout,LoRA action expert,本地 2 卡短跑 | action-MAE:base vs +milestone(GT) vs control;>base → 进 I2 |
| **I2 · 漂移检查** | 测语言跟随/原任务是否退化(KI 是否顶住) | 退化 → 切备选 A(零初始化 cross-attn) |
| **I3 · 真预测器 + 集群** | 换真 LMWM milestone(非 GT),集群微调 | SR vs LaWM 98.6% |

**先 GT milestone 隔离"注入机制是否有效",再换真预测器**——避免预测误差污染注入方案的结论(沿用 next_milestone_vla_validation_plan 的 GT-first 铁律)。

---

## 6. 关键引用
- **π*0.6 / RECAP**:arXiv:2511.14759(prefix token 注入新条件 §V-B、动作前摆位、CFG dropout、KI 采纳 §V-A)
- **Knowledge Insulation**:arXiv:2505.23705(注意力层 sg() Eq.5-6、离散替身信号、backbone 不冻)
- **π0.5**:arXiv:2504.16054(子任务 ℓ̂ 预测 + 后训 joint action expert)
- **FLARE** 2505.15659 · **DreamVLA**(分块掩码)2507.04447 · **LaWAM**(AdaLN cross-attn)2606.15768 · **DiT adaLN-Zero** 2212.09748 · **Flamingo 门控 xattn** 2204.14198 · **Act2Goal** 2512.23541 · **LCB/OpenHelix**(prompt>FT)2405.04798/2505.03912 · **FAST** 2501.09747

> 调研备注:本轮 WebSearch/WebFetch 后端不可用,两路均通过 HF paper_search + curl 直取 arXiv 全文核对;OpenHelix 2.13-vs-1.74 与 InternVLA-A1.5 指针为既有引用未独立复核。
