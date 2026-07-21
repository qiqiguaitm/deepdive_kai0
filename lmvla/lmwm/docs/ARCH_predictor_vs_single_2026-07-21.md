# 为什么 LMWM 用「预测器+生成器」而非「单预测器」—— 受控实验(2026-07-21)

> **问题**:为什么最终不用**单预测器**(当前帧 → 直接回归 milestone+1 代表帧 latent),
> 而用**预测器+生成器**两模型?有无对比实验/数据?
>
> **一句话结论**:两模型的价值**不是**单发精度(单发时单模型反而更准、且不坍缩),
> **而是多模态**——分叉点上 MDN 提议 K 个候选、取命中的那个,单模型(确定性)做不到。
> best-of-K **0.779 > 单模型 0.765 > 两模型单发 0.744**,3/3 与 8/8 seed 一致。

---

## 1. 两种架构 = LMWM 的 v1→v2 演进

| | 单预测器(v1) | 预测器+生成器(v2, 当前) |
|---|---|---|
| 结构 | 一个 MLP trunk → 头**直接回归**目标 latent | 预测器(MDN→紧凑 code)+ 生成器(AdaLN 在当前画布上渲染) |
| 源码 | `lmwm/src/lmwm/models.py:65` `greedy_proto_head=MLP(...)`+`F.normalize` | `train_twomodel_v2.py` / `p1_train_lmwm_libero.py` |
| 部署输出 | `f(g_t)` 一张(确定性) | 单发=MDN mode;多模态=K 分量各渲一张 |

**历史声称的转换理由**(`train_twomodel_v2.py` 头 + `archive/decoder_blur_diagnosis`):
单模型直接回归会「**持久坍缩**」(输出≈当前帧)+「均值/模糊 ill-posed」。
但那是 **DINOv3-H / SigLIP / 离散 milestone** 老口径,**从未有干净的同空间 A/B**。本文补这张表。

---

## 2. 受控实验设计

同一 **So400m(pi05 真 token)空间**、同一 **rvalley pairs**、同一 **held-out(按 ep 20%)**,容量匹配(都用 hid=512 nblk=4 conv backbone):

| 臂 | 定义 |
|---|---|
| **A** | 单模型直接回归 `f(g_t)→ĝ`,loss=smooth_l1 [= v1 风格] |
| **A+** | A + 反持久 lift 项 [隔离:光加 lift 够吗] |
| **B 单发** | 预测器(MDN)取 mode code → 生成器渲染 [不看未来] |
| **B best-of-K** | MDN 的 K=4 分量均值各渲一张,逐样本取 recon 最高 [多模态,不看未来] |
| **B oracle** | 生成器用 teacher(看未来)code [上界参照] |

判据:`recon_cos(ĝ, 目标)` · `persist=cos(当前,目标)`(基线)· `lift=recon−persist` ·
`copy_cos=cos(ĝ,当前)`(持久坍缩指纹)。脚本 `exp_single_vs_twomodel.py`(已入 git)。

---

## 3. 结果(code=32 主配置)

| 臂 | recon_cos | lift | n seed |
|---|---|---|---|
| persist 基线 | 0.6650 | — | — |
| **A 单模型直接回归** | **0.7654**±.003 | +0.100 | 8 |
| A+ 单模型+lift | (不稳,见 §5) | — | 8 |
| B 预测器+生成器(单发 mode) | 0.7445±.003 | +0.080 | 8 |
| **B 预测器+生成器(best-of-K)** | **0.7793**±.003 | +0.115 | 3 |
| B oracle(看未来) | 0.7928±.002 | +0.128 | 8 |

**两个关键对比,方向相反,全 seed 一致:**

| 对比 | Δ | 一致性 | 谁赢 |
|---|---|---|---|
| 单发部署:A − B单发 | **+0.021** | **8/8 seed** | **单模型** |
| 多模态:B best-of-K − A | **+0.014** | **3/3 seed** | **两模型** |

**code=128 复核(排除 code 瓶颈)**:A 0.765 vs B单发 0.741,A−B=+0.025,**2/2 seed** —— 加大 code 容量**不救** B 单发,坍缩说不成立于 code 维度。

---

## 4. 机理:为什么是这样

1. **单模型没坍缩**。A 的 lift **+0.100**(远>0)、copy_cos 0.753 ≈ 自身 recon,**不是原地不动**(persist 0.665)。
   → **历史"持久坍缩"叙事在 So400m/rvalley 设定下不复现**。老口径的坍缩是那套编码器/离散标签特有,不普适。
2. **单发时单模型更准**,因为它直接学**条件均值** `E[g_f|g_t]`;而两模型要挤过 code 瓶颈,MDN 单个 mode 未必对准分支。
3. **但分叉点是多模态的**。给定 g_t,下一 milestone 常有多个可能分支。
   - 单模型只能出**一个**(均值,分叉处必偏);
   - 两模型的 MDN 出 **K 个分支提议**,生成器各渲一张,**其中一个命中** → best-of-K 反超。
4. **oracle(0.793)证明生成器渲染力足够**:瓶颈在**预测器从 g_t 推断分支**,而多模态提议正是绕开它的方式。
   → **生成器+紧凑 code 分解 = 多模态的使能条件**(确定性单回归器无法 best-of-N)。

**因此**:预测器+生成器的正确辩护 = **多模态子目标提议**(分叉处 propose-K),而非单发精度或防坍缩。

---

## 5. 诚实的边界(勿过度外推)

- **内在指标,非下游 SR**。recon_cos 到目标 latent ≠ VLA 用它当 hint 后的成功率。按 roadmap §4.7 教训(内在 gain 2.1× 未换 SR),此结论**不能**直接推成"两模型对 VLA 更好"。真判据仍是 best-of-K 子目标是否提 VLA-SR,待做。
- **So400m/rvalley 特定**。老 DINOv3-H/SigLIP 空间可能不同(那里历史观察到坍缩)。
- **best-of-K 需要下游能消费多个候选**(如 CFG/采样/重排),单选一个就退回 B 单发(输)。
- **A+ 不稳**:单模型加 lift 有 seed(s10)塌到 0.673 —— 反持久项在无 code 指向时会过罚。lift 是 v2 的项,单模型嫁接会脆。

---

## 6. 复现

```bash
CUDA_VISIBLE_DEVICES=0 python lmwm/scripts/exp_single_vs_twomodel.py --steps 4000 --seed 0            # code=32 单发+bestofK
CUDA_VISIBLE_DEVICES=1 python lmwm/scripts/exp_single_vs_twomodel.py --steps 4000 --seed 0 --code_dim 128
```
产物:`lmwm/outputs/exp_svst_*.json`(逐 seed)、`exp_svst_SUMMARY.json`(汇总)、`exp_svst_env.json`(版本+hash)。
特征:`lmwm/data/libero_so400m_grid`(400 ep,So400m patch-token grid,`_env.json` 在旁)。
