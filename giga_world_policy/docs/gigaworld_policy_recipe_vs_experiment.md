# GigaWorld-Policy 官方 recipe 对照与本实验诊断（叠衣服 visrobot01）

> 调研日期 2026-06-04。目的:本实验(visrobot01 叠衣服 full-FT)观察到 **held-out 开环 action MAE 平坦且劣于 "stay-put" 基线、train `action_loss` 长期在 ~0.3**,据此对照 GigaWorld-Policy 官方论文/recipe,判断是否异常并给出下一步。
> 结论先行:**这不是 pipeline bug,而是 recipe 偏离官方 + 评估口径不对位**。核心差异是本实验**缺 embodied 预训练阶段**、**loss 权重用 1:1(官方后训练为 5:1)**、且**用开环 MAE 评估(官方用闭环成功率)**。

---

## 0. 本实验现象(已核实)

- 训练:`visual_loss` 0.30→~0.18 并缓慢改善;held-out 视频指标缓慢变好(`psnr~19`,`ssim 0.65→0.72`,`temporal 1.9→1.3` 趋向 1.0)。
- 但 `action_loss` 快速降到 ~0.3 后长期在 **0.29–0.31** 徘徊(平滑后 train total 仍在缓降,非硬 plateau)。
- held-out 开环 action MAE **平坦且劣于 stay-put 基线**:模型 mae@1=0.144 / mae@48=0.24,stay-put 基线 mae@1=0.000(因 `action[0]==当前state`)/ mae@48=0.175。

### 五维 bug 排查(均已否定,非 pipeline bug)
1. ✅ checkpoint/resume:action head 权重 6k→30k 变化 11.6%(确实在训练、且跨 resume 持久化)。
2. ✅ eval 重构:GT 经 `denormalize_action`+`add_state_to_action` round-trip 误差 = 0.000000。
3. ✅ train↔infer 约定:timestep 布局、共享 `sigma`、`flow_shift=5`、scheduler(UniPC,无 dynamic shift)、`num_frames`(视频 T_latent=2≈5帧,两侧一致)全部一致。
4. ✅ 推理积分:denoise 步数 10/30/60 结果不变 → 非"步数太少"。
5. ✅ 条件依赖:同一 ckpt 对 4 个不同 window 的预测 std=0.285(≠0)→ 模型确实用了输入,非"忽略条件"。

---

## 1. 官方 GigaWorld-Policy vs 本实验 —— 对照表

| 维度 | 官方 GigaWorld-Policy(论文 2603.17240) | 本实验(visrobot01_fold) | 差异 |
|---|---|---|---|
| **初始化** | Wan2.2-5B → **embodied video-only 预训练**(≈10000h, 6000 GPU·h, bs256) → 任务后训练 | **raw Wan2.2-TI2V-5B** 直接联合训练(`checkpoints/` 仅有 raw Wan + T5,无官方 embodied-pretrained 权重) | ❗**缺整段 embodied 预训练** |
| **训练阶段** | 两阶段:**预训练只优化 video**,后训练才加 action | 一上来 video+action 联合 | ❗ |
| **loss 权重** | `ℒ=λ_v·ℒ_video+λ_a·ℒ_action`,**后训练 λ_action=5, λ_video=1**(原文"emphasizing action prediction while retaining the video-consistency regularizer") | **1:1**(`parse_losses` 直接 `sum`) | ❗**action 欠加权 5×** |
| **优化器/lr** | AdamW(β1=0.85,β2=0.9),cosine **1e-4→1e-6** | CAME8Bit,warmup→**8.6e-5**→0 cosine | lr 量级吻合(印证 lr 非主因);优化器不同 |
| **后训练数据** | 仅 **50 demos/任务** | ~2098 episodes(visrobot01_train) | 本实验数据更多,但缺预训练 |
| **评估** | **闭环 Success Rate**(real **0.83**、sim RoboTwin2.0 **0.86**;graded:抓0.5+放0.5,20 trials/任务,≤5次尝试) | **开环 action MAE**(及视频 PSNR/SSIM) | ❗**评估口径不对位** |
| **video 分支** | **辅助/可选**:action-only 推理直接出控制;ablation Δ=0→SR0.60,**Δ=12→0.83**(video 监督 +0.23 SR) | 同架构(action 主、video 辅,causal mask:action 不 attend future-video) | 一致 |
| **action_chunk** | 48 | 48 | 一致 |
| **future stride Δ** | 12 | (训练 stride=4 采样;eval exec_horizon=16) | — |

---

## 2. 核心结论

**(1) "video 快 / action 慢" 符合 WAM 趋势 —— 且官方设计预设了这一点。**
video 从 Wan 迁移(快)、action 从零(慢)是结构性必然。官方为此加了两个补偿:(a) 大规模 **embodied video-only 预训练**先把 backbone 喂熟动力学;(b) 后训练 **5:1 强加权 action**。**本实验把两个补偿都省了**(raw Wan + 1:1) → action 慢/弱是 recipe 偏离的预期后果,**不是 bug**。

**(2) `action_loss≈0.3` 不能直接判定"已到 floor"。** 论文只报闭环 SR、不报 loss 数值,无法对标量级;但官方靠"预训练+5:1"才到 0.83 SR,本实验缺这两项 → 0.3 偏高且收敛慢更可能是 **recipe 不足**,而非任务 floor。

**(3) 开环 action MAE 劣于 stay-put ≠ 模型失败。** 官方**完全用闭环 Success Rate**。叠衣服是准静态任务,stay-put(输出当前位姿)是极强基线(mae@1 天然=0)。**判定 GigaWorld-Policy 类模型必须用闭环 rollout 成功率**;开环 MAE 仅作 sanity check。

---

## 3. 下一步建议(有官方依据,按杠杆排序)

1. **改 `λ_action=5`(5:1)** —— 与官方后训练一致,直接针对 action 欠监督。最高杠杆、改动小(`forward_step` 给 `action_loss` 乘权重,或加 config 字段)。
2. **评估改闭环 rollout 成功率** —— 开环 MAE 仅作 sanity,别作判定(尤其准静态任务)。
3. **(中期)补 embodied video-only 预训练**,或设法获取官方 embodied-pretrained checkpoint 作 init —— 这是官方 action 能力的主要来源,本实验最大的缺口。
4. (可选)优化器/lr 对齐:AdamW(0.85,0.9)、peak 1e-4。

---

## 4. 来源

- GigaWorld-Policy 论文:[HF papers/2603.17240](https://huggingface.co/papers/2603.17240) · [arXiv abs](https://arxiv.org/abs/2603.17240) · [arXiv html](https://arxiv.org/html/2603.17240)
- 官方项目页:[gigaai-research.github.io/GigaWorld-Policy](https://gigaai-research.github.io/GigaWorld-Policy/) · 官方代码:[github.com/open-gigaai/giga-world-policy](https://github.com/open-gigaai/giga-world-policy)
- 关联工作:[GigaWorld-0 (2511.19861)](https://huggingface.co/papers/2511.19861)、[GigaBrain-0 (2510.19430)](https://huggingface.co/papers/2510.19430)、[GigaBrain-0.5M (2602.12099)](https://huggingface.co/papers/2602.12099)
- 同类参考:[Unified World Models (2504.02792)](https://huggingface.co/papers/2504.02792)、[Video Prediction Policy (2412.14803)](https://huggingface.co/papers/2412.14803)、[τ0-WM (2606.01027)](https://huggingface.co/papers/2606.01027)

> 注:对照表中"官方"数值来自论文/项目页(部分经 WebFetch 小模型摘取,关键项为原文引述);本实验数值来自 `runs/visrobot01_fold_aihc_latent*` 训练日志与 `eval_watch` 输出。loss 绝对量级官方未公开,故 §2(2) 标注为"不可直接对标"。
