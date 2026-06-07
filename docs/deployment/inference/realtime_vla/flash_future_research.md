# Realtime-VLA FLASH — 深度研究方向 (deepdive_kai0 视角)

> **目的**: 在通读 [Realtime-VLA FLASH](https://dexmal.github.io/realtime-vla-flash/) (arXiv:2605.13778, Niu et al. 2026) + 源码 (`/data1/tim/workspace/realtime-vla-flash`, openpi/pi0 衍生) 之后, 结合 deepdive_kai0 已有的 realtime-VLA 工作 (`roadmap.md` / `strategy.md` / `v1_triton_log.md` / `layer_b_plan.md` / `ee_stability_layer1.md`) 与本仓库近期诊断结论 (开环/因果混淆、夹爪校准漂移、视觉消融), 提出**可深度研究的后续课题**。
>
> **建立**: 2026-06-07
> **代码**: `realtime-vla-flash/` (root) · 模型 `huggingface.co/Dexmal/RealtimeVLA-Flash`
> **关联**: [`roadmap.md`](roadmap.md) §3.1 阶段5 "#3 Flash 推测推理" (本文把它从"研究性、边际有限"升级为有明确抓手的课题) · [[reference_vision_ablation_openloop]] · `../../strategy/rlt_implementation_plan.md` · `../../strategy/dagger_implementation_plan.md`

---

## 0. FLASH 是什么 (源码级精读)

FLASH = **首个面向 diffusion/flow-matching VLA 的"投机推理"(speculative inference) 框架**, 思想类比 LLM 的 speculative decoding: 用一个极轻 draft 先猜整条 action chunk, 再用原模型 Action Expert 验证, 只执行"被接受的前缀", 失败则回退全量推理。

| 组件 | 源码 | 机制 |
|---|---|---|
| **Draft 模型** | `src/openpi/models_pytorch/draft.py: DraftChunkHead` | **单层 Gemma decoder** over prefix embedding(VLM 已算好的 img+lang+state token)。`_action_queries` (Embedding[chunk_m]) + `_state_token` (Linear 32→h) + 1×`GemmaDecoderLayer` + `_action_head` (Linear→out_dim=7=6pose+1grip)。**一次前向出整条 chunk, 无 flow-matching 去噪迭代** → 这是提速来源。 |
| **接受准则** | `spec_pi0_pytorch._compute_radius_prefix_acceptance` | **半径前缀接受**: `dist=‖draft−verified‖`, 接受满足 `dist≤τ_radius` 的最长前缀。 |
| **相位感知回退** | `_detect_verify_gripper_switch_any_k` + `_truncate_accepted_prefix_on_gripper_switch` | **夹爪开合事件处截断**: chunk 内夹爪穿越开/合阈值 → 在该处截断接受前缀、强制全量验证。即**绝不在夹爪状态切换(精度关键事件)上做投机**。 |
| **拼接** | `_stitch_radius_prefix_output` | 接受的 draft 前缀 + 全量输出尾段拼成最终 chunk。 |
| **Triton serving** | `scripts/spec/triton/{pi0_spec_infer,triton_pi0_runtime}.py` | 定制 kernel, draft 路径 **7.8ms(2 视角)**, 任务级 **3.04× 加速**。 |
| **Draft 训练** | `enc_cache.py` → `spec_draft_train.py` | 冻结 backbone, 先 dump prefix-embedding cache(sliding-chunk-shift 采样), 再蒸馏 DraftChunkHead 回归 pi0 的 action chunk。成本极低。 |

**结果**: 全量 58.0ms → flash 7.8ms / 任务均 19.1ms; LIBERO 仅掉 0.3pt 成功率; 66.8% 重规划轮由 flash 路径完成; 唯一在 15 m/min 传送带动态任务上拿到非零成功率。

**为何与 deepdive_kai0 高度契合**: FLASH 基座是 **openpi/pi0**, 与 kai0(pi0/pi05)同源; kai0 已有 V1 Triton 路径(P50≈32ms/8×, `v1_triton_log.md`)与 5090 wgmma 调优(`layer_b_plan.md`)。FLASH 的 draft 几乎可直接挂到 kai0 现有栈上。

---

## 1. 课题总览 (按优先级)

| # | 课题 | 类型 | 价值 | 与本仓库结论的关联 |
|---|---|---|---|---|
| **R1** | FLASH 移植到 kai0 pi05 + 接 V1 Triton | 工程+研究 | <10ms 闭环 → 高 Hz 重规划 | roadmap 阶段5 落地 |
| **R2** | **夹爪/相位门控的视觉强验证** | 研究 ⭐ | 治"抓空开环" | [[reference_vision_ablation_openloop]] |
| **R3** | draft↔full 散度作在线不确定性/OOD 信号 | 研究 ⭐ | 主动学习 + RLT 触发 | dagger/rlt plan |
| **R4** | 125Hz 投机重规划 取代 RTC blend | 研究 | 去抖、去 chunk 陈旧 | `ee_stability_layer1.md` RTC |
| **R5** | 投机接受率作开环/因果混淆在线探针 | 研究 | 免费部署门禁 | 视觉消融 SNR |
| **R6** | 5090(Blackwell)kernel 移植 + roofline | 工程 | 复用已有 wgmma 工作 | `layer_b_plan.md` |
| **R7** | draft 作"轻量校正头"绕开重训 | 研究 | 救夹爪漂移数据 | 夹爪校准漂移结论 |
| **R8** | 把投机推理推广到 X-VLA(Florence2) | 研究 | 更慢的模型更受益 | `reference_xvla_inference` |

---

## 2. 课题详述

### R1 — FLASH 移植到 kai0 pi05, 接入 V1 Triton 路径 (落地基座)
- **做什么**: 用 `enc_cache.py`+`spec_draft_train.py` 从 **pi05 flatten-fold ckpt** 蒸馏 `DraftChunkHead`(注意 kai0 是 14D joint / EE6D 输出, `out_dim` 需从 LIBERO 的 7 改成 kai0 的 per-step 维度); 把 `pi0_spec_infer.py` 的投机循环接到 kai0 `serve_policy_v1.py` 的 Triton runtime。
- **为何**: kai0 现 P50≈32ms(3Hz 推理 + RTC 插值)。FLASH 把单轮压到 ~10ms → 可把推理频率拉到几十 Hz, 这是 R2/R4/R5 的前置。
- **实验**: LIBERO 先复现 3.04×; 再在 kai0 vis val 上测 draft 接受率 + 半径分布; 真机 A/B(FLASH vs 现 V1)看任务完成 + EE 抖动。
- **风险/坑**: pi05 vs pi0 的 Action Expert 结构差异; kai0 sm_120 与 FLASH kernel(4090/4090D 调过)的 autotune 不兼容(复用 `layer_b_plan.md` 经验)。

### R2 — 夹爪/相位门控的"视觉强验证" ⭐ (把 FLASH 相位逻辑接到本仓库核心病)
- **观察**: FLASH 已经**在夹爪开合处禁止投机、强制全量验证**(`_truncate_accepted_prefix_on_gripper_switch`)。而本仓库视觉消融实测: **坏模型在夹爪通道几乎不看画面**(置黑 SNR 5.7× vs 好模型 26.6×), 真机"抓空也往下做"。两件事指向同一处: **抓取瞬间是精度+视觉关键事件**。
- **研究问题**: 把"夹爪切换处强制全量"升级为"**强制全量 + 强制重新编码视觉**"(smooth 段可用缓存/降采样视觉省时, 抓取段必须吃最新全分辨率帧)。**能否用这种"相位自适应视觉预算"在不重训主模型的前提下缓解开环抓取失败?**
- **实验**: 在 trace 里按相位统计 draft 接受/回退; 真机对比"全程等额视觉" vs "抓取段视觉强验证"的抓取成功率。
- **价值**: 把推理框架变成**闭环安全机制**, 而非单纯提速。

### R3 — draft↔full 散度作在线不确定性 / OOD / "需要人介入"信号 ⭐
- **机制**: 接受准则里的 `dist=‖draft−full‖`(半径)天然是**每步免费的置信度**: draft 与全量分歧大 = 模型不确定 / 当前帧 OOD。
- **研究问题**: (a) 半径散度是否与抓取失败 / 新场景 / 夹爪漂移帧相关? (b) 能否用"高散度段"**自动触发 DAgger 采集**(active learning, 只在模型没把握处要人接管)或 **RLT critical-phase 介入**(见 `rlt_implementation_plan.md`)?
- **实验**: 离线在 vis val + 真机 trace 上算 per-step 半径, 与失败标注做相关; 设阈值跑一次"散度触发 DAgger"闭环。
- **价值**: 把 FLASH 副产物变成 dagger/rlt 两条 plan 的**触发器**, 三者打通。

### R4 — 125Hz 投机重规划是否可**取代 RTC chunk-blend**
- **背景**: kai0 现在 3Hz 推理 + `StreamActionBuffer` min-jerk 重叠混合(`ee_stability_layer1.md`), 混合带来过 euler-wrap、attractor freeze、走3退1 等一系列 deploy 层补丁。
- **研究问题**: FLASH 若能 100+Hz 每个控制步重规划, **是否可直接丢弃 chunk 重叠混合**(每步都是新鲜 chunk 的第 0 帧)→ 从根上消除陈旧/混合伪影?
- **实验**: 仿真/真机对比 "3Hz+RTC blend" vs "≥30Hz FLASH no-blend" 的 EE 抖动 FFT 主频(沿用 [[feedback_real_machine_oscillation_data_tail]] 的判别)、折返比、任务速度。
- **风险**: 高 Hz 下 proprio 反馈时序与固件 t_motion 必须重新标定(roadmap 真机测试2)。

### R5 — 投机接受率作"开环/因果混淆"在线探针
- **联系**: 本仓库已有**离线**视觉消融门禁(置黑 MAE 比值)。draft 蒸馏自主模型, 若主模型开环(无视视觉), draft 也会无视视觉 → 其**接受率/半径分布可能成为开环的在线指纹**。
- **研究问题**: FLASH 部署统计(接受率、夹爪强制率 `gripper_force_rate`、半径分布)能否在线区分"闭环好模型"与"开环坏模型", 作为**比 MAE 更敏感的实时健康度**?
- **实验**: 对好/坏两个已知 ckpt 跑 FLASH, 比较接受统计与离线视觉消融 SNR 的一致性。

### R6 — FLASH Triton kernel 移植到 5090 (Blackwell) + roofline
- **做什么**: FLASH kernel 在 4090/4090D 调优(`exp/plot_pi0_roofline_4090d.py`)。kai0 已在 5090 上做过 V1 重 autotune + wgmma 升级(`layer_b_plan.md`)。把两边 kernel 互相借鉴, 出 5090 roofline。
- **价值**: 纯工程但能把 7.8ms 进一步压到 Blackwell 上限, 且沉淀可复用 kernel 资产。

### R7 — draft 作"轻量校正头", 绕开主模型重训
- **联系**: 本仓库刚定位 5-18~5-21 **夹爪校准漂移**导致主模型不可用(action↔物理映射错)。重训全模型成本高。
- **研究问题**: draft 头(~百 K 参数, 蒸馏即得)能否被训成一个**校准/校正头**——在不动 pi05 主体的前提下, 用少量正确标定数据把夹爪输出映射拉回真机 firmware? (与 RLT residual 思路互补, 但走推理框架而非 RL)。
- **风险**: draft 设计初衷是"快而粗", 当校正头需验证其精度足够; 可能更适合作 RLT 的 state encoder。

### R8 — 投机推理推广到 X-VLA / 更慢模型
- **联系**: kai0 另有 X-VLA(Florence2 + flow-matching, `reference_xvla_inference`), 比 pi05 更慢更重。FLASH 思路 model-agnostic(只要是 diffusion/flow VLA)。
- **研究问题**: 给 X-VLA 蒸一个 DraftChunkHead 是否能拿到比 pi05 更大的相对加速(基数更慢)? Florence2 prefix embedding 与 Gemma draft 的对接如何设计?

---

## 3. 建议起步顺序

1. **R1**(基座, 2–3 周): 先在 LIBERO 复现, 再蒸 kai0 pi05 draft、接 V1 Triton、出 kai0 接受率基线。
2. 并行 **R5**(几乎零成本, 复用 R1 的统计)验证"接受率↔开环"假说 → 立刻反哺部署门禁。
3. **R2 + R3**(研究主线): 把 FLASH 从"提速"升级为"闭环安全 + 主动学习触发", 直接打本仓库核心病(开环抓取)。
4. R4 / R6 / R7 / R8 视 R1 结果再排。

> **一句话立项理由**: FLASH 与 kai0 同源(pi0), 落地成本低; 而它的两个机制——**半径接受**(免费不确定性)与**夹爪相位强验证**(精度事件保护)——恰好正面命中本仓库这轮诊断出的**开环抓取/视觉脱钩**根问题。提速是表, 闭环安全与在线健康度才是对 deepdive_kai0 更大的价值。
