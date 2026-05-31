# TAC v2 是否有效? — pure_200 上 tac_v2 vs 非 TAC baseline (2026-05-29)

> **主线问题**: bug 修复后的 TAC (`pi05_flatten_fold_a_new_pure_200_tac_v2`) 相比同数据的非 TAC baseline (`task_a_new_pure_200`) 到底有没有效果? 真机上"看不出区别"。
>
> **结论先行**: **有效,但只在它设计的"喂前缀 (RTC)"场景**。
> - 无前缀单步精度: TAC v2 在过拟合 native val 上比 baseline 略差 ~18% (正则化代价)。
> - 喂干净前缀 (TAC 真正的训练目标): **只有 TAC v2 能用前缀**——d=8 时 TAC postfix MAE 改善 +14.4% (降到 0.0044, 比 baseline 无前缀的 0.0049 还低); 同样喂前缀 baseline 直接 OOD 崩溃 (-595%)。
> - 之前"TAC 全面更差 2–4×"是**双重错误**: ① 连续帧采样假象; ② 用"无前缀"指标评判一个"有前缀"模型 (错误 regime)。
>
> **关联**: [data_scale_vs_quality_vis_v2_full_vs_pure_200.md §A.3](data_scale_vs_quality_vis_v2_full_vs_pure_200.md) (TAC pi0.py:335 convention bug) · [task_a_new_pure_200_new_norm_results.md](../history/experiments/task_a_new_pure_200_new_norm_results.md) (baseline)

---

## 1. 对比对象 (同数据, 唯一变量 = TAC)

| 维度 | baseline | TAC v2 |
|---|---|---|
| config | `pi05_flatten_fold_a_new_pure_200_js`* | `pi05_flatten_fold_a_new_pure_200_tac_v2` |
| model | `Pi0Config(pi05=True)` | `Pi0Config(pi05=True, tac_enabled=True, tac_max_delay=6)` |
| 数据 | A_new_pure_200 (200 ep) | A_new_pure_200 (200 ep) — **同数据** |
| init | mixed_1_clean | pi05_base (init 差异 50k step 后基本消除, 见 data_scale §A.2) |
| steps | 50k | 50k |
| 本地 ckpt | `ckpt_v0/task_a_new_pure_200_step49999` | `ckpt_v0/pi05_flatten_fold_a_new_pure_200_tac_v2_step49999` |

\* 本地 sidecar 用 `pi05_flatten_fold_a_new_pure_1200` + `override_asset_id=a_new_pure_200` 加载 (同架构)。两者 **norm_stats 字节级相同** (Δmean=Δstd=0) → 归一化一致, 非加载问题。

TAC 机制 (pi0.py `compute_loss`): 每样本随机延迟 d∈[0,6], 前缀 [0:d] 用干净 GT 动作 + per-token time=0, 后段 [d:] 用采样 time, loss 只算后段。**推理 `sample_actions` 代码与非 TAC 完全一致** → TAC 只改了训练学到的权重。

---

## 2. 三轮评测

### 2.1 单步 clean MAE (标准协议, 已验证可信)

`eval_val_action_mse.py`, A_new_pure_200_val 全 20 ep × linspace 120 帧, random noise:

| H | baseline | TAC v2 | 比值 |
|---:|---:|---:|---:|
| @1 | 0.0064 | 0.0079 | 1.23× |
| @10 | 0.0072 | 0.0085 | 1.18× |
| @25 | 0.0075 | 0.0085 | 1.14× |
| @50 | 0.0078 | 0.0092 | 1.18× |

baseline @1=0.0064 与训练 inline-eval 0.0065 吻合 → 流水线验证通过。TAC v2 各 horizon 均匀差 ~18%, **非崩溃**。native val 本身过拟合 (data_scale Lesson #2), 正则化模型 (TAC) 在其上略差是预期。

### 2.2 ⚠️ 连续帧假象 (方法论教训)

最初用"连续中段帧 + fixed noise"采样, 得 baseline @1=0.0013 / TAC @1=0.0024 (1.85×)、@50 差 4×。改标准 linspace 协议后 baseline @1 回到 0.0064 (=训练值)。**连续中段帧既压低绝对值, 又放大模型间相对差距** (clean MAE 相对差从真实 1.23× 夸大到 1.85×; @50 夸大到 4×)。同批连续帧采的 P1/P2 (chunk 连续性/噪声方差) 同样被夸大, **不作为判据**。

### 2.3 ⭐ Faithful prefix-conditioned (TAC 真实场景)

`eval_tac_faithful.py`, 4 ep × 10 帧, fixed noise。给模型喂干净 GT 前缀 [0:d] (per-token time=0, clamp), 只去噪后段, 量后段 [d:50] 相对 GT 的 raw-space MAE:

| 延迟 d | 模型 | 无前缀 | 喂干净前缀 | 改善 |
|---:|---|---:|---:|---:|
| 8 | baseline | 0.0049 | **0.0338** | **-594.8%** 💥 |
| 8 | **TAC v2** | 0.0052 | **0.0044** | **+14.4%** ✅ |
| 16 | baseline | 0.0049 | 0.0439 | -790.1% 💥 |
| 16 | TAC v2 | 0.0052 | 0.0067 | -29.5% |

读法:
- **给 baseline 喂干净前缀 = 灾难** (postfix MAE 炸 7–9×)。非 TAC 模型训练时从没见过"前缀=干净 GT、time=0"的输入 → 彻底 OOD。
- **只有 TAC v2 能用前缀**: d=8 喂前缀让 postfix MAE 降到 **0.0044**, 不仅优于自己无前缀 (0.0052), 甚至优于 baseline 无前缀最好成绩 (0.0049)。
- TAC `tac_max_delay=6` → 甜区是**短前缀 d≤~8**; d=16 已超训练范围开始退化 (-29.5%), 但远好于 baseline 的 -790%。

---

## 3. 对部署 (RTC) 的意义

真机 RTC (见 `docs/deployment/inference/rtc_implementation.md`) 每次 replan 把上一 chunk 作为 `prev_action_chunk` 喂入、`latency_k=8`。这正落在 TAC 甜区 (d≤~8):
- **TAC v2**: 能稳定利用已提交前缀 → chunk 切换更连贯。
- **baseline + 强 RTC 引导**: 引导把模型拉向 prev_chunk, 但模型本身对"已提交前缀"是 OOD → 容易出不稳定。

→ 真机"看不出区别"是因为单看无前缀行为二者接近; **TAC 的价值要在 RTC 开启 + 有延迟时才显现**。建议真机 A/B 用 `rtc_apply.sh` 对照 `off` vs `on`, 比较 TAC v2 与 baseline 在 RTC 下的 chunk 平滑度/抖动。

---

## 4. 方法与可复现

| 产物 | 路径 |
|---|---|
| 标准 clean MAE | `train_scripts/kai/eval/eval_val_action_mse.py` (+ `OPENPI_EXTRA_CONFIG` sidecar) |
| P1/P2 诊断 (连续帧, 仅参考) | `train_scripts/kai/eval/eval_tac_diagnostic.py` |
| ⭐ faithful prefix-conditioned | `train_scripts/kai/eval/eval_tac_faithful.py` |
| 结果 JSON | `/tmp/cleanmae_{baseline,tac_v2}.json`, `/tmp/faithful_{baseline,tac_v2}.json` |

**代码改动 (向后兼容)**: `src/openpi/models/pi0.py` 的 `Pi0.sample_actions` 加了 keyword-only `tac_prefix=None, tac_delay=0` (默认关闭, 非 TAC 路径逐字节不变; 唯一外部调用 `scripts/train.py` 不传; `Pi0RTC` 有自己的 `sample_actions` 不受影响)。eval 用 `pol._sample_actions(..., tac_prefix=, tac_delay=)` 调用。

**两个踩坑** (复现必读):
1. **模型内部动作空间 ≠ 简单 z-score**: agilex `data_transforms` 会重排/变换动作。前缀必须用 `pol._input_transform({...,"actions":raw_gt})["actions"]` 拿模型空间; 对比必须用 `pol._output_transform` 回 raw。手动 `(a-mean)/std` 会得假性 0.52 MAE。
2. **只有 `policy._model` 经 `module_jit` (即 `pol._sample_actions`/`pol.infer`) 的实例可信**; eager / 重新 load / 重新 `nnx.split` 的实例都出垃圾 (0.52)。
