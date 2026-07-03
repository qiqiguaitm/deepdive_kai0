# pi05 v4 AWBC — DCT frequency-loss vs baseline 训练结果记录

**状态**: ✅ 两者均训练完成 (49999 / 50000, 100%)，2026-07-03 完成平滑度 + 离线 MAE 对照
**核心问题**: VLANeXt 频域 DCT loss 是否真的提升了动作输出平滑度，且不损精度？
**结论**: ✅ **DCT 在 4 项平滑度指标上一致改善（chunk 内方向反转 −14.6% 最强），离线 MAE 与 baseline 持平（≤+0.9%，噪声内）。零精度代价换平滑度提升。真机为终判（未做）。**

---

## 1. 背景与实验设计

两个 run 是**严格的单变量对照**：除 `use_dct_loss` 外逐字段一致（同 v4 数据 / 同 init / 同 hparams / 同 50k step / 同 cnsh 8×A100）。

| 项 | baseline (`pi05_v4_awbc`) | DCT (`pi05_v4_awbc_dct`) |
|---|---|---|
| 数据 | `A_v4_base_dagger` (2017 ep / 2.385M frames, v4 修正夹爪 action 范围) | 同 |
| init | warm-start `pi05_base/params` | 同 |
| model | `Pi0Config(pi05=True)` | `Pi0Config(pi05=True, **use_dct_loss=True**)` ⭐唯一变量 |
| AWBC prompt | `prompt_from_task=True`，正/负优势 prompt（discretize top-30% 为 positive） | 同 |
| LR / step / bs | Cosine warmup1k/peak1.5e-5/decay50k/end1.5e-6, 50k, bs128/fsdp8 | 同 |
| 集群 / job | cnsh robot-task 8×A100 | cnsh robot-task 8×A100 |

**DCT loss 机理**：沿 action-chunk 时间轴做 DCT-II 变换，对高频系数施加惩罚（论文默认权重 low/high = 1.0/0.2，总权重 0.1），压制 chunk 内动作的高频往复抖动。默认关闭，仅此 config 启用，向后兼容。plan: `future_plans/plans/vlanext_dct_then_soft_connection_plan.md` Step 1。

**收敛**：DCT run loss 0.686 → 49999 时 `dct_loss=0.0028` / `main_loss≈0.0031`，健康单调收敛。两 run 均保留 ckpt `10000/20000/30000/40000/49999`。

⚠️ **inline-eval 无 MAE 曲线**：两 run 的 inline-eval val (`vis_v2_merged_val`) 无优势 prompt → 每步报 `[inline-eval] failed: Prompt is required`，故训练时无 logged MAE（非训练问题）。下方 MAE 为训练后离线补测。

---

## 2. 平滑度对照 (eval_oscillation_diag.py)

**设置**: A_v4_base_dagger 6 ep，prompt=`Flatten and fold the cloth. Advantage: positive`，stride=10 (replan 节奏)，本地 2×A100 分别推理。指标定义见 `train_scripts/kai/eval/eval_oscillation_diag.py`（12 个非夹爪关节 JIDX）。

| 指标 | 含义 | baseline | DCT | Δ | 方向 |
|---|---|---|---|---|---|
| **chunk_rev_fixed** | chunk 内每关节方向反转数（↓更平滑，最直接抖动度量） | 12.87 | **10.99** | **−14.6%** | ✅ |
| **reversal_rate** | replan 接缝掉头率（↓更平滑） | 0.322 | **0.301** | **−6.4%** | ✅ |
| **chunk_ng_fixed** | chunk 内净位移/路程（↑更平滑，越接近 1 越少往复） | 0.536 | **0.560** | +4.4% | ✅ |
| **exec_ng_fixed** | 模拟执行轨迹净/总位移（↑更平滑） | 0.0650 | **0.0668** | +2.8% | ✅ |
| gt_ng | GT 参考（净/总，越接近越好） | 0.0895 | 0.0895 | — | 参考 |
| noise_chunk_mae | 多模态坍缩探针（fixed vs random noise 差异） | 0.0206 | 0.0206 | 持平 | — |

**读法**: 4 项平滑度指标**全部朝"更平滑"方向、无一反例** → DCT 频域高频惩罚起作用。最强信号是 chunk 内方向反转 **−14.6%**（每关节每 50 步 chunk 平均少约 1.9 次掉头），正是 DCT loss 直接压制的量。`noise_chunk_mae` 持平 → 平滑非靠动作坍缩换来。

⚠️ **残余**: `exec_ng_fixed` 两者仍远低于 `gt_ng`（0.067 vs 0.0895）→ 模拟执行轨迹的"走一步退一步"只被缓解未消除，残余抖动集中在 replan 接缝处。

---

## 3. 离线 MAE 对照 (eval_val_action_mse.py)

**设置**: v4 mini-val（A_v4_base_dagger 均匀采样 15 ep × 40 帧/ep，正优势 prompt），归一化动作空间 MAE。⚠️ **held-in**（取自训练集）→ 仅反映拟合质量非泛化；两模型同数据同评故横向可比，但绝对值偏乐观。

| Horizon | baseline | DCT | Δ |
|---|---|---|---|
| H=1 | 0.0090 | 0.0090 | 0.0% |
| H=10 | 0.0154 | 0.0155 | +0.6% |
| H=25 | 0.0201 | 0.0202 | +0.5% |
| H=50 | 0.0234 | 0.0236 | +0.9% |

**DCT 与 baseline MAE 几乎完全一致**（各 horizon ≤+0.9%，采样噪声内）。

---

## 4. 综合结论

| 维度 | 结果 |
|---|---|
| 平滑度 | ✅ 全面改善（reversal −14.6% / 接缝掉头 −6.4% / chunk 净总 +4.4%） |
| 精度 (MAE) | ⚖️ 与 baseline 持平（≤+0.9%） |
| 净判断 | **DCT 用几乎零精度代价换来一致平滑度提升，达到设计目的** |

两条独立证据链闭合：`noise_chunk_mae` 持平 → 平滑非靠动作坍缩；离线 MAE 持平 → 平滑非靠牺牲跟踪精度。

⚠️ **保留项**: (1) MAE 为 held-in，非泛化；(2) `exec_ng` 仍低于 GT，残余接缝抖动未消除；(3) **真机是最终判据（未做）** —— offline 平滑改善能否转化为真机抓衣角更稳，需真机对照验证。

---

## 5. ckpt 路径（best = final 49999，两 run 均保留 10k/20k/30k/40k/49999）

- **baseline**: `kai0/checkpoints/pi05_v4_awbc/pi05_v4_awbc/49999/params`
- **DCT** ⭐: `kai0/checkpoints/pi05_v4_awbc_dct/pi05_v4_awbc_dct/49999/params`

（绝对路径前缀 `/vePFS/tim/workspace/deepdive_kai0/`，cnsh 本地，root-owned；step 目录已补 sidecar `train_config.json` 供离线 eval 用。）
