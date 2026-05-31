# PyTorch vs JAX 框架对比 — Eval 方法论 + 踩坑 Postmortem (2026-05-31)

> **范围**: 记录 `A_mirror200_pi05_pytorch` 框架对比实验的**完整测试流程、踩的坑、以及方法论教训**。结果数字在 [`../history/experiments/task_a_new_pure_200_new_norm_results.md`](../history/experiments/task_a_new_pure_200_new_norm_results.md) §8;本文档专注 **process / 诊断方法 / 错误轨迹**,供后续做框架/方法对比时避坑。
>
> **一句话**: 同协议三方对比证实 PyTorch 比 JAX 真差 (@50 4.1×),但**主因不是 EMA** (假说被 model-soup 证伪),且**最初的 "7.4×" 是跨协议伪对比**。过程中踩了 6 个坑。

---

## 0. 最终可信结论 (全部同协议 `eval_val_action_mse.py`, 20ep×200f, A_new_pure_200_val)

| ckpt | @1 | @10 | @25 | @50 |
|---|---:|---:|---:|---:|
| **JAX pure_200** | **0.0066** | 0.0074 | 0.0078 | **0.0085** |
| PyTorch plain 50k | 0.0100 | 0.0174 | 0.0258 | 0.0350 |
| PyTorch soup (≈EMA) | 0.0101 | 0.0175 | 0.0259 | 0.0349 |
| **Δ (PyTorch/JAX)** | +52% | +135% | +231% | **+312% (4.1×)** |

- ✅ **EMA 不是主因**: soup ≈ plain (差<1%)。
- ✅ **PyTorch 真差于 JAX**: @50 4.1×,且 gap 随 horizon 单调放大 → 指向 **flow-matching sampler/denoising 实现差异**(rollout 误差累积),待定位。
- ✅ **JAX 同协议 @1=0.0066 ≈ §7 训练记录 0.0065** → 数字自洽可信。

---

## 1. 测试流程 (正确姿势, 复用模板)

### 1.1 model-soup 模拟 EMA (证伪/证实 EMA 假说的廉价工具)

```bash
# 脚本: train_scripts/kai/eval/model_soup_ema_probe.py
# 均匀平均末段 N 个 ckpt 模拟 EMA(decay) 的有效窗 ≈ 1/(1-decay) 步
python train_scripts/kai/eval/model_soup_ema_probe.py \
  --ckpt-root <run_dir> --steps 40000,42000,44000,46000,48000,50000 \
  --ref-step 50000 --out <repo_root>/_soup_40k_50k    # ⚠️ 输出到 tim 可写目录, 不要进 root-owned checkpoints/
```
- EMA(0.9999) 有效窗 ≈ 10000 步 → 取末 10k 的 6 个 ckpt 平均。
- **diff 探针验证 soup 真的平均了**: `|soup−50k| ≈ 0.5×|40k−50k|` (本次 action_in_proj.weight 2.3e-5 vs 4.9e-5)。

### 1.2 同协议 eval (三方都用同一脚本/参数)

```bash
# 脚本真实路径: train_scripts/kai/eval/eval_val_action_mse.py  (不是 kai0/scripts/)
# flag: --config / --ckpt / --val / --n-sample-frames / --prompt / --out
export OPENPI_DATA_HOME=<repo>/openpi_cache
export CUDA_VISIBLE_DEVICES=0          # 容器只暴露 1 张 GPU = index 0
<repo>/kai0/.venv/bin/python train_scripts/kai/eval/eval_val_action_mse.py \
  --config pi05_pytorch_a_new_pure_200 \
  --ckpt <ckpt_dir> --val <val_dir> --n-sample-frames 200 \
  --prompt "Flatten and fold the cloth." --out <repo>/_eval_xxx.json
```
- `create_trained_policy` 自动检测 `model.safetensors` → PyTorch 分支;无则 JAX 分支。**同一脚本评 PyTorch + JAX,保证协议一致**。
- JAX ckpt 自带 `assets/<asset_id>/norm_stats.json`,PyTorch ckpt 没有 → 靠 config repo_id 解析(本次复用同数据集 JAX ckpt 的那份)。

---

## 2. 踩的坑 (6 个, 按出现顺序)

| # | 坑 | 现象 | 根因 | 修复 |
|---|---|---|---|---|
| **1** | **未实测数字写进文档** | §8.4.6 一度填了 soup MAE | 假说推断当结论写,数字没真跑出来 | 撤回,改"待核实";**实测出来才写** |
| **2** | **eval 脚本路径错** | `can't open kai0/scripts/eval_val_action_mse.py` exit 2 | 脚本真实在 `train_scripts/kai/eval/`,不在 `kai0/scripts/` | 用真实路径 |
| **3** | **eval flag 全错** | argparse 报错 / 没跑 | 用了 `--config-name/--ckpt-path`,真实是 `--config/--ckpt/--val/--n-sample-frames/--prompt` | 先 `grep add_argument` 确认 flag |
| **4** | **CUDA_VISIBLE_DEVICES=1 回落 CPU** | log 出现 `No CUDA runtime` + `cpp_CppMicroGemm` CPU autotune,极慢 | 容器只暴露 1 张 GPU (index 0),device 1 不存在 | 设 `=0` 或不设 (policy 默认 auto cuda) |
| **5** | **soup 写进 root-owned 目录** | `PermissionError: .../checkpoints/.../soup_40k_50k` | 训练在容器内以 root 跑,`checkpoints/` 是 root 所有,tim 无法建子目录 | soup 输出改到 tim 可写的 repo 根 |
| **6** | **跨协议伪对比 (最严重)** | §8.3 得出 "PyTorch 比 JAX 差 7.4×" | 拿 PyTorch **训练 inline-eval** (0.0646) 比 JAX **训练 inline-eval** (0.0087),两者 eval 协议未对齐 + ckpt 未同协议重测 | 同一 50k ckpt 独立 eval 实为 0.0350,真 gap 4.1× |

### 坑 6 详解 (核心方法论教训)

同一个 PyTorch 50k ckpt:
- 训练时 inline-eval 记录: @50 = **0.0646**
- 独立 `eval_val_action_mse.py` (20ep×200f): @50 = **0.0350**
- **差 1.8×** —— inline-eval 与独立 eval 的采样/帧选取/val 子集不同。

→ 拿 "PyTorch 训练 inline 0.0646" 比 "JAX 训练 inline 0.0087" = **苹果比橘子**,得出的 7.4× 是假的。真实 gap 必须 **同一脚本、同参数、同 val,把两个 ckpt 都重测一遍**。

---

## 3. 假说证伪记录 (EMA 不是主因)

| 步骤 | 内容 | 结果 |
|---|---|---|
| 推断 | `train_pytorch.py:513` 写死 `"EMA is not supported"`;JAX 用 ema_decay=0.9999 → 怀疑 EMA 缺失是主因 | (代码事实成立) |
| 误判 | 把 "EMA 缺失是主因" 当结论写进文档 (还配了机理: EMA 平滑长 horizon) | 听起来合理 |
| 证伪 | model-soup (≈EMA) 实测 @50=0.0349 ≈ plain 0.0350 (差<1%) | **EMA 假说被打脸** |
| 收益 | soup 廉价确定地证伪假说,**避免了白做 EMA patch + 68h 重训** | ✅ 用对了诊断工具 |

**教训**: 机理推断再合理也要先证再写。soup 是诊断 EMA 类假说的正确廉价工具 —— 改代码重训之前先用它验证。

---

## 4. 部署包 (本次顺带产出)

`kai0/checkpoints/A_mirror200_pi05_pytorch_50k_deploy/` (7.0G, gitignore'd 因在 checkpoints/ 下) —— PyTorch plain 50k 真机 spot-check 包:
- `model.safetensors` (size 校验 = 源) + `metadata.pt` + `assets/a_new_pure_200/norm_stats.json` (md5 `52f3bf5f…`, 与 JAX 源一致) + `DEPLOY_README.md`
- 剥掉了 13.5G optimizer.pt (推理不需要)
- ⚠️ caveat: PyTorch @50 比 JAX 差 4.1× + pure_200 仅 2 天数据 OOD-prone → 这是 spot-check 不是最优部署候选;最优折叠候选是 JAX `task_a_pure200_base_pi05_step49999` (@50=0.0085)

---

## 5. 待办 (定位 PyTorch 真实缺陷, 非 EMA)

| 优先级 | 行动 | 目的 |
|---|---|---|
| ⭐⭐⭐ | 逐行对比 `pi0_pytorch.sample_actions` vs JAX `pi0.sample_actions` denoising loop (num_steps / dt 符号 / noise→action 方向) | gap 随 horizon 放大 → 最可能是采样器 |
| ⭐⭐ | 关掉 PyTorch train-time image aug (crop/rotate/color) 重训一小段,看 @1 是否回到 ~0.0066 | aug 拉低 in-distribution 锐度? |
| — | 真机 spot-check `_deploy_pytorch_pure200_50k` | 确认 offline gap 是否传导到真机 |

---

## 6. 通用 checklist (下次做框架/方法对比前过一遍)

- [ ] **eval 脚本路径 + flag**: 先 `grep add_argument <脚本>` 确认, 别猜
- [ ] **GPU 可见性**: 确认容器暴露几张卡, `CUDA_VISIBLE_DEVICES` 别超范围 (否则回落 CPU)
- [ ] **同协议**: 所有对比 ckpt 用**同一 eval 脚本 + 同参数 + 同 val** 重测; 训练 inline-eval 不能跨 ckpt/跨框架比
- [ ] **写文件权限**: 产物输出到自己可写目录, 别进 root-owned checkpoints/
- [ ] **数字先实测再写**: 假说/推断标 "待验证", 不填占位数字
- [ ] **机理假说先用廉价工具证伪** (如 soup 之于 EMA), 再决定改代码/重训
