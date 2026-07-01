# TaskP_local_continuous 训练结果 — 连续夹爪 X-VLA (✅ 完成, loss-only)

> **时间**: 2026-07-01, ~75 min
> **硬件**: sim01 (本地), 1×RTX 5090 32GB (GPU1)
> **训练任务**: 本地跑 TaskP_local_continuous config (action_mode=agibot_ee6d, 20000 步单卡)
> **启动命令**:
> ```bash
> CUDA_VISIBLE_DEVICES=1 python xvla/launch/xvla_train.py \
>   --config TaskP_local_continuous --output_dir xvla/ckpts/xvla_taskp_continuous \
>   --batch_size 6 --grad_checkpointing --workers 4
> ```

---

## 0. 一句话结论

**连续夹爪 X-VLA 训练可行。** Task_P 数据用 `joint_to_ee6d.py --continuous` 保留遥操作原始夹爪值 [0, 0.08]m，用 `agibot_ee6d` action space (MSE loss, 不置零 proprio 夹爪通道) 单卡微调 20000 步:

- 全程 **无 NaN / 无 OOM / 无发散**, 峰值显存 **~27.4 GB** (batch=6, grad-ckpt)
- **Loss 健康收敛**: 79.3 (step 0 freeze) → 5.74 (step 1000 解冻) → 0.42 (5k) → **0.11 (step 18000, 最低)**, plateau mean 0.20
- 速度: **~4.0-7.7 it/s** (解冻后 ~4.0), 全局约 **75 min**
- ⚠️ **loss-only, 无 MAE eval** — 离线 vision-ablation 或真机 A/B 判模型质量。

→ 连续夹爪消除了二值化信息损失，模型能感知当前夹爪开度，理论上可学到跨 chunk 保持闭合。

---

## 1. 实验设定

| 参数 | 值 |
|---|---|
| config_name | `TaskP_local_continuous` |
| action_mode | **agibot_ee6d** (MSE on gripper, 不置零 proprio 夹爪) |
| init | `xvla/xvla_ckpts` (lerobot **xvla-base**) |
| dataset | `TaskP_ee6d_continuous/2026-04-21` — 1 日期, 100 ep, **30175 rows** → static-skip 后 **23777 samples** |
| domain_id | 23 |
| prompt | `"pick and place in box"` |
| action | EE6D 20D, gripper **连续值 [0, 0.08]m** (MSE loss), action_qdur=2.0, chunk=30 |
| use_proprio | True (官方默认) |
| param_groups | 4group_official |
| freeze | 前 1000 步冻 vlm+transformer_core |
| steps | 20,000 |
| batch_size | 6 (实测 ~27.4 GB / 32 GB) |
| lr / warmup / schedule | 1e-4 / 2000 / constant |
| 图像 | ImageNet norm + ColorJitter(0.2) |
| 精度 | bf16 mixed |
| gradient_checkpointing | ✅ (VLM) |
| static_skip | ✅ |
| 数据转换 | `joint_to_ee6d.py --continuous`: 保留原始夹爪 clip [0, 0.08], 不二值化 |

### 与 binary 版 (TaskP_local) 差异

| | Binary (ee6d) | Continuous (agibot_ee6d) |
|---|---|---|
| 夹爪值 | {0, 1} 二值 | [0, 0.08]m 连续 |
| 夹爪 loss | BCEWithLogitsLoss | MSELoss |
| 夹爪 preprocess | **置零** (模型盲) | **不置零** (模型感知) |
| 夹爪 postprocess | sigmoid → (0,1) | 原值透传 |
| domain_id | 22 | 23 |
| 数据目录 | TaskP_ee6d | TaskP_ee6d_continuous |

---

## 2. Loss 曲线

### 2.1 关键里程碑

| step | loss | gnorm | 阶段 |
|:---:|---:|---:|---|
| 0 | 79.32 | 5018.3 | 🧊 全冻 |
| 200 | 30.19 | 2591.1 | 🧊 |
| 500 | 7.12 | 670.5 | 🧊 |
| **1000** | **5.74** | 605.6 | 🔓 **解冻** |
| 2000 | 1.41 | 273.7 | 解冻后快速收敛 |
| 3000 | 0.95 | 101.1 | |
| 5000 | 0.42 | 67.0 | |
| 8000 | 0.31 | 140.1 | |
| 10000 | 0.14 | 63.2 | |
| 12000 | 0.16 | 49.2 | |
| 15000 | 0.19 | — | plateau |
| 18000 | **0.11** | 38.0 | ⭐ 最低点 |
| **20000 (final)** | **~0.17** | ~34 | plateau mean=0.20 |

### 2.2 关键观察

- **无 NaN / 发散**: gnorm 冻结期最高 5168, 解冻后 21–605, 全程可控
- **解冻后陡降**: 5.74→1.41 在 1000 步内, 与 binary 版同模式
- **快速 plateau**: ~5k 步达 0.42, ~10k 步达 0.14, 之后在 0.09–0.47 范围震荡
- **末段仍健康**: last 200 steps mean=0.20, 无过拟合反弹

---

## 3. 与 Binary 版 Loss 对照

> ⚠️ 两版 loss 函数不同 (BCE vs MSE), **数值不可直接对比**。趋势和收敛形态更有意义。

| step | Binary (BCE, ee6d) | Continuous (MSE, agibot_ee6d) |
|:---:|---:|---:|
| 0 | — | 79.32 |
| 200 | 23.22 | 30.19 |
| 1000 | 4.26 | 5.74 |
| 2000 | 1.49 | 1.41 |
| 5000 | 0.41 | 0.42 |
| 10000 | 0.18 | 0.14 |
| 18000 | 0.16 | 0.11 |
| 20000 (plateau mean) | ~0.28 | ~0.20 |

**趋势一致**: 两版收敛形态高度相似 — freeze 期快速下降 → 解冻后陡降 → ~5k 进入 plateau → 末段低位震荡。连续版 plateau 略低 (0.20 vs 0.28), 但不可解读为"更好", 仅说明 MSE 在连续夹爪空间的拟合难度与 BCE 在二值空间相当。

---

## 4. 性能与资源

| 指标 | 值 |
|---|---|
| 训练全时 | ~75 min (20000 步) |
| 速度 (冻结期) | ~7.7 it/s (bs 6) |
| 速度 (解冻后) | ~4.0 it/s (稳定) |
| 峰值显存 | ~27.4 GB / 32 GB (GPU1, bs=6) |
| 训练 GPU | GPU 1 (GPU 0 被其他任务占用) |

---

## 5. Checkpoint 路径

```
本地 (sim01):
  xvla/ckpts/xvla_taskp_continuous/
    ├── config.json
    ├── step_002000/state_dict.pt  (3.52 GB)
    ├── step_004000/…
    ├── step_006000/…
    ├── step_008000/…  (每 2000 步一存)
    ├── step_010000/…
    ├── step_012000/…
    ├── step_014000/…
    ├── step_016000/…
    ├── step_018000/…
    └── step_final/state_dict.pt   ⭐

部署 (已 repack):
  /data1/DATA_IMP/checkpoints/ckpt_xvla/xvla_taskp_continuous_step_final/
    ├── state_dict.pt
    ├── config.json
    └── sidecar.json  (action_mode=agibot_ee6d, deploy_binarize_gripper=false)
```

---

## 6. 结论与下一步

### ✅ 验证通过
1. **agibot_ee6d action space 完全可用**: MSE 连续夹爪 loss 稳定收敛
2. **joint_to_ee6d.py --continuous 数据正确**: 夹爪 43-66 个唯一值, 覆盖 [0, 0.08]m
3. **模型加载兼容**: action_mode override 0 missing/unexpected keys
4. **和 binary 版同等稳定**: 无 NaN/OOM, 收敛形态一致

### ⭐ 连续版理论优势 (待真机验证)
- **proprio 不置零夹爪** → 模型感知当前夹爪开度 → 可学到跨 chunk 保持闭合
- **连续输出** → 夹爪指令可以是任意中间值 → 匹配物体宽度

### 下一步
- **真机部署**: `./xvla/start_xvla_from_ckpt.sh xvla_taskp_continuous_step_final --execute --trace`
  - serve 端自动从 sidecar 读 `deploy_binarize_gripper: false` 关二值化
  - 或显式: `XVLA_SERVER_ARGS='--no-binarize_gripper'`
- 离线 vision-ablation (SNR 门禁)
- 对比 binary vs continuous 真机效果
