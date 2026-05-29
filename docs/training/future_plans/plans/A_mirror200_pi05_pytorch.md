# A_mirror200_pi05_pytorch — pure_200 PyTorch 原生训练对照实验

> **实验目的**: 在 pure_200 数据 (mirror 增强, NEW SOTA 数据集) 上跑 **PyTorch 原生训练**, 与已有 **JAX 原生** ckpt (`task_a_new_pure_200_new_norm`, MAE@1=0.0065) 形成 1:1 对照, **隔离 "训练框架 (JAX vs PyTorch)" 这个变量**.
>
> **更大背景**: 配合 v8 数据 audit 分析 ([../../analysis/vis_v2_full_data_audit.md](../../analysis/vis_v2_full_data_audit.md)), 在数据已知干净/已知 work 的 pure_200 上验证 PyTorch 路径是否能等价 reproduce 真机表现, 然后才能信赖 PyTorch 用于其他实验.
>
> **状态**: ⏳ pending
> **日期**: 2026-05-27
> **关联文档**:
> - 已有 JAX ckpt 报告: [`task_a_new_pure_200_new_norm_results.md`](../../history/experiments/task_a_new_pure_200_new_norm_results.md) (MAE@1=0.0065, SOTA)
> - PyTorch 训练 series 旁路 plan: [`pytorch_native_vis_v2_full.md`](pytorch_native_vis_v2_full.md) (vis_v2_full 数据 R1/R2 PyTorch)
> - 数据侧 audit: [`vis_v2_full_data_audit.md`](../../analysis/vis_v2_full_data_audit.md)

---

## 1. 实验配置

| 项 | 值 |
|---|---|
| **Exp name** | `A_mirror200_pi05_pytorch` |
| **Config name** | `pi05_pytorch_a_new_pure_200` (待加 config.py) |
| **Model** | `pi0_config.Pi0Config(pi05=True)` |
| **训练框架** | **PyTorch DDP** (`scripts/train_pytorch.py`) ⭐ 关键差异 |
| **Init** | `pi05_base` (从 JAX ckpt 一次性转 PyTorch weights) |
| **Dataset** | `A_new_pure_200` (200 ep `-new` 精选 + hflip mirror, ~150K frames) |
| Dataset 路径 (cnsh) | `/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200` |
| Dataset 路径 (cnbj) | `/vePFS-North-E/vis_robot/dataset/KAI0/Task_A/self_built/A_new_pure_200` |
| **Val** | `A_new_pure_200_val` (26 ep, native val 与 JAX 版完全一致) |
| Prompt | `"Flatten and fold the cloth."` |
| `use_delta_joint_actions` | False (absolute, 与 JAX 版一致) |
| **LR schedule** | Cosine, warmup_steps=1_000, peak_lr=**1.5e-5**, decay_steps=50_000, decay_lr=**1.5e-6** |
| **EMA decay** | 0.9999 |
| **Steps** | **50,000** |
| **Batch size** | **128** |
| **Cluster** | 8× NVIDIA GPU (单节点 FSDP, 与 JAX SOTA 配置一致) |
| `fsdp_devices` | 8 |
| `pytorch_training_precision` | `bfloat16` |
| `pytorch_weight_path` | 从 JAX `pi05_base` 转换一次产出的 PyTorch safetensors |
| `inline_eval_n_frames` | 200 (与 JAX SOTA 一致) |
| `inline_eval_every` | 每 4k step |

### 1.1 与 JAX SOTA 配置对比 (隔离唯一变量)

| 维度 | JAX SOTA (`task_a_new_pure_200_new_norm`) | **本实验 (PyTorch)** | 差异 |
|---|---|---|---|
| 框架 | JAX/Flax NNX (`scripts/train.py`) | **PyTorch DDP** (`scripts/train_pytorch.py`) | ⭐ **唯一变量** |
| Model | pi05 | pi05 | 同 |
| Dataset | A_new_pure_200 (200 ep + mirror) | 同 | 同 |
| Init | mixed_1_clean | pi05_base | ⚠️ JAX SOTA 是 mixed_1_clean, 本实验改 pi05_base 因为 (a) PyTorch 直接 init pi05_base 简单 (b) [task_a_new_pure_200_new_norm_results §6](../../history/experiments/task_a_new_pure_200_new_norm_results.md) 实测 pi05_base + 50k 与 mixed_1_clean + 50k final MAE 几乎一致 (0.0065 vs 0.0065) |
| Batch | 120 | 128 | 微差 (8 多, 与 vis_v2_full 一致) |
| LR / step / EMA | 同 | 同 | 同 |
| 集群 | js02 8×A800 | 8× NVIDIA GPU | 类似 |

### 1.2 启动命令模板

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py pi05_pytorch_a_new_pure_200 \
  --exp_name A_mirror200_pi05_pytorch \
  --save_interval 2000
```

### 1.3 待加 config (`kai0/src/openpi/training/config.py`)

```python
TrainConfig(
    name="pi05_pytorch_a_new_pure_200",
    model=pi0_config.Pi0Config(pi05=True),
    data=LerobotAgilexDataConfig(
        repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200",
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
    ),
    pytorch_weight_path="/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot/pi05_base",
    pytorch_training_precision="bfloat16",
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
    ),
    ema_decay=0.9999,
    num_train_steps=50_000,
    keep_period=10_000,
    save_interval=2_000,
    num_workers=8,
    batch_size=128,
    fsdp_devices=8,
    inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val",
    inline_eval_n_frames=200,
    inline_eval_every=4,
),
```

---

## 2. 期望对照点

### 2.1 与已有 JAX pure_200 ckpt 对照

| 项 | JAX SOTA | **PyTorch (本)** 期望 | 验证含义 |
|---|---|---|---|
| Final MAE@1 | 0.0065 (49999) | **应 ≈ 0.0065 (±10%)** | PyTorch 训练等价性 ✓ |
| @50 | 0.0079 | **应 ≈ 0.0079** | long-horizon 等价 |
| 真机表现 | 待测 (JAX ckpt 还没真机 deploy 过, 但 PyTorch 转换 ckpt 应已部分测) | 真机应稳定夹取 | 验证 PyTorch 训练不会引入新问题 |
| 训练时长 | ~16h (js02 resume 22k→49999) | ~30-50h (PyTorch 一般慢 2-3×) | accept |

### 2.2 与已有 PyTorch 转换 ckpt 对照 (隔离 "训练 vs 转换")

如果有现成 JAX→PyTorch 转换的 ckpt:
- JAX→PyTorch 转换 ckpt: 精度有损失 (用户已观察)
- **PyTorch 原生训练 ckpt (本实验)**: 应**优于**转换 ckpt
- 这个对照直接量化 "JAX→PyTorch 转换损失" 的大小

### 2.3 与 v8 数据 audit 衔接

如果 PyTorch 在 pure_200 上能 reproduce JAX SOTA (MAE 一致 + 真机稳定), 则:
- **PyTorch 路径被 validate**
- 后续可以信赖 PyTorch 训练用于其他 dataset (vis_v2_full / A_0423_0527 等)
- 与 `pytorch_native_vis_v2_full.md` (R1/R2) 协同 — 三个 PyTorch 实验形成 series

如果 PyTorch 真机不如 JAX SOTA → PyTorch 路径有问题 (不只是数据问题), 需要排查 `train_pytorch.py` 实现

---

## 3. 实施步骤

| Step | 内容 | ETA |
|---|---|---:|
| T1 | 把 §1.3 config 加到 `config.py` | 0.5h |
| T2 | 从 JAX `pi05_base` 转 PyTorch 权重 (`pytorch_weight_path` 指向); 验证 PI0Pytorch.from_pretrained 路径 | 2-4h |
| T3 | 1 GPU smoke test (~1k step, 看 loss 曲线 + 速度 + 无 NaN) | 4-6h |
| T4 | 8 GPU full run 50k step | ~30-50h |
| T5 | Offline eval on A_new_pure_200_val 26 ep + 跨 val (vis_v2_val50, 与 JAX SOTA 一致) | 2h |
| T6 | 与 JAX SOTA 数字对照 + 写 results.md | 1h |
| T7 | (可选) 部署 ckpt 到 sim01 真机测试, 与 JAX 转换 ckpt 对照 | 1 day |

---

## 4. 风险 + 应对

| # | 风险 | 应对 |
|---|---|---|
| 1 | PyTorch 训练 final MAE > JAX SOTA (loss landscape 实现差异) | 接受 ≤10% 差距; >10% 则 debug `train_pytorch.py` |
| 2 | PyTorch 训练慢 (50k step 可能 ~30-50h vs JAX 16h) | 接受, 用 bf16 + DDP + grad_ckpt 优化 |
| 3 | `pytorch_weight_path` init 转换损失 | T2 先做 weight 对比 (init ckpt 一致性 check) |
| 4 | 8 GPU FSDP 配置问题 | 参考 `pytorch_native_vis_v2_full.md` §10.8 已 validated 模板 |
| 5 | OOM (pi05 ~3B + batch 128 + bf16) | grad_ckpt + 必要时 batch 64 + accum 2 |

---

## 5. 决策点

- **D1**: T3 smoke test 接受标准: 同 step loss 差 < 5% vs JAX (如有同 step JAX log 可对); 速度 < 3× JAX
- **D2**: T5 final MAE 接受标准: vs JAX SOTA MAE@1 = 0.0065 ± 10% (即 [0.00585, 0.00715])
- **D3**: T7 真机表现 ≥ JAX 转换 ckpt → PyTorch 路径 validated, 后续实验可用 PyTorch

---

## 6. 后续 (T6/T7 完成后)

| 后续 | 用途 |
|---|---|
| 移到 `history/experiments/A_mirror200_pi05_pytorch_results.md` | 实验完成后归档 |
| 更新 `00_training_history.md` 排行榜 | 加 PyTorch 行 |
| 触发 `pytorch_native_vis_v2_full.md` R1/R2 启动? | 如果 pure_200 PyTorch 路径 validated, R1/R2 可放心启动 |
